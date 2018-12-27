import sys
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
from numpy.random import *
from pylab import *
from PIL import Image
import tempfile
import random
from region_to_bbox import region_to_bbox
import tensorflow as tf
from siamese_net import SiameseNet


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def getSubWinTracking(img, pos, modelSz, originalSz, avgChans):
    if originalSz is None:
        originalSz = modelSz

    sz = originalSz
    im_sz = img.shape
    # make sure the size is not too small
    assert min(im_sz[:2]) > 2, "the size is too small"
    c = (np.array(sz) + 1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c[1])
    context_xmax = context_xmin + sz[1] - 1
    context_ymin = round(pos[0] - c[0])
    context_ymax = context_ymin + sz[0] - 1
    left_pad = max(0, int(-context_xmin))
    top_pad = max(0, int(-context_ymin))
    right_pad = max(0, int(context_xmax - im_sz[1] + 1))
    bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))

    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)

    if top_pad or left_pad or bottom_pad or right_pad:
        r = np.pad(img[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[0])
        g = np.pad(img[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[1])
        b = np.pad(img[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[2])
        r = np.expand_dims(r, 2)
        g = np.expand_dims(g, 2)
        b = np.expand_dims(b, 2)

        # h, w = r.shape
        # r1 = np.zeros([h, w, 1], dtype=np.float32)
        # r1[:, :, 0] = r
        # g1 = np.zeros([h, w, 1], dtype=np.float32)
        # g1[:, :, 0] = g
        # b1 = np.zeros([h, w, 1], dtype=np.float32)
        # b1[:, :, 0] = b

        img = np.concatenate((r, g, b ), axis=2)

    im_patch_original = img[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
    if not np.array_equal(modelSz, originalSz):
        im_patch = cv2.resize(im_patch_original, modelSz)
        # im_patch_original = im_patch_original/255.0
        # im_patch = transform.resize(im_patch_original, modelSz)*255.0
        # im = Image.fromarray(im_patch_original.astype(np.float))
        # im = im.resize(modelSz)
        # im_patch = np.array(im).astype(np.float32)
    else:
        im_patch = im_patch_original

    # im_patch = im_patch[:, :, ::-1]
    # im_patch[:, :, 0] = im_patch[:, :, 0] - 103.939
    # im_patch[:, :, 1] = im_patch[:, :, 1] - 116.779
    # im_patch[:, :, 2] = im_patch[:, :, 2] - 123.68
    return im_patch, im_patch_original

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y

def trackerEval(score, sx, targetPosition, window, hp,design):
    # responseMaps = np.transpose(score[:, :, :, 0], [1, 2, 0])
    responseMaps = score[:,:,:,0]
    upsz = design['score_sz']*hp['response_up']
    # responseMapsUp = np.zeros([opts['scoreSize']*opts['responseUp'], opts['scoreSize']*opts['responseUp'], opts['numScale']])
    responseMapsUP = []

    if hp['scale_num'] > 1:
        currentScaleID = int(hp['scale_num']/2)
        bestScale = currentScaleID
        bestPeak = -float('Inf')

        for s in range(hp['scale_num']):
            if hp['response_up'] > 1:
                responseMapsUP.append(cv2.resize(responseMaps[s, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC))
            else:
                responseMapsUP.append(responseMaps[s, :, :])

            thisResponse = responseMapsUP[-1]

            if s != currentScaleID:
                thisResponse = thisResponse*hp['scale_penalty']

            thisPeak = np.max(thisResponse)
            if thisPeak > bestPeak:
                bestPeak = thisPeak
                bestScale = s

        responseMap = responseMapsUP[bestScale]
    else:
        responseMap = cv2.resize(responseMaps[0, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
        bestScale = 0

    responseMap = responseMap - np.min(responseMap)
    responseMap = responseMap/np.sum(responseMap)

    responseMap = (1-hp['window_influence'])*responseMap+hp['window_influence']*window
    rMax, cMax = np.unravel_index(responseMap.argmax(), responseMap.shape)
    pCorr = np.array((rMax, cMax))
    dispInstanceFinal = pCorr-int(upsz/2)
    dispInstanceInput = dispInstanceFinal*design['tot_stride']/hp['response_up']
    dispInstanceFrame = dispInstanceInput*sx/design['search_sz']
    newTargetPosition = targetPosition+dispInstanceFrame
    # print(bestScale)

    return newTargetPosition, bestScale

def makeScalePyramid(im, targetPosition, in_side_scaled, out_side, avgChans, numScale):
    """
    computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
    and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.
    """
    in_side_scaled = np.round(in_side_scaled)
    max_target_side = int(round(in_side_scaled[-1]))
    min_target_side = int(round(in_side_scaled[0]))
    beta = out_side / float(min_target_side)
    # size_in_search_area = beta * size_in_image
    # e.g. out_side = beta * min_target_side
    search_side = int(round(beta * max_target_side))
    search_region, _ = getSubWinTracking(im, targetPosition, (search_side, search_side),
                                              (max_target_side, max_target_side), avgChans)

    assert round(beta * min_target_side) == int(out_side)

    tmp_list = []
    tmp_pos = ((search_side - 1) / 2., (search_side - 1) / 2.)
    for s in range(numScale):
        target_side = round(beta * in_side_scaled[s])
        tmp_region, _ = getSubWinTracking(search_region, tmp_pos, (out_side, out_side), (target_side, target_side),
                                               avgChans)
        tmp_list.append(tmp_region)

    pyramid = np.stack(tmp_list)

    return pyramid

def _init_video(video):
    root_dataset = '/home/xiaobai/dataset/VOT18/'
    video += '/'
    video_folder = os.path.join(root_dataset, video,'color')
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(root_dataset, video,'color', '') + s for s in frame_name_list]
    frame_name_list.sort()
    #with Image.open(frame_name_list[0]) as img:
    img = Image.open(frame_name_list[0])
    frame_sz = np.asarray(img.size)
    frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(root_dataset,video, 'groundtruth.txt')
    gt_tmp = np.genfromtxt(gt_file, delimiter=',')
    if len(gt_tmp.shape) < 2:
        gt_tmp = np.genfromtxt(gt_file)

    n_frames = len(frame_name_list)
    gt = np.zeros((n_frames,4))
    gt[:, 0] = np.min(gt_tmp[:, 0:8:2], axis=1)
    gt[:, 1] = np.min(gt_tmp[:, 1:8:2], axis=1)
    gt[:, 2] = np.max(gt_tmp[:, 0:8:2], axis=1)
    gt[:, 3] = np.max(gt_tmp[:, 1:8:2], axis=1)
    gt[:, 2] = gt[:, 2] - gt[:, 0]
    gt[:, 3] = gt[:, 3] - gt[:, 1]
    if n_frames > len(gt):
        frame_name_list = frame_name_list[:len(gt)]

    return gt, frame_name_list, frame_sz, n_frames

def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/float(np.size(new_distances)) * 100.0

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/float(np.size(new_distances))

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

def getOpts(opts):
    print("config opts...")

    opts['numScale'] = 3
    opts['scaleStep'] = 1.0375
    opts['scalePenalty'] = 0.9745
    # opts['scalePenalty'] = 1/0.9745
    opts['scaleLr'] = 0.59
    opts['responseUp'] = 16
    opts['windowing'] = 'cosine'
    opts['wInfluence'] = 0.176
    opts['exemplarSize'] = 127
    opts['instanceSize'] = 239
    opts['scoreSize'] = 17
    opts['totalStride'] = 8
    opts['contextAmount'] = 0.5
    opts['trainWeightDecay'] = 5e-04
    opts['stddev'] = 0.03
    opts['subMean'] = False

    opts['video'] = 'vot15_bag'
    opts['modelPath'] = './models/'
    opts['modelName'] = opts['modelPath']+"model_tf.ckpt"
    opts['summaryFile'] = './data_track/'+opts['video']+'_20170518'

    return opts


def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

if __name__ == '__main__':
    # titles = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark', 'CarScale', 'Coke', 'Couple', 'Crossing', 'David',
    #           'David2',
    #           'David3', 'Deer', 'Dog1', 'Doll', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace', 'Football',
    #           'Football1',
    #           'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Ironman', 'Jogging_1','Jogging_2', 'Jumping', 'Lemming', 'Liquor',
    #           'Matrix', 'Mhyang',
    #           'MotorRolling', 'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Subway',
    #           'Suv', 'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman']

    titles = os.listdir('/home/xiaobai/dataset/VOT18')
    titles.sort()
    hp = {}
    hp['scale_min'] = 0.2
    hp['window_influence'] = 0.25
    hp['z_lr'] = 0.01
    hp['scale_max'] = 5
    hp['scale_step'] = 1.04
    hp['scale_num'] = 3
    hp['scale_penalty'] = 0.97
    hp['response_up'] = 8
    hp['scale_lr'] = 0.59

    evaluation = {}
    evaluation['start_frame'] = 0
    evaluation['n_subseq'] = 1
    evaluation['stop_on_failure'] = 0
    evaluation['dist_threshold'] = 20

    design = {}
    design['exemplar_sz'] = 127
    design['search_sz'] = 239
    design['tot_stride'] = 4
    design['context'] = 0.5
    design['pad_with_image_mean'] = True
    design['windowing'] = 'cosine_sum'
    design['score_sz'] = 33
    design['trainBatchSize'] = 8


    opts = {}
    opts = getOpts(opts)

    exemplarOp = tf.placeholder(tf.float32, [1, design['exemplar_sz'], design['exemplar_sz'], 3])
    instanceOp = tf.placeholder(tf.float32, [hp['scale_num'], design['search_sz'], design['search_sz'], 3])
    exemplarOpBak = tf.placeholder(tf.float32, [design['trainBatchSize'], design['exemplar_sz'], design['exemplar_sz'], 3])
    instanceOpBak = tf.placeholder(tf.float32, [design['trainBatchSize'], design['search_sz'], design['search_sz'], 3])
    isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')

    sn = SiameseNet()

    scoreOp = sn.buildTrainNetwork(exemplarOpBak, instanceOpBak, opts)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./summaryFile/summary')
    sess = tf.Session()
    saver.restore(sess, './ckpt/ckpt/model_epoch49.ckpt')

    zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts, isTrainingOp)

    nv = len(titles)
    speed = np.zeros(nv * evaluation['n_subseq'])
    precisions = np.zeros(nv * evaluation['n_subseq'])
    precisions_auc = np.zeros(nv * evaluation['n_subseq'])
    ious = np.zeros(nv * evaluation['n_subseq'])
    lengths = np.zeros(nv * evaluation['n_subseq'])
    for i in range(nv):
        titles[i] = 'car1'
        gt, frame_name_list, frame_sz, n_frames = _init_video(titles[i])
        starts = np.rint(np.linspace(0, n_frames - 1, evaluation['n_subseq'] + 1))
        starts = starts[0:evaluation['n_subseq']]
        final_score_sz = hp['response_up'] * (design['score_sz'] - 1) + 1
        for j in range(evaluation['n_subseq']):
            idx = i * evaluation['n_subseq'] + j
            start_frame = int(starts[j])
            gt_ = gt[start_frame:, :]
            frame_name_list_ = frame_name_list[start_frame:]
            pos_x, pos_y, target_w, target_h = region_to_bbox(gt_[0])
            '''----------------------tracking single video------------------------'''
            num_frames = np.size(frame_name_list)
            scale_factors = hp['scale_step'] ** np.linspace(-np.ceil(hp['scale_num'] / 2), np.ceil(hp['scale_num'] / 2),
                                                            hp['scale_num'])
            hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
            penalty = np.transpose(hann_1d) * hann_1d
            penalty = penalty / np.sum(penalty)
            context = design['context'] * (target_w + target_h)
            z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
            scalez = design['exemplar_sz'] / z_sz
            im = cv2.imread(frame_name_list[start_frame], cv2.IMREAD_COLOR)
            avgChans = np.mean(im, axis=(0, 1))

            zCrop, _ = getSubWinTracking(im, [pos_y, pos_x], (design['exemplar_sz'], design['exemplar_sz']),
                                         (np.around(z_sz), np.around(z_sz)), avgChans)

            dSearch = (design['search_sz'] - design['exemplar_sz']) / 2
            pad = dSearch / scalez
            sx = z_sz + 2 * pad

            minSx = 0.2 * sx
            maxSx = 5.0 * sx
            winSz = design['score_sz'] * hp['response_up']

            hann = np.hanning(winSz).reshape(winSz, 1)
            window = hann.dot(hann.T)

            window = window / np.sum(window)
            scales = np.array([hp['scale_step'] ** k for k in
                               range(int(np.ceil(hp['scale_num'] / 2.0) - hp['scale_num']),
                                     int(np.floor(hp['scale_num'] / 2.0) + 1))])
            zCrop = np.expand_dims(zCrop, axis=0)
            zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop.astype('float')})
            zFeat1 = np.transpose(zFeat, [1, 2, 3, 0])
            zFeat = zFeat1.copy()
            #zFeatConstantOp = tf.constant(zFeat, dtype=tf.float32)
            zFeatConstantOp = tf.placeholder(tf.float32,(6,6,256,1))
            scoreOp = sn.buildInferenceNetwork(instanceOp, zFeatConstantOp, opts, isTrainingOp)
            writer.add_graph(sess.graph)

            bBoxes = np.zeros([len(frame_name_list) - start_frame, 4])
            bBoxes[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
            targetSize = np.array([target_h, target_w])
            t_start = time.time()
            for im_id in range(start_frame + 1, len(frame_name_list)):
                im = cv2.imread(frame_name_list[im_id], cv2.IMREAD_COLOR)
                scaledInstance = sx * scales
                scaledTarget = np.array([targetSize * scale_i for k, scale_i in enumerate(scales)])
                xCrops = makeScalePyramid(im, [pos_y, pos_x], scaledInstance, design['search_sz'], avgChans,
                                          hp['scale_num'])
                score = sess.run(scoreOp, feed_dict={instanceOp: xCrops,zFeatConstantOp:zFeat1})
                #np.max(score)
                newTargetPosition, newScale = trackerEval(score, round(sx), [pos_y, pos_x], window, hp, design)
                targetPosition = newTargetPosition
                sx = max(minSx, min(maxSx, (1 - hp['scale_lr']) * sx + hp['scale_lr'] * scaledInstance[newScale]))

                targetSize = (1 - hp['scale_lr']) * targetSize + hp['scale_lr'] * scaledTarget[newScale]

                rectPosition = targetPosition - targetSize / 2.
                tl = tuple(np.round(rectPosition).astype(int)[::-1])
                br = tuple(np.round(rectPosition + targetSize).astype(int)[::-1])


                imDraw = im.astype(np.uint8)
                cv2.rectangle(imDraw, tl, br, (0, 255, 255), thickness=3)
                cv2.imshow("tracking", imDraw)
                cv2.waitKey(1)
                bBoxes[im_id-start_frame, :] = tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]
                pos_y = tl[1] + (br[1] - tl[1])/2
                pos_x = tl[0] + (br[0] - tl[0])/2





            t_elapsed = time.time() - t_start
            speed_i = num_frames / t_elapsed

            speed[idx] = speed_i


            lengths[idx], precisions[idx], precisions_auc[idx], ious[idx] = _compile_results(gt_, bBoxes,evaluation['dist_threshold'])
            print str(i) + ' -- ' + titles[i] + \
                  ' -- Precision: ' + "%.2f" % precisions[idx] + \
                  ' -- Precisions AUC: ' + "%.2f" % precisions_auc[idx] + \
                  ' -- IOU: ' + "%.2f" % ious[idx] + \
                  ' -- Speed: ' + "%.2f" % speed[idx] + ' --'
            print



    tot_frames = np.sum(lengths)
    mean_precision = np.sum(precisions * lengths) / tot_frames
    mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
    mean_iou = np.sum(ious * lengths) / tot_frames
    mean_speed = np.sum(speed * lengths) / tot_frames
    print '-- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(tot_frames) + ' frames) --'
    print ' -- Precision ' + "(%d px)" % evaluation['dist_threshold'] + ': ' + "%.2f" % mean_precision + \
          ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc + \
          ' -- IOU: ' + "%.2f" % mean_iou + \
          ' -- Speed: ' + "%.2f" % mean_speed + ' --'
    print