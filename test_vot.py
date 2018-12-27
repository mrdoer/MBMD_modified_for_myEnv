import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from core.model_builder import build_man_model
from object_detection.core import box_list
from object_detection.core import box_list_ops
from PIL import Image
import scipy.io as sio
import cv2
import os
from region_to_bbox import region_to_bbox
import time
import random
from siamese_net import SiameseNet
from siamese_utils import *
from tracking_utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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


def main(_):
    init_training = True
    config_file = 'model/ssd_mobilenet_tracking.config'
    # checkpoint_dir = '../model/server13_alov'
    # checkpoint_dir = '../model/ssd_mobilenet_alov2'
    # checkpoint_dir = '../model/server13_alov2'
    # checkpoint_dir = '../model/server14_alov2'
    # checkpoint_dir = '../model/server13_alov_2.2'
    # checkpoint_dir = '../model/server13_alov2.4.0'
    # checkpoint_dir = '../model/ssd_mobilenet_alov2.4.1'
    # checkpoint_dir = '../model/server12_alov2.4.3'
    checkpoint_dir = 'model/dump'

    model_config, train_config, input_config, eval_config = get_configs_from_pipeline_file(config_file)
    model = build_man_model(model_config=model_config, is_training=False)
    model_scope = 'model'

    initFeatOp, initInputOp = build_init_graph(model, model_scope, reuse=None)
    initConstantOp = tf.placeholder(tf.float32, [1, 1, 1, 512])
    pre_box_tensor, scores_tensor, input_cur_image = build_box_predictor(model, model_scope,initConstantOp, reuse=None)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    variables_to_restore = tf.global_variables()
    restore_model(sess, model_scope, checkpoint_dir, variables_to_restore)

    hp = {}
    hp['scale_min'] = 0.2
    hp['window_influence'] = 0.175  # 0.175
    hp['z_lr'] = 0.0102  # 0.0102
    hp['scale_max'] = 5
    hp['scale_step'] = 1.0470  # 1.0470
    hp['scale_num'] = 3
    hp['scale_penalty'] = 0.9825  # 0.9825
    hp['response_up'] = 8
    hp['scale_lr'] = 0.68  # 0.68

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
    instanceOp = tf.placeholder(tf.float32, [3, design['search_sz'], design['search_sz'], 3])
    exemplarOpBak = tf.placeholder(tf.float32,
                                   [design['trainBatchSize'], design['exemplar_sz'], design['exemplar_sz'], 3])
    instanceOpBak = tf.placeholder(tf.float32, [design['trainBatchSize'], design['search_sz'], design['search_sz'], 3])
    isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')

    sn = SiameseNet()

    scoreOp = sn.buildTrainNetwork(exemplarOpBak, instanceOpBak, opts)
    variables_to_restore = tf.global_variables()
    variables_to_restore = [var for var in variables_to_restore if (var.name.startswith("siamese") or var.name.startswith("adjust"))]
    saver = tf.train.Saver(var_list=variables_to_restore)
    saver.restore(sess, './ckpt/alexnet4241/model_epoch49.ckpt')

    zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts, isTrainingOp)
    zFeatConstantOp = tf.placeholder(tf.float32, (6, 6, 256, 1))

    scoreOp = sn.buildInferenceNetwork(instanceOp, zFeatConstantOp, opts, isTrainingOp)

    image_root = '/home/xiaobai/dataset/VOT18/'
    titles = os.listdir(image_root)
    titles.sort()
    titles = [title for title in titles if not title.endswith("txt")]
    precisions = np.zeros(len(titles))
    precisions_auc = np.zeros(len(titles))
    ious = np.zeros(len(titles))
    lengths = np.zeros(len(titles))
    speed = np.zeros(len(titles))
    for title_id in range(len(titles)):

        title = titles[title_id]
        #title = 'butterfly'
        image_path = image_root + title+'/color/'
        gt_path = image_root +title+ '/groundtruth.txt'
        try:
            gt_tmp = np.loadtxt(gt_path)
        except:
            gt_tmp = np.loadtxt(gt_path, delimiter=',')
        num_frames = gt_tmp.shape[0]
        gt = np.zeros((num_frames,4))
        gt[:,0] = np.min(gt_tmp[:,0:8:2],axis=1)
        gt[:,1] = np.min(gt_tmp[:,1:8:2], axis=1)
        gt[:,2] = np.max(gt_tmp[:,0:8:2], axis=1)
        gt[:,3] = np.max(gt_tmp[:,1:8:2], axis=1)
        gt[:,2] = gt[:,2] - gt[:,0]
        gt[:,3] = gt[:,3] - gt[:,1]
        #num_frames = gt.shape[0]
        frame_list = os.listdir(image_path)
        frame_list = [frame for frame in frame_list if frame.endswith('jpg')]
        frame_list.sort()
        # init_img = Image.open(image_path + '0001.jpg')
        init_img = Image.open(image_path + frame_list[0])
        init_gt = gt[0]
        init_gt = [init_gt[1], init_gt[0], init_gt[1]+init_gt[3], init_gt[0]+init_gt[2]] # ymin xmin ymax xmax

        # init_gt = [x-20 for x in ini_gt]
        init_img_array = np.array(init_img)
        im = np.array(init_img)

        show_res(init_img_array, np.array(init_gt, dtype=np.int32), '2')

        expand_channel = False
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)
            expand_channel = True

        gt_boxes = np.zeros((1,4))
        gt_boxes[0,0] = init_gt[0] / float(init_img.height)
        gt_boxes[0,1] = init_gt[1] / float(init_img.width)
        gt_boxes[0,2] = init_gt[2] / float(init_img.height)
        gt_boxes[0,3] = init_gt[3] / float(init_img.width)

        init_img_array = crop_init_array(init_img, gt_boxes)
        last_gt = init_gt

        initfeatures1 = sess.run(initFeatOp, feed_dict={initInputOp: init_img_array})

        target_w = init_gt[3] - init_gt[1]
        target_h = init_gt[2] - init_gt[0]
        pos_y = (init_gt[0] + init_gt[2]) / 2.0
        pos_x = (init_gt[1] + init_gt[3]) / 2.0
        context = design['context'] * (target_w + target_h)
        z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
        scalez = design['exemplar_sz'] / z_sz
        avgChans = np.mean(init_img_array, axis=(0, 1))

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
        zFeat = np.transpose(zFeat, [1, 2, 3, 0])
        zFeat_ori = zFeat.copy()

        save_path = 'result_img/' +title + '/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        update_count = 0
        avg_score = 0
        bBoxes = np.zeros((num_frames,4))
        bBoxes[0,:] = init_gt
        t_start = time.time()
        targetSize = np.array([target_h, target_w])
        t_start = time.time()
        for i in range(1,num_frames):
            frame_path = image_path + frame_list[i]
            cur_ori_img = Image.open(frame_path)
            if expand_channel:
                cur_ori_img = np.array(cur_ori_img)
                cur_ori_img = np.expand_dims(cur_ori_img, axis=2)
                cur_ori_img = np.repeat(cur_ori_img, repeats=3, axis=2)
                cur_ori_img = Image.fromarray(cur_ori_img)
            cur_ori_img_array = np.array(cur_ori_img)

            scaledInstance = sx * scales
            scaledTarget = np.array([targetSize * scale_i for k, scale_i in enumerate(scales)])
            xCrops = makeScalePyramid(cur_ori_img_array, [pos_y, pos_x], scaledInstance,
                                      design['search_sz'], avgChans,
                                      hp['scale_num'])
            score = sess.run(scoreOp, feed_dict={instanceOp: xCrops, zFeatConstantOp: zFeat})
            # np.max(score)
            newTargetPosition, newScale = trackerEval(score, round(sx), [pos_y, pos_x], window,
                                                      hp, design)
            targetPosition = newTargetPosition
            sx = max(minSx, min(maxSx,
                                     (1 - hp['scale_lr']) *  sx + hp['scale_lr'] * scaledInstance[
                                         newScale]))

            targetSize = (1 - hp['scale_lr']) * targetSize + hp['scale_lr'] * scaledTarget[newScale]

            rectPosition = targetPosition - targetSize / 2.
            tl = tuple(np.round(rectPosition).astype(int)[::-1])
            br = tuple(np.round(rectPosition + targetSize).astype(int)[::-1])
            pos_y = tl[1] + (br[1] - tl[1]) / 2
            pos_x = tl[0] + (br[0] - tl[0]) / 2

            target_h = targetSize[0]
            target_w = targetSize[1]
            last_gt = np.int32(
                [pos_y - target_h / 2.0, pos_x - target_w / 2.0, pos_y + target_h / 2.0, pos_x + target_w / 2.0])

            cropped_img, last_gt_norm, win_loc, scale = crop_search_region(cur_ori_img, last_gt, 300, mean_rgb=128)
            cur_img_array = np.array(cropped_img)
            detection_box, scores = sess.run([pre_box_tensor, scores_tensor],
                                                  feed_dict={input_cur_image: cur_img_array,
                                                             initConstantOp: initfeatures1})
            # detection_box = detection_box[0]

            detection_box[:, 0] = detection_box[:, 0] * scale[0] + win_loc[0]
            detection_box[:, 1] = detection_box[:, 1] * scale[1] + win_loc[1]
            detection_box[:, 2] = detection_box[:, 2] * scale[0] + win_loc[0]
            detection_box[:, 3] = detection_box[:, 3] * scale[1] + win_loc[1]

            cur_ori_img_array = np.array(cur_ori_img)
            for tmp in range(10):
                distance1 = (detection_box[tmp,0] + detection_box[tmp,2]) /2.0 - pos_y
                distance2 = (detection_box[tmp,1] + detection_box[tmp,3]) /2.0 - pos_x
                distance = np.sqrt(distance1**2 + distance2**2)
                if distance < min(target_w/4.0,target_h/4.0) and scores[0,tmp] > 0.5: #2.0
                    detection_box = detection_box[tmp]
                    pos_y = (detection_box[0] + detection_box[2])/2.0
                    pos_x = (detection_box[1] + detection_box[3]) / 2.0
                    target_h = detection_box[2] - detection_box[0]
                    target_w = detection_box[3] - detection_box[1]

                    context = design['context'] * (target_w + target_h)
                    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
                    scalez = design['exemplar_sz'] / z_sz
                    dSearch = (design['search_sz'] - design['exemplar_sz']) / 2
                    pad = dSearch / scalez
                    sx = z_sz + 2 * pad
                    targetSize = np.array([target_h, target_w])

                    last_gt = np.int32(detection_box)

                    #if np.mod(i,2) == 0:
                    gt_boxes = np.zeros((1, 4))
                    gt_boxes[0, 0] = last_gt[0] / float(cur_ori_img.height)
                    gt_boxes[0, 1] = last_gt[1] / float(cur_ori_img.width)
                    gt_boxes[0, 2] = last_gt[2] / float(cur_ori_img.height)
                    gt_boxes[0, 3] = last_gt[3] / float(cur_ori_img.width)
                    init_img_array = crop_init_array(cur_ori_img, gt_boxes)

                    # target_w = targetSize[1]
                    # target_h = targetSize[0]
                    # context = design['context'] * (target_w + target_h)
                    # z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
                    # zCrop, _ = getSubWinTracking(cur_ori_img_array[:, :, -1::-1], [pos_y, pos_x], (design['exemplar_sz'], design['exemplar_sz']),
                    #                              (np.around(z_sz), np.around(z_sz)), avgChans)
                    # zCrop = np.expand_dims(zCrop, axis=0)
                    # zFeat2 = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop.astype('float')})
                    # zFeat2 = np.transpose(zFeat2, [1, 2, 3, 0])
                    #
                    # #if np.sum(zFeat_ori * zFeat2) > 5.0:
                    # zFeat = zFeat2*hp['z_lr'] + zFeat*(1-hp['z_lr'])

                    break

            show_res(cur_ori_img_array, np.array(last_gt, dtype=np.int32), '2')
            # pos_y = (last_gt[0] + last_gt[2]) / 2.0
            # pos_x = (last_gt[1] + last_gt[3]) / 2.0
            # if np.mod(i,2) == 0:
            #     target_w = targetSize[1]
            #     target_h = targetSize[0]
            #     context = design['context'] * (target_w + target_h)
            #     z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
            #     zCrop, _ = getSubWinTracking(cur_ori_img_array[:, :, -1::-1], [pos_y, pos_x], (design['exemplar_sz'], design['exemplar_sz']),
            #                                  (np.around(z_sz), np.around(z_sz)), avgChans)
            #     zCrop = np.expand_dims(zCrop, axis=0)
            #     zFeat2 = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop.astype('float')})
            #     zFeat2 = np.transpose(zFeat2, [1, 2, 3, 0])
            #     if np.sum(zFeat_ori * zFeat2) > 5.0:
            #         zFeat = zFeat2*hp['z_lr'] + zFeat*(1-hp['z_lr'])



            # x1 = pos_x - targetSize[1]/2.0
            # y1 = pos_y - targetSize[0]/2.0
            # x2 = pos_x + targetSize[1]/2.0
            # y2 = pos_y + targetSize[0]/2.0

            #bBoxes[i,:] = np.int32([y1,x1,y2,x2])
            bBoxes[i, :] = last_gt

        t_elapsed = time.time() - t_start
        speed_i = num_frames / t_elapsed

        speed[title_id] = speed_i
        bBoxes[:,2] = bBoxes[:,2] - bBoxes[:,0]
        bBoxes[:,3] = bBoxes[:,3] - bBoxes[:,1]
        bBoxes_ = np.zeros((num_frames,4))
        bBoxes_[:,0] = bBoxes[:,1]
        bBoxes_[:,1] = bBoxes[:,0]
        bBoxes_[:,2] = bBoxes[:,3]
        bBoxes_[:,3] = bBoxes[:,2]

        lengths[title_id], precisions[title_id], precisions_auc[title_id], ious[title_id] = _compile_results(gt.astype(np.float32), bBoxes_.astype(np.float32),
                                                                                         evaluation['dist_threshold'])
        print str(title_id) + ' -- ' + titles[title_id] + \
              ' -- Precision: ' + "%.2f" % precisions[title_id] + \
              ' -- Precisions AUC: ' + "%.2f" % precisions_auc[title_id] + \
              ' -- IOU: ' + "%.2f" % ious[title_id] + \
              ' -- Speed: ' + "%.2f" % speed[title_id] + ' --'
        print

    tot_frames = np.sum(lengths)
    mean_precision = np.sum(precisions * lengths) / tot_frames
    mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
    mean_iou = np.sum(ious * lengths) / tot_frames
    mean_speed = np.sum(speed * lengths) / tot_frames
    print '-- Overall stats (averaged per frame) on ' + str(len(titles)) + ' videos (' + str(tot_frames) + ' frames) --'
    print ' -- Precision ' + "(20 px)"  + ': ' + "%.2f" % mean_precision + \
          ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc + \
          ' -- IOU: ' + "%.2f" % mean_iou + \
          ' -- Speed: ' + "%.2f" % mean_speed + ' --'
    print

if __name__ == '__main__':
    tf.app.run()
