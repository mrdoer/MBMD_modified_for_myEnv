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
from vggm import vggM
from sample_generator import *
from tracking_utils import *
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def extract_regions(image,samples,crop_size=107,padding=16,shuffle=False):
    regions = np.zeros((samples.shape[0], crop_size,crop_size,3), dtype = 'uint8')
    for t in range(samples.shape[0]):
        regions[t] = crop_image(image,samples[t],crop_size,padding)

    regions = regions - 128.
    return regions

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

    try:
        assert dist >= 0
        assert dist != float('Inf')
    except AssertionError:
        print a,b

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

    mdnet = vggM()
    imageOp = tf.placeholder(dtype=tf.float32, shape=(20, 107, 107, 3))
    outputsOp = mdnet.vggM(imageOp)

    featInputOp = tf.placeholder(dtype=tf.float32, shape=(128, 3, 3, 512))
    labelOp = tf.placeholder(dtype=tf.float32, shape=(128, 2))
    lrOp = tf.placeholder(tf.float32, )
    logitsOp = mdnet.classification(featInputOp)
    lossOp = mdnet.loss(logitsOp, labelOp)
    optimizer_vggm1 = tf.train.MomentumOptimizer(learning_rate=lrOp, momentum=0.9)
    trainable_vars_vggm = tf.trainable_variables()
    vggMTrainableVars1 = [var for var in trainable_vars_vggm if (var.name.startswith("VGGM") )]
    trainVGGMGradOp1 = optimizer_vggm1.compute_gradients(lossOp, var_list=vggMTrainableVars1)
    trainVGGMOp = optimizer_vggm1.apply_gradients(trainVGGMGradOp1)

    # optimizer_vggm2 = tf.train.MomentumOptimizer(learning_rate=lrOp*10.0, momentum=0.9)
    # vggMTrainableVars2 = [var for var in trainable_vars_vggm if
    #                       (var.name.startswith("VGGM/layer6"))]
    # trainVGGMGradOp2 = optimizer_vggm2.compute_gradients(lossOp, var_list=vggMTrainableVars2)
    # trainVGGMOp2 = optimizer_vggm2.apply_gradients(trainVGGMGradOp2)
    # trainVGGMOp = tf.group(trainVGGMOp1,trainVGGMOp2)

    imageOp1 = tf.placeholder(dtype=tf.float32, shape=(256, 107, 107, 3))
    featOp = mdnet.extractFeature(imageOp1)

    all_vars = tf.global_variables()
    vggMVars = [var for var in all_vars if (var.name.startswith("VGGM"))]
    vggMVarsRestore = [var for var in all_vars if (var.name.startswith("VGGM") and not var.name.endswith("Momentum:0"))]
    vggMSaver = tf.train.Saver(var_list=vggMVarsRestore)

    init_fn = tf.variables_initializer(var_list=vggMVars)
    sess.run(init_fn)

    image_root = '/home/xiaobai/dataset/VOT18_long/'
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
        gt = gt_tmp.copy()
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

        pos_examples = gen_samples(SampleGenerator('gaussian', init_img.size, 0.1, 1.2),gt[0], 500, [0.7, 1])
        pos_regions = extract_regions(im,pos_examples)
        pos_regions = pos_regions[:,:,:,::-1]

        neg_examples = np.concatenate([
            gen_samples(SampleGenerator('uniform', init_img.size, 1, 2, 1.1),gt[0], 5000 // 2, [0, 0.5]),
            gen_samples(SampleGenerator('whole', init_img.size, 0, 1.2, 1.1),gt[0], 5000 // 2, [0, 0.5])])
        neg_regions = extract_regions(im,neg_examples)
        neg_regions = neg_regions[:,:,:,::-1]

        vggMSaver.restore(sess,'./ckpt/VGGM/vggMParams.ckpt')


        neg_features = np.zeros((5000,3,3,512))
        pos_features = np.zeros((500,3,3,512))
        num_iter = 5000 / 256
        for t in range(num_iter):
            neg_features[t*256:(t+1)*256,:,:,:] = sess.run(featOp,feed_dict={imageOp1:neg_regions[t*256:(t+1)*256,:,:,:]})
        residual = 5000 - 256*num_iter
        tmp = 256 / residual + 1
        tmp1 = np.tile(neg_regions[num_iter*256:,:,:,:],(tmp,1,1,1))
        tmp1 = sess.run(featOp,feed_dict={imageOp1:tmp1[:256,:,:,:]})
        neg_features[num_iter*256:,:,:,:] = tmp1[:residual,:,:,:]

        num_iter = 500 / 256
        for t in range(num_iter):
            pos_features[t*256:(t+1)*256,:,:,:] = sess.run(featOp,feed_dict={imageOp1:pos_regions[t*256:(t+1)*256,:,:,:]})
        residual = 500 - 256*num_iter
        tmp = 256 / residual + 1
        tmp1 = np.tile(pos_regions[num_iter*256:,:,:,:],(tmp,1,1,1))
        tmp1 = sess.run(featOp,feed_dict={imageOp1:tmp1[:256,:,:,:]})
        pos_features[num_iter*256:,:,:,:] = tmp1[:residual,:,:,:]
        labels1 = np.array([0, 1])
        labels1 = np.reshape(labels1, (1, 2))
        labels1 = np.tile(labels1, (32, 1))
        labels2 = np.array([1, 0])
        labels2 = np.reshape(labels2, (1, 2))
        labels2 = np.tile(labels2, (96, 1))
        labels = np.concatenate((labels1, labels2), axis=0)

        for iter in range(30):
            pos_feat = np.random.randint(0,500,32)
            pos_feat = pos_features[pos_feat]
            neg_feat = np.random.randint(0,5000,96)
            neg_feat = neg_features[neg_feat]
            featInputs = np.concatenate((pos_feat,neg_feat), axis=0)

            _,loss1,logits1 = sess.run([trainVGGMOp,lossOp,logitsOp],feed_dict={featInputOp:featInputs,labelOp:labels,lrOp: 0.0001})

        tmp1 = np.random.randint(0,500,50)
        pos_feat_record = pos_features[tmp1,:,:,:]
        tmp1 = np.random.randint(0,5000,200)
        neg_feat_record = neg_features[tmp1,:,:,:]

        save_path = 'result_img/' +title + '/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        bBoxes = np.zeros((num_frames,4))
        bBoxes[0,:] = init_gt
        t_start = time.time()
        target_w = init_gt[3] - init_gt[1]
        target_h = init_gt[2] - init_gt[0]
        for i in range(1,num_frames):
            frame_path = image_path + frame_list[i]
            cur_ori_img = Image.open(frame_path)
            if expand_channel:
                cur_ori_img = np.array(cur_ori_img)
                cur_ori_img = np.expand_dims(cur_ori_img, axis=2)
                cur_ori_img = np.repeat(cur_ori_img, repeats=3, axis=2)
                cur_ori_img = Image.fromarray(cur_ori_img)
            cur_ori_img_array = np.array(cur_ori_img)

            cropped_img, last_gt_norm, win_loc, scale = crop_search_region(cur_ori_img, last_gt, 300, mean_rgb=128)
            cur_img_array = np.array(cropped_img)
            detection_box_ori, scores = sess.run([pre_box_tensor, scores_tensor],
                                                  feed_dict={input_cur_image: cur_img_array,
                                                             initConstantOp: initfeatures1})
            # detection_box = detection_box[0]

            detection_box_ori[:, 0] = detection_box_ori[:, 0] * scale[0] + win_loc[0]
            detection_box_ori[:, 1] = detection_box_ori[:, 1] * scale[1] + win_loc[1]
            detection_box_ori[:, 2] = detection_box_ori[:, 2] * scale[0] + win_loc[0]
            detection_box_ori[:, 3] = detection_box_ori[:, 3] * scale[1] + win_loc[1]

            search_box1 = detection_box_ori[:20]
            search_box = np.zeros_like(search_box1)
            search_box[:,1] = search_box1[:,0]
            search_box[:,0] = search_box1[:,1]
            search_box[:,2] = search_box1[:,3]
            search_box[:,3] = search_box1[:,2]
            haha = np.ones_like(search_box[:,2]) *3
            search_box[:,2] = search_box[:,2] - search_box[:,0]
            search_box[:,3] = search_box[:,3] - search_box[:,1]
            search_box[:,2] = np.maximum(search_box[:,2],haha)
            search_box[:,3] = np.maximum(search_box[:,3],haha)
            haha2 = np.zeros_like(search_box[:, 0])
            search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
            search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
            haha = np.ones_like(search_box[:,2]) * cur_ori_img.width -1 - search_box[:,2]
            search_box[:, 0] = np.minimum(search_box[:, 0], haha)
            haha2 = np.ones_like(search_box[:,3]) * cur_ori_img.height -1 - search_box[:,3]
            search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

            search_regions = extract_regions(cur_ori_img_array, search_box)
            search_regions = search_regions[:,:,:,::-1]
            mdnet_scores = sess.run(outputsOp,feed_dict={imageOp:search_regions})
            mdnet_scores = mdnet_scores[:,1]
            max_idx = np.argmax(mdnet_scores)
            if mdnet_scores[max_idx] < 0:
                overlap = []
                for t in range(20):
                    x1 = max(last_gt[1], detection_box_ori[t, 1])
                    y1 = max(last_gt[0], detection_box_ori[t, 0])
                    x2 = min(last_gt[3], detection_box_ori[t, 3])
                    y2 = min(last_gt[2], detection_box_ori[t, 2])
                    tmp = (x2 - x1) * (y2 - y1)
                    if tmp > 0:
                        overlap.append(tmp / float((last_gt[2] - last_gt[0]) * (last_gt[3] - last_gt[1]) + (
                                detection_box_ori[t, 3] - detection_box_ori[t, 1]) * (
                                detection_box_ori[t, 2] - detection_box_ori[t, 0]) - tmp))

                rank = np.argsort(scores)
                k = 20
                candidates = rank[0, -k:]
                pixel_count = np.zeros((k,))
                for ii in range(k):
                    bb = detection_box_ori[candidates[ii], :].copy()
                    x1 = max(last_gt[1], bb[1])
                    y1 = max(last_gt[0], bb[0])
                    x2 = min(last_gt[3], bb[3])
                    y2 = min(last_gt[2], bb[2])
                    pixel_count[ii] = (x2 - x1) * (y2 - y1) / float(
                        (last_gt[2] - last_gt[0]) * (last_gt[3] - last_gt[1]) + (bb[3] - bb[1]) * (bb[2] - bb[0]) - (
                                x2 - x1) * (y2 - y1))

                threshold = 0.4
                passed = pixel_count > (threshold)
                if np.sum(passed) > 0:
                    candidates_left = candidates[passed]
                    max_idx = candidates_left[np.argmax(scores[0, candidates_left])]
                else:
                    max_idx = 0

            detection_box = detection_box_ori[max_idx]

            if scores[0, max_idx] < 0.3:
                search_gt = (np.array(last_gt)).copy()
                # search_gt = last_gt.copy()
                search_gt[0] = cur_ori_img.height / 2.0 - (last_gt[2] - last_gt[0]) / 2.0
                search_gt[2] = cur_ori_img.height / 2.0 + (last_gt[2] - last_gt[0]) / 2.0
                search_gt[1] = cur_ori_img.width / 2.0 - (last_gt[3] - last_gt[1]) / 2.0
                search_gt[3] = cur_ori_img.width / 2.0 + (last_gt[3] - last_gt[1]) / 2.0

                cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                                   mean_rgb=128)
                cur_img_array1 = np.array(cropped_img1)
                detection_box_ori1, scores1 = sess.run([pre_box_tensor, scores_tensor],
                                                   feed_dict={input_cur_image: cur_img_array1,
                                                              initConstantOp: initfeatures1})
                if scores1[0, 0] > 0.8:
                    scores = scores1.copy()
                    detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
                    detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
                    detection_box_ori = detection_box_ori1.copy()
                    max_idx = 0

                    search_box1 = detection_box_ori[:20]
                    search_box = np.zeros_like(search_box1)
                    search_box[:, 1] = search_box1[:, 0]
                    search_box[:, 0] = search_box1[:, 1]
                    search_box[:, 2] = search_box1[:, 3]
                    search_box[:, 3] = search_box1[:, 2]
                    haha = np.ones_like(search_box[:, 2]) *3

                    search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
                    search_box[:, 3] = search_box[:, 3] - search_box[:, 1]
                    search_box[:, 2] = np.maximum(search_box[:, 2], haha)
                    search_box[:, 3] = np.maximum(search_box[:, 3], haha)
                    haha2 = np.zeros_like(search_box[:, 0])
                    search_box[:,0] = np.maximum(search_box[:,0],haha2)
                    search_box[:,1] = np.maximum(search_box[:,1],haha2)

                    haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
                    search_box[:, 0] = np.minimum(search_box[:, 0], haha)
                    haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
                    search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

                    search_regions = extract_regions(cur_ori_img_array, search_box)
                    mdnet_scores = sess.run(outputsOp, feed_dict={imageOp: search_regions})
                    mdnet_scores = mdnet_scores[:, 1]
                    max_idx = np.argmax(mdnet_scores)
                    if mdnet_scores[max_idx] < 0:
                        max_idx = 0
                    detection_box = detection_box_ori[max_idx]

            if scores[0,max_idx] > 0.8:
                gt_tmp = np.array([detection_box[1],detection_box[0],detection_box[3]-detection_box[1],detection_box[2]-detection_box[0]])
                pos_examples1 = gen_samples(SampleGenerator('gaussian', cur_ori_img.size, 0.1, 1.2), gt_tmp, 50, [0.7, 1])
                pos_regions1 = extract_regions(cur_ori_img_array, pos_examples1)
                pos_regions1 = pos_regions1[:, :, :, ::-1]
                neg_examples2 = np.zeros((50,4))
                count = 0
                t = 0
                while count < 50 and t < 100:
                    x1 = max(detection_box[1], detection_box_ori[t,1])
                    y1 = max(detection_box[0],detection_box_ori[t,0])
                    x2 = min(detection_box[3],detection_box_ori[t,3])
                    y2 = min(detection_box[2],detection_box_ori[t,2])
                    tmp1 = (x2-x1)*(y2-y1)
                    tmp = tmp1 / float((detection_box[2]-detection_box[0])*(detection_box[3]-detection_box[1]) + (detection_box_ori[t,2]-detection_box_ori[t,0]) * (detection_box_ori[t,3]-detection_box_ori[t,1]) - tmp1)
                    if tmp < 0.5 and (detection_box_ori[t,3]-detection_box_ori[t,1]) > 0 and (detection_box_ori[t,2] - detection_box_ori[t,0]) > 0:
                        neg_examples2[count,0] = detection_box_ori[t,1]
                        neg_examples2[count,1] = detection_box_ori[t,0]
                        neg_examples2[count,2] = detection_box_ori[t,3] - detection_box_ori[t,1]
                        neg_examples2[count,3] = detection_box_ori[t,2] - detection_box_ori[t,0]
                        if neg_examples2[count,0] < 0:
                            neg_examples2[count,0] = 0
                        if neg_examples2[count,1] < 0:
                            neg_examples2[count,1] = 0
                        if neg_examples2[count,2] < 1:
                            neg_examples2[count,2] = 1
                        if neg_examples2[count,3] < 1:
                            neg_examples2[count,3] = 1
                        if neg_examples2[count,0] > cur_ori_img.width-1-neg_examples2[count,2]:
                            neg_examples2[count,0] = cur_ori_img.width-1-neg_examples2[count,2]
                        if neg_examples2[count,1] > cur_ori_img.height-1-neg_examples2[count,3]:
                            neg_examples2[count,1] = cur_ori_img.height-1-neg_examples2[count,3]
                        count += 1

                    t+=1

                neg_examples1 = gen_samples(SampleGenerator('uniform', cur_ori_img.size, 1.5, 1.2), gt_tmp, 200-neg_examples2.shape[0], [0, 0.5])
                neg_examples1 = np.concatenate((neg_examples1,neg_examples2), axis=0)
                neg_regions1 = extract_regions(cur_ori_img_array, neg_examples1)
                neg_regions1 = neg_regions1[:, :, :, ::-1]

                tmp_regions = np.concatenate((pos_regions1,neg_regions1,pos_regions1[:6]),axis=0)
                feat1 = sess.run(featOp,feed_dict={imageOp1:tmp_regions})
                pos_feat1 = feat1[:50,:,:,:]
                neg_feat1 = feat1[50:250,:,:,:]
                pos_feat_record = np.concatenate((pos_feat_record,pos_feat1),axis=0)
                neg_feat_record = np.concatenate((neg_feat_record,neg_feat1),axis=0)

                if pos_feat_record.shape[0] > 5*50+1:
                    pos_feat_record = pos_feat_record[50:,:,:,:]
                    neg_feat_record = neg_feat_record[200:,:,:,:]


            if np.mod(i,10) == 0:
                for iter in range(15):
                    pos_feat = np.random.randint(0, pos_feat_record.shape[0], 32)
                    pos_feat = pos_feat_record[pos_feat]
                    neg_feat = np.random.randint(0, neg_feat_record.shape[0], 96)
                    neg_feat = neg_feat_record[neg_feat]
                    featInputs = np.concatenate((pos_feat, neg_feat), axis=0)

                    _, loss1, logits1 = sess.run([trainVGGMOp, lossOp, logitsOp],
                                                 feed_dict={featInputOp: featInputs, labelOp: labels, lrOp: 0.0002})

            if scores[0,max_idx] < 0.3:
                x_c = (detection_box[3]+detection_box[1])/2.0
                y_c = (detection_box[0]+detection_box[2])/2.0
                w1 = last_gt[3]-last_gt[1]
                h1 = last_gt[2]-last_gt[0]
                x1 = x_c - w1/2.0
                y1 = y_c - h1/2.0
                x2 = x_c + w1/2.0
                y2 = y_c + h1/2.0
                last_gt = np.float32([y1,x1,y2,x2])
            else:
                last_gt = detection_box
                target_w = detection_box[3]-detection_box[1]
                target_h = detection_box[2]-detection_box[0]

            if last_gt[0] <0:
                last_gt[0] = 0
                last_gt[2] = target_h
            if last_gt[1] < 0:
                last_gt[1] = 0
                last_gt[3] = target_w
            if last_gt[2] > cur_ori_img.height:
                last_gt[2] = cur_ori_img.height-1
                last_gt[0] = cur_ori_img.height-1-target_h
            if last_gt[3] > cur_ori_img.width:
                last_gt[3] = cur_ori_img.width-1
                last_gt[1] = cur_ori_img.width-1 - target_w

            target_h = last_gt[2] - last_gt[0]
            target_w = last_gt[3] - last_gt[1]


            show_res(cur_ori_img_array, np.array(last_gt, dtype=np.int32), '2', score=scores[0,max_idx],frame_id=i,all_frame=num_frames)

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

        index1 = []
        for t in range(gt.shape[0]):
            if np.isnan(gt[t,0]):
                index1.append(t)
        gt = np.delete(gt,index1,axis=0)
        bBoxes_ = np.delete(bBoxes_,index1,axis=0)

        lengths[title_id], precisions[title_id], precisions_auc[title_id], ious[title_id] = _compile_results(gt.astype(np.float32), bBoxes_.astype(np.float32),20)
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
