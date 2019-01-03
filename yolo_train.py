import os
import time
import config
import numpy as np
from PIL import Image
import tensorflow as tf
from dataReader import Reader
from model.yolo3_model import yolo
from collections import defaultdict
from yolo_predict import yolo_predictor
from utils import draw_box, load_weights, letterbox_image, voc_ap
import tensorlayer as tl
from tensorflow.python.client import device_lib
from tensorflow.python import debug as tf_debug
import numpy

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
beta1 = config.beta1
alpha = config.alpha
beta = config.beta


def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num

def train():
    """
    Introduction
    ------------
        训练模型
    """
    # gpu_num = check_available_gpus()
    #
    # for gpu_id in range(int(gpu_num)):
    # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):

        # with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
    # with tf.variable_scope(tf.get_variable_scope(), reuse=False):

    #-----------------------train_data-------------------------
    train_reader = Reader('train', config.data_dir, config.anchors_path2, config.num_classes,
                          input_shape=config.input_shape, max_boxes=config.max_boxes)
    train_data = train_reader.build_dataset(config.train_batch_size)
    is_training = tf.placeholder(tf.bool, shape=[])
    iterator = train_data.make_one_shot_iterator()
    images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()

    #-----------------------  definition-------------------------
    images.set_shape([None, config.input_shape, config.input_shape, 3])
    bbox.set_shape([None, config.max_boxes, 5])
    grid_shapes = [config.input_shape // 32, config.input_shape // 16, config.input_shape // 8]
    lr_images = tf.image.resize_images(images, size=[config.input_shape // 4, config.input_shape // 4], method=0, align_corners=False)
    lr_images.set_shape([None, config.input_shape // 4, config.input_shape // 4, 3])
    bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], 3, 5 + config.num_classes])
    bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], 3, 5 + config.num_classes])
    bbox_true_52.set_shape([None, grid_shapes[2], grid_shapes[2], 3, 5 + config.num_classes])
    bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]


    #------------------------summary + draw-----------------------------------
    tf.summary.image('input1', images, max_outputs=3)
    draw_box(images, bbox)

    #------------------------------model---------------------------------
    model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path2, config.classes_path,
                 config.pre_train)
    # with tf.variable_scope("train_var"):
    # g_img1 = model.GAN_g1(lr_images)
    # print(g_img1.outputs)
    # tf.summary.image('img', g_img1.outputs, 3)
    # g_img2 = model.GAN_g2(g_img1)
    # print(model.g_variables)
    # net_g1 = model.GAN_g1(lr_images, is_train=True)
    with tf.variable_scope("model_gd"):
        net_g1 = model.GAN_g(lr_images, is_train=True, mask=False)
        net_g = model.GAN_g(lr_images, is_train=True, reuse=True, mask=True)

        d_real = model.yolo_inference(images, config.num_anchors / 3, config.num_classes, training=True)
        tf.get_variable_scope().reuse_variables()
        d_fake = model.yolo_inference(net_g.outputs, config.num_anchors / 3, config.num_classes, training=True)

    #---------------------------d_loss---------------------------------
    d_loss1 = model.yolo_loss(d_real, bbox_true, model.anchors, config.num_classes, 1, config.ignore_thresh)
    d_loss2 = model.yolo_loss(d_fake, bbox_true, model.anchors, config.num_classes, 0, config.ignore_thresh)
    d_loss = d_loss1 + d_loss2
    l2_loss = tf.losses.get_regularization_loss()
    d_loss += l2_loss

    #--------------------------g_loss------------------------------------
    adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake[3]), logits=d_fake[3])
    # adv_loss = 1e-3 * tf.reduce_sum(adv_loss) / tf.cast(tf.shape(d_fake[3])[0], tf.float32)
    adv_loss = tf.reduce_sum(adv_loss) / tf.cast(tf.shape(d_fake[3])[0], tf.float32)
    mse_loss1 = tl.cost.mean_squared_error(net_g1.outputs, images, is_mean=True)
    mse_loss1 = tf.reduce_sum(mse_loss1) / tf.cast(tf.shape(net_g1.outputs)[0], tf.float32)
    mse_loss2 = tl.cost.mean_squared_error(net_g.outputs, images, is_mean=True)
    mse_loss2 = tf.reduce_sum(mse_loss2) / tf.cast(tf.shape(net_g.outputs)[0], tf.float32)
    mse_loss = mse_loss1 + mse_loss2
    # clc_loss = 2e-6 * d_loss2
    clc_loss = model.yolo_loss(d_fake, bbox_true, model.anchors, config.num_classes, 1, config.ignore_thresh)
    g_loss = mse_loss + adv_loss + clc_loss
    l2_loss = tf.losses.get_regularization_loss()
    g_loss += l2_loss

    #----------------summary loss-------------------------
    # tf.summary.image('img', images, 3)
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('g_loss', g_loss)
    merged_summary = tf.summary.merge_all()

    #----------------------optimizer---------------------------
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(config.learning_rate, global_step, decay_steps=2000, decay_rate = 0.8)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # 如果读取预训练权重，则冻结darknet53网络的变量
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # print(tf.all_variables())
    with tf.control_dependencies(update_ops):
        if config.pre_train:
            # aaa = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            train_varg1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model_gd/generator/generator1')
            train_varg2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model_gd/generator/generator2')
            train_varg = train_varg1 + train_varg2
            # print(train_varg)
            train_vard = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model_gd/yolo_inference/discriminator')
            # print(train_vard)

            train_opg = optimizer.minimize(loss=g_loss, global_step=global_step, var_list=train_varg)
            train_opd = optimizer.minimize(loss=d_loss, global_step=global_step, var_list=train_vard)
        else:
            train_opd = optimizer.minimize(loss=d_loss, global_step=global_step)
            train_opg = optimizer.minimize(loss=g_loss, global_step=global_step)

    #-------------------------session-----------------------------------
    init = tf.global_variables_initializer()
    # tl.layers.print_all_variables()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        ckpt = tf.train.get_checkpoint_state(config.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('restore model', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        if config.pre_train is True:
            load_ops = load_weights(tf.global_variables(scope='darknet53'), config.darknet53_weights_path)
            sess.run(load_ops)
        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        dloss_value = 0
        gloss_value = 0
        for epoch in range(config.Epoch):
            for step in range(int(config.train_num / config.train_batch_size)):
                start_time = time.time()
                train_dloss, summary, global_step_value, _ = sess.run([d_loss, merged_summary, global_step, train_opd], {is_training : True})
                train_gloss, summary, global_step_value, _ = sess.run([g_loss, merged_summary, global_step, train_opg], {is_training : True})
                dloss_value += train_dloss
                gloss_value += train_gloss
                duration = time.time() - start_time
                examples_per_sec = float(duration) / config.train_batch_size
                print(global_step_value)

    #------------------------print(epoch)--------------------------
                format_str1 = ('Epoch {} step {},  train dloss = {} train gloss = {} ( {} examples/sec; {} ''sec/batch)')
                print(format_str1.format(epoch, step, dloss_value / global_step_value, gloss_value / global_step_value, examples_per_sec, duration))
                # print(format_str1.format(epoch, step, train_dloss, train_gloss, examples_per_sec, duration))

    #----------------------------summary loss------------------------
                summary_writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag = "train dloss", simple_value = train_dloss)]), global_step = step)
                summary_writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag = "train gloss", simple_value = train_gloss)]), global_step = step)
                summary_writer.add_summary(summary, step)
                summary_writer.flush()

    #--------------------------save model------------------------------
            # 每3个epoch保存一次模型
            if epoch % 3 == 0:
                checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = global_step)




def eval(model_path, min_Iou = 0.5, yolo_weights=None):
    """
    Introduction
    ------------
        计算模型在coco验证集上的MAP, 用于评价模型
    """
    ground_truth = {}
    class_pred = defaultdict(list)
    gt_counter_per_class = defaultdict(int)
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    input_image = tf.placeholder(shape=[None, 192, 192, 3], dtype=tf.float32)

    # model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path,
    #              config.pre_train)
    with tf.variable_scope("model_gd"):
        # g_img = model.GAN_g(input_image, mask=True)

        predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path2)
        boxes, scores, classes = predictor.predict(input_image, input_image_shape)
        for ele1 in tf.trainable_variables():
            print(ele1.name)
    val_Reader = Reader("val", config.data_dir, config.anchors_path2, config.num_classes, input_shape=config.input_shape,
                        max_boxes=config.max_boxes)
    image_files, bboxes_data = val_Reader.read_annotations()
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)
        else:
            saver = tf.train.Saver()
            model_file = tf.train.latest_checkpoint(model_path)
            print(model_file)
            saver.restore(sess, model_file)
        for index in range(len(image_files)):
            val_bboxes = []
            image_file = image_files[index]
            file_id = os.path.split(image_file)[-1].split('.')[0]
            for bbox in bboxes_data[index]:
                left, top, right, bottom, class_id = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
                class_name = val_Reader.class_names[int(class_id)]
                bbox = [float(left), float(top), float(right), float(bottom)]
                val_bboxes.append({"class_name": class_name, "bbox": bbox, "used": False})

                gt_counter_per_class[class_name] += 1
                # print(gt_counter_per_class[class_name])

            ground_truth[file_id] = val_bboxes
            # for ele1 in tf.trainable_variables():
            #     print(ele1.name)

            image = Image.open(image_file)
            resize_image = letterbox_image(image, (192, 192))
            image_data = np.array(resize_image, dtype=np.float32)
            image_data /= 255.
            image_data = np.expand_dims(image_data, axis=0)
            # print(image_data)


            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })
            # print(out_boxes)
            print(image_file)
            print(out_scores)
            print(out_classes)

            print("detect {}/{} found boxes: {}".format(index, len(image_files), len(out_boxes)))

            for o, c in enumerate(out_classes):
                predicted_class = val_Reader.class_names[c]
                box = out_boxes[o]
                score = out_scores[o]

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                bbox = [left, top, right, bottom]
                class_pred[predicted_class].append({"confidence": str(score), "file_id": file_id, "bbox": bbox})
                # print(class_pred[predicted_class])
        # print(ground_truth['000000026204'])

    # 计算每个类别的AP
    sum_AP = 0.0
    count_true_positives = {}
    for class_index, class_name in enumerate(sorted(gt_counter_per_class.keys())):
        count_true_positives[class_name] = 0
        predictions_data = class_pred[class_name]
        # 该类别总共有多少个box
        nd = len(predictions_data)

        tp = [0] * nd  # true positive
        fp = [0] * nd  # false positive
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction['file_id']
            ground_truth_data = ground_truth[file_id]
            bbox_pred = prediction['bbox']
            Iou_max = -1
            gt_match = None
            for obj in ground_truth_data:
                if obj['class_name'] == class_name:
                    bbox_gt = obj['bbox']
                    bbox_intersect = [max(bbox_pred[0], bbox_gt[0]), max(bbox_gt[1], bbox_pred[1]),
                                      min(bbox_gt[2], bbox_pred[2]), min(bbox_gt[3], bbox_pred[3])]
                    intersect_weight = bbox_intersect[2] - bbox_intersect[0] + 1
                    intersect_high = bbox_intersect[3] - bbox_intersect[1] + 1
                    if intersect_high > 0 and intersect_weight > 0:
                        union_area = (bbox_pred[2] - bbox_pred[0] + 1) * (bbox_pred[3] - bbox_pred[1] + 1) + (
                                    bbox_gt[2] - bbox_gt[0] + 1) * (
                                                 bbox_gt[3] - bbox_gt[1] + 1) - intersect_weight * intersect_high
                        Iou = intersect_high * intersect_weight / union_area
                        if Iou > Iou_max:
                            Iou_max = Iou
                            gt_match = obj
            if Iou_max > min_Iou:
                if not gt_match['used'] and gt_match is not None:
                    tp[idx] = 1
                    gt_match['used'] = True
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1
        # 计算精度和召回率
        sum_class = 0
        for idx, val in enumerate(fp):
            fp[idx] += sum_class
            sum_class += val
        sum_class = 0
        for idx, val in enumerate(tp):
            tp[idx] += sum_class
            sum_class += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = tp[idx] / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = tp[idx] / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap
    MAP = sum_AP / len(gt_counter_per_class) * 100
    print("The Model Eval MAP: {}".format(MAP))

    # val_Reader = Reader("val", config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes)
    # image_files, bboxes_data, labels, labels_text, _, _ = val_Reader.read_annotations(config.data_dir, config.voc2007test_dir)
    # print(image_files)
    #
    # with tf.Session() as sess:
    #     if yolo_weights is not None:
    #         with tf.variable_scope('predict'):
    #             boxes, scores, classes = predictor.predict(input_image, input_image_shape, is_reuse=False)
    #             print("qwert")
    #         load_op = load_weights(tf.global_variables(scope = 'predict'), weights_file = yolo_weights)
    #         sess.run(load_op)
    #         print("ppppppppppppppppppp")
    #     else:
    #         saver = tf.train.Saver()
    #         model_file = tf.train.latest_checkpoint(model_path)
    #         saver.restore(sess, model_file)
    #     for index in range(len(image_files)):
    #         val_bboxes = []
    #         image_file = image_files[index]
    #         file_id = os.path.split(image_file)[-1].split('.')[0]
    #         bid = 0
    #         # print(bboxes_data[index])
    #         # print(bboxes_data[index+1])
    #         for bbox in bboxes_data:
    #             # print(bbox)
    #             left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    #
    #             # class_id = labels[bid]
    #             class_name = labels_text[bid]
    #             bid += 1
    #             bbox = [left, top, right, bottom]
    #             val_bboxes.append({"class_name" : class_name, "bbox": bbox, "used": False})
    #             gt_counter_per_class[class_name] += 1
    #         ground_truth[file_id] = val_bboxes
    #         image = Image.open(image_file)
    #         resize_image = letterbox_image(image, (416, 416))
    #         image_data = np.array(resize_image, dtype = np.float32)
    #         image_data /= 255.
    #         image_data = np.expand_dims(image_data, axis = 0)
    #         print("ooooooooooooooo")
    #         out_boxes, out_scores, out_classes = sess.run(
    #             [boxes, scores, classes],
    #             feed_dict = {
    #                 input_image : image_data,
    #                 input_image_shape : [image.size[1], image.size[0]]
    #             })
    #         # print(out_boxes)
    #         print("detect {}/{} found boxes: {}".format(index, len(image_files), len(out_boxes)))
    #         for o, c in enumerate(out_classes):
    #             predicted_class = val_Reader.class_names[c]
    #             # print(c)
    #             box = out_boxes[o]
    #             score = out_scores[o]
    #
    #             top, left, bottom, right = box
    #             # print(box)
    #             top = max(0, np.floor(top + 0.5).astype('int32'))
    #             # print(top)
    #             left = max(0, np.floor(left + 0.5).astype('int32'))
    #             # print(left)
    #             bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    #             # print(bottom)
    #             right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    #             # print(right)
    #
    #             bbox = [left, top, right, bottom]
    #             # print(bbox)
    #             class_pred[predicted_class].append({"confidence": str(score), "file_id": file_id, "bbox": bbox})
    #             # print(predicted_class)
    #             # print(class_pred['dog'])
    #
    # # 计算每个类别的AP
    # sum_AP = 0.0
    # count_true_positives = {}
    # for class_index, class_name in enumerate(sorted(gt_counter_per_class.keys())):
    #     # print(gt_counter_per_class.keys())
    #     # print(class_name)
    #     class_name = class_name.decode('utf-8')
    #     print(class_name)
    #
    #     count_true_positives[class_name] = 0
    #     predictions_data = class_pred[class_name]
    #     # predictions_data = class_pred['train']
    #     # print(predictions_data)
    #     # 该类别总共有多少个box
    #     nd = len(predictions_data)
    #     # print(nd)
    #     tp = [0] * nd  # true positive
    #     fp = [0] * nd  # false positive
    #     for idx, prediction in enumerate(predictions_data):
    #         file_id = prediction['file_id']
    #         ground_truth_data = ground_truth[file_id]
    #         # print(ground_truth_data)
    #         bbox_pred = prediction['bbox']
    #         print(bbox_pred)
    #         Iou_max = -1
    #         gt_match = None
    #         for obj in ground_truth_data:
    #             if obj['class_name'].decode('utf-8') == class_name:
    #                 # print(obj['class_name'].decode('utf-8'))
    #                 bbox_gt = obj['bbox']
    #                 print(bbox_gt)
    #                 bbox_intersect = [max(bbox_pred[0], bbox_gt[0]), max(bbox_gt[1], bbox_pred[1]), min(bbox_gt[2], bbox_pred[2]), min(bbox_gt[3], bbox_pred[3])]
    #                 # print(bbox_intersect)
    #                 intersect_weight = bbox_intersect[2] - bbox_intersect[0] + 1
    #                 # print(intersect_weight)
    #                 intersect_high = bbox_intersect[3] - bbox_intersect[1] + 1
    #                 if intersect_high > 0 and intersect_weight > 0:
    #                     union_area = (bbox_pred[2] - bbox_pred[0] + 1) * (bbox_pred[3] - bbox_pred[1] + 1) + (bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1) - intersect_weight * intersect_high
    #                     Iou = intersect_high * intersect_weight / union_area
    #                     if Iou > Iou_max:
    #                         Iou_max = Iou
    #                         gt_match = obj
    #         if Iou_max > min_Iou:
    #             if not gt_match['used'] and gt_match is not None:
    #                 tp[idx] = 1
    #                 gt_match['used'] = True
    #             else:
    #                 fp[idx] = 1
    #         else:
    #             fp[idx] = 1
    #     # 计算精度和召回率
    #     sum_class = 0
    #     for idx, val in enumerate(fp):
    #         fp[idx] += sum_class
    #         sum_class += val
    #     sum_class = 0
    #     for idx, val in enumerate(tp):
    #         tp[idx] += sum_class
    #         sum_class += val
    #     rec = tp[:]
    #     for idx, val in enumerate(tp):
    #         rec[idx] = tp[idx] / gt_counter_per_class[class_name]
    #     prec = tp[:]
    #     for idx, val in enumerate(tp):
    #         prec[idx] = tp[idx] / (fp[idx] + tp[idx])
    #
    #     ap, mrec, mprec = voc_ap(rec, prec)
    #     sum_AP += ap
    # MAP = sum_AP / len(gt_counter_per_class) * 100
    # print("The Model Eval MAP: {}".format(MAP))

if __name__ == "__main__":
    train()
    # 计算模型的Map
    # eval(config.model_dir, yolo_weights = config.yolo3_weights_path)
    # eval(config.model_dir, yolo_weights=None)


