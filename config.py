train_batch_size = 32

val_batch_size = 1
num_parallel_calls = 4
input_shape = 192
max_boxes = 20
jitter = 0.3
hue = 0.1
sat = 1.0
cont = 0.8
bri = 0.1
norm_decay = 0.99
norm_epsilon = 1e-3
pre_train = True
num_anchors = 9
num_classes = 80
training = True
ignore_thresh = .5
learning_rate = 0.001
train_num = 118287
val_num = 5000
# Epoch = 9
Epoch = 50
obj_threshold = 0.3
nms_threshold = 0.5
gpu_index = "0"
# log_dir = './logs'
log_dir = './logs/logs-1'

data_dir = './model_data'
voc_dir = 'VOCROOT'
voc2007_dir = 'VOCROOT/VOC2007'
voc2007test_dir = 'VOCROOT/VOC2007TEST'
voc2012_dir = 'VOCROOT/VOC2012'

model_dir = './test_model/model.ckpt-1'
pre_train_yolo3 = False
yolo3_weights_path = './model_data/yolov3.weights'
darknet53_weights_path = './model_data/darknet53.weights'
anchors_path = './model_data/yolo_anchors.txt'

anchors_path0 = './model_data/yolo_anchors0.txt'
# anchors_path1 = './model_data/yolo_anchors1.txt'
anchors_path2 = './model_data/yolo_anchors2.txt'


classes_path = './model_data/coco_classes.txt'
train_data_file = './dataset/coco/train2017'
val_data_file = './dataset/coco/val2017'
train_annotations_file = './dataset/annotations/instances_train2017.json'
val_annotations_file = './dataset/annotations/instances_val2017.json'


beta1 = 0.9
alpha = 0.001
beta = 0.01

