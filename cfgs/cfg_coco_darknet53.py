'''config file for darknet53-coco'''


# anchors
ANCHORS = [[], [], []]
# backbone
BACKBONE_TYPE = 'darknet53'
PRETRAINED_MODEL_PATH = ''
IS_MULTI_GPUS = True
ADDED_MODULES_WEIGHT_INIT_METHOD = None
# dataset
DATASET_ROOT_DIR = ''
MAX_NUM_GT_BOXES = 50
NUM_CLASSES = 81
NUM_WORKERS = 8
PIN_MEMORY = True
BATCHSIZE = 16
CLSNAMESPATH = 'names/coco.names'
USE_COLOR_JITTER = False
IMAGE_NORMALIZE_INFO = {'mean_rgb': (0.0, 0.0, 0.0), 'std_rgb': (0.0, 0.0, 0.0)}
# loss function
CLS_LOSS_SET = {}
REG_LOSS_SET = {}
# optimizer
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
LR_ADJUST_EPOCHS = [9, 12]
MAX_EPOCHS = 12
IS_USE_WARMUP = True
NUM_WARMUP_STEPS = 500
GRAD_CLIP_MAX_NORM = 35
GRAD_CLIP_NORM_TYPE = 2
# image size
IMAGESIZE_DICT = {'LONG_SIDE': 1333, 'SHORT_SIDE': 800}
# record
TRAIN_BACKUPDIR = 'yolov3_darknet53_trainbackup_coco'
TRAIN_LOGFILE = 'yolov3_darknet53_trainbackup_coco/train.log'
TEST_BACKUPDIR = 'yolov3_darknet53_testbackup_coco'
TEST_LOGFILE = 'yolov3_darknet53_testbackup_coco/test.log'
TEST_BBOXES_SAVE_PATH = 'yolov3_darknet53_testbackup_coco/yolov3_darknet53_detection_results_coco.json'
SAVE_INTERVAL = 1