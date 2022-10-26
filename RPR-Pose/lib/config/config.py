# -------------------------------------------------------------
# 2022/10/25
# Written by Xinyu Nan (nan_xinyu@126.com)
# -------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = True#False #
_C.PIN_MEMORY = True   #
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'RelativePosPose'
_C.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = '' #
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.IMG_SIZE = [256, 256]
_C.MODEL.IMG_CHANNELS = 3
_C.MODEL.SIGMA = 2
_C.AUX_ALPHA = 0.00001 #
_C.MODEL.HEAD_INPUT = 256 #
_C.MODEL.DIM = 2 #
_C.MODEL.INIT = False #
_C.MODEL.EXTRA = CN(new_allowed=True) #
_C.MODEL.PATCH_SIZE = [16, 16]
_C.MODEL.HIDDEN_SIZE = [64, 128, 256]

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False #
_C.LOSS.TOPK = 8 #
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHTS = False ###
_C.LOSS.TYPE = 'KLDiscretLoss'
_C.LOSS.LABEL_SMOOTHING = 0.1 #

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = '' #
_C.DATASET.SELECT_DATA = False
_C.DATASET.TRAIN_RATIO = 1.0 ###
_C.DATASET.TEST_RATIO = 1.0  ###

# training data augmentation
_C.DATASET.FLIP = True 
_C.DATASET.SCALE_FACTOR = 0.25 #
_C.DATASET.ROT_FACTOR = 30     #
_C.DATASET.PROB_HALD_BODY = 0.0 #
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False #

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1 #
_C.TRAIN.LR_STEP = [90, 110] #
_C.TRAIN.LR = 0.001 #

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN_MOMENTUM = 0.9 #
_C.TRAIN.WD = 0.0001    #
_C.TRAIN.NESTREROV  =False #
_C.TRAIN.GAMMA1 = 0.99  #
_C.TRAIN.GAMMA2 = 0.0   #

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 250

_C.TRAIN.RESUME = False  #
_C.TRAIN.CHECKPOINT = '' #

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False  ###
_C.TEST.SHIFT_HEATMAP = False #

_C.TEST.USE_GT_BBOX = True ###
_C.TEST.BLUR_KERNEL = 11 #

# nms
_C.TEST.IMAGE_THRE = 0.1 #
_C.TEST.NMS_THRE = 0.6 #
_C.TEST.SOFT_NMS = False #
_C.TEST.OKS_THRE = 0.5 #
_C.TEST.IN_VIS_THRE = 0.0 #
_C.TEST.COCO_BBOX_FILE = '' #
_C.TEST.BBOX_THRE = 1.0 #
_C.TEST.MODEL_FILE = '' #

# PCKH
_C.TEST.PCKH_THRE = 0.5

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False #
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False ###
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False

def update_config(cfg):
    # RPR-Pose/
    cfg.CUR_DIR = osp.dirname(osp.abspath(__file__))
    cfg.ROOT_DIR = osp.join(cfg.CUR_DIR, '.')

    # PRP-Pose/output
    cfg.OUTPUT_DIR = osp.join(cfg.ROOT_DIR, 'output')
    
    # PRP-Pose/output/log
    cfg.LOG_DIR = osp.join(cfg.OUTPUT_DIR, 'log')

    cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, 'dataset')

    cfg.DATASET_ROOT = osp.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )
    
    cfg.freeze()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

    
