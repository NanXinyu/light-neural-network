from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import math

import cv2
from cv2 import transform
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

logger = logging.getLogger(__name__)

class Data(Dataset):
    def __init__(
        self, 
        cfg, 
        root,
        image_set,
        is_train,
        transform = None,
    ):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.image_size = np.array(cfg.MODEL.IMG_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db =[]

        self.loss = cfg.LOSS.TYPE

        # target related
        self.num_h = cfg.MODEL.IMG_SIZE[0]//cfg.MODEL.PATCH_SIZE[0]
        self.num_w = cfg.MODEL.IMG_SIZE[1]//cfg.MODEL.PATCH_SIZE[1]
        self.num = self.num_h * self.num_w
        self.patch = cfg.MODEL.PATCH_SIZE
    
    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_die, *args, **kwargs):
        raise NotImplementedError
    
    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.rand() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None
        
        selected_joints = np.array(selected_joints, dtype = np.float32)
        center = selected_joints.mean(axis = 0)[:2]

        left_top = np.amin(selected_joints, axis = 0)
        right_bottom = np.amax(selected_joints, axis = 0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h *self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype = np.float32
        )

        scale = scale * 1.5

        return center, scale
    
    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)
        
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        target, target_weight = self.generate_token_target(joints, joints_vis)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        return input, target, target_weight

    def generate_token_target(self, joints, joints_vis):
        '''
            :param joints: [num_joints, 3]
            :param joints_vis: [num_joints, 3]
            :return: target, target_weight(1: visible, 0:invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype = np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
            
        target = np.zeros((self.num_joints, self.num), dtype = np.float32)
        
        tmp_size = self.sigma * 3
       
        div = np.full((self.num), self.num_h, dtype = np.float32) 
        offset_x = np.full((self.num), self.patch[1] * 0.5, dtype = np.float32)
        offset_y = np.ones((self.num), self.patch[0] * 0.5, dtype = np.float32)
        
        for joint_id in range(self.num_joints):
            target_weight[joint_id] = \
                self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
            if target_weight[joint_id] == 0:
                continue
            
            mu_x = np.full((self.num), joints[joint_id][0], dtype = np.float32)
            mu_y = np.full((self.num), joints[joint_id][1], dtype = np.float32)
            
            x = (np.arange(0, self.num, 1, np.float32) // div) * self.patch[1] + offset_x
            y = (np.arange(0, self.num, 1, np.float32) % div) * self.patch[0] + offset_y

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id] = (np.exp(-0.5 * (((x - mu_x)**2 + (y - mu_y)**2) / self.sigma**2)))/(np.pi*2*(self.sigma**2))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        mu_x = joint[0]
        mu_y = joint[1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size +1), int(mu_y + tmp_size +1)]
        if ul[0] >= (self.image_size[0]) or ul[1] >= self.image_size[1] \
                 or br[0] < 0 or br[1] < 0:
            target_weight = 0
        
        return target_weight

    def select_data(self,db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected     