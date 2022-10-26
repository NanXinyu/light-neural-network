# --------------------------------------------------
# 2022/10/25
# Written by Xinyu Nan (nan_xinyu@126.com)
# --------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import logging

import torch
import torch.nn as nn
# from config.base import _C as cfg

logger = logging.getLogger(__name__)

class SqueezeExcitation(nn.Module):
    '''
    input: [batch_size, channels, height, width]
    output: [batch_size, channels, height, width]
    '''
    def __init__(
        self,
        in_channels,
        squeeze_channels,
        activation = nn.ReLU,
        scale_activation = nn.Sigmoid
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * x

class Conv_Norm_Activation(nn.Module):
    '''
    input: [batch_size, channels, height, width]
    -> Conv2d
    -> Normlization (BatchNorm2d / LayerNorm)
    -> Activation (ReLU / GELU)
    '''
    def __init__(
        self,
        in_size,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        groups = 1,
        norm = 'bn',
        activation = 'relu'
    ):
        super(Conv_Norm_Activation, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias = False)
        
        if norm == 'bn':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'ln':
            self.norm_layer = nn.LayerNorm([out_channels, in_size[0], in_size[1]])
        
        if activation == 'relu':
            self.activation_layer = nn.ReLU(inplace = True)
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        
    def forward(self, x):

        x = self.conv_layer(x) 

        x = self.norm_layer(x)

        x = self.activation_layer(x)

        return x

class SimpleClassifier(nn.Module):
    '''
    input: [batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH]
    -> hidden layers
    output: [batch_size, NUM_JOINTS, IMG_HEIGHT, IMG_WIDTH]
    '''
    def __init__(
        self,
        img_channels, #cfg.MODEL.IMG_CHANNELS
        hidden_size, #cfg.MODEL.HIDDEN_SIZE
        num_joints, # cfg.MODEL.NUM_JOINTS
    ):
        super(SimpleClassifier, self).__init__()
        
        layers = []
        layers.append(Conv_Norm_Activation(None, img_channels, hidden_size[0], kernel_size=1))
        
        for i in range(len(hidden_size)-1):
            layers.append(Conv_Norm_Activation(None, hidden_size[i], hidden_size[i], kernel_size=3))
            layers.append(Conv_Norm_Activation(None, hidden_size[i], hidden_size[i+1], kernel_size=1))
        
        for i in range(len(hidden_size)-1):
            layers.append(SqueezeExcitation(hidden_size[-1-i], hidden_size[-1-i]//4))
            layers.append(Conv_Norm_Activation(None, hidden_size[-1-i], hidden_size[-2-i], kernel_size=1))

        layers.append(Conv_Norm_Activation(None, hidden_size[0], num_joints, kernel_size= 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x)
        return x 

class KeypointsPredictor(nn.Module):
    '''
    input: [batch_size, NUM_JOINTS, IMG_H, IMG_W]
    -> DW SE PW
    -> AVGPOOL
    -> output: [batch_size, patches, 1]
    '''
    def __init__(
        self,
        img_size, # cfg.MODEL.IMG_SIZE
        patch_size, # cfg.MODEL.PATCH_SIZE
    ):
        super().__init__()
        self.patches = (img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])
        self.pw = Conv_Norm_Activation(None, self.patches, self.patches, kernel_size=1)
        self.SE_layer = SqueezeExcitation(self.patches, self.patches // 4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.predict = nn.Softmax(dim = 1)

    def forward(self, x):
        for i in range(1):
            size = [x.shape[2]//2, x.shape[3]//2]
            dw = Conv_Norm_Activation(size, self.patches, self.patches, kernel_size=3, stride=2, groups=self.patches, norm='ln', activation='gelu')
            x = dw(x)
            y = self.SE_layer(x)
            y = self.pw(y)
            x = x + y
        
        x = self.avgpool(x)
        x = x.reshape(-1, self.patches, 1)
        # not direct
        # x = self.predict(x)
        return x

class RelativePosNet(nn.Module):
    '''
    input: [batch_size, img_channels, img_h, img_w]
    -> classifier: [batch_size, num_joints, img_h, img_w]
    -> predictor:
    -> patchstack: [batch_size, patches, patch_h, patch_w] * num_joints
    -> predict:[batch_size, patches, 1] * num_joints
    -> output: [batch_size, num_joints, patches]
    '''
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.classifier = SimpleClassifier(cfg.MODEL.IMG_CHANNELS, cfg.MODEL.HIDDEN_SIZE, cfg.MODEL.NUM_JOINTS)
        self.Predictor = KeypointsPredictor(cfg.MODEL.IMG_SIZE, cfg.MODEL.PATCH_SIZE)
        self.new_channels = (cfg.MODEL.IMG_SIZE[0]// cfg.MODEL.PATCH_SIZE[0])*(cfg.MODEL.IMG_SIZE[1] // cfg.MODEL.PATCH_SIZE[1])
        self.img_size = cfg.MODEL.IMG_SIZE
        self.patch_size = cfg.MODEL.PATCH_SIZE
        self.num_joints = cfg.MODEL.NUM_JOINTS

    def PatchStack(self, x):
        # new_channel = (cfg.IMG_SIZE[0]// cfg.PATCH_SIZE[0])*(cfg.IMG_SIZE[1] // cfg.PATCH_SIZE[1])
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.transpose(2, 3)
        x = x.reshape(B, C*H//self.patch_size[0], self.patch_size[0], W)
        x = x.transpose(2, 3)
        x = x.reshape(B, self.new_channels, self.patch_size[0], self.patch_size[1])
        return x

    def forward(self, x):
        x = self.classifier(x)
        joints_pred = []
        for i in range(self.num_joints):
            joint = x[:,i,:,:].reshape(-1, 1, self.img_size[0], self.img_size[1])
            joint = self.PatchStack(joint)
            joints_pred.append(self.Predictor(joint))

        joints_pred = torch.stack(joints_pred, dim = 1)
        joints_pred = joints_pred.reshape(-1, self.num_joints, self.new_channels)
        print(joints_pred.shape)
        return joints_pred

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.normal_(m.weight, std=0.001)
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_pose_net(cfg, is_train):
    model = RelativePosNet(cfg)
    if is_train:
        model.init_weights()
    return model