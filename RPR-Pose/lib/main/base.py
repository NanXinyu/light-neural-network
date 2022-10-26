# ----------------------------------------------
# 2022/10/25
# Written by Xinyu Nan(nan_xinyu@126.com)
# ----------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time 
import logging
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.transforms import flip_back_token
from utils.transforms import transform_preds
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)

def regression(token, cfg):
    '''
    Each patch has been predicted a probability token:
    input token: [batch_size, num_joints, patches]
    regress token to predicted coordinate.
    output: [batch_size, num_joints, [x, y]]
    '''
    assert isinstance(token, np.nadarray), \
        'token should be np.ndarray'
    assert token.ndim == 3, 'token should be 3-ndim.'

    img_height, img_width = cfg.MODEL.IMG_SIZE[0], cfg.MODEL.IMG_SIZE[1]
    patch_height, patch_width = cfg.MODEL.PATCH_SIZE[0], cfg.MODEL.PATCH_SIZE[1]
    n_h = img_height//patch_height,
    n_w = img_width//patch_width

    batch_size = token.shape[0]
    num_joints = token.shape[1]
    patches = token.shape[2]
    
    center = [0.5*img_height, 0.5*img_width]
    assert (n_w*n_h) == patches, 'patches num should be equal to width num * height num'
    
    theta = np.empty(patches)
    dist = np.empty(patches)
    for i in range(patches):
        y = 0.5*patch_height + (i % n_h)*patch_height
        x = 0.5*patch_width + (i // n_h)*patch_width
        theta[i] = math.atan2((y - center[0]), (x - center[1]))
        dist[i] = math.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    # [patches] -> [batch_size, num_joints, patches]
    theta = np.tile(theta, (batch_size, num_joints, 1)).astype(np.float32)
    dist = np.tile(dist, (batch_size, num_joints, 1)).astypr(np.float32)

    # [batch_size, num_joints, patches] -> [batch_size, num_joints ,1]
    pred_theta = (theta*token).sum(axis = 2).reshape(batch_size, num_joints, 1)
    pred_dist = (dist*token).sum(axis = 2).reshape(batch_size, num_joints, 1)
    
    # pred_coordinate: [batch_size, num_joints, [x,y]]
    pred_coord = np.tile(center, (batch_size, num_joints, 1)).astype(np.float32)
    pred_coord[:,:,0] += pred_dist[:,:,0] * np.cos(pred_theta[:,:,0])
    pred_coord[:,:,1] += pred_dist[:,:,0] * np.sin(pred_theta[:,:,0])

    return pred_coord


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

def Trainer(config, train_loader, model, criterion, optimizer, epoch,
    output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # model output = [batch_size, num_joints, patches]
        output = model(input)

        target = target.cuda(non_blocking = True)
        target_weight = target_weight.cuda(non_blocking = True).float()

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def Validater(config, val_loader, val_dataset, model, criterion, output_dir,
        tb_log_dir, writer_dict = None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL_NUM_JOINTS, 3), dtype = np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            output = model(input)

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_flipped = model(input_flipped)
                output_flipped = flip_back_token(output_flipped.cpu().numpy(), val_dataset.flip_pairs, config)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                    
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, 0:-1] = \
                        output_flipped.clone()[:, :, 1:]   
                # not direct
                # else:
                # ouput = (output+output_filpped)*0.5
                output = (F.softmax(output, dim = 2) + F.softmax(output_flipped, dim =2))*0.5
            else:
                # not direct
                # else:
                # output = output
                output = F.softmax(output, dim = 2)

            target = target.cuda(non_blocking = True)
            target_weight = target_weight.cuda(non_blocking = True).float()

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            maxvals = output.max(2, keepdim = True)
            output_coord = regression(output, config)
            output_coord = output_coord.cpu().numpy()
            preds = output_coord.copy()

            # Transform back
            for i in range(output_coord.shape[0]):
                preds[i] = transform_preds(
                    output_coord[i], c[i], s[i], [config.IMG_SIZE[0], config.IMG_SIZE[1]]
                )
                
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, preds, output, prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        '''
        config.MODEL.NAME
        '''
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1
    
    return perf_indicator