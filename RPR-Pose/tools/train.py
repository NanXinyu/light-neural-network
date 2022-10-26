from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function

import argparse
import os
import pprint
import  shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter #

import _init_paths
from config import cfg
from config import update_config
from main.loss import NMTNORMCritierion, KLDiscretLoss
from main.base import Trainer
from main.base import Validater
from utils.utils import get_optimizer, save_checkpoint, create_logger
from main.model import get_pose_net
import dataset 

def main():
    update_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, 'train')
    
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    model = get_pose_net(cfg, is_train = True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(os.path.join(this_dir, '../lib/main/model.py'), final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir = tb_log_dir),
        'train_global_steps':0,
        'valid_global_steps':0,
    }

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total number of parameters: % d" % pytorch_total_params)

    model = torch.nn.DataParallel(model, device_ids = cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if cfg.LOSS.TYPE == 'NMTNORMCritierion':
        criterion = NMTNORMCritierion(label_smoothing=cfg.LOSS.LABEL_SMOOTHING).cuda()
    elif cfg.LOSS.TYPE == 'KLDiscretLoss':
        criterion = KLDiscretLoss().cuda()       

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET,
        transforms.Compose([
            transforms.ToTensor(), 
            normalize,])
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST.SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle = cfg.TRAIN_SHUFFLE,
        num_workers = cfg.WORKERS,
        pin_memory = cfg.PIN_MEMORY
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU*(len(cfg.GPUS)),
        shuffle = False,
        num_workers = cfg.WORKERS,
        pin_memory = cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        Trainer(
            cfg, train_loader, model, criterion, optimizer, epoch,
            final_output_dir, tb_log_dir, writer_dict
            ) 

        perf_indicator = Validater(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )   

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
    
    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()



    


