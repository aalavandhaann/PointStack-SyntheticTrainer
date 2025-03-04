import os
import argparse
import datetime
import numpy as np
import random
import shutil
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from core.builders import build_dataset, build_network, build_optimizer
from utils.runtime_utils import cfg, cfg_from_yaml_file, validate, get_nn_module_cuda, get_method
from utils.vis_utils import visualize_numpy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed number')
    parser.add_argument('--val_steps', type=int, default=1, help='perform validation every n steps')
    parser.add_argument('--pretrained_ckpt', type = str, default = None, help='path to pretrained ckpt')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    exp_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy2(args.cfg_file, exp_dir)

    return args, cfg

def main():

    args, cfg = parse_config()        
    random_seed = cfg.RANDOM_SEED # Setup seed for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Build Dataloader
    train_dataset = build_dataset(cfg, split = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.OPTIMIZER.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=min(cfg.OPTIMIZER.BATCH_SIZE, 24), pin_memory=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg.OPTIMIZER.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=12)

    '''
        Do not have a batch_size of value greater than 1 for the validation dataset. From my observation, the increased batch_size is not 
        giving a good performance of mIOU and accuracy. Hence, we set the batch_size to 1.
    '''
    val_dataset = build_dataset(cfg, split='test')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=min(cfg.OPTIMIZER.BATCH_SIZE, 24), pin_memory=True)

    # Build Network and Optimizer
    net = build_network(cfg)
    net, device = get_nn_module_cuda(net, cfg.GPU_COUNT)

    pretrained_state_dict, pretrained_opt_state_dict = None, None
    if args.pretrained_ckpt is not None:
        pretrained_state_dict = torch.load(args.pretrained_ckpt)['model_state_dict']
        pretrained_opt_state_dict = torch.load(args.pretrained_ckpt)['optimizer_state_dict']
        for k, v in net.state_dict().items():
            item = pretrained_state_dict.get(k, None)
            # item_module = pretrained_state_dict.get(f'module.{k}', None)

            if(item is not None):
                if (v.shape != item.shape):
                    del pretrained_state_dict[k]
            # elif(item_module is not None):
            #     print('saved with dataparallel ', v.shape, item_module.shape)
            #     if (v.shape != item_module.shape):
            #         del pretrained_state_dict[f'module.{k}']
            else: 
                print('KEY NOT FOUND IN LOADED MODEL CHECKPOINT', k)

        net.load_state_dict(pretrained_state_dict, strict = False)

    opt, scheduler = build_optimizer(cfg, net.parameters(), len(train_dataloader))    
    if(pretrained_opt_state_dict):
        opt.load_state_dict(pretrained_opt_state_dict)
    
    from torch.utils.tensorboard import SummaryWriter
    ckpt_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'ckpt'
    tensorboard_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'tensorboard'

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    writer = SummaryWriter(tensorboard_dir)

    min_loss = 1e20
    max_acc = 0

    steps_cnt = 0
    epoch_cnt = 0


    for epoch in tqdm(range(1, cfg.OPTIMIZER.MAX_EPOCH + 1), dynamic_ncols=True):
        opt.zero_grad()
        net.zero_grad()
        net.train()
        loss = 0
        training_iteration = 0
        for original_data_dic in tqdm(train_dataloader, dynamic_ncols=True):

            data_dic = {}
            for dkey in original_data_dic.keys():
                # print(f'KEY: {dkey}, SIZE: {data_dic[dkey].shape}')
                if(not original_data_dic[dkey].is_cuda):
                    data_dic[dkey] = original_data_dic[dkey].to(device)#cuda()
                else:
                    data_dic[dkey] = original_data_dic[dkey]

            data_dic = net(data_dic)

            # Check if the network is multi-gpu, otherwise use the single-gpu network as handled in exception
            loss, loss_dict = get_method(net, 'get_loss')(data_dic, smoothing = True, is_segmentation = cfg.DATASET.IS_SEGMENTATION)
            # try:
            #     loss, loss_dict = net.get_loss(data_dic, smoothing = True, is_segmentation = cfg.DATASET.IS_SEGMENTATION)
            # except AttributeError:
            #     loss, loss_dict = net.module.get_loss(data_dic, smoothing = True, is_segmentation = cfg.DATASET.IS_SEGMENTATION)

            # loss = loss
            loss.backward()
            steps_cnt += 1
            
            # if (steps_cnt)%(cfg.OPTIMIZER.GRAD_ACCUMULATION) == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            opt.step()
            opt.zero_grad()#Originally was here
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            writer.add_scalar('steps/loss', loss, steps_cnt)
            writer.add_scalar('steps/lr', lr, steps_cnt)
            
            for k,v in loss_dict.items():
                writer.add_scalar('steps/loss_' + k, v, steps_cnt)

            # if(training_iteration > 5):
            #     break
            # training_iteration+=1
        
        if (epoch % args.val_steps) == 0:
            # Check if the network is multi-gpu, otherwise use the single-gpu network as handled in exception
            val_dict = validate(net, val_dataloader, get_method(net,'get_loss'), device, is_segmentation = cfg.DATASET.IS_SEGMENTATION, num_classes = cfg.DATASET.NUM_CLASS, part_wise_ious=True)
            
            print(f'\n{"="*20} Epoch {epoch+1} {"="*20}')

            if cfg.DATASET.IS_SEGMENTATION:
                writer.add_scalar('epochs/val_miou', val_dict['miou'], epoch_cnt)
                writer.add_scalar('epochs/val_accuracy', val_dict.get('accuracy', 0), epoch_cnt)
                print('Val mIoU: ', val_dict['miou'])
                part_wise_ious = val_dict['miou_part_wise']#getattr(val_dict, )
                for part_id, part_miou in enumerate(part_wise_ious):
                    writer.add_scalar(f'epochs/val_miou_{train_dataset.seg_classes_list[part_id]}', part_miou, epoch_cnt)
                    print(f'Val mIoU({train_dataset.seg_classes_list[part_id]}): {part_miou}')
    
            else:
                writer.add_scalar('epochs/val_loss', val_dict['loss'], epoch_cnt)
                writer.add_scalar('epochs/val_acc', val_dict['acc'], epoch_cnt)
                writer.add_scalar('epochs/val_acc_avg', val_dict['acc_avg'], epoch_cnt)
                print('Val Loss: ', val_dict['loss'], 'Val Accuracy: ', val_dict['acc'], 'Val Avg Accuracy: ', val_dict['acc_avg'])

                for k,v in val_dict['loss_dic'].items():
                    writer.add_scalar('epochs/val_loss_'+ k, v, epoch_cnt)

            epoch_cnt += 1

            
            if cfg.DATASET.IS_SEGMENTATION:
                if val_dict['miou'] > max_acc:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        }, ckpt_dir / 'ckpt-best.pth')
                    
                    max_acc = val_dict['miou']
            else:

                if val_dict['acc'] > max_acc:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': val_dict['loss'],
                        }, ckpt_dir / 'ckpt-best.pth')
                    
                    max_acc = val_dict['acc']

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    }, ckpt_dir / 'ckpt-last.pth')

if __name__ == '__main__':
    # import torch.multiprocessing
    # torch.multiprocessing.set_start_method("spawn", force=False)
    main()

