import os
import yaml
import torch
import datetime

import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from pathlib import Path

import torch.nn.functional as F
import torch.multiprocessing

'''
    When DataParallel is used then the methods of a module are exposed only as net.module.method
    To handle this we need a method
'''
def get_method(net: torch.nn.Module, method: str):
    flag = getattr(net, method, None)
    if(not flag):
        return getattr(net.module, method)
    return getattr(net, method)

def get_device():
    return torch.device(torch.cuda.current_device() if (torch.cuda.is_available()) else "cpu")  
'''
    Method to get the module tied to multi-gpu or a single gpu or cpu
'''
def get_nn_module_cuda(net: torch.nn.Module, ngpu: int=1)->torch.nn.DataParallel:
    device = get_device()
    if(ngpu < 2):
        return net.to(device)#.cuda()

    if(torch.multiprocessing.get_start_method() != 'spawn'):
        message = 'You need to set the cuda start method to "spawn"'
        message = f'{message}\nSimply add the code \nimport torch.multiprocessing\ntorch.multiprocessing.set_start_method("spawn")'
        message = f'{message}\nThis needs to be done the moment after torch package is imported'
        # raise RuntimeError(message)


    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")  
    # net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))
    net = torch.nn.DataParallel(net, device_ids=list(range(min(torch.cuda.device_count(), ngpu))))
    net = net.to(device)
    return net, device

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
cfg.GPU_COUNT = 1

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
        self.avg = self.sum / self.count

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def validate(net, testloader, criterion, device, is_segmentation = False, num_classes=50, *, part_wise_ious=False):
    net.eval()
    num_params = get_n_params(net)

    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = []
    mious = []
    all_part_iou = np.zeros((0, num_classes), dtype=float)

    accuracy = []
    shape_ious = 0.0
    count  = 0.0

    with torch.no_grad():
        for batch_idx, original_data_dic in enumerate(tqdm(testloader, dynamic_ncols=True)):
            start_time = datetime.datetime.now()
            
            data_dic = {}
            for dkey in original_data_dic.keys():
                # print(f'KEY: {dkey}, SIZE: {data_dic[dkey].shape}')
                if(not original_data_dic[dkey].is_cuda):
                    data_dic[dkey] = original_data_dic[dkey].to(device)#cuda()
                else:
                    data_dic[dkey] = original_data_dic[dkey]

            data_dic = net(data_dic)
            time_cost.append(float((datetime.datetime.now() - start_time).total_seconds()))
              

            if is_segmentation:
                if(part_wise_ious):
                    miou, all_part_iou_values = get_method(net, 'compute_overall_iou')(data_dic['pred_score_logits'], data_dic['seg_id'], num_classes = num_classes, part_wise_ious=True)
                else:
                    miou = get_method(net, 'compute_overall_iou')(data_dic['pred_score_logits'], data_dic['seg_id'], num_classes = num_classes)
                   
                # total iou of current batch in each process:
                batch_ious = data_dic['pred_score_logits'].new_tensor([np.sum(miou)], dtype=torch.float64)  # same device with seg_pred

                # prepare seg_pred and target for later calculating loss and acc:
                seg_pred = data_dic['pred_score_logits'].contiguous().view(-1, num_classes) # ShapeNetPart has 50 classes

                target = data_dic['seg_id'].view(-1, 1)[:, 0]
                # Loss

                pred_choice = seg_pred.data.max(1)[1]  # b*n
                correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

                shape_ious += batch_ious.item()  # count the sum of ious in each iteration
                count += data_dic['pred_score_logits'].shape[0]  # count the total number of samples in each iteration
                accuracy.append(correct.item() / (data_dic['pred_score_logits'].shape[0] * data_dic['points'].shape[1]))  # append the accuracy of each iteration
                mious.append(miou)
                if(part_wise_ious):
                    all_part_iou = np.vstack((all_part_iou, all_part_iou_values))
            else:
                preds = data_dic['pred_score_logits'].max(dim=1)[1]
                test_true.append(data_dic['cls_id'].cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                total += data_dic['cls_id'].size(0)
                correct += preds.eq(data_dic['cls_id']).sum().item()
                loss, loss_dict = criterion(data_dic)
                test_loss += loss.item()
        
        total_samples = len(time_cost)
        time_cost = np.mean(time_cost[total_samples//5:total_samples*4//5])
        
        if is_segmentation:
            overall_miou = np.mean(mious)
            overall_acc = np.mean(accuracy)
            overall_ins_acc = shape_ious * 1.0 / count
            
            try:
                miou = float("%.3f" % (100. * overall_miou))
            except TypeError:
                miou = float("%.3f" % (100. * overall_miou[0]))
            
            try:
                accuracy = float("%.3f" % (100. * overall_acc))
            except TypeError:
                accuracy = float("%.3f" % (100. * overall_acc[0]))
            if(not part_wise_ious):
                all_part_iou = np.zeros((1, num_classes), dtype=float)
            return {
                    "miou": miou,
                    "miou_part_wise": np.mean(all_part_iou, axis=0)*100.0,
                    "accuracy": accuracy, 
                    "time": time_cost,
                    "num_params": num_params,
                    'peak_memory': torch.cuda.memory_stats('cuda')['allocated_bytes.all.peak']
                }
        else:
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)


            return {
                "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
                "loss_dic": loss_dict,
                "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
                "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
                "time": time_cost,
                "num_params": num_params,
                'peak_memory': torch.cuda.memory_stats('cuda')['allocated_bytes.all.peak']
            }

def validate_voting(net, testloader, device = 'cuda', num_repeat = 300, num_vote = 10):
    net.eval()
    best_acc = 0
    best_mean_acc = 0
    pointscale = PointcloudScale(scale_low=0.85, scale_high=1.15)

    for i in tqdm(range(num_repeat)):
        test_true = []
        test_pred = []

        for batch_idx, data_dic in enumerate(tqdm(testloader)):
            pred = 0
            for v in range(num_vote):
                new_data = data_dic
                if v > 0:
                    new_data['points'] = pointscale(new_data['points'])
                with torch.no_grad():
                    pred += F.softmax(net(data_dic)['pred_score_logits'], dim=1)  # sum 10 preds
            pred /= num_vote  # avg the preds!
            pred_choice = pred.max(dim=1)[1]
            test_true.append(data_dic['cls_id'].cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = 100. * metrics.accuracy_score(test_true, test_pred)
        test_mean_acc = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_mean_acc > best_mean_acc:
            best_mean_acc = test_mean_acc
        outstr = 'Voting %d, test acc: %.3f, test mean acc: %.3f,  [current best(all_acc: %.3f mean_acc: %.3f)]' % \
                 (i, test_acc, test_mean_acc, best_acc, best_mean_acc)
        print(outstr)
    best_acc = best_acc*100
    best_mean_acc = best_mean_acc*100
    print('Final Test Acc: ', best_acc)
    print('Final Test Avg Acc: ', best_mean_acc)

    val_dict = {
        'acc': best_acc,
        'acc_avg': best_mean_acc
    }

    return val_dict

class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.device = get_device()

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(self.device))#.cuda())

        return pc
