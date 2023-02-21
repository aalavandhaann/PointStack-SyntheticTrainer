import numpy as np
import open3d as o3d

import torch
from tqdm import tqdm
import datetime
from utils.runtime_utils import get_device

def visualize_numpy(pc_numpy, colors = None, window_name='3D Window'):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_numpy)
    try:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    except:
        pass

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(point_cloud)
    ctr = vis.get_view_control()
    ctr.set_up((1, 0, 0))
    ctr.set_front((0, 1, 0))

    vis.run()
    
    # o3d.visualization.draw_geometries([point_cloud])
    
def visualize_part(net, testloader):
    # color_choices = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #3
    #                  (1, 0, 0), (0, 1, 0), #5
    #                  (1, 0, 0), (0, 1, 0), #7
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #11
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #15
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), # 18
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #21
    #                  (1, 0, 0), (0, 1, 0), #23
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), #27
    #                  (1, 0, 0), (0, 1, 0), #29
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), # 35
    #                  (1, 0, 0), (0, 1, 0), #37
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), #40
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), #43
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), #46
    #                  (1, 0, 0), (0, 1, 0), (0, 0, 1), #49
    # ]
    # parts = [
    #     'Background', 'Left Hand', 'Right Eye', 'Right Leg', 'Left Eye', 'Left Cheek',
    #     'Rest Of Face', 'Right Ear', 'Left Leg', 'Chest', 'Left Ear', 'Left Feet', 'Right Arm',
    #     'Right Hand', 'Forehead', 'Right Cheek', 'Abdomen', 'Lips', 'Right Feet', 'Nose', 'Left Arm'
    #     ]
    color_choices_np = np.array(
        [
            [0,	0,	0, 255],#	Background
            [0,	0,	204, 255],#	Left Hand
            [0,	0,	255, 255],#	Right Eye
            [0,	102,	0, 255],#	Right Leg
            [0,	255,	0, 255],#	Left Eye
            [0,	255,	255, 255],#	Left Cheek
            [128,	128,	128, 255],#	RestOfFace
            [135,	206,	250, 255],#	Right Ear
            [139,	0,	0, 255],#	Left Leg
            [139,	69,	19, 255],#	Chest
            [144,	238,	144, 255],#	Left Ear
            [192,	192,	192, 255],#	Left Feet
            [240,	230,	140, 255],#	RightArm
            [245,	245,	220, 255],#	Right Hand
            [255,	0,	0, 255],#	Forehead
            [255,	0,	255, 255],#	Right Cheek
            [255,	153,	0, 255],#	Abdomen
            [255,	204,	153, 255],#	Lips
            [255,	215,	0, 255],#	Right Feet
            [255,	255,	0, 255],#	Nose
            [255,	255,	240, 255],#	Left Arm
        ], dtype=float)
    color_choices = color_choices_np[:,:3] / 255.0

    net.eval()

    with torch.no_grad():
        for batch_idx, original_data_dic in enumerate(tqdm(testloader)):
            # if ((batch_idx % 10 == 0) and (batch_idx > 1260)):
            if ((batch_idx % 10 == 0)):
                data_dic = {}
                device = get_device()
                for dkey in original_data_dic.keys():
                    # print(f'KEY: {dkey}, SIZE: {data_dic[dkey].shape}')
                    if(not original_data_dic[dkey].is_cuda):
                        data_dic[dkey] = original_data_dic[dkey].to(device)#cuda()
                    else:
                        data_dic[dkey] = original_data_dic[dkey]

                data_dic = net(data_dic)        
                points = data_dic['points'].squeeze(0).cpu().numpy()
                label = data_dic['seg_id'].squeeze(0).cpu().numpy()
                pred = torch.argmax(data_dic['pred_score_logits'], dim = -1).squeeze(0).cpu().numpy()

                color_list = np.zeros_like(points)
                for i, pred_id in enumerate(pred):
                    color_list[i] = color_choices[int(pred_id)]            
                visualize_numpy(points, colors=color_list, window_name='Prediction')

                color_list = np.zeros_like(points)
                for i, label_id in enumerate(label):
                    color_list[i] = color_choices[int(label_id)]            
                visualize_numpy(points, colors=color_list, window_name='Original')
