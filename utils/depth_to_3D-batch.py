# (c) 2018 Fraunhofer IOSB
# see mini-rgbd-license.txt for licensing information

from os.path import join, exists
from os import makedirs
import pathlib
import tqdm
import numpy as np
import scipy
import cv2

import pymeshlab

np.seterr(divide='ignore', invalid='ignore')

DOWN_SAMPLE_SIZE: int = 30000 # Use -1 for no downsampling

PLY_HEADER_WITH_NORMALS_COLORS = 'ply\nformat ascii 1.0\ncomment Created by labeled_geometry.py\nelement vertex {0}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nproperty uchar label\nend_header'
PLY_FORMAT_WITH_NORMALS_COLORS = '%f %f %f %f %f %f %d %d %d %d %d'

VIDYA_SEGMENTATION_COLORS = np.array(
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
        ], dtype=int)

GENERATE_FOR_SMIL = 'smil'
GENERATE_FOR_VIDYA= 'vidya'
GENERATE_MODE = GENERATE_FOR_VIDYA # choices are GENERATE_FOR_SMIL, GENERATE_FOR_VIDYA

 
if(GENERATE_MODE == GENERATE_FOR_SMIL):
    # USE THE BELOW PATHS FOR SMIL
    BASE_PATH = pathlib.Path(__file__).parent.parent.joinpath('data').joinpath('SMIL').resolve()
    DEPTH_IMAGES = BASE_PATH.joinpath('01').joinpath('depth').resolve()
    OUTPUT_FOLDER = DEPTH_IMAGES.parent.joinpath('ply').resolve()
else:
    # USE THE BELOW PATHS FOR VIDYA
    BASE_PATH = pathlib.Path('/home/ashok/Pictures/Vidya/dataset/')
    DEPTH_IMAGES = BASE_PATH.joinpath('DEPTH').resolve()
    OUTPUT_FOLDER = BASE_PATH.joinpath('PLY').resolve()

try:
    camera_intrinsic_path: pathlib.Path = DEPTH_IMAGES.joinpath('camera_intrinsics.npy')
    CAMERA_INTRINSICS = np.load(f'{camera_intrinsic_path.resolve()}')
    fx, fy = CAMERA_INTRINSICS[0, 0], CAMERA_INTRINSICS[1, 1]
    cx, cy = CAMERA_INTRINSICS[0, 2].astype(float), CAMERA_INTRINSICS[1, 2].astype(float)
except FileNotFoundError:
    # camera calibration used for generation of depth
    fx = 588.67905803875317
    fy = 590.25690113005601
    cx = 322.22048191353628
    cy = 237.46785983766890

def get2DFrom3D(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float)->np.ndarray:
    x_coords = np.floor(((points_3d[:, 0] * fx) / (points_3d[:, 2])) + cx).astype(int)
    y_coords = np.floor(((points_3d[:, 1] * fy) / (points_3d[:, 2])) + cy).astype(int)

    x_coords[x_coords < 0] = 0
    y_coords[y_coords < 0] = 0

    return np.vstack((y_coords, x_coords))

def project3D(rgb_image_path: pathlib.Path, depth_image_path: pathlib.Path, 
              output_folder: pathlib.Path, 
              fx: float, fy: float, cx: float, cy: float, *, 
              segmentation_image: pathlib.Path=None, 
              down_sample_size: int = -1, segmentation_colors: np.ndarray=np.zeros((0, 0, 4 ))):
    global PLY_HEADER_WITH_NORMALS_COLORS
    global PLY_FORMAT_WITH_NORMALS_COLORS

    output_path = output_folder.joinpath(f'{depth_image_path.stem}.ply')
    
    rgb_im = cv2.imread(f'{rgb_image_path}', cv2.IMREAD_COLOR)
    rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGBA)

    # load depth image using OpenCV - can be replaced by any other library that loads image to numpy array
    depth_im = cv2.imread(f'{depth_image_path}', -1)

    im_width, im_height = depth_im.shape
    
    # create tuple containing image indices
    indices = tuple(np.mgrid[:depth_im.shape[0],:depth_im.shape[1]].reshape((2,-1)))   

    # Get the depth values from the depth image using the indices
    raveled_depth = depth_im[indices].ravel()
    raveled_seg_rgb: np.ndarray = np.zeros((0, 0, 3))
    
    #Create an empty (im_width * im_height) x 3 array for the 3d points
    pts3D = np.zeros((indices[0].size, 3))
    #The z value is depth at a pixel / the distance of that pixel from kinect camera
    pts3D[:, 2] = raveled_depth / np.max(raveled_depth) #1000.
    
    # Get the 3D x coordinate from the 2D pixel x by offseting 
    # the center x (cx) from the pixel 
    # Multiply this by the depth value of that pixel
    # Finally divide the whole thing by field of view X (fx)
    pts3D[:, 0] = ((np.asarray(indices).T[:, 1] - cx) * pts3D[:, 2]) / fx
    # Similar to the 3D X coordinate calculate the 3D Y coordinate 
    pts3D[:, 1] = ((np.asarray(indices).T[:, 0] - cy) * pts3D[:, 2]) / fy

    # write to .obj file
    output_folder.mkdir(exist_ok=True, parents=True)

    #Given a segmentation or mask image choose only non-black pixels
    if(segmentation_image):
        seg_rgb_im = cv2.imread(f'{segmentation_image}', cv2.IMREAD_COLOR)
        seg_rgb_im = cv2.cvtColor(seg_rgb_im, cv2.COLOR_BGR2RGBA)
        raveled_seg_rgb = seg_rgb_im.reshape(im_height*im_width, 4)

        mask_region_indices = ~np.all(raveled_seg_rgb == [0, 0, 0, 255], axis=-1)
        pts3D = pts3D[mask_region_indices]

    ms: pymeshlab.MeshSet = pymeshlab.MeshSet()
    m: pymeshlab.Mesh = pymeshlab.Mesh(pts3D)
    ms.add_mesh(m, depth_image_path.stem)
    ms.compute_normal_for_point_clouds(k=10, flipflag=True)   

    if(down_sample_size > -1):
        # point cloud sampling here to restrict for 2048 or 4096 or as much as you need
        ms.generate_simplified_point_cloud(samplenum=down_sample_size)
        
    m = ms.current_mesh()

    #The 3d points in the point cloud
    v: np.ndarray = m.vertex_matrix()
    #The 3D normals in the point cloud
    n: np.ndarray = m.vertex_normal_matrix()

    #Based on the final 3D positions (after segmentation, downsampling) get their respective 2d coordinates again
    coords_2d: np.ndarray = get2DFrom3D(v, fx, fy, cx, cy)
    #Based on the 2d coordinates find their respective colors in the segmentation or RGB image again
    c: np.ndarray = seg_rgb_im[coords_2d[0], coords_2d[1]]    

    #Time to fill the labels of each point in the point cloud
    #This is done by going through the colors of each point and assign the respective class index
    l: np.ndarray = np.ones((v.shape[0], 1))#np.ones((raveled_rgb.shape[0], 1))

    #If a segmentation palette is not given to choose the label index from
    #Then just give 0 for black and 1 for white
    if(not segmentation_colors.shape[0]): 
        segmentation_colors = np.array([[0 ,0, 0, 255], [1, 1, 1, 255]])
    
    # Create the kd tree for searching through the segmentation colors
    segmentation_palette = scipy.spatial.cKDTree(segmentation_colors)
    # Find the indices to use as label index for each point with the segmentation color
    _, indices = segmentation_palette.query(c)
    indices = indices.reshape(indices.shape[0], 1)
    l[:,] = indices[:,]

    # print(np.max(l), np.min(l))
    # print(v.shape)
    
    ply_np_data = np.hstack((v, n, c, l))

    PLY_HEADER_WITH_NORMALS_COLORS = PLY_HEADER_WITH_NORMALS_COLORS.format(ply_np_data.shape[0])
    np.savetxt(f'{output_path}', ply_np_data, header=PLY_HEADER_WITH_NORMALS_COLORS, comments='', encoding='ASCII', fmt=PLY_FORMAT_WITH_NORMALS_COLORS)

    # save the current mesh
    # ms.save_current_mesh(f'{output_path}', binary=False, save_vertex_normal=True, save_vertex_color=True)



    # with open(join(output_folder, f'{}.obj'), 'w') as file:
    #     for pt in pts3D:
    #         file.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))


def smilBatch(depth_folder: pathlib.Path, output_folder: pathlib.Path, fx: float, fy: float, cx: float, cy: float):
    glob_result: pathlib.Path.glob = depth_folder.glob('*.png')
    output_folder.mkdir(parents=True, exist_ok=True)
    glob_result = list(glob_result)

    for d_png in tqdm.tqdm(glob_result, dynamic_ncols=True, total=len(glob_result)):
        rgb_name: str = '_'.join(d_png.stem.split('_')[:2])
        rgb_image: pathlib.Path = d_png.parent.parent.joinpath('rgb').joinpath(f'{rgb_name}.png')
        project3D(rgb_image, d_png, output_folder, fx, fy, cx, cy, down_sample_size=DOWN_SAMPLE_SIZE)
        break

def vidyaBatch(depth_folder: pathlib.Path, output_folder: pathlib.Path, fx: float, fy: float, cx: float, cy: float):
    glob_result: pathlib.Path.glob = depth_folder.glob('*.png')
    output_folder.mkdir(parents=True, exist_ok=True)
    glob_result = list(glob_result)

    for d_png in tqdm.tqdm(glob_result, dynamic_ncols=True, total=len(glob_result)):
        rgb_name: str = d_png.stem.split('-')[0]
        rgb_image: pathlib.Path = d_png.parent.parent.joinpath('RGB').joinpath(f'{rgb_name}-RGB.png')
        seg_rgb_image: pathlib.Path = d_png.parent.parent.joinpath('SEGMENTATION').joinpath(f'{rgb_name}-SEGMENTATION.png')
        project3D(rgb_image, d_png, output_folder, 
                  fx, fy, cx, cy, 
                  down_sample_size=DOWN_SAMPLE_SIZE, 
                  segmentation_image=seg_rgb_image,
                  segmentation_colors=VIDYA_SEGMENTATION_COLORS)
        # break

if __name__ == '__main__':
    if(GENERATE_MODE == GENERATE_FOR_SMIL):
        print('GENERATE SMIL')
        smilBatch(DEPTH_IMAGES, OUTPUT_FOLDER, fx, fy, cx, cy)
    else:
        print('GENERATE VIDYA')
        vidyaBatch(DEPTH_IMAGES, OUTPUT_FOLDER, fx, fy, cx, cy)
    # glob_result: pathlib.Path.glob = DEPTH_IMAGES.glob('*.png')
    # OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    # glob_result = list(glob_result)
    # for d_png in tqdm.tqdm(glob_result, dynamic_ncols=True, total=len(glob_result)):
    #     # rgb_name: str = '_'.join(d_png.stem.split('_')[:2])
    #     rgb_image: pathlib.Path = d_png.parent.parent.joinpath('rgb').joinpath(f'{rgb_name}.png')
    #     project3D(rgb_image, d_png, OUTPUT_FOLDER)
    #     break