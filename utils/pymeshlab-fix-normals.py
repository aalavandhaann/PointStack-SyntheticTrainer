import numpy as np
import pathlib
import tqdm
import pymeshlab

TXT_MESH_LOCATIONS_DIR: pathlib.Path = pathlib.Path('../data/syntheticpartnormal-NEW/pediatrics-TXT')
# TXT_MESH_LOCATIONS_DIR: pathlib.Path = pathlib.Path('../data/syntheticpartnormal-NEW/real-ply')


def fixPointCloud(point_cloud_path: pathlib.Path, save_dir:pathlib.Path=None, ply_save_location: pathlib.Path = None)->pathlib.Path:
    npy_path: pathlib.Path = save_dir.joinpath(f"{point_cloud_path.stem}.npy") if(save_dir) else point_cloud_path.parent.joinpath(f"{point_cloud_path.stem}.ply")

    ms: pymeshlab.MeshSet = pymeshlab.MeshSet()
    m: pymeshlab.Mesh = None

    if(point_cloud_path.suffix == '.txt'):
        np_data: np.ndarray = np.loadtxt(point_cloud_path)   
        # ms.load_new_mesh(f'{point_cloud_path}', strformat='X Y Z Nx Ny Nz', separator='SPACE')
        ms.load_new_mesh(f'{point_cloud_path}', strformat='X Y Z', separator='SPACE')        
        m = ms.current_mesh()
    else:
        ms.load_new_mesh(point_cloud_path)
        m = ms.current_mesh()
        np_data = np.zeros((m.vertex_matrix().shape[0], 7))

    ms.compute_normal_for_point_clouds(k=10, flipflag=True)
    m: pymeshlab.Mesh = ms.current_mesh()
    v: np.ndarray = m.vertex_matrix()
    n: np.ndarray = m.vertex_normal_matrix()
    np_data[:,:3] = v
    np_data[:,3:6] = n

    if(not npy_path.exists()):
        np.save(npy_path, np_data)
    if(ply_save_location):
        ms.save_current_mesh(f'{ply_save_location.joinpath(f"{npy_path.stem}.ply")}',binary=False, save_vertex_normal=True)
    
    return npy_path

def fixNormalsBatch(glob_location: pathlib.Path, save_to_dir: pathlib.Path, ply_files_dir: pathlib.Path = None)->None:
    point_cloud_files = list(glob_location.glob('*.txt'))
    if(not len(point_cloud_files)):
        point_cloud_files = list(glob_location.glob('*.ply'))
        
    if(not len(point_cloud_files)):
        raise RuntimeError('No txt or ply files found')
        
    save_to_dir.mkdir(exist_ok=True, parents=True)
    
    if(ply_files_dir):
        ply_files_dir.mkdir(exist_ok=True, parents=True)

    for point_cloud_path in tqdm.tqdm(point_cloud_files, dynamic_ncols=True):
        fixPointCloud(point_cloud_path, save_to_dir, ply_files_dir)
        # break


if __name__ == '__main__':
    npy_files_dir: pathlib.Path = TXT_MESH_LOCATIONS_DIR.parent.joinpath('pediatrics-NPY')
    ply_files_dir: pathlib.Path = None#TXT_MESH_LOCATIONS_DIR.parent.joinpath('pymeshlab-PLY')
    fixNormalsBatch(
        TXT_MESH_LOCATIONS_DIR, 
        npy_files_dir,
        ply_files_dir)