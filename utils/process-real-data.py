import numpy as np
import pathlib

main_path: pathlib.Path = pathlib.Path('../data/syntheticpartnormal/real').absolute()
ply_names: list = ['00001', '00002', '00012', '00015', '00049', '00090']
for ply_name in ply_names:
    ply_path: pathlib.Path = main_path.joinpath(f'{ply_name}-BodySeparation.ply')
    txt_path: pathlib.Path = main_path.joinpath(f'{ply_path.stem}.txt')
    np_data: np.ndarray = np.loadtxt(ply_path, skiprows=17)
    np_data = np_data[:,:6]
    print(np_data.shape)
    np_data = np.hstack((np_data, np.zeros((np_data.shape[0], 1))))
    print(np_data.shape)
    np.savetxt(txt_path, np_data, fmt='%f %f %f %f %f %f %f' )
