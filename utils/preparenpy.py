import pathlib
from easydict import EasyDict
from runtime_utils import cfg, cfg_from_yaml_file

from preparesyntheticdata import prepareDataSet

CONFIG_YAML_FILE: pathlib.Path = pathlib.Path('../cfgs/syntheticpartnormal/syntheticpartnormal.yaml')
#The relative path where the synthetic data exists
SYNTHETIC_DATASET_ORIGIN: pathlib.Path = pathlib.Path('../data/syntheticpartnormal-NEW/pediatrics-NPY')
SYNTHETIC_DATASET_DESTINATION: pathlib.Path = pathlib.Path('../data/syntheticpartnormal-NEW/pediatrics-NPY')


if __name__ == '__main__':
    ply_files_list: pathlib.Path.glob = SYNTHETIC_DATASET_ORIGIN.glob('*.npy')
    cfg: EasyDict = cfg_from_yaml_file(CONFIG_YAML_FILE, cfg)
    prepareDataSet(ply_files_list, SYNTHETIC_DATASET_DESTINATION, config=cfg, process_normals=False)