import shutil
import pathlib
import json
import numpy as np
import tqdm
import sklearn.model_selection

from easydict import EasyDict

from runtime_utils import cfg, cfg_from_yaml_file

CONFIG_YAML_FILE: pathlib.Path = pathlib.Path('../cfgs/syntheticpartnormal/syntheticpartnormal.yaml')
#The relative path where the synthetic data exists
SYNTHETIC_DATASET_ORIGIN: pathlib.Path = pathlib.Path('./PLY')
SYNTHETIC_DATASET_DESTINATION: pathlib.Path = pathlib.Path('../data/syntheticpartnormal/pediatrics')

'''
    ply_files: pathlib.Path.glob a generator that has contains the iteration to find all ply files to use as dataset
    dataset_dir: pathlib.Path a posix path where the files need to be stored
    split: tuple a tuple with float values to direct the splitting of dataset into training, testing, and validation
    config: EasyDict instance that conforms to the yaml structure used as configuration in PointStack (refer to cfgs/partnormal/partnormal.yaml for example)
'''
def prepareDataSet(ply_files: pathlib.Path.glob, dataset_destination_dir: pathlib.Path, *, split:tuple = (0.7, 0.2, 0.1), config: EasyDict = None) -> tuple:
    
    def saveNPAndAppend(ply_file: pathlib.Path, destination_dir: pathlib.Path, append_to: list) -> tuple:
        save_dataset_path = destination_dir.joinpath(f'{ply_file.stem}.txt')
        dataset_json_entry = f'shape_data/{destination_dir.name}/{save_dataset_path.stem}'

        if(not save_dataset_path.exists()):
            np_data = np.loadtxt(ply_file, skiprows=12)
            np_data = np_data[np_data[:,-1] > 0]
            if(np_data.shape[0]):
                np.savetxt(save_dataset_path, np_data, fmt='%f')
            else:
                dataset_json_entry = None          
        if(dataset_json_entry):
            append_to.append(dataset_json_entry)
        return dataset_json_entry, append_to

    def saveJSON(save_json_path: pathlib.Path, entries: list) -> None:
        f = open(save_json_path, 'w')
        f.write(json.dumps(entries))
        f.close()
    
    shuffled_train_list = []
    shuffled_test_list = []
    shuffled_val_list = []

    synsetoffset2category_file: pathlib.Path =  dataset_destination_dir.parent.joinpath('synsetoffset2category.txt')
    train_test_split_dir: pathlib.Path =  dataset_destination_dir.parent.joinpath('train_test_split')
    shuffled_train_file_list_json: pathlib.Path = train_test_split_dir.joinpath('shuffled_train_file_list.json')
    shuffled_test_file_list_json: pathlib.Path = train_test_split_dir.joinpath('shuffled_test_file_list.json')
    shuffled_val_file_list_json: pathlib.Path = train_test_split_dir.joinpath('shuffled_val_file_list.json')

    ply_files_list = list(ply_files)

    # First split the data into training, and testing_temp (e.g 0.7, 0.2 + 0.1)
    data_train, data_test_temp, _, _ = sklearn.model_selection.train_test_split(ply_files_list, ply_files_list, test_size=split[1] + split[2], random_state=config.RANDOM_SEED)
    
    # Now split the training_temp data to training, and validation (e.g 0.2, 0.1)
    data_test, data_val, _, _ = sklearn.model_selection.train_test_split(data_test_temp, data_test_temp, test_size= split[2] / (split[1] + split[2]), random_state=config.RANDOM_SEED)
    
    # if(dataset_destination_dir.exists()):
    #     shutil.rmtree(dataset_destination_dir, ignore_errors=True)
    
    if(train_test_split_dir.exists()):
        shutil.rmtree(train_test_split_dir, ignore_errors=True)

    if(not dataset_destination_dir.exists()):
        dataset_destination_dir.mkdir(parents = True, exist_ok = True)
    
    if(not train_test_split_dir.exists()):
        train_test_split_dir.mkdir(parents=True, exist_ok=True)

    for i, ply_file in tqdm.tqdm(enumerate(data_train), desc='Training Datasets...'):
        _, _ = saveNPAndAppend(ply_file, dataset_destination_dir, shuffled_train_list)
    
    for i, ply_file in tqdm.tqdm(enumerate(data_test), desc='Testing Datasets...'):
        _, _ = saveNPAndAppend(ply_file, dataset_destination_dir, shuffled_test_list)

    for i, ply_file in tqdm.tqdm(enumerate(data_val), desc='Validation Datasets...'):
        _, _ = saveNPAndAppend(ply_file, dataset_destination_dir, shuffled_val_list)
    
    saveJSON(shuffled_train_file_list_json, shuffled_train_list)
    saveJSON(shuffled_test_file_list_json, shuffled_test_list)
    saveJSON(shuffled_val_file_list_json, shuffled_val_list)

    f = open(synsetoffset2category_file, 'w')
    f.write(f'{dataset_destination_dir.stem.capitalize()}\t{dataset_destination_dir.stem}')
    f.close()

if __name__ == '__main__':
    ply_files_list: pathlib.Path.glob = SYNTHETIC_DATASET_ORIGIN.glob('*.ply')
    cfg: EasyDict = cfg_from_yaml_file(CONFIG_YAML_FILE, cfg)
    prepareDataSet(ply_files_list, SYNTHETIC_DATASET_DESTINATION, config=cfg)
    
    
    