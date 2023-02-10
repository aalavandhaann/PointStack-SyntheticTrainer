import pathlib
import numpy as np
import json
import tqdm

DATASETS_DIRECTORY = pathlib.Path(__file__).parent.absolute().joinpath('../data/partnormal/').absolute()
TRAINING_FILES_JSON = DATASETS_DIRECTORY.joinpath('train_test_split/shuffled_train_file_list.json')
TESTING_FILES_JSON = DATASETS_DIRECTORY.joinpath('train_test_split/shuffled_test_file_list.json')
VALIDATION_FILES_JSON = DATASETS_DIRECTORY.joinpath('train_test_split/shuffled_val_file_list.json')

def getFilesCount(json_path: pathlib.Path) -> tuple:    
    def resolve_path(pathstr: str, json_path: pathlib.Path) -> pathlib.Path:
        temppath = pathlib.Path(pathstr)
        actual_path: pathlib.Path = json_path.parents[1].joinpath(temppath.parts[-2], f'{temppath.parts[-1]}.txt')
        return actual_path
        
    f = open(json_path, 'r')
    jsdata = json.load(f)
    f.close()
    jsdata = [resolve_path(p, json_path) for p in jsdata]
    jsdata = np.array(jsdata)
    return jsdata, jsdata.shape[0]

def getLabelsInDataset(json_path: pathlib.Path) -> np.ndarray:
    jsdata, _ = getFilesCount(json_path)
    np_labels = np.array([])

    for pointcloudpath in tqdm.tqdm(jsdata):
        np_data = np.loadtxt(pointcloudpath) 
        np_labels = np.append(np_labels, np.unique(np_data[:, -1]))

    return np_labels    

if __name__ == '__main__':
    _, num_training_files = getFilesCount(TRAINING_FILES_JSON)
    _, num_testing_files = getFilesCount(TESTING_FILES_JSON)
    _, num_validation_files = getFilesCount(VALIDATION_FILES_JSON)
    sum_total = num_training_files + num_testing_files + num_validation_files

    # training_labels = getLabelsInDataset(TRAINING_FILES_JSON)
    # testing_labels = getLabelsInDataset(TESTING_FILES_JSON)
    # validation_labels = getLabelsInDataset(VALIDATION_FILES_JSON)

    # all_labels = np.array([])
    # all_labels = np.append(all_labels, training_labels)
    # all_labels = np.append(all_labels, testing_labels)
    # all_labels = np.append(all_labels, validation_labels)
    # all_labels = np.sort(np.unique(all_labels))

    print(f'DATASETS INFO:-> TRAINING: {num_training_files}, TESTING: {num_testing_files}, VALIDATION: {num_validation_files}, SUM TOTAL: {sum_total}')
    print(f'DATASETS DISTRIBUTION PERCENTAGE:-> TRAINING: {(num_training_files/sum_total)*100}%, TESTING: {(num_testing_files/sum_total)*100.0}%, VALIDATION: {(num_validation_files/sum_total)*100}%')
    # print(f'LABELS :: {all_labels}, MIN: {np.min(all_labels)}, MAX: {np.max(all_labels)}')