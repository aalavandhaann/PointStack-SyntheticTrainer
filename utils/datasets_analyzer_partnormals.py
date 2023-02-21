import pathlib
import numpy as np
import json
import tqdm

DATASETS_DIRECTORY = pathlib.Path(__file__).parent.absolute().joinpath('../data/syntheticpartnormal/').absolute()
TRAINING_FILES_JSON = DATASETS_DIRECTORY.joinpath('train_test_split/shuffled_train_file_list.json')
TESTING_FILES_JSON = DATASETS_DIRECTORY.joinpath('train_test_split/shuffled_test_file_list.json')
VALIDATION_FILES_JSON = DATASETS_DIRECTORY.joinpath('train_test_split/shuffled_val_file_list.json')
SEGMENTATION_LABELS  = [ 'Background', 'Left Hand', 'Right Eye', 'Right Leg', 'Left Eye', 'Left Cheek',
        'Rest Of Face', 'Right Ear', 'Left Leg', 'Chest', 'Left Ear', 'Left Feet', 
        'Right Arm', 'Right Hand', 'Forehead', 'Right Cheek', 'Abdomen', 'Lips',
        'Right Feet', 'Nose', 'Left Arm'
        ]


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

def getLabelsInDataset(json_path: pathlib.Path, num_classes=21) -> np.ndarray:
    jsdata, _ = getFilesCount(json_path)
    np_labels = np.array([])
    np_counts = np.zeros((num_classes))
    total_points = 0
    for pointcloudpath in tqdm.tqdm(jsdata, dynamic_ncols=True):
        np_data = np.loadtxt(pointcloudpath) 
        values, counts = np.unique(np_data[:, -1], return_counts=True)
        np_labels = np.append(np_labels, values)
        np_counts[values.astype(int)] += counts
        total_points += np_data.shape[0]

    return np_labels, np_counts, total_points

if __name__ == '__main__':
    _, num_training_files = getFilesCount(TRAINING_FILES_JSON)
    _, num_testing_files = getFilesCount(TESTING_FILES_JSON)
    _, num_validation_files = getFilesCount(VALIDATION_FILES_JSON)
    sum_total = num_training_files + num_testing_files + num_validation_files

    training_labels, training_counts, total_training_points = getLabelsInDataset(TRAINING_FILES_JSON)
    testing_labels, testing_counts, total_testing_points = getLabelsInDataset(TESTING_FILES_JSON)
    validation_labels, validation_counts, total_validation_points = getLabelsInDataset(VALIDATION_FILES_JSON)
    sum_total_points = total_training_points + total_testing_points + total_validation_points

    all_labels = np.array([])
    all_labels = np.append(all_labels, training_labels)
    all_labels = np.append(all_labels, testing_labels)
    all_labels = np.append(all_labels, validation_labels)
    all_labels = np.sort(np.unique(all_labels))
    all_counts = training_counts + testing_counts + validation_counts


    
    line1 = f'DATASETS INFO:-> TRAINING: {num_training_files}, TESTING: {num_testing_files}, VALIDATION: {num_validation_files}, SUM TOTAL: {sum_total}'
    line2 = f'DATASETS DISTRIBUTION PERCENTAGE:-> TRAINING: {(num_training_files/sum_total)*100}%, TESTING: {(num_testing_files/sum_total)*100.0}%, VALIDATION: {(num_validation_files/sum_total)*100}%'
    line3 = f'LABELS :: {all_labels}, MIN: {np.min(all_labels)}, MAX: {np.max(all_labels)}'
    line4 = f'Counts: {all_counts}'
    line5 = f'Segmentation Labels:  {SEGMENTATION_LABELS}'
    line6 = f'Segmentation Distribution %: {(all_counts.astype(float) / float(sum_total_points))*100.0}'

    print(line1)
    print(line2)
    print(line3)
    print(line4)
    print(line5)
    print(line6)

    f = open('dataset-info.log', 'w')
    f.write(f'{line1}\n{line2}\n{line3}\n{line4}\n{line5}\n{line6}')
    f.close()
