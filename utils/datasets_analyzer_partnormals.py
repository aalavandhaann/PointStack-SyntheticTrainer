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


class DatasetInfo():

    _sum_total_files: int
    _sum_total_points: int

    _num_training_files: int
    _num_testing_files: int
    _num_validation_files: int    

    _segmentation_labels: np.ndarray

    _labels_in_dataset: np.ndarray
    _count_segmentation_labels: np.ndarray
    
    
    _training_json: pathlib.Path
    _testing_json: pathlib.Path
    _validation_json: pathlib.Path


    def __init__(self, segmentation_labels: list, training_json: pathlib.Path, testing_json: pathlib.Path, validation_json: pathlib.Path) -> None:
        self._segmentation_labels = np.array(segmentation_labels)
        self._training_json = training_json
        self._testing_json = testing_json
        self._validation_json = validation_json
        self._process()

    
    def _process(self)->None:
        _, self._num_training_files = self._getFilesCount(self._training_json)
        _, self._num_testing_files = self._getFilesCount(self._testing_json)
        _, self._num_validation_files = self._getFilesCount(self._validation_json)

        training_labels, training_counts, total_training_points = self._getLabelsInDataset(TRAINING_FILES_JSON)
        testing_labels, testing_counts, total_testing_points = self._getLabelsInDataset(TESTING_FILES_JSON)
        validation_labels, validation_counts, total_validation_points = self._getLabelsInDataset(VALIDATION_FILES_JSON)

        self._sum_total_files = self._num_training_files + self._num_testing_files + self._num_validation_files
        self._sum_total_points = total_training_points + total_testing_points + total_validation_points

        all_labels = np.array([])
        all_labels = np.append(all_labels, training_labels)
        all_labels = np.append(all_labels, testing_labels)
        all_labels = np.append(all_labels, validation_labels)

        self._labels_in_dataset = np.sort(np.unique(all_labels))
        self._count_segmentation_labels = training_counts + testing_counts + validation_counts

    def _getFilesCount(self, json_path: pathlib.Path) -> tuple:    
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

    def _getLabelsInDataset(self, json_path: pathlib.Path, num_classes=21) -> np.ndarray:
        jsdata, _ = self._getFilesCount(json_path)
        np_labels = np.array([])
        np_counts = np.zeros((num_classes))
        total_points = 0
        for i, pointcloudpath in tqdm.tqdm(enumerate(jsdata), dynamic_ncols=True):
            np_data = np.loadtxt(pointcloudpath) 
            values, counts = np.unique(np_data[:, -1], return_counts=True)
            np_labels = np.append(np_labels, values)
            np_counts[values.astype(int)] += counts
            total_points += np_data.shape[0]
            if(i > 3):
                break

        return np_labels, np_counts, total_points

    def _distribution(self) -> tuple:
        training_distribution: int = (self._num_training_files/self._sum_total_files)*100
        testing_distribution: int = (self._num_testing_files/self._sum_total_files)*100.0
        validation_distribution: int = (self._num_validation_files/self._sum_total_files)*100
        return training_distribution, testing_distribution, validation_distribution
    
    def _distributionPoints(self) -> np.ndarray:
        return ((self._count_segmentation_labels.astype(float) / float(self._sum_total_points))*100.0).astype(int)

    def __str__(self) -> str:
        train_percent, test_percent, validate_percent = self._distribution()
        line1 = f'DATASETS INFO:-> TRAINING: {self._num_training_files}, TESTING: {self._num_testing_files}, VALIDATION: {self._num_validation_files}, SUM TOTAL: {self._sum_total_files}'
        line2 = f'DATASETS DISTRIBUTION PERCENTAGE:-> TRAINING: {train_percent}%, TESTING: {test_percent}%, VALIDATION: {validate_percent}%'
        line3 = f'LABELS :: {self._labels_in_dataset}, MIN: {np.min(self._labels_in_dataset)}, MAX: {np.max(self._labels_in_dataset)}'
        line4 = f'Counts: {self._count_segmentation_labels}'
        line5 = f'Segmentation Labels:  {SEGMENTATION_LABELS}'
        line6 = f'Segmentation Distribution %: {self._distributionPoints()}'

        return f'{line1}\n{line2}\n{line3}\n{line4}\n{line5}\n{line6}'
    
    def saveLOG(self, save_name: str, destination_dir: pathlib.Path)->pathlib.Path:
        file_path: pathlib.Path = destination_dir.joinpath(f'{save_name}-dataset-info.log')
        f = open(file_path, 'w')
        f.write(str(self))
        f.close()
        return file_path
    
    def graph(self, name: str, destination_dir: pathlib.Path, mode: str = 'files'):
        if(mode == 'files'):
            files_distribution = np.ndarray(self._distribution())
        elif (mode == 'points'):
            points_distribution = self._distributionPoints()





if __name__ == '__main__':

    d: DatasetInfo = DatasetInfo(SEGMENTATION_LABELS, TRAINING_FILES_JSON, TESTING_FILES_JSON, VALIDATION_FILES_JSON)
    # d.saveLOG(DATASETS_DIRECTORY.stem, pathlib.Path(__file__))
    d.saveLOG('HELLO', pathlib.Path(__file__).parent)
    
    
    # f = open(f'{DATASETS_DIRECTORY.stem}-dataset-info.log', 'w')
    # f.write(f'{line1}\n{line2}\n{line3}\n{line4}\n{line5}\n{line6}')
    # f.close()
