import pathlib
import numpy as np
import tqdm

SEGMENTATION_LABELS: list  = [ 'Background', 'Left Hand', 'Right Eye', 'Right Leg', 'Left Eye', 'Left Cheek',
        'Rest Of Face', 'Right Ear', 'Left Leg', 'Chest', 'Left Ear', 'Left Feet', 
        'Right Arm', 'Right Hand', 'Forehead', 'Right Cheek', 'Abdomen', 'Lips',
        'Right Feet', 'Nose', 'Left Arm'
        ]

REDUCED_SEGMENTATION_CATEGORIES: list = [
        'background', 'arm', 'leg', 'face', 'thorax'
]

SEGMENTATION_CATEGORIES: dict = {
    'background': 0, 'arm': 1, 'hand': 1, 
    'leg': 2, 'feet': 2, 
    'eye': 3, 'cheek': 3, 'ear': 3, 'forehead': 3, 'face': 3,  'lips': 3, 'nose': 3,
    'abdomen': 4, 'chest': 4
    }


CLASSES_21_path: pathlib.Path = pathlib.Path('../data/syntheticpartnormal/pediatrics-NPY-21')
CLASSES_5_path: pathlib.Path = pathlib.Path('../data/syntheticpartnormal/pediatrics-NPY')

def partialMatch(searchFor: str, names: list)->str:
        return [s for s in names if searchFor.lower() in s.lower() or s.lower() in searchFor.lower()][0]

def reduceClasses(npy_files: pathlib.Path, destination_npy_files: pathlib.Path, map_from: list, map_to_names: list, map_to: dict) ->None:
        seg_category_keys: list = list(map_to.keys())
        remapped_seg_indices: np.ndarray = np.array([])

        for i, seg_name in enumerate(map_from):
                mapped_key = partialMatch(seg_name, seg_category_keys)
                remapped_seg_indices = np.append(remapped_seg_indices, map_to[mapped_key])

        for i, npy_file in tqdm.tqdm(enumerate(npy_files)):
                save_npy_path: pathlib.Path = destination_npy_files.joinpath(f'{npy_file.stem}.npy')
                np_data: np.ndarray = np.load(npy_file)
                np_data[:,-1] = remapped_seg_indices[np_data[:,-1].astype(int)]
                np.save(save_npy_path, np_data)

if __name__ == '__main__':
        # reduceClasses(CLASSES_21_path.glob('*.npy'), CLASSES_5_path, SEGMENTATION_LABELS, REDUCED_SEGMENTATION_CATEGORIES, SEGMENTATION_CATEGORIES)
        test_np_data = np.load(CLASSES_5_path.joinpath('10905-PLY.npy'))
        print(np.unique(test_np_data[:,6]))