import pathlib
import numpy as np
import tqdm

SAMPLE_SIZE: int = 2048

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


# choice = np.random.choice(min(len(seg), self.npoints), self.npoints, replace=True)

CLASSES_21_path: pathlib.Path = pathlib.Path('../data/syntheticpartnormal/pediatrics-NPY-21')
CLASSES_5_path: pathlib.Path = pathlib.Path('../data/syntheticpartnormal/pediatrics-NPY')

def partialMatch(searchFor: str, names: list)->str:
        return [s for s in names if searchFor.lower() in s.lower() or s.lower() in searchFor.lower()][0]

def reduceClasses(
                npy_files: pathlib.Path.glob, destination_npy_files: pathlib.Path, 
                map_from: list, map_to_names: list, map_to: dict,
                reduce_to_sample_size: int = 2048) ->None:
        seg_category_keys: list = list(map_to.keys())
        remapped_seg_indices: np.ndarray = np.array([])

        # How many points to write in final point cloud per segmentation category
        count_points_per_segmentation: float = int(np.floor((1.0 / len(map_to_names)) * reduce_to_sample_size))
        destination_npy_files.mkdir(parents=True, exist_ok=True)

        errors = ''
        npy_files = list(npy_files)
        for i, seg_name in enumerate(map_from):
                mapped_key = partialMatch(seg_name, seg_category_keys)
                remapped_seg_indices = np.append(remapped_seg_indices, map_to[mapped_key])

        for i, npy_file in tqdm.tqdm(enumerate(npy_files), total=len(npy_files), dynamic_ncols=True):
                # Load the 21 classes npy file
                save_npy_path: pathlib.Path = destination_npy_files.joinpath(f'{npy_file.stem}.npy')
                np_data: np.ndarray = np.load(npy_file)
                final_np_data: np.ndarray = np.zeros((0, np_data.shape[1]))
                
                # Remap the segmentation ids from 0->21 to 0->5
                np_data[:,-1] = remapped_seg_indices[np_data[:,-1].astype(int)]
                
                #Reduce the #N points in the point cloud to #reduce_to_sample_size
                #Also ensure to a maximum the points are distributed equally
                for seg_id, mapped_name in enumerate(map_to_names):
                        seg_np_data_full: np.ndarray = np_data[(np_data[:, -1] == seg_id)]
                        seg: np.ndarray = seg_np_data_full[:,-1]
                        
                        npoints: int = min(len(seg), count_points_per_segmentation)
                        choice = np.random.choice(npoints, npoints, replace=True)
                        data_points_seg: np.ndarray = seg_np_data_full[choice]                       
                        
                        try:
                                # Ensure there are only one segmentation id in this subset
                                assert np.unique(data_points_seg[:,-1]).shape[0] == 1
                                # Ensure the segmentation ids in this subset matches the correct segmentation id
                                assert np.unique(data_points_seg[:,-1])[0] == seg_id
                        except AssertionError:
                                errors = f'{"#"*50}\n'
                                errors = f'{npy_file.stem}\n'
                                errors = f'{errors}\n{mapped_name} has no points'
                        
                        final_np_data = np.vstack((final_np_data, data_points_seg))
                        
                
                assert final_np_data.shape[0] < reduce_to_sample_size
                assert final_np_data.shape[0] > 0

                np.save(save_npy_path, final_np_data)

        f = open(destination_npy_files.parent.joinpath('synthetic-reduction-errors.log'), 'w')
        f.write(errors)
        f.close()

if __name__ == '__main__':
        reduceClasses(
                CLASSES_21_path.glob('*.npy'), CLASSES_5_path, SEGMENTATION_LABELS, 
                REDUCED_SEGMENTATION_CATEGORIES, SEGMENTATION_CATEGORIES, 
                SAMPLE_SIZE)
        # test_np_data = np.load(CLASSES_5_path.joinpath('10905-PLY.npy'))
        # print(np.unique(test_np_data[:,6]))