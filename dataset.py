import trimesh
import os 
import torch
from torch.utils.data import Dataset
from path import Path
from sample_normalize_points import SamplePoints




class ModelNet10Datset(Dataset):
    def __init__(self, root_dir, valid=False, folder="train"):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        pcd = trimesh.load(file)
        sampled_pc = trimesh.sample.sample_surface(pcd,5000)
        pc = sampled_pc[0]
        k=SamplePoints(pc)
        pc=k.farthestPointSample()
        pc=k.normalize(pc)
        if self.valid == False:
            pc=k.RandomRotation(pc)
            pc=k.RandomNoise(pc)
        pc=k.totensor(pc)
        return pc

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']

        pointcloud = self.__preproc__(pcd_path)

        return {'pointcloud': pointcloud, 'category': self.classes[category]}