import torch
import torchvision
import torch.nn as nn
import os


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class PlantData(torch.utils.data.Dataset):
    def __init__(self, csv, image_path, image_transform, data_transform, targets_transform):
        self.csv = csv
        self.image_path = image_path
        self.image_transform = image_transform
        self.data_transform = data_transform
        self.targets_transform = targets_transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_path, str(self.csv.iloc[index, 0]) + '.jpeg')
        image = torchvision.io.read_image(image_path)

        transformed = self.image_transform(image)

        additional = self.data_transform(torch.tensor(self.csv.iloc[index, 1:-6], dtype=torch.float32))

        targets = self.targets_transform(torch.tensor(self.csv.iloc[index, -6:], dtype=torch.float32))

        return transformed, additional, targets
    
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class PlantDataTest(torch.utils.data.Dataset):
    def __init__(self, csv, image_path, image_transform, data_transform):
        self.csv = csv
        self.image_path = image_path
        self.image_transform = image_transform
        self.data_transform = data_transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_path, str(self.csv.iloc[index, 0]) + '.jpeg')
        image = torchvision.io.read_image(image_path)

        transformed = self.image_transform(image)

        additional = self.data_transform(torch.tensor(self.csv.iloc[index, 1:], dtype=torch.float32))

        return transformed, additional
