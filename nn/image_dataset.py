import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image

class TrainImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_parquet(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.class_dict = self.get_class_dict()
        print(self.class_dict)


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        label = self.class_dict[label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)

        return sample


    def get_classes(self):
        return self.img_labels["class"].unique()


    def get_class_weights(self):
        value_counts = self.img_labels["class"].value_counts(normalize=True)
        class_dict = self.get_class_dict()
        print(value_counts, self.get_class_dict())
        inverse = 1 - value_counts
        values = (inverse - value_counts.mean()) / (value_counts.max() - value_counts.min())

        weights = [0] * len(class_dict)
        for index in range(len(class_dict)):
            # print(values[index], values.index[index])
            weights[class_dict[values.index[index]]] = values[index]

        return weights


    def get_class_dict(self):
        classes = self.get_classes()
        return {class_name: index for index, class_name in enumerate(classes)}
