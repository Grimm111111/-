import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SkeletonDataset(Dataset):
    def __init__(self, joint_data, bone_data, labels):
        self.joint_data = (joint_data - joint_data.mean()) / joint_data.std()  # 标准化
        self.bone_data = (bone_data - bone_data.mean()) / bone_data.std()  # 标准化
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        joint = self.joint_data[idx]
        bone = self.bone_data[idx]
        label = self.labels[idx]
        return torch.tensor(joint, dtype=torch.float32), torch.tensor(bone, dtype=torch.float32), torch.tensor(label, dtype=torch.long)



def load_data(joint_path, bone_path, label_path, batch_size=16, shuffle=True):
    # 加载数据
    joint_data = np.load(joint_path)
    bone_data = np.load(bone_path)
    labels = np.load(label_path)

    # 创建数据集和数据加载器
    dataset = SkeletonDataset(joint_data, bone_data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
