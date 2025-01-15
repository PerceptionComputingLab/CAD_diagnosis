import os
import random
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from einops import rearrange
import nibabel as nib
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import zoom

from architecture import spatio_temporal_semantic_learning
import functions as funcs
import optimization as opt_fn
from config import opt

class clinically_credible_augmentation(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape

    def data_resize(self, original_data):
        data, label = original_data['cpr_volume'], original_data['label']
        inp_size = data.shape

        if data.shape[0] == self.input_shape[0]:
            return original_data

        data_zoom_factors = (
        self.input_shape[0] / inp_size[0], self.input_shape[1] / inp_size[1], self.input_shape[2] / inp_size[2])
        resized_data = zoom(data, data_zoom_factors, order=3)
        label_zoom_factors = (self.input_shape[0] / inp_size[0])
        resized_label = zoom(label, label_zoom_factors, order=1)
        return {"cpr_volume": resized_data, "label": resized_label}

    def data_generator(self, foreground_data, background_data):
        foreground_data = self.data_resize(foreground_data)
        background_data = self.data_resize(background_data)

        f_data, f_label = foreground_data['cpr_volume'], foreground_data['label']
        b_data, b_label = background_data['cpr_volume'], background_data['label']

        ret_data = np.full((256, 64, 64), -1024, dtype=np.int32)
        ret_label = f_label
        b_indices = np.where(b_label == 0)[0]
        ret_data[b_indices, :, :] = b_data[b_indices, :, :]
        f_indices = np.where(f_label > 0)[0]
        ret_data[f_indices, :, :] = f_data[f_indices, :, :]
        ret_label = np.where(ret_label > 0, ((ret_label - 1) % 3) + 1, ret_label)
        return {'cpr_volume': ret_data, 'label': ret_label}

    def read_data(self, volumes_file, labels_file):

        nii_file = nib.load(volumes_file)
        affine_matrix = nii_file.affine
        ret_volumes = nii_file.get_fdata()
        ret_volumes = ret_volumes.transpose(2, 0, 1)
        ret_volumes = np.array(ret_volumes)
        ret_labels = np.loadtxt(labels_file).astype(np.int32)

        return {'cpr_volume': ret_volumes, 'label': ret_labels}, affine_matrix

    def write_data(self, data_info, idx, augmented_root, affine_matrix):

        volumes = data_info['cpr_volume']
        labels = data_info['label']
        nii_image = nib.Nifti1Image(volumes, affine=affine_matrix)
        nib.save(nii_image, os.path.join(augmented_root, f'volumes/gen_{idx}.nii'))
        labels_str = '\n'.join(map(str, labels))
        with open(os.path.join(augmented_root, f'labels/gen_{idx}.txt'), 'w') as file:
            file.write(labels_str)

        return

    def forward(self, generated_num, original_root=r'original_data_root', augmented_root=r'augmented_data_root'):

        volumes_root = os.path.join(original_root, 'volumes/')
        labels_root = os.path.join(original_root, 'labels/')
        volumes_file_list = os.listdir(volumes_root)
        volumes_file_list = sorted(volumes_file_list)
        labels_file_list = os.listdir(labels_root)
        labels_file_list = sorted(labels_file_list)
        file_total = len(volumes_file_list)

        for i in range(generated_num):

            selected_f_idx, selected_d_idx = random.randint(0, file_total - 1), random.randint(0, file_total - 1)
            volumes_f_file = os.path.join(volumes_root, volumes_file_list[selected_f_idx])
            labels_f_file = os.path.join(labels_root, labels_file_list[selected_f_idx])
            volumes_d_file = os.path.join(volumes_root, volumes_file_list[selected_d_idx])
            labels_d_file = os.path.join(labels_root, labels_file_list[selected_d_idx])

            ori_f_data, affine_matrix = self.read_data(volumes_f_file, labels_f_file)
            ori_b_data, affine_matrix = self.read_data(volumes_d_file, labels_d_file)

            gen_data = self.data_generator(ori_f_data, ori_b_data)
            self.write_data(gen_data, i, augmented_root, affine_matrix)

        return


class cubic_sequence_data(data.Dataset):
    def __init__(self, dataset_root, pattern='training', train_ratio=0.8, input_shape=[256,64,64], window=[300, 900]):

        self.volumes_root = os.path.join(dataset_root, 'volumes/')
        self.labels_root = os.path.join(dataset_root, 'labels/')
        self.input_shape, self.window = input_shape, [window[0] - window[1] / 2, window[0] + window[1] / 2]

        self.volumes_file_list = os.listdir(self.volumes_root)
        self.volumes_file_list = sorted(self.volumes_file_list)
        self.labels_file_list = os.listdir(self.labels_root)
        self.labels_file_list = sorted(self.labels_file_list)
        self.file_total = len(self.volumes_file_list)
        if pattern == 'training':
            self.data_start, self.data_end = 0, int(self.file_total * train_ratio)
        else :
            self.data_start, self.data_end = int(self.file_total * train_ratio), self.file_total
        self.length = self.data_end - self.data_start
        return

    def read_data(self, volumes_file, labels_file):

        nii_file = nib.load(volumes_file)
        ret_volumes = nii_file.get_fdata()
        if ret_volumes.shape[0] == ret_volumes.shape[1]:
            ret_volumes = ret_volumes.transpose(2, 0, 1)
        ret_volumes = np.array(ret_volumes)
        ret_labels = np.loadtxt(labels_file).astype(np.int32)

        return ret_volumes, ret_labels

    def detection_targets(self, labels_data):

        boxes, labels = [], []
        start, label, length, last = None, 0, self.input_shape[0], -1

        for i in range(labels_data.shape[0]):
            if start is not None:
                if labels_data[i] != last:
                    boxes.append([(start + 1) / length, min((i + 1) / length, 1.0)])
                    labels.append(label - 1)
                    if labels_data[i] != 0:
                        start, label, last = i, labels_data[i], labels_data[i]
                    else: start, label, last = None, 0, -1
                else:
                    continue
            else:
                if labels_data[i] == 0:
                    start, label, last = None, 0, -1
                else:
                    start, label, last = i, labels_data[i], labels_data[i]

        if start is not None:
            boxes.append([(start + 1) / length, 1.0])
            labels.append(label - 1)

        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.torch.float32)
        return {"labels": labels, "boxes": boxes}

    def __getitem__(self, index):

        volumes_file = os.path.join(self.volumes_root, self.volumes_file_list[index])
        labels_file = os.path.join(self.labels_root, self.labels_file_list[index])
        ret_volumes, ret_labels = self.read_data(volumes_file, labels_file)
        ret_volumes = funcs.normalize_ct_data(ret_volumes, hu_min=self.window[0], hu_max=self.window[1])
        return {'image': torch.tensor(ret_volumes,dtype=torch.float32), 'target': self.detection_targets(ret_labels)}

    def __len__(self):
        return self.length


def collate_fn(batch):

    images, targets = [], []
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    images = torch.stack(images, dim=0)

    return images, targets