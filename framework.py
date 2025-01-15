import os
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader

from architecture import spatio_temporal_semantic_learning
from config import opt
import functions as funcs
import optimization as opt_fn
import augmentation as aug


class sc_net_framework:

    def __init__(self, pattern='pre_training', state_dict_root=None):

        if pattern == "pre_training":
            self.model_pattern = "training"
            self.model_num_classes = opt.net_params["num_classes"][0]
        elif pattern == "fine_tuning":
            self.model_pattern = "training"
            self.model_num_classes = opt.net_params["num_classes"][1]
        else:
            self.model_pattern = "testing"
            self.model_num_classes = opt.net_params["num_classes"][1]

        self.model = self.get_model()
        self.state_dict_root = state_dict_root
        if self.state_dict_root is not None:
            self.pre_training_load()

        if pattern != 'inference':

            self.data_root = opt.data_params["dataset_root"]
            self.train_ratio = opt.data_params["train_ratio"]
            self.input_shape = opt.net_params["input_shape"]
            self.window_lw = opt.data_params["window_lw"]
            self.batch_size = opt.data_params["batch_size"]

            self.loss_fn = self.get_loss_fn()
            self.dataLoader_train, self.dataLoader_eval = self.get_dataloader()

    def get_model(self):
        return spatio_temporal_semantic_learning(
            num_classes=self.model_num_classes,
            pattern=self.model_pattern,
            ret_map=opt.net_params["ret_map"],
            in_channels=opt.net_params["in_channels"],
            _3d_cube_selection=opt.sc_params["_3d_cube_selection"],
            temporal_conv_levels=opt.sc_params["temporal_conv_levels"],
            temporal_conv_maps=opt.sc_params["temporal_conv_maps"],
            temporal_feature_channels=opt.sc_params["temporal_feature_channels"],
            temporal_embedding_dim=opt.sc_params["temporal_embedding_dim"],
            temporal_transfromer_param=opt.sc_params["temporal_transfromer_param"],
            temporal_class_dim=opt.sc_params["temporal_class_dim"],
            spatial_conv_levels=opt.od_params["spatial_conv_levels"],
            spatial_conv_maps=opt.od_params["spatial_conv_maps"],
            spatial_3dconv_layers=opt.od_params["spatial_3dconv_layers"],
            spatial_2dconv_layers=opt.od_params["spatial_2dconv_layers"],
            spatial_2d_weight=opt.od_params["spatial_2d_weight"],
            spatial_3d_weight=opt.od_params["spatial_3d_weight"],
            spatial_proj_channels=opt.od_params["spatial_proj_channels"],
            spatial_embedding_shape=opt.od_params["spatial_embedding_shape"],
            spatial_transfromer_param=opt.od_params["spatial_transfromer_param"],
            spatial_num_query=opt.od_params["spatial_num_query"],
            spatial_od_dim_list=opt.od_params["spatial_od_dim_list"]
        )

    def get_loss_fn(self):
        return opt_fn.spatio_temporal_contrast_loss(
            num_classes=self.model_num_classes,
            seq_length=opt.net_params["cubeseq_length"],
            eos_coef=opt.data_params["eos_coef"]
        )

    def get_dataloader(self):
        dataset_training = aug.cubic_sequence_data(
            dataset_root=self.data_root,
            pattern='training',
            train_ratio=self.train_ratio,
            input_shape=self.input_shape,
            window=self.window_lw)
        dataset_testing = aug.cubic_sequence_data(
            dataset_root=self.data_root,
            pattern='testing',
            train_ratio=self.train_ratio,
            input_shape=self.input_shape,
            window=self.window_lw)
        return DataLoader(dataset_training, batch_size=self.batch_size, shuffle=True, collate_fn=aug.collate_fn),\
               DataLoader(dataset_testing, batch_size=self.batch_size, shuffle=False, collate_fn=aug.collate_fn)

    def pre_training_load(self, ):

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.state_dict_root)

        pretrained_dict_filtered = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_dict_filtered[k] = v

        model_dict.update(pretrained_dict_filtered)
        self.model.load_state_dict(model_dict)