import numpy as np
from einops import rearrange, repeat
import torch
from torch import nn, einsum

import functions as funcs
import optimization as opt_fn


class Residual_Connection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Layer_Normal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def conv3x3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Conv3d(nn.Module):
    def __init__(self, in_channels, conv_levels, f_maps):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(conv_levels):
            if i == 0:
                in_planes, out_planes = in_channels, f_maps[i]
            else:
                in_planes, out_planes = f_maps[i - 1], f_maps[i]
            self.layers.append(conv3x3x3(in_planes, out_planes, stride=1))
            self.layers.append(nn.BatchNorm3d(out_planes))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1))

    def forward(self, x):
        b, l, n_l, n_h, n_w = x.shape
        x = rearrange(x, 'b (l c) n_l n_h n_w -> (b l) c n_l n_h n_w', c=1)
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, '(b l) c n_l n_h n_w  -> b l c n_l n_h n_w', l=l)
        return x


class temporal_flattening_projection(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()

        self.flattening = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.projection = nn.Linear(embedding_dim[0], embedding_dim[1])

    def forward(self, x):
        b, l, c, n_l, n_h, n_w = x.shape
        x = rearrange(x, 'b l c n_l n_h n_w -> (b l) c n_l n_h n_w')
        x = self.flattening(x)
        x = rearrange(x, 'bl c n_l n_h n_w -> bl (c n_l n_h n_w)')
        x = self.projection(x)
        x = rearrange(x, '(b l) d -> b l d', b=b)
        return x


class temporal_correlation_analysis(nn.Module):
    def __init__(self, embedding_dim, head_num, encoder_num):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(encoder_num):
            self.layers.append(
                nn.TransformerEncoderLayer(embedding_dim, nhead=head_num, dropout=0.1, activation='relu',
                                           batch_first=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class sampling_point_classify(nn.Module):
    def __init__(self, num_class, dim_list):
        super().__init__()

        self.MLP = MLP_Block(dim_list[0], dim_list[1], num_class)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x, p):
        b, l, n = x.shape
        x = rearrange(x, 'b l n -> (b l) n')
        x = self.MLP(x)
        if p != 'training':
            x = self.soft_max(x)
        x = rearrange(x, '(b l) n -> b l n', b=b)
        return x


class temporal_semantic_learning(nn.Module):
    def __init__(self, *, num_classes, pattern, cube_size, num_cubes, cubes_step, in_channels, conv_levels, conv_maps,
                 feature_channels, embedding_dim, transfromer_param, class_dim):
        super().__init__()

        self.pattern = pattern
        self.cube_size, self.num_cubes, self.cubes_step = cube_size, num_cubes, cubes_step
        self._3dcnn = Conv3d(in_channels=in_channels, conv_levels=conv_levels, f_maps=conv_maps)
        self.flattening_projection = temporal_flattening_projection(in_channels=feature_channels[0],
                                                                    out_channels=feature_channels[1],
                                                                    embedding_dim=embedding_dim)
        self.temporal_correlation_analysis = temporal_correlation_analysis(embedding_dim=embedding_dim[1],
                                                                           head_num=transfromer_param[0],
                                                                           encoder_num=transfromer_param[1])
        self.softmax_classify = sampling_point_classify(num_class=num_classes, dim_list=[class_dim[0], class_dim[1]])

    def forward(self, img):
        b, c, n_h, n_w = img.shape
        x = funcs._3d_cubes_selection(img, cube_size=self.cube_size, num_cubes=self.num_cubes, step=self.cubes_step, batch_size=b)
        x = self._3dcnn(x)
        x = self.flattening_projection(x)
        x = self.temporal_correlation_analysis(x)
        x = self.softmax_classify(x, self.pattern)
        return x


class _2d_extraction_block(nn.Module):
    def __init__(self, *, in_channels, out_channels, _2d_conv_num, feature_weight):
        super().__init__()

        self.feature_weight = feature_weight
        self.layers = nn.ModuleList([])
        self.layers.append(conv3x3(in_channels, out_channels, stride=1))
        self.layers.append(conv3x3(out_channels, out_channels, stride=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding=0))

    def _get_2d_view_feature(self, _3d_features):

        b, c, n_l, n_h, n_w = _3d_features.shape
        view1 = _3d_features[:, :, :, n_h // 2]
        view2 = _3d_features[:, :, :, :, n_w // 2]
        view3 = _3d_features[:, :, np.arange(n_l)[:, None], np.arange(n_h), np.arange(n_w)]
        view4 = _3d_features[:, :, np.arange(n_l)[:, None], np.arange(n_h), np.arange(n_w - 1, -1, -1)]
        ret_feature = torch.cat((view1.unsqueeze(1), view2.unsqueeze(1), view3.unsqueeze(1), view4.unsqueeze(1)), dim=1)
        return ret_feature

    def _2d_maps_to_3d_maps(self, _2d_features):

        b, v, c, n_l, n_h = _2d_features.shape
        ret_features = _2d_features.unsqueeze(-1).repeat(1, 1, 1, 1, 1, n_h)
        feature_weight = torch.tensor(self.feature_weight).view(1, 4, 1, 1, 1, 1)
        ret_features = (ret_features * feature_weight).sum(dim=1)
        return ret_features

    def forward(self, x):
        b, c, n_l, n_h, n_w = x.shape
        x = self._get_2d_view_feature(x)
        x = rearrange(x, 'b n c n_l n_h -> (b n) c n_l n_h')
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, '(b n) c n_l n_h -> b n c n_l n_h', b=b)
        x = self._2d_maps_to_3d_maps(x)
        return x


class _3d_extraction_block(nn.Module):
    def __init__(self, *, in_channels, out_channels, _3d_conv_num):
        super().__init__()

        self.layers = nn.ModuleList([])
        for j in range(_3d_conv_num):
            if j == 0:
                self.layers.append(conv3x3x3(in_channels, out_channels, stride=1))
            else:
                self.layers.append(conv3x3x3(out_channels, out_channels, stride=1))
        self.layers.append(nn.BatchNorm3d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=0))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class feature_extraction_3d(nn.Module):
    def __init__(self, *, in_channels, conv_levels, f_maps, conv_num_3d, conv_num_2d, _2d_weight, _3d_weight):
        super().__init__()

        self.conv_levels = conv_levels
        self._3d_weight = _3d_weight

        self._3d_extraction_blocks = [
            _3d_extraction_block(in_channels=f_maps[i - 1] if i > 0 else in_channels,
                                 out_channels=f_maps[i],
                                 _3d_conv_num=conv_num_3d[i])
            for i in range(self.conv_levels)
        ]

        self._2d_extraction_blocks = [
            _2d_extraction_block(in_channels=f_maps[i - 1] if i > 0 else in_channels,
                                 out_channels=f_maps[i],
                                 _2d_conv_num=conv_num_2d[i],
                                 feature_weight=_2d_weight
                                 )
            for i in range(self.conv_levels)
        ]

    def forward(self, x):
        b, n_l, n_h, n_w = x.shape
        x = rearrange(x, '(b c) n_l n_h n_w -> b c n_l n_h n_w', c=1)
        x_3d = None

        for i in range(self.conv_levels):
            if i == 0:
                x_3d = self._3d_extraction_blocks[i](x)
                x_2d = self._2d_extraction_blocks[i](x)
            else :
                x_2d = self._2d_extraction_blocks[i](x_3d)
                x_3d = self._3d_extraction_blocks[i](x_3d)
                x_3d = self._3d_weight * x_3d + (1 - self._3d_weight) * x_2d

        return x_3d


class spatial_flattening_projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.flattening = nn.Conv3d(in_channels[0], out_channels[0], kernel_size=1)
        self.projection = nn.Linear(in_channels[1], out_channels[1])

    def forward(self, x):
        b, c, n_l, n_h, n_w = x.shape
        x = rearrange(x, 'b (c ct1 ct2) n_l n_h n_w  -> (b c) ( n_l n_h ct1 n_w ct2)', ct1=2, ct2=2)
        x = self.projection(x)
        x = rearrange(x, '(b c) n -> b c n', b=b)
        return x


class bounding_box_prediction(nn.Module):
    def __init__(self, num_class, dim_list):
        super().__init__()

        self.class_prediction = MLP_Block(dim_list[0], dim_list[1], num_class)
        self.boxes_prediction = MLP_Block(dim_list[0], dim_list[1], 2)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x, p):
        b, l, c = x.shape
        x = rearrange(x, 'b l c -> (b l) c')
        x1 = self.class_prediction(x)
        x2 = self.boxes_prediction(x).sigmoid()
        if p != 'training':
            x1 = self.soft_max(x1)
        x1 = rearrange(x1, '(b l) n -> b l n', b=b)
        x2 = rearrange(x2, '(b l) n -> b l n', b=b)
        return x1, x2


class spatial_semantic_learning(nn.Module):
    def __init__(self, *, num_classes, pattern, in_channels, conv_levels, f_maps, conv_num_3d, conv_num_2d, _2d_weight,
                 _3d_weight, proj_channels, embedding_shape, num_layers, num_query, od_dim_list):
        super().__init__()

        self.num_query = num_query
        self.pattern = pattern

        self.feature_3d = feature_extraction_3d(in_channels=in_channels, conv_levels=conv_levels, f_maps=f_maps,
                                                conv_num_3d=conv_num_3d, conv_num_2d=conv_num_2d, _2d_weight=_2d_weight,
                                                _3d_weight=_3d_weight)
        self.flattening_projection = spatial_flattening_projection(in_channels=[proj_channels[0], proj_channels[1]],
                                                                   out_channels=[proj_channels[2], proj_channels[3]])
        self.embedding_maker = nn.Embedding(embedding_shape[0], embedding_shape[1])
        self.transformer_architecture = nn.Transformer(d_model=embedding_shape[1],
                                                       num_encoder_layers=num_layers[0],
                                                       num_decoder_layers=num_layers[1])
        self.object_detection = bounding_box_prediction(num_class=num_classes, dim_list=od_dim_list)

    def forward(self, img):
        b, n_l, n_h, n_w = img.shape

        x = self.feature_3d(img)
        emb_f = self.flattening_projection(x)
        emb_q = self.embedding_maker(torch.randint(0, self.num_query, (b, self.num_query)))
        emb_f, emb_q = emb_f.transpose(0, 1), emb_q.transpose(0, 1)
        x = self.transformer_architecture(emb_f, emb_q).transpose(0, 1)
        xc, xb = self.object_detection(x, self.pattern)

        return xc, xb


class spatio_temporal_semantic_learning(nn.Module):
    def __init__(self, *, num_classes, pattern, ret_map, in_channels, _3d_cube_selection, temporal_conv_levels,
                 temporal_conv_maps, temporal_feature_channels, temporal_embedding_dim, temporal_transfromer_param,
                 temporal_class_dim, spatial_conv_levels, spatial_conv_maps, spatial_3dconv_layers,
                 spatial_2dconv_layers, spatial_2d_weight, spatial_3d_weight, spatial_proj_channels,
                 spatial_embedding_shape, spatial_transfromer_param, spatial_num_query, spatial_od_dim_list
                 ):
        super().__init__()

        self.num_classes = num_classes + 1
        self.pattern = pattern
        self.ret_map = ret_map

        self.sampling_point_framework = temporal_semantic_learning(
            num_classes=self.num_classes,
            pattern=pattern,
            cube_size=_3d_cube_selection[1],
            num_cubes=_3d_cube_selection[0],
            cubes_step=_3d_cube_selection[2],
            in_channels=in_channels,
            conv_levels=temporal_conv_levels,
            conv_maps=temporal_conv_maps,
            feature_channels=temporal_feature_channels,
            embedding_dim=temporal_embedding_dim,
            transfromer_param=temporal_transfromer_param,
            class_dim=temporal_class_dim
        )

        self.object_detection_framework = spatial_semantic_learning(
            num_classes=self.num_classes,
            pattern=pattern,
            in_channels=in_channels,
            conv_levels=spatial_conv_levels,
            f_maps=spatial_conv_maps,
            conv_num_3d=spatial_3dconv_layers,
            conv_num_2d=spatial_2dconv_layers,
            _2d_weight=spatial_2d_weight,
            _3d_weight=spatial_3d_weight,
            proj_channels=spatial_proj_channels,
            embedding_shape=spatial_embedding_shape,
            num_layers=spatial_transfromer_param,
            num_query=spatial_num_query,
            od_dim_list=spatial_od_dim_list
        )

    def sc_pred2map(self, pred_xs):
        ret_outputs = {
            "pred_logits": pred_xs
        }
        return ret_outputs

    def od_pred2map(self, pred_xc, pred_xb):
        ret_outputs = {
            "pred_logits": pred_xc,
            "pred_boxes": pred_xb
        }
        return ret_outputs

    def forward(self, img):

        xs = self.sampling_point_framework(img)
        xc, xb = self.object_detection_framework(img)

        if self.ret_map == True:
            if self.pattern == 'training':
                return self.od_pred2map(xc, xb), self.sc_pred2map(xs)
            else :
                return self.od_pred2map(xc, xb)
        else:
            if self.pattern == 'training':
                return xc, xb, xs
            else:
                return xc, xb