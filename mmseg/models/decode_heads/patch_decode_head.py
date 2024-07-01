# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy
from ..utils import resize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
class patch_BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg. Default: None.
        threshold (float): Threshold for binary segmentation in the case of
            `num_classes==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_cls (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 corruption_threshold=0.3,
                 corruption_step=2,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super().__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_channels=in_channels
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.corruption_threshold = corruption_threshold
        self.corruption_step = corruption_step

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert `seg_logits` into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_cls, dict):
            self.loss_cls = MODELS.build(loss_cls)
        elif isinstance(loss_cls, (list, tuple)):
            self.loss_cls = nn.ModuleList()
            for loss in loss_cls:
                self.loss_cls.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_cls must be a dict or sequence of dict,\
                but got {type(loss_cls)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            """
            upsampled_inputs
            [0] : [bs, 64, 128, 256]
            [1] : [bs, 128, 128, 256]
            [2] : [bs, 256, 128, 256]
            [3] : [bs, 512, 128, 256]
            """
            inputs = torch.cat(upsampled_inputs, dim=1)  # inputs : [bs, channel합, 128, 256]
           
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        results= self.forward(inputs)        
        losses = self.total_loss(results, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # seg_logits = self.forward(inputs)
        #return self.predict_by_feat(seg_logits, batch_img_metas) # resize해서 output image와 resolution 맞추는 용도
        return self.forward(inputs)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    def viz_seg(self, arr):
        arr = arr.cpu().numpy()
        # 색상 맵 정의
        cmap_binary = mcolors.ListedColormap(['white', 'blue'])
        cmap_custom = mcolors.ListedColormap(['white', 'blue', 'black'])

        # 서브플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # [0, :, :] 시각화
        ax1.imshow(arr[0], cmap=cmap_binary, vmin=0, vmax=1)
        ax1.set_title('[0, :, :]')
        ax1.axis('off')

        # [1, :, :] 시각화
        ax2.imshow(arr[1], cmap=cmap_custom, vmin=0, vmax=2)
        ax2.set_title('[1, :, :]')
        ax2.axis('off')

        # 색상바 추가
        cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_binary), ax=ax1, ticks=[0, 1], orientation='horizontal')
        cbar1.set_ticklabels(['White (0)', 'Blue (1)'])

        cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_custom), ax=ax2, ticks=[0, 1, 2], orientation='horizontal')
        cbar2.set_ticklabels(['White (0)', 'Blue (1)', 'Black (2)'])

        plt.tight_layout()

        # 파일로 저장
        plt.savefig('visualization_after.png', dpi=300)
    def total_loss(self, results: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #########
        #ky
        #########
        gt_labels = []
        for data_samples in batch_data_samples:
            gt_label = data_samples.gt_sem_seg.data.squeeze(0)
            gt_label = torch.stack((torch.div(gt_label, 4, rounding_mode='floor'), torch.remainder(gt_label, 4)), dim=0)
            gt_labels.append(gt_label)
        gt_labels = torch.stack(gt_labels, dim=0)

        gt_labels = self._stack_batch_gt(batch_data_samples)
        #16x16x2로 임시 reshape
        """
        gt_labels = self._stack_batch_gt(batch_data_samples)
        gt_labels = gt_labels.float()
        gt_labels = F.interpolate(gt_labels, size=(16, 16), mode='bilinear', align_corners=False).repeat(1, 2, 1, 1).long().contiguous()
        seg_label = gt_labels[:,0,...].unsqueeze(dim=1)
        corruption_label = gt_labels[:,1,...].unsqueeze(dim=1) /  gt_labels[:,1,...].max()# 0,1,2 -> 0, 0.5, 1
        """
        loss = dict()
        # 2,16,16을 가정한 코드입니다.
        # seg_label = gt_labels[:,0,...].unsqueeze(dim=1)
        # corruption_label = gt_labels[:,1,...].unsqueeze(dim=1) /  gt_labels[:,1,...].max()
        if gt_labels.sum() == 0:
            seg_label = torch.zeros_like(gt_labels)
            corruption_label = torch.zeros_like(gt_labels)
        else:
            prepare_for_seg_label = gt_labels.clone()
            prepare_for_seg_label[prepare_for_seg_label != 0] += 1
            seg_label = torch.div(prepare_for_seg_label, 2, rounding_mode='trunc')

            prepare_for_corruption_label = seg_label.clone()
            prepare_for_corruption_label[prepare_for_corruption_label != 0] -= 1
            corruption_label = gt_labels - prepare_for_corruption_label*2
            # corruption_label[corruption_label != 0] += 1
            corruption_label = corruption_label / self.corruption_step

        if self.sampler is not None:
            seg_weight = self.sampler.sample(results[0], seg_label)
        else:
            seg_weight = None
        loss['loss_corruption'] = F.smooth_l1_loss(results[1].sigmoid(),corruption_label)
        seg_GT_mask = (results[1] > self.corruption_threshold).long().view(-1, 1, 16, 16).contiguous()
        seg_pred_mask = (results[1] > self.corruption_threshold).long().repeat(1, self.num_classes, 1, 1).contiguous()

        seg_GT = (seg_label*seg_GT_mask).squeeze(1)
        seg_pred = (results[0]*seg_pred_mask).squeeze(1)
        seg_GT = torch.where(seg_GT_mask.squeeze(1) == 0, torch.tensor(255, dtype=seg_GT.dtype, device=seg_GT.device), seg_GT)
        seg_pred = torch.where(seg_pred_mask.squeeze(1) == 0, torch.tensor(255, dtype=seg_pred.dtype, device=seg_pred.device), seg_pred)

        if not isinstance(self.loss_cls, nn.ModuleList):
            losses_decode = [self.loss_cls]
        else:
            losses_decode = self.loss_cls
        for loss_cls in losses_decode:
            if loss_cls.loss_name not in loss:
                loss[loss_cls.loss_name] = loss_cls(
                    seg_pred,
                    seg_GT,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_cls.loss_name] += loss_cls(
                    seg_pred,
                    seg_GT,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # loss['acc_seg'] = accuracy(results[0], seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
