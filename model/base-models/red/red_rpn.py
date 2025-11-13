# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec, cat
import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.utils.registry import Registry
from mepu.model.rew.transformer_ae import TransformerAE

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")

def build_rpn_head(cfg, input_shape):
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class OFFLINE_AE_RPNHead(nn.Module):
    @configurable
    def __init__(self, *, in_channels: int, enable_rew: bool):
        super().__init__()
        self.enable_rew = enable_rew
        if self.enable_rew:
            # Project features into AE embedding space
            self.proj_dim = 64
            self.proj = nn.Conv2d(in_channels, self.proj_dim, kernel_size=1)
            self.recon = TransformerAE(
                embed_dim=self.proj_dim,
                num_layers=2,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
            )
            self.unproj = nn.Conv2d(self.proj_dim, in_channels, kernel_size=1)
        # Initialize conv weights
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_ch = [s.channels for s in input_shape]
        assert len(set(in_ch)) == 1, "All FPN levels must share channel count"
        return {
            "in_channels": in_ch[0],
            "enable_rew": cfg.OPENSET.ENABLE_REW,
        }

    def forward(self, features: List[torch.Tensor]):
        recons_error_by_level = []
        recons_error_map_by_level = []
        for x in features:
            if self.enable_rew:
                z = self.proj(x.detach())
                recon_z = self.recon(z)
                recon_feat = self.unproj(recon_z)
                err_map = ((recon_feat - x.detach()) ** 2).mean(dim=1, keepdim=True).sqrt()
                recons_error_map_by_level.append(err_map)
                recons_error_by_level.append(
                    err_map.expand(x.size(0), x.size(1), err_map.size(2), err_map.size(3))
                )
        return recons_error_by_level, recons_error_map_by_level


@PROPOSAL_GENERATOR_REGISTRY.register()
class OFFLINE_AE_RPN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[int, int],
        post_nms_topk: Tuple[int, int],
        nms_thresh: float,
        min_box_size: float,
        anchor_boundary_thresh: float,
        loss_weight: Union[float, Dict[str, float]],
        box_reg_loss_type: str,
        smooth_l1_beta: float,
        enable_rew: bool,
        in_loop_density: bool,
        density_dim: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta

        self.enable_rew = enable_rew
        self.in_loop_density = in_loop_density
        if self.enable_rew and self.in_loop_density:
            D = density_dim
            self.density_head = nn.Sequential(
                nn.Linear(1, D),
                nn.ReLU(),
                nn.Linear(D, D // 2),
                nn.ReLU(),
                nn.Linear(D // 2, 1),
            )

    @torch.no_grad()
    def assign_soft_label(
        self,
        images: ImageList,
        instances: List[Instances],
        err_map_by_lvl: List[torch.Tensor],
    ):
        """
        For each GT box in each image, compute a soft unknown score from
        the first-level AE error map and attach it as `Instances.soft_labels`.
        """
        maps0 = err_map_by_lvl[0]  # (N,1,h_feat,w_feat)
        for img_idx, inst in enumerate(instances):
            h, w = images.image_sizes[img_idx]
            err = maps0[img_idx:img_idx+1]        # [1,1,h_feat,w_feat]
            err_up = F.interpolate(
                err, size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]                            # [H,W]
            boxes = inst.gt_boxes.tensor        # [M,4]
            softs = []
            for x1, y1, x2, y2 in boxes:
                x1i, y1i = max(int(x1), 0), max(int(y1), 0)
                x2i, y2i = min(int(x2), w), min(int(y2), h)
                roi = err_up[y1i:y2i, x1i:x2i]
                softs.append(roi.mean() if roi.numel() else torch.tensor(0.0, device=err_up.device))
            inst.soft_labels = torch.stack(softs) if softs else torch.empty((0,), device=err_up.device)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "in_features": cfg.MODEL.RPN.IN_FEATURES,
            "head": build_rpn_head(cfg, [input_shape[f] for f in cfg.MODEL.RPN.IN_FEATURES]),
            "anchor_generator": build_anchor_generator(cfg, [input_shape[f] for f in cfg.MODEL.RPN.IN_FEATURES]),
            "anchor_matcher": Matcher(
                cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
            ),
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "pre_nms_topk": (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST),
            "post_nms_topk": (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.LOSS_WEIGHT * cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT,
            },
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
            "enable_rew": cfg.OPENSET.ENABLE_REW,
            "in_loop_density": cfg.OPENSET.REW.IN_LOOP_DENSITY,
            "density_dim": cfg.OPENSET.REW.DENSITY_DIM,
        }

    def _subsample_labels(self, label):
        pos, neg = subsample_labels(label, self.batch_size_per_image, self.positive_fraction, 0)
        label.fill_(-1)
        label.scatter_(0, pos, 1)
        label.scatter_(0, neg, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        anchors = Boxes.cat(anchors)
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]

        all_labels, all_boxes, all_classes, all_idxs = [], [], [], []
        for size, boxes_i, classes_i in zip(image_sizes, gt_boxes, gt_classes):
            mq = retry_if_cuda_oom(pairwise_iou)(boxes_i, anchors)
            idxs, labs = retry_if_cuda_oom(self.anchor_matcher)(mq)
            labs = labs.to(device=boxes_i.device)
            labs = self._subsample_labels(labs)
            all_labels.append(labs)
            all_boxes.append(boxes_i[idxs].tensor)
            all_classes.append(classes_i[idxs])
            all_idxs.append(idxs)
        return all_labels, all_boxes, all_classes, all_idxs

    def losses(self, recons_error_by_level: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        if self.enable_rew:
            errs = cat(recons_error_by_level, dim=1)
            losses["loss_ae"] = (errs**2).mean()
        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        # 1) Extract features + anchors
        feats = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(feats)

       
        err_by_lvl, err_map_by_lvl = self.rpn_head(feats)

     
        flat_errs = [e.permute(0,2,3,1).flatten(1) for e in err_by_lvl]

        if self.training:
          
            losses = self.losses(flat_errs)

           
            if self.enable_rew and self.in_loop_density:
                maps = [m.permute(0,2,3,1).flatten(1) for m in err_map_by_lvl]
                pix = torch.cat(maps, dim=1)
                p_unk = self.density_head(pix.view(-1,1)).view_as(pix)
                _, _, matched_classes, _ = self.label_and_sample_anchors(anchors, gt_instances)
                cls_tensor = torch.stack(matched_classes, dim=0)
                unk = (cls_tensor == 80).float()
                pixel_masks, ptr = [], 0
                for lvl, m in enumerate(err_map_by_lvl):
                    N, _, H, W = m.shape
                    M = H * W
                    A = self.anchor_generator.num_anchors[lvl]
                    sl = unk[:, ptr : ptr + M * A]
                    pixel_masks.append(
                        sl.view(N, M, A).any(dim=2).float()
                    )
                    ptr += M * A
                mask = torch.cat(pixel_masks, dim=1)

                losses["loss_density"] = F.binary_cross_entropy_with_logits(p_unk, mask)
            return losses
        else:
         
            self.assign_soft_label(images, gt_instances, err_map_by_lvl)
          
            for inst in gt_instances:
                inst.pred_boxes = inst.gt_boxes  
                
                inst.scores = torch.ones(len(inst.gt_boxes), device=inst.gt_boxes.tensor.device)
            return gt_instances
            
