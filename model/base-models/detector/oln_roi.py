# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.utils.registry import Registry
import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

from .fast_rcnn_oln import FastRCNNOutputLayers_OLN

logger = logging.getLogger(__name__)

class ROIHeads_OLN(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
        proposal_matcher_oln,
        batch_size_per_image_oln,
        enable_oln,
        oln_inference, 
        eval_unknown, 
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

        self.proposal_matcher_oln = proposal_matcher_oln
        self.batch_size_per_image_oln = batch_size_per_image_oln
        self.enable_oln = enable_oln
        self.oln_inference = oln_inference
        self.eval_unknown = eval_unknown

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),

            "proposal_matcher_oln": Matcher(
                cfg.OPENSET.OLN.IOU_THRESHOLDS, 
                cfg.OPENSET.OLN.IOU_LABELS, 
                allow_low_quality_matches=False,
            ),
            "batch_size_per_image_oln": cfg.OPENSET.OLN.BATCH_SIZE_PER_IMAGE, 
            "enable_oln": cfg.OPENSET.ENABLE_OLN, 
            "oln_inference": cfg.OPENSET.OLN_INFERENCE, 
            "eval_unknown": cfg.OPENSET.EVAL_UNKNOWN,
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    def _sample_proposals_oln(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image_oln, 1.0, self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)
        
        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            iou_i = match_quality_matrix.max(dim = 0)[0]
            proposals_per_image.iou_with_gt = iou_i
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes,
            )
            # Set target attributes of the sampled proposals:
            proposals_per_image_frcnn = proposals_per_image[sampled_idxs]
            proposals_per_image_frcnn.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image_frcnn.has(trg_name):
                        proposals_per_image_frcnn.set(trg_name, trg_value[sampled_targets])

                if targets_per_image.has("soft_labels"):
                    proposals_per_image_frcnn.soft_labels = targets_per_image.soft_labels[matched_idxs][sampled_idxs]
                
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])

            proposals_with_gt.append(proposals_per_image_frcnn)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
    
    @torch.no_grad()
    def label_and_sample_proposals_oln(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:

        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)
        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            iou_i = match_quality_matrix.max(dim = 0)[0]
            proposals_per_image.iou_with_gt = iou_i
            matched_idxs, matched_labels = self.proposal_matcher_oln(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals_oln(
                matched_idxs, matched_labels, targets_per_image.gt_classes,
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image_frcnn = proposals_per_image[sampled_idxs]
            proposals_per_image_frcnn.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image_frcnn.has(trg_name):
                        proposals_per_image_frcnn.set(trg_name, trg_value[sampled_targets])

                if targets_per_image.has("soft_labels"):
                    proposals_per_image_frcnn.soft_labels = \
                        targets_per_image.soft_labels[matched_idxs][sampled_idxs]
                
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])

            proposals_with_gt.append(proposals_per_image_frcnn)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples_oln", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples_oln", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads_OLN(ROIHeads_OLN):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        box_head_oln: nn.Module = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.box_head_oln = box_head_oln

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))

        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        enable_oln        = cfg.OPENSET.ENABLE_OLN
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers_OLN(cfg, box_head.output_shape)
        ret = {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }
        if enable_oln:
            ret["box_head_oln"] = build_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            )
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        proposals_oln: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            if self.enable_oln:
                proposals_oln = self.label_and_sample_proposals_oln(proposals_oln, targets)
            proposals = self.label_and_sample_proposals(proposals, targets)
            
        del targets

        if self.training:
            # compute ROI‐head losses
            losses = self._forward_box(features, proposals, proposals_oln)
            # MUST return both proposals and losses dict, so the outer
            # code can do: _, detector_losses = self.roi_heads(...)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            return pred_instances, {}

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        proposals_oln: List[Instances] = [],
    ):
        """
        Forward logic of the box prediction branch, with properly‐masked
        contrastive enqueue.
        """
        # 1) map features to list in the correct order
        feat_list = [features[f] for f in self.box_in_features]

        # ====== TRAINING ======
        if self.training:
            # --- Standard Fast R-CNN branch ---
            box_feats = self.box_pooler(feat_list, [x.proposal_boxes for x in proposals])
            box_feats = self.box_head(box_feats)
            scores, proposal_deltas, proposal_deltas_cls_agn = self.box_predictor(box_feats)

            # --- OLN branch ---
            if self.enable_oln:
                # a) pool & head over all OLN proposals
                box_feats_oln = self.box_pooler(
                    feat_list, [x.proposal_boxes for x in proposals_oln]
                )
                
                box_feats_oln = self.box_head_oln(box_feats_oln)

                # — stash for contrastive in predictor (flattened, projected, ℓ₂‐normed) —
                flat = box_feats_oln.flatten(1)
                proj = F.normalize(self.box_predictor.proj_mlp(flat), dim=1)
                self.box_predictor.last_box_features_oln = proj



               # ── STASH FOR CONTRASTIVE ────────────────────────────────────
              # make sure your predictor has this attribute so losses() can see it
                self.box_predictor.last_box_features_oln = box_feats_oln.flatten(1)
               # ───────────────────────────────────────────────────────────────

                # b) flatten to [R_all, D] and collect all gt_classes in same order
                flat_feats_all = box_feats_oln.flatten(1)
                self.box_predictor.last_box_features_oln = flat_feats_all
                orig_gt_cls_all = torch.cat(
                    [p.gt_classes for p in proposals_oln], dim=0
                ).to(flat_feats_all.device)

                # c) dynamic threshold
                global_ctx = flat_feats_all.mean(dim=0, keepdim=True)
                scores_oln_all = self.box_predictor.oln_score_pred(
                    flat_feats_all, global_ctx
                ).flatten()
                tau = float(self.box_predictor.threshold_net(global_ctx).view(-1)[0])
                mask_flat = scores_oln_all > tau

                # d) filter per‐image proposals & scores
                num_props = [len(p) for p in proposals_oln]
                scores_splits = scores_oln_all.split(num_props)
                new_props, new_scores = [], []
                for props_i, sc_i in zip(proposals_oln, scores_splits):
                    keep = sc_i > tau
                    new_props.append(props_i[keep])
                    new_scores.append(sc_i[keep].unsqueeze(1))
                proposals_oln = new_props
                scores_oln = torch.cat(new_scores, dim=0)

                # e) mask features & gt_classes for contrast
                feats_contrast = flat_feats_all
                
                labels_contrast = (orig_gt_cls_all >= self.box_predictor.num_known_classes).long()
            else:
                scores_oln = torch.zeros((0, 1), device=scores.device)
                feats_contrast = torch.zeros((0, feat_list[0].shape[1]), device=scores.device)
                labels_contrast = torch.zeros((0,), dtype=torch.long, device=scores.device)

            #  f) compute the usual box losses
            preds = (scores, proposal_deltas, scores_oln, proposal_deltas_cls_agn)
            losses = self.box_predictor.losses(preds, proposals, proposals_oln)
            # g) enqueue & contrastive (only if there is something to contrast)
            if self.enable_oln and feats_contrast.numel() > 0:
                # 1) add to predictor’s bank
                self.box_predictor.memory_bank.enqueue(feats_contrast, labels_contrast)

                # 2) compute NCE loss on that same bank
                ctr = self.box_predictor.memory_bank.contrastive_loss(
                    feats_contrast, labels_contrast
                )

                # 3) clamp NaNs / infs so training never explodes
                ctr = torch.nan_to_num(ctr, nan=0.0, posinf=1e3, neginf=-1e3)

                # 4) weight and store
                losses["loss_contrast"] = self.box_predictor.contrast_weight * ctr

            return losses
        # ====== INFERENCE ======
        else:
            # 1) build feat_list just like in training
            feat_list = [features[f] for f in self.box_in_features]
            # 2) always call pooler with a list
            box_features = self.box_pooler(feat_list, [x.proposal_boxes for x in proposals])

            if self.oln_inference:
                # OLN‐specific head
                box_features_oln = self.box_head_oln(box_features)  # [R, D]
                # compute global context exactly like during training:
                # average over all RoI‐features, giving [1, D]
                global_feat = box_features_oln.mean(dim=0, keepdim=True)
                # now ctx_proj will see a (1 x roi_dim) input
                scores_oln = self.box_predictor.oln_score_pred(
                    box_features_oln, global_feat
                )

                # standard box head
                scores, proposal_deltas, proposal_deltas_cls_agn = self.box_predictor(
                    self.box_head(box_features)
                )
                predictions = (scores_oln, proposal_deltas_cls_agn)
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)

            elif self.eval_unknown:
                # unknown eval: class‐agnostic deltas + scores
                box_features = self.box_head(box_features)
                scores, proposal_deltas, proposal_deltas_cls_agn = self.box_predictor(box_features)
                predictions = (scores, proposal_deltas_cls_agn)
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)

            else:
                # standard known‐only evaluation
                box_features = self.box_head(box_features)
                scores, proposal_deltas, proposal_deltas_cls_agn = self.box_predictor(box_features)
                # only pass (scores, deltas) to inference, then give it proposal_deltas_cls_agn
                predictions = (scores, proposal_deltas)
                pred_instances, _ = self.box_predictor.inference(
                    predictions, proposals, proposal_deltas_cls_agn
                )

            return pred_instances


