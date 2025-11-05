Here’s your fully cleaned & ready-to-run script:

Every rew → RED (case-insensitive)
Every mepu → removed (folders, comments, configs)
All paths fixed to use pure RED + SOWOD
Kept perfect indentation & comments for each task

bash#!/bin/bash
# SOWOD training script (RED-based, MEPU-free)
# All REW → RED, all MEPU paths removed

# ===================================
# Task 1
# ===================================

# Generate pseudo labels using FreeSOLO
python tools/gen_pseudo_label_new.py --proposal_path proposals/proposals_freesolo.json \
	--data_path datasets/sowod --save_path datasets/sowod/Annotations/pseudo_label_fs.json \
	--keep_type num --num_keep 5 --known_cls_num 19 --num_vis 5 --data_split t1_train

# Update Density Head models of RED using known object labels in Task 1
python train_net.py --dist-url auto --num-gpus 2 --config config/RED/red_t1_sowod.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red/sowod_t1 \
	MODEL.WEIGHTS training_dir/changes_in_model/red/model_final.pth

# Assign RED scores for unknown pseudo labels
python train_net.py --eval-only --inference-red --resume --dist-url auto --num-gpus 2 \
	--config config/RED/red_t1_sowod.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red/sowod_t1 \
	DATASETS.TEST '("sowod_train_t1_fs",)' \
	OPENSET.OUTPUT_PATH_RED datasets/sowod/Annotations/pseudo_label_fs.json

# Train object detectors using known + unknown pseudo labels (self-training)
python train_net.py --resume --dist-url auto --num-gpus 2 --config config/SOWOD/t1/self-train.yaml \
	DATASETS.TRAIN '("sowod_train_t1_fs",)' \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t1-self-train \
	OPENSET.RED.GAMMA 4.0

# Inference using OLN for self-training
python train_net.py --resume --eval-only --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t1/self-train.yaml \
	DATASETS.TEST '("sowod_train_t1",)' \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t1-self-train \
	OPENSET.OLN_INFERENCE True \
	OPENSET.INFERENCE_SELT_TRAIN True \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t1-self-train/model_final.pth

# Generate new pseudo labels using OLN proposals
python tools/gen_pseudo_label_new.py \
	--proposal_path training_dir/changes_in_model/sowod/fs-t1-self-train/inference/inference_results.json \
	--data_path datasets/sowod --save_path datasets/sowod/Annotations/pseudo_label_st.json \
	--keep_type percent --percent_keep 0.3 --known_cls_num 19 --data_split t1_train

# Assign RED scores for new unknown pseudo labels
python train_net.py --eval-only --inference-red --resume --dist-url auto --num-gpus 2 \
	--config config/RED/red_t1_sowod.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red/sowod_t1 \
	DATASETS.TEST '("sowod_train_t1_st",)' \
	OPENSET.OUTPUT_PATH_RED datasets/sowod/Annotations/pseudo_label_st.json

# Final training with refreshed pseudo labels
python train_net.py --resume --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t1/self-train.yaml \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t1-self-train \
	OPENSET.RED.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t1-self-train/model_final.pth


# ===================================
# Task 2
# ===================================

python train_net.py --dist-url auto --num-gpus 2 --config config/RED/red_t2_sowod.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red/sowod_t2 \
	MODEL.WEIGHTS training_dir/changes_in_model/red/model_final.pth

python train_net.py --resume --eval-only --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t2/train.yaml \
	DATASETS.TEST '("sowod_t2_train_and_ft",)' \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t2-train \
	OPENSET.OLN_INFERENCE True \
	OPENSET.INFERENCE_SELT_TRAIN True \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t1-self-train/model_final.pth

python tools/gen_pseudo_label_new.py \
	--proposal_path training_dir/changes_in_model/sowod/fs-t2-train/inference/inference_results.json \
	--data_path datasets/sowod --save_path datasets/sowod/Annotations/pseudo_label_st.json \
	--keep_type percent --percent_keep 0.3 --known_cls_num 40 --data_split t2_train_and_ft

python train_net.py --eval-only --inference-red --resume --dist-url auto --num-gpus 2 \
	--config config/RED/red_t2_sowod.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red/sowod_t2 \
	DATASETS.TEST '("sowod_train_t2_st",)' \
	OPENSET.OUTPUT_PATH_RED datasets/sowod/Annotations/pseudo_label_st.json

python train_net.py --resume --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t2/train.yaml \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t2-train \
	OPENSET.RED.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t1-self-train/model_final.pth

python train_net.py --resume --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t2/ft.yaml \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t2-ft \
	OPENSET.RED.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t2-train/model_final.pth


# ===================================
# Task 3
# ===================================

python train_net.py --dist-url auto --num-gpus 2 --config config/RED/red_t3_sowod.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red/sowod_t3 \
	MODEL.WEIGHTS training_dir/changes_in_model/red/model_final.pth

python train_net.py --resume --eval-only --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t3/train.yaml \
	DATASETS.TEST '("sowod_t3_train_and_ft",)' \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t3-train \
	OPENSET.OLN_INFERENCE True \
	OPENSET.INFERENCE_SELT_TRAIN True \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t2-ft/model_final.pth

python tools/gen_pseudo_label_new.py \
	--proposal_path training_dir/changes_in_model/sowod/fs-t3-train/inference/inference_results.json \
	--data_path datasets/sowod --save_path datasets/sowod/Annotations/pseudo_label_st.json \
	--keep_type percent --percent_keep 0.3 --known_cls_num 60 --data_split t3_train_and_ft

python train_net.py --eval-only --inference-red --resume --dist-url auto --num-gpus 2 \
	--config config/RED/red_t3_sowod.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red/sowod_t3 \
	DATASETS.TEST '("sowod_train_t3_st",)' \
	OPENSET.OUTPUT_PATH_RED datasets/sowod/Annotations/pseudo_label_st.json

python train_net.py --resume --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t3/train.yaml \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t3-train \
	OPENSET.RED.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t2-ft/model_final.pth

python train_net.py --resume --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t3/ft.yaml \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t3-ft \
	OPENSET.RED.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t3-train/model_final.pth


# ===================================
# Task 4 (Final Incremental Step)
# ===================================

python train_net.py --resume --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t4/train.yaml \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t4-train \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t3-ft/model_final.pth

python train_net.py --resume --dist-url auto --num-gpus 2 \
	--config config/SOWOD/t4/ft.yaml \
	OUTPUT_DIR training_dir/changes_in_model/sowod/fs-t4-ft \
	MODEL.WEIGHTS training_dir/changes_in_model/sowod/fs-t4-train/model_final.pth