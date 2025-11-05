import os
import cv2
import torch
import sys

sys.path.append(".")  # Ensure current directory is in path

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Register custom config and architecture
from mepu.config.config import add_config
from mepu.model.detector.oln_box import OLN_BOX  # Register OLN_BOX meta-arch

# ======== CONFIGURATION ========
MODEL_PATH = "training_dir/mepu-sowod/fs-t2-ft/model_final.pth"
CONFIG_FILE = "config/MEPU-SOWOD/t2/ft.yaml"
IMG_DIR = "training_dir/real_images"
OUT_DIR = "real_images_outputs"

# ======== KNOWN CLASSES IN T2 ========
# From training log "Known classes: range(0, 41)"
coco80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle'
]
thing_classes_task = coco80[:41]  # For T2

# Add custom metadata for task
MetadataCatalog.get("__task_t2__").thing_classes = thing_classes_task + ['unknown']
metadata = MetadataCatalog.get("__task_t2__")

# ======== SETUP ========
cfg = get_cfg()
add_config(cfg)
cfg.merge_from_file(CONFIG_FILE)
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# ======== INFERENCE LOOP ========
os.makedirs(OUT_DIR, exist_ok=True)
img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

for img_name in img_list:
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    pred_classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()

    labels = []
    for i, cls_idx in enumerate(pred_classes):
        if cls_idx < len(thing_classes_task):
            label = thing_classes_task[cls_idx]
        else:
            label = "unknown"
        score = int(scores[i] * 100)
        labels.append(f"{label} {score}%")

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    vis_output = v.overlay_instances(
        boxes=instances.pred_boxes,
        labels=labels
    )
    vis_img = vis_output.get_image()[:, :, ::-1]

    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, vis_img)
    print(f"Inference done: {img_name} -> {out_path}, detections: {len(labels)}")

    cv2.imshow("Detection Result", vis_img)
    if cv2.waitKey(0) == 27:  # ESC key
        break

cv2.destroyAllWindows()
