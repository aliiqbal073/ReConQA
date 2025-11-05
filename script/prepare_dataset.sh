# DATA_DIR=datasets/s-owod
# COCO_DIR=datasets/coco

# # # make neccesary dirs
# rm $DATA_DIR -rf
# echo "make dirs"
# mkdir -p $DATA_DIR
# mkdir -p $DATA_DIR/Annotations
# # mkdir -p $DATA_DIR/JPEGImages
# mkdir -p $DATA_DIR/ImageSets
# mkdir -p $DATA_DIR/ImageSets/Main

# # cp data
# # make use you have $COCO_DIR
# echo "copy coco images"
# cp -r $COCO_DIR/train2017 $DATA_DIR/JPEGImages 
# cp $COCO_DIR/val2017/* $DATA_DIR/JPEGImages/

# echo "convert coco annotation to voc"
# python tools/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_train2017.json
# python tools/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_val2017.json

# echo "copy owod spilit files"
# cp ./dataset_splits/s-owod/ImageSets/Main/* $DATA_DIR/ImageSets/Main/


#!/bin/bash

# Define the absolute path to the repo root (parent of script directory)
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

DATA_DIR=$ROOT_DIR/datasets/s-owod
COCO_DIR=$ROOT_DIR/datasets/coco

echo "Root: $ROOT_DIR"
echo "COCO: $COCO_DIR"
echo "Target: $DATA_DIR"

# Clean and recreate folders
rm -rf $DATA_DIR
echo "Creating directories..."
mkdir -p $DATA_DIR/Annotations
mkdir -p $DATA_DIR/JPEGImages
mkdir -p $DATA_DIR/ImageSets/Main

# Copy images
echo "Copying COCO images..."
cp -r $COCO_DIR/train2017/* $DATA_DIR/JPEGImages/
cp -r $COCO_DIR/val2017/* $DATA_DIR/JPEGImages/

# Convert COCO annotations to VOC format
echo "Converting annotations..."
python $ROOT_DIR/tools/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_train2017.json
python $ROOT_DIR/tools/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_val2017.json

# Copy OWOD split files
echo "Copying split files..."
cp $ROOT_DIR/dataset_splits/s-owod/ImageSets/Main/* $DATA_DIR/ImageSets/Main/



