# unsupervised pretraining of REW using COCO train2017
# pretrained SoCo backbone can be downloaded in https://github.com/hologerry/SoCo
python train_net.py --resume --dist-url auto --num-gpus 2 --config config/RED/red_pretrain.yaml \
	OUTPUT_DIR training_dir/changes_in_model/red \
    
    