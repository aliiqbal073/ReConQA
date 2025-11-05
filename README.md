Steps for Test train Split of the dataset can be followed from: https://github.com/frh23333/mepu-owod

Installation:
conda create -n ReconQA python=3.8
conda activate ReconQA
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e . 
cd .. 
pip install -r requirements.txt

Pretraining of RED module:
sh script/train_red.sh

S-OWOD benchmark Training:
sh script/train_fs.sh

Evaluation on M-OWOD benchmark:
sh script/eval_owod.sh
