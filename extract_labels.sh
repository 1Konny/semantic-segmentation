DATASET=$1
YOUR_FOLDER=$2
YOUR_SAVE_DIR=$3

if [ $DATASET == "Cityscapes" ]; then
    SNAPSHOT=./pretrained_models/cityscapes_best.pth
elif [ $DATASET == "KITTI" ]; then
    SNAPSHOT=./pretrained_models/kitti_best.pth
fi

CUDA_VISIBLE_DEVICES=0 python demo_folder.py --demo-folder $YOUR_FOLDER --snapshot $SNAPSHOT --save-dir $YOUR_SAVE_DIR
