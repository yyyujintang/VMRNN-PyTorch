export CUDA_VISIBLE_DEVICES=6

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

EX_NAME="mmnist/mamba_1gpu_bs16_lr1e-4_2000ep_2222_${CURRENT_TIME}"

nohup python tools/train.py \
    --config_file configs/mmnist/VMRNN-D.py \
    --dataname mmnist \
    --batch_size 16 \
    --epoch 2000 \
    --overwrite \
    --lr 1e-4 \
    --ex_name "$EX_NAME"  \
    --find_unused_parameters >> logs/${EX_NAME}_run.log 2>&1 &
