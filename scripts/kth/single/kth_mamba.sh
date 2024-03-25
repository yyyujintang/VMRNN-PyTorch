export CUDA_VISIBLE_DEVICES=0

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

EX_NAME="kth/mamba_1gpu_ps2_bs4_lr1e-4_dp6_100ep_${CURRENT_TIME}"

nohup python tools/train.py \
    --config_file configs/kth/VMRNN-B.py \
    --dataname kth \
    --batch_size 4 \
    --epoch 100 \
    --overwrite \
    --lr 1e-4 \
    --fps \
    --ex_name "$EX_NAME"  \
    --find_unused_parameters >> logs/${EX_NAME}_run.log 2>&1 &
