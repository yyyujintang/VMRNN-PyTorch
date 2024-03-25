export CUDA_VISIBLE_DEVICES=4

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

EX_NAME="kth40/mamba_1gpu_ps2_bs1_lr1e-4_100ep_cos_${CURRENT_TIME}"

nohup python tools/train.py \
    --config_file configs/kth/VMRNN-B.py \
    --dataname kth40 \
    --batch_size 1 \
    --val_batch_size 1 \
    --epoch 100 \
    --overwrite \
    --lr 1e-4 \
    --fps \
    --ex_name "$EX_NAME"  \
    --find_unused_parameters >> logs/${EX_NAME}_run.log 2>&1 &