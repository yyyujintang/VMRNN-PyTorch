#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
export PORT=42001
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="kth40/mamba_4gpu_ps2_bs1_lr1e-4_100ep_one_${CURRENT_TIME}"


nohup $PYTHON -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py --dist \
    --config_file configs/kth/VMRNN-B.py \
    --dataname kth40 \
    --ex_name $EX_NAME \
    --batch_size 1 \
    --val_batch_size 1 \
    --epoch 100 \
    --overwrite \
    --lr 1e-4 \
    --fps \
    --find_unused_parameters \
    --seed 42 --launcher="pytorch" >> logs/${EX_NAME}_run.log 2>&1 &
