#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
export PORT=29002
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,5,6,7

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="taxibj/mamba_4gpu_bs8_lr1e-3_100ep_${CURRENT_TIME}"

nohup $PYTHON -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py --dist \
    -d taxibj \
    --config_file configs/taxibj/VMRNN-B.py \
    --ex_name $EX_NAME \
    --batch_size 8 \
    --epoch 100 \
    --overwrite \
    --find_unused_parameters \
    --seed 42 --launcher="pytorch" >> logs/${EX_NAME}_run.log 2>&1 &
