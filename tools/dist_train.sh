#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
export PORT=29002
export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="taxibj/mamba_1gpu_ps4_bs16_lr4e-4_dp4_200ep_${CURRENT_TIME}"

# WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
LOG_DIR=$(echo ${CFG%.*} | sed "s/configs\///g")

nohup $PYTHON -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py --dist \
    --config_file configs/mmnist/PredRNNpp.py \
    --ex_name $EX_NAME \
    --batch_size 8 \
    --epoch 100 \
    --overwrite \
    --find_unused_parameters \
    --seed 42 --launcher="pytorch" >> logs/${EX_NAME}_run.log 2>&1 & &

