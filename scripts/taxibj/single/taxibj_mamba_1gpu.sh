export CUDA_VISIBLE_DEVICES=6

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

EX_NAME="taxibj/mamba_1gpu_bs16_lr4e-4_200ep_test_drop_${CURRENT_TIME}"

# nohup python tools/train.py \
#     --config_file configs/taxibj/VMRNN-B.py \
#     --dataname taxibj \
#     --batch_size 16 \
#     --epoch 200 \
#     --overwrite \
#     --lr 4e-4 \
#     --fps \
#     --ex_name "$EX_NAME"  \
#     --find_unused_parameters >> logs/${EX_NAME}_run.log 2>&1 &

python tools/train.py \
    --config_file configs/taxibj/VMRNN-B.py \
    --dataname taxibj \
    --batch_size 16 \
    --epoch 200 \
    --overwrite \
    --lr 4e-4 \
    --fps \
    --ex_name "$EX_NAME"  \
    --find_unused_parameters 
