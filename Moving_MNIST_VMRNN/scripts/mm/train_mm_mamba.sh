export CUDA_VISIBLE_DEVICES=5
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="mminst_VMRNN_500_5e-5_${CURRENT_TIME}"
nohup python train.py \
    --res_dir ./exps/${EX_NAME} \
    --model VMRNN-D \
    --epochs 500 \
    --lr 5e-5 \
    --train_batch_size 8 \
    --valid_batch_size 8 \
    --test_batch_size 8 \
    --epoch_valid 5 \
    --log_train 200 \
    --log_valid 200 >> logs/${EX_NAME}_run.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
# EX_NAME="mminst_VMRNN_1539_461_5e-5_${CURRENT_TIME}"
# nohup python train.py \
#     --res_dir ./exps/${EX_NAME} \
#     --model VMRNN-D \
#     --epochs 461 \
#     --lr 5e-5 \
#     --train_batch_size 8 \
#     --valid_batch_size 8 \
#     --checkpoint_path exps/mminst_VMRNN_2024-02-04-11-58/model/trained_model_state_dict \
#     --test_batch_size 8 \
#     --log_train 100 \
#     --log_valid 100 >> logs/${EX_NAME}_run.log 2>&1 &