# export CUDA_VISIBLE_DEVICES=0
# python tools/test.py \
#     -d mmnist \
#     -c configs/mmnist/VMRNN-D.py \
#     --batch_size 1 \
#     --val_batch_size 4 \
#     --overwrite \
#     --ex_name work_dirs/mmnist/mamba_4gpu_bs4_lr1e-4_200ep_2024-03-05-02-33

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 tools/test.py --launcher="pytorch" -d mmnist -c configs/mmnist/VMRNN-D.py --dist --method VMRNN_d --ex_name work_dirs/mmnist/mamba_1gpu_bs16_lr1e-4_200ep_2222_2024-03-06-21-45
# python tools/test.py -d mmnist -c configs/mmnist/VMRNN-D.py --method VMRNN_d --ex_name work_dirs/mmnist/mamba_4gpu_bs16_lr1e-4_2000ep_2222_cos_2024-03-07-01-18
# PORT=28001 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh configs/mmnist/VMRNN-B.py 4 work_dirs/mmnist/mamba_4gpu_bs16_lr1e-4_2000ep_2222_cos_2024-03-07-01-18 -d mmnist
python -m torch.distributed.launch tools/test.py --launcher="pytorch" -d mmnist -c configs/mmnist/VMRNN-D.py --dist --method VMRNN_d --ex_name work_dirs/mmnist/mamba_1gpu_bs16_lr1e-4_200ep_2222_2024-03-06-21-45
#