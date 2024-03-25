method = 'VMRNN_D'
# model
depths_downsample = '2,2'
depths_upsample = '2,2'
num_heads = '4,8'
patch_size = 2
window_size = 4
embed_dim = 128
# training
lr = 1e-4
batch_size = 4
# sched = 'onecycle'
sched = 'cosine'
warmup_epoch = 5