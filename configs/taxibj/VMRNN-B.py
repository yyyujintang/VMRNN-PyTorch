method = 'VMRNN_B'
# model
patch_size = 4
depths = 12
heads_number = [8]
window_size = 8
drop_rate = 0.
attn_drop_rate = 0.
drop_path_rate = 0.1
embed_dim = 128
num_heads = 4
# training
batch_size = 4
lr = 1e-4
sched = 'cosine'
warmup_epoch = 5
