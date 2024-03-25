method = 'SwinLSTM_B'
# model
patch_size = 2
depths = 6
# heads_number = [8]
window_size = 4
drop_rate = 0.
attn_drop_rate = 0.
drop_path_rate = 0.1
batch_size = 8
# 
embed_dim = 128
num_heads = 8
# training
# batch_size = 4
lr = 1e-4
# lr = 5e-4
sched = 'cosine'
warmup_epoch = 5
# sched = 'onecycle'
