always_save_checkpoint = True
wandb_log = True # override via command line if you like
wandb_offline = False
wandb_project = 'anon'
wandb_entity = "anon"
use_idr_torch = False
# data
block_size = 2048
batch_size = 24
gradient_accumulation_steps = 20 # total bsize ~= 1M tokens
audiotxt_datasets = ['libriheavy', 'stinystories', 'swiki_interleaved']
train_audiotxt_datasets_probs = [0.37, 0.53, 0.1]
train_splits = ['large', 'train']
val_splits = ['dev-clean', 'dev-other']
audio_tokens = 'mhubert25hzl11'
txt_tokens = 'smollm'
p_strategies = [0., 0., 0., 0., 1.] # stage 1 does only interleaving
num_workers = 16
pred_txt_in_interleaved = True
# model
backbone = "HuggingFaceTB/SmolLM-135M"
warm_init = True
rope_theta = 100000.0 # increase RoPE base frequency for long-context handling as in SpiritLM
n_audio_in_layers = 2
n_audio_out_layers = 2
layer_wa_audio = True
layer_selwa_audio = True
selwa_downproj = 0
raw_speech_residual = True
freeze_backbone = True
freeze_txt_inout = True
# adamw optimizer
audio_learning_rate = 3e-3
min_audio_lr = 3e-4
max_iters = 1000 # train for ~1B tokens
lr_decay_iters = max_iters
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
dadapt = False
# learning rate decay settings
decay_lr = False
decay_audio_lr = True
schedule = 'trapezoidal'
warmup_iters = 100  # how many steps to warm up for
warmdown_iters = 200
# eval stuff
eval_interval = 100
eval_iters = 10
log_interval = 10
# misc
compile = True
out_dir = "out/ours_135m_2stg"
