always_save_checkpoint = True
wandb_log = True # override via command line if you like
wandb_offline = False
wandb_project = 'anon'
wandb_entity = "anon"
use_idr_torch = False
# data
block_size = 4096
batch_size = 20
gradient_accumulation_steps = 24 # total bsize ~= 1M tokens
audio_datasets = ['ls960', 'librilight-large', 'people', 'stinystories', 'swiki', 'tedlium', 'voxpopuli']
txt_datasets = ['finewebedu', 'cosmopedia2', 'pythonedu', 'finemath']
audiotxt_datasets = ['libriheavy', 'stinystories', 'swiki_interleaved']
train_txt_datasets_probs = val_txt_datasets_probs = [0.7, 0.15, 0.08, 0.06] # taken from https://github.com/huggingface/smollm/blob/main/pre-training/smollm1/config_smollm1_135M.yaml
val_audio_datasets_probs = [1/3, 0., 1/3, 0., 0., 1/3, 0.] # only ls960, people, and tedlium have validation sets
train_audiotxt_datasets_probs = [0.37, 0.53, 0.1]
train_splits = ['large', 'train']
val_splits = ['val', 'dev-clean', 'dev-other']
audio_tokens = 'mhubert25hzl11'
txt_tokens = 'smollm'
p_strategies = [1., 1., 0., 0., 1.] # we do speech only, text only, and interleaving
num_workers = 16
pred_txt_in_interleaved = True
# model
backbone = "HuggingFaceTB/SmolLM-360M"
warm_init = True
rope_theta = 100000.0 # increase RoPE base frequency for long-context handling as in SpiritLM
n_audio_in_layers = 2
n_audio_out_layers = 2
layer_wa_audio = True
layer_selwa_audio = True
selwa_downproj = 0
raw_speech_residual = True
entropy_reg = 0.01
# adamw optimizer
learning_rate = 3e-4
min_lr = learning_rate
max_iters = 1000 # train for ~1B tokens
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
dadapt = False
# learning rate decay settings
decay_lr = True # although we don't really do it, see ***
schedule = 'trapezoidal'
warmup_iters = 100  # how many steps to warm up for
warmdown_iters = 200
lr_decay_iters = 1000 # or whatever stage 1 lasts, namely, *** we don't decay the lr, but we do it for compatibility with ckpt_warmdown_at arg
# eval stuff
eval_interval = 100
eval_iters = 10
log_interval = 10
# misc
compile = True
out_dir = "out/ours_360m_2stg"
resume_from = "last"
force_load_last = True
load_optimizer = False