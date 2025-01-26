# Output path for training runs. Each training run makes a new directory in here.
output_dir = '/content/diffusion-pipe-colab/data/output'

# Dataset config file.
dataset = 'examples/dataset.toml'

# training settings
epochs = 100
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 4
gradient_clipping = 1.0
warmup_steps = 100

# eval settings

eval_every_n_epochs = 1
eval_before_first_step = true
eval_micro_batch_size_per_gpu = 1
eval_gradient_accumulation_steps = 1

# misc settings

save_every_n_epochs = 10
checkpoint_every_n_minutes = 120
activation_checkpointing = true
partition_method = 'parameters'
save_dtype = 'bfloat16'
caching_batch_size = 1
steps_per_print = 1
video_clip_mode = 'single_middle'

[model]
type = 'hunyuan-video'
transformer_path = '/content/diffusion-pipe-colab/models/hunyuan_video_720_cfgdistill_bf16.safetensors'
vae_path = '/content/diffusion-pipe-colab/models/hunyuan_video_vae_bf16.safetensors'
llm_path = '/content/diffusion-pipe-colab/models/llava-llama-3-8b-text-encoder-tokenizer'
clip_path = '/content/diffusion-pipe-colab/models/clip-vit-large-patch14'
dtype = 'bfloat16'
transformer_dtype = 'float8'
timestep_sample_method = 'logit_normal'

[adapter]
type = 'lora'
rank = 32
dtype = 'bfloat16'

[optimizer]
type = 'adamw_optimi'
lr = 2e-5
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8

