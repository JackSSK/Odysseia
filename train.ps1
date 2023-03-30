# Activate python venv
.\sd-scripts\venv\Scripts\activate

$n_workers = 1
$model_name = "glock_design_v0"
$pretrained_model = "stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.ckpt"  
$train_data_dir = "data/Glock17/img/" 
$reg_data_dir = "" 


################################## Network settings ##################################

# networks.lora -> LoRA; lycoris.kohya -> LyCORIS（LoCon、LoHa）
$network_module = "networks.lora" 

# pretrained weights for LoRA network
$network_weights = "" 

# 4~128
$network_dim = 32 

# network alpha，similar with network dim or small values.
# Consider increase learning ratio if using small values
$network_alpha = 32 


################################## Train related params ##################################
# 64i, 64j
$resolution = "512,512"
$batch_size = 1 
$max_train_epoches = 10 
$save_every_n_epochs = 2 
$train_unet_only = 0 
$train_text_encoder_only = 0 

# Modify offset value if having outputs too bright or too dark: ~0.1
$noise_offset = 0 
# keep heading N tokens when shuffling caption tokens
$keep_tokens = 0 


################################## Learning ratio ##################################
$lr = "1e-4"
$unet_lr = "1e-4"
$text_encoder_lr = "1e-5"
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
$lr_scheduler = "cosine_with_restarts" 
$lr_warmup_steps = 0
$lr_restart_cycles = 1


################################## Optimizer params ##################################
$use_8bit_adam = 1 # use 8bit adam optimizer?
$use_lion = 0 # use lion optimizer?


################################## LyCORIS params ##################################
# LyCORIS network algo: lora / loha 
$algo = "lora" 
$conv_dim = 4
$conv_alpha = 4


################################## Output settings ##################################
$output_name = $model_name
# ckpt, pt, safetensors
$save_model_as = "safetensors"


################################## Others ##################################
# arb min resolution
$min_bucket_reso = 256 

# arb max resolution
$max_bucket_reso = 1024 

$persistent_data_loader_workers = 0 
$clip_skip = 2 


$Env:HF_HOME = "huggingface"
$ext_args = [System.Collections.ArrayList]::new()

if ($train_unet_only) {
  [void]$ext_args.Add("--network_train_unet_only")
}

if ($train_text_encoder_only) {
  [void]$ext_args.Add("--network_train_text_encoder_only")
}

if ($network_weights) {
  [void]$ext_args.Add("--network_weights=" + $network_weights)
}

if ($reg_data_dir) {
  [void]$ext_args.Add("--reg_data_dir=" + $reg_data_dir)
}

if ($use_8bit_adam) {
  [void]$ext_args.Add("--use_8bit_adam")
}

if ($use_lion) {
  [void]$ext_args.Add("--use_lion_optimizer")
}

if ($persistent_data_loader_workers) {
  [void]$ext_args.Add("--persistent_data_loader_workers")
}

if ($network_module -eq "lycoris.kohya") {
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("conv_dim=$conv_dim")
  [void]$ext_args.Add("conv_alpha=$conv_alpha")
  [void]$ext_args.Add("algo=$algo")
}

if ($noise_offset) {
  [void]$ext_args.Add("--noise_offset=$noise_offset")
}

# run train
accelerate launch --num_cpu_threads_per_process=8 "sd-scripts/train_network.py" `
  --enable_bucket `
  --pretrained_model_name_or_path=$pretrained_model `
  --train_data_dir=$train_data_dir `
  --output_dir="./output" `
  --logging_dir="./logs" `
  --resolution=$resolution `
  --network_module=$network_module `
  --max_train_epochs=$max_train_epoches `
  --learning_rate=$lr `
  --unet_lr=$unet_lr `
  --text_encoder_lr=$text_encoder_lr `
  --lr_scheduler=$lr_scheduler `
  --lr_warmup_steps=$lr_warmup_steps `
  --lr_scheduler_num_cycles=$lr_restart_cycles `
  --network_dim=$network_dim `
  --network_alpha=$network_alpha `
  --output_name=$output_name `
  --train_batch_size=$batch_size `
  --save_every_n_epochs=$save_every_n_epochs `
  --mixed_precision="fp16" `
  --save_precision="fp16" `
  --seed="1337" `
  --cache_latents `
  --clip_skip=$clip_skip `
  --prior_loss_weight=1 `
  --max_token_length=225 `
  --caption_extension=".txt" `
  --save_model_as=$save_model_as `
  --min_bucket_reso=$min_bucket_reso `
  --max_bucket_reso=$max_bucket_reso `
  --keep_tokens=$keep_tokens `
  --max_data_loader_n_workers=$n_workers `
  --xformers --shuffle_caption $ext_args
Write-Output "Train finished"
Read-Host | Out-Null ;