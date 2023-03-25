```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
"../../../../stable-diffusion-webui/models/Stable-diffusion/"
export dataset_name="data/pokemon"

accelerate launch --mixed_precision="fp16"  train_txt2img.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model"




  # ```bash
  # export MODEL_NAME="CompVis/stable-diffusion-v1-4"
  # export TRAIN_DIR="path_to_your_dataset"
  # export OUTPUT_DIR="path_to_save_model"
  #
  # accelerate launch train_txt2img.py \
  #   --pretrained_model_name_or_path=$MODEL_NAME \
  #   --train_data_dir=$TRAIN_DIR \
  #   --use_ema \
  #   --resolution=512 --center_crop --random_flip \
  #   --train_batch_size=1 \
  #   --gradient_accumulation_steps=4 \
  #   --gradient_checkpointing \
  #   --mixed_precision="fp16" \
  #   --max_train_steps=15000 \
  #   --learning_rate=1e-05 \
  #   --max_grad_norm=1 \
  #   --lr_scheduler="constant" --lr_warmup_steps=0 \
  #   --output_dir=${OUTPUT_DIR}
