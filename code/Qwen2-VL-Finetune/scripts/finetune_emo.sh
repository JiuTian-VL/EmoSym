#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="/data4/EmoSym/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export TOKENIZERS_PARALLELISM=false

deepspeed --include localhost:0,1,2,3 --master_port 25670 src/training/train.py \
    --use_liger False \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'cls_head', 'con_head', 'EMO_embed', 'CTX_embed', 'text_encoder']" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir ./EmoSet_aug_lora_context_equal_1_FI_7B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --min_pixels $((4 * 28 * 28)) \
    --max_pixels $((512 * 28 * 28)) \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps  740\
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --training_dataset EmoSet \
    --context True \
    --qwen_aug True
#128

# deepspeed --include localhost:3,4,5,6 --master_port 25670 src/training/train.py \
#     --use_liger False \
#     --lora_enable True \
#     --use_dora False \
#     --lora_namespan_exclude "['lm_head', 'embed_tokens', 'cls_head', 'con_head', 'EMO_embed', 'CTX_embed', 'text_encoder']" \
#     --lora_rank 64 \
#     --lora_alpha 128 \
#     --lora_dropout 0.05 \
#     --num_lora_modules -1 \
#     --deepspeed scripts/zero2.json \
#     --model_id $MODEL_NAME \
#     --data_path /path/to/your/training/data.json \
#     --image_folder /path/to/your/image/folder \
#     --remove_unused_columns False \
#     --freeze_vision_tower True \
#     --freeze_llm False \
#     --tune_merger False \
#     --bf16 True \
#     --fp16 False \
#     --disable_flash_attn2 False \
#     --output_dir /data4/Zyj_MM/Training_EMO/EmoSet_wo_context \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 128 \
#     --gradient_accumulation_steps 1 \
#     --min_pixels $((4 * 28 * 28)) \
#     --max_pixels $((512 * 28 * 28)) \
#     --learning_rate 1e-4 \
#     --merger_lr 1e-5 \
#     --vision_lr 2e-6 \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --gradient_checkpointing True \
#     --report_to tensorboard \
#     --lazy_preprocess True \
#     --save_strategy "steps" \
#     --save_steps 185 \
#     --save_total_limit 10 \
#     --dataloader_num_workers 4 \
#     --training_dataset EmoSet \
#     --context False