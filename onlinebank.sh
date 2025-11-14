#!/bin/bash

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_base) output_base="$2"; shift ;;   # 获取输出路径
        *) echo "Unknown parameter passed: $1"; exit 1 ;;  # 参数检查
    esac
    shift
done

# 输出所用的 output_base 路径
echo "Using output base: $output_base"

# 启动前四个推理进程，并行执行
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 0& 
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 1& 
CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 2& 
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 3& 
wait
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 4& 
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 5&
CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 6& 
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file accelerate/default_config.yaml /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/inference.py --emotion_idx 7& 
wait
mv /zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img $output_base
mv $output_base/img $output_base/image
CUDA_VISIBLE_DEVICES=5 python /zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src/training/data_filter.py --output_base "$output_base"
