
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

# 等待所有推理进程完成
