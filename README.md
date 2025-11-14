<div align="center">
<h2 class="papername"> EmoSym: A Symbiotic Framework for Unified Emotional Understanding<br>and Generation via Latent Reasoning </h2>
<div>
    <a href="https://scholar.google.com.hk/citations?user=0GtAUPoAAAAJ&hl=zh-CN&oi=sra" target="_blank">Yijie Zhu</a>,
    <a href="https://orcid.org/0009-0001-1562-804X" target="_blank">Yibo Lyu</a>,
    <a href="https://zitongyu.github.io/" target="_blank">Zitong Yu*</a>, 
    <a href="https://rshaojimmy.github.io/OrionLab/" target="_blank">Rui Shao*</a>,
    <a href="https://kaiyangzhou.github.io/" target="_blank">Kaiyang Zhou</a>,
    <a href="http://faculty.hitsz.edu.cn/guanweili" target="_blank">Liqiang Nie</a>
</div>

School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen<br>
Great Bay University<br>
*Corresponding author<br>
[![arXiv](https://img.shields.io/badge/arXiv-2407.14439-b31b1b.svg?logo=arxiv)](https://github.com/JiuTian-VL/EmoSym)


</div>

</div>

## :fire: If you find this work useful for your research, please kindly cite our paper and star our repo.

## :fire: Updates
- [11/2025] The code is released!
- [07/2025] EmoSym has been accepted by ACM MM 2025!

## :fire: Introduction

This is the github repository of *EmoSym: A Symbiotic Framework for Unified Emotional Understanding and Generation via Latent Reasoning*. In this work, we introduce EmoSym, a symbiotic framework that models emotional understanding and generation by leveraging latent reasoning.

The whole framework of EmoSym:

<div align="center">
<img src='assets/model.png' width='100%'>
</div>

## Setup
To create the conda environment needed to run the code, run the following command:

```
conda env create -f environment/env.yaml
```


## Usage

### Preliminary
[EmoSet](https://vcc.tech/EmoSet) is needed to train in this network. You should use GPt-4o to expand the dataset! 
Please download the following models from Hugging Face:
Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct
(Or the specific variants used in your experiments.)
Before training, manually update the dataset-loading code so that the file path correctly points to your local EmoSet directory.

### Training
To train our model, follow these steps:

Step 1 — Emotional Understanding Finetuning

Run the initial finetuning script:
```
bash code/Qwen2-VL-Finetune/scripts/finetune_emo.sh
```
Step 2 — Reinforcement Learning Finetuning
Modify the rf_model_path in the script to the checkpoint obtained in Step 1, then run:
```
bash code/Qwen2-VL-Finetune/scripts/finetune_RL.sh 
```
Step 3 — Joint Training
Run the joint training pipeline:
```
sh joint_training.sh
```
### Evaluation
#### Emotional Understanding
To evaluate emotional understanding, run the following script:
```
python code/Qwen2-VL-Finetune/src/training/evalutaion.py
```
#### Emotional Generation
For emotional generation evaluation, refer to the official EmoGen evaluation pipeline
https://github.com/JingyuanYY/EmoGen/tree/master
1. Compute the emotion space using the EmoGen method.
2. Evaluate generated samples with the same metrics used in EmoGen (e.g., emotion alignment, diversity, intensity consistency).

## Citation
If you find this work useful, please kindly cite our paper:
```
@inproceedings{zhu2025emosym,
  title={EmoSym: A Symbiotic Framework for Unified Emotional Understanding and Generation via Latent Reasoning},
  author={Zhu, Yijie and Lyu, Yibo and Yu, Zitong and Shao, Rui and Zhou, Kaiyang and Nie, Liqiang},
  booktitle={Proceedings of the 33nd ACM International Conference on Multimedia},
  year={2025}
}
```

