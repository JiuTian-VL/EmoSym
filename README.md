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

Alternatively, install the requirements from `requirements.txt`

## Usage

### Preliminary
[EmoSet](https://vcc.tech/EmoSet) is needed to train in this network. You should use GPt-4o to expand the dataset! 

### Training
To train our network, follow these steps:

First, manually modify the code related to reading EmoSet and change the file location to the location where your EmoSet is located. You should download Qwen2-VL-2B/B/7B-Instruct from Hugging Face.
Start to train your own network for the first step:
```
bash code/Qwen2-VL-Finetune/scripts/finetune_emo.sh
```
Second, we train the model with our step 2; you should modify the rf_model_path to the step 1.
```
bash code/Qwen2-VL-Finetune/scripts/finetune_RL.sh 
Thirdly, start to train your own network:
```
accelerate training/main.py
```

Finally, generate emotional image:
```
python training/inference.py
```
You can modify config/config.yaml to change some details.
## :pencil: Citation

```bib
@inproceedings{zhu2025emosym,
  title={EmoSym: A Symbiotic Framework for Unified Emotional Understanding and Generation via Latent Reasoning},
  author={Zhu, Yijie and Lyu, Yibo and Yu, Zitong and Shao, Rui and Zhou, Kaiyang and Nie, Liqiang},
  booktitle={Proceedings of the 33nd ACM International Conference on Multimedia},
  year={2025}
}
