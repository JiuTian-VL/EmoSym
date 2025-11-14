import argparse
import logging
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import shutil
import warnings
from pathlib import Path
import datetime
import numpy as np
import pickle
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from training.model_file.qwen2_emo import *
from model import *
# from inference import inference, generate
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
import json
import diffusers
# import cv2
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
import sys
import warnings
sys.path.append("../")
import argparse
import logging
import math
import os
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.checkpoint
import transformers
import datetime
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import yaml
import json
# from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from qwen_vl_utils import process_vision_info
from model_file.qwen2_emo import *
from termcolor import colored
import torch.optim as optim
import clip

from transformers import AutoProcessor, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
import pdb  
from training.model_file.qwen2_RL import *


class EmoSet(Dataset):
    ATTRIBUTES_MULTI_CLASS = [
        'scene', 'facial_expression', 'human_action', 'brightness', 'colorfulness',
    ]
    ATTRIBUTES_MULTI_LABEL = [
        'object'
    ]
    NUM_CLASSES = {
        'brightness': 11,
        'colorfulness': 11,
        'scene': 254,
        'object': 409,
        'facial_expression': 6,
        'human_action': 264,
    }
    EXTRA_CLASS = [
        'scene_aug', 'object_aug',
    ]
    def __init__(self,
                 data_root,
                 num_emotion_classes,
                 phase,
                 min_pixels, 
                 max_pixels, 
                 processor,
                 context,
                 qwen_aug,
                 ):
        # pdb.set_trace()
        assert num_emotion_classes in (8, 2)
        assert phase in ('train', 'val', 'test')
        self.transforms_dict = self.get_data_transforms()
        self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)
        self.info = self.get_info(data_root, num_emotion_classes)

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = processor
        self.context = context
        self.qwen_aug = qwen_aug
        
        if phase == 'train':
            self.transform = self.transforms_dict['train']
        elif phase == 'val':
            self.transform = self.transforms_dict['val']
        elif phase == 'test':
            self.transform = self.transforms_dict['test']
        else:
            raise NotImplementedError
        # pdb.set_trace()
        data_store = json.load(open(os.path.join(data_root, f'{phase}.json')))
        self.data_store = [
            [
                self.info['emotion']['label2idx'][item[0]],
                item[1].split('/')[-1].rsplit('.', 1)[0],
                os.path.join(data_root, item[1]),
                os.path.join(data_root, item[2])
            ]
            for item in data_store
        ]
        # pdb.set_trace()
    def __len__(self):
        return len(self.data_store)

    @classmethod
    def get_data_transforms(cls):
        transforms_dict = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 'train': transforms.Compose(
            # [
            # transforms.Resize(256),
            # transforms.RandomRotation(15),
            # transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            # transforms.ToTensor(),  # Convert to tensor after all other transformations
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #     ]
            # ),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return transforms_dict

    def get_info(self, data_root, num_emotion_classes):
        assert num_emotion_classes in (8, 2)
        info = json.load(open(os.path.join(data_root, 'info.json')))
        if num_emotion_classes == 8:
            pass
        elif num_emotion_classes == 2:
            emotion_info = {
                'label2idx': {
                    'amusement': 0,
                    'awe': 0,
                    'contentment': 0,
                    'excitement': 0,
                    'anger': 1,
                    'disgust': 1,
                    'fear': 1,
                    'sadness': 1,
                },
                'idx2label': {
                    '0': 'positive',
                    '1': 'negative',
                }
            }
            info['emotion'] = emotion_info
        else:
            raise NotImplementedError

        return info

    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image
    def load_image_by_path_wiout_tranform(self, path):
        image = Image.open(path).convert('RGB')
        return image
    def load_annotation_by_path(self, path):
        json_data = json.load(open(path))
        return json_data
    
    def process_qwen_format(self, image_paths, prompt):
        content_list = []
        # if isinstance(image_paths, str):
        #     image_paths = [image_paths]

        prompt = format_string(prompt)
        query_txt_with_prompt = self.processor.tokenizer(prompt, truncation=True, max_length=450, padding=False, return_tensors=None, add_special_tokens=False)
        text = self.processor.tokenizer.decode(query_txt_with_prompt['input_ids'])
        text += '\nSummarize above image and sentence in one word: '
        
        for path in [image_paths]:
            if len(path) > 0:
                content_list.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                )
        content_list.append(
            {"type": "text", "text": text},
        )

        messages = [
            {
                "role": "user",
                "content": content_list
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[CTX][EMO]" if self.context else "[EMO]."}
                ]
            },
        ]

        return messages

    def __getitem__(self, item):
        emotion_label_idx, image_id, image_path, annotation_path = self.data_store[item]
        image = self.load_image_by_path(image_path)
        image_diff = self.load_image_by_path_wiout_tranform(image_path)
        annotation_data = self.load_annotation_by_path(annotation_path)
        data = {'image_id': image_id, 'image': image, 'emotion_label_idx': emotion_label_idx}
        
        prompt = 'What is the emotion of the image?'
        messages = self.process_qwen_format(image_path, prompt)
        data.update({'message': messages})


        for attribute in self.ATTRIBUTES_MULTI_CLASS:
            # if empty, set to -1, else set to label index
            attribute_label_idx = -1
            if attribute in annotation_data:
                # pdb.set_trace()
                # attribute_label_idx = self.info[attribute]['label2idx'][str(annotation_data[attribute])]
                # data.update({'scene_text': str(annotation_data[attribute])})
                if str(annotation_data[attribute]) in self.info[attribute]['label2idx']:
                    attribute_label_idx = self.info[attribute]['label2idx'][str(annotation_data[attribute])]
            data.update({f'{attribute}_label_idx': attribute_label_idx})
            if attribute == 'scene':
                if attribute in annotation_data:
                    data.update({'scene_text': str(annotation_data[attribute])})
                else:
                    data.update({'scene_text': 'scene'})


        for attribute in self.ATTRIBUTES_MULTI_LABEL:
            # if empty, set to 0, else set to 1
            assert attribute == 'object'

            num_classes = self.NUM_CLASSES[attribute]
            attribute_label_idx = torch.zeros(num_classes)
            
            if attribute in annotation_data:
                for label in annotation_data[attribute]:
                    data.update({'object_text': label})
                    if label in self.info[attribute]['label2idx']:
                        attribute_label_idx[self.info[attribute]['label2idx'][label]] = 1
            else:
                data.update({'object_text': 'object'})
            data.update({f'{attribute}_label_idx': attribute_label_idx})
        # pdb.set_trace()
        data.update({
            'scene_aug': str(annotation_data['scene_aug']) if 'scene_aug' in annotation_data else 'scene',
            'object_aug': str(annotation_data['object_aug']) if 'object_aug' in annotation_data else 'object',
            'qwen_aug': str(annotation_data['qwen_aug']) if 'qwen_aug' in annotation_data else ''
                })
        if not image_diff.mode == "RGB":
            image_diff = image_diff.convert("RGB")

        # img = np.array(image_diff).astype(np.uint8)
        # image = Image.fromarray(img)
        # image = image.resize((512, 512))
        # image = self.flip_transform(image)
        # image = np.array(image).astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        # pixel_values = torch.from_numpy(image).permute(2, 0, 1)
        # data.update({'pixel_values': pixel_values})
        return data
def parse_args(pretrained_model_name_or_path, emotion, train_data_dir, learnable_property, max_train_steps,
               num_train_epochs, att_rate, threshold, seed, emo_rate,
               learning_rate, output_dir, model, num_fc_layers, model_base, model_path, need_LN=False, need_ReLU=False, need_Dropout=False):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=model,
        help="choose the model use to map",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=threshold,
        help="choose the model use to map",
    )
    parser.add_argument(
        "--num_fc_layers",
        type=int,
        default=num_fc_layers,
        help="If the model is MLP, how many fully connected layers do you need?",
    )
    parser.add_argument(
        "--att_rate",
        type=float,
        default=att_rate,
    )
    parser.add_argument(
        "--emo_rate",
        type=float,
        default=emo_rate,
    )
    parser.add_argument(
        "--need_LN",
        type=bool,
        default=need_LN,
    )
    parser.add_argument(
        "--need_ReLU",
        type=bool,
        default=need_ReLU,
    )
    parser.add_argument(
        "--need_Dropout",
        type=bool,
        default=need_Dropout,
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=pretrained_model_name_or_path,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=emotion,
        help="Emotion to learn.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=train_data_dir, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<dummy>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="cat", help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=list, default=learnable_property,
                        help="Choose between 'object' and 'scene'")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=seed, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=num_train_epochs)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=max_train_steps,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="<dummy>",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20000000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", default=True,
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_path,
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=model_base,
    )
    parser.add_argument(
        "--context",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--joint_train",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args
import logging

def create_logger(output_dir, accelerator, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # 根据需求设置日志级别

    # 创建文件处理器（确保目录存在）
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    file_handler = logging.FileHandler(f"{output_dir}/logs/{name}.log")
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    # 主进程添加控制台输出
    if accelerator.is_local_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(console_handler)

    return logger
def load_model_(args):
    
    device = args.device
    kwargs = {"device_map": args.device}
    kwargs['torch_dtype'] = torch.float16
    kwargs['_attn_implementation'] = 'flash_attention_2'
    processor = AutoProcessor.from_pretrained(args.model_base)
    model = Qwen2VLEmoRL.from_pretrained(args.model_base, low_cpu_mem_usage=True, joint_train=True, **kwargs)
    processor.tokenizer.add_tokens(["[EMO]"])
    model.config.EMO_token_id = processor.tokenizer("[EMO]", add_special_tokens=False).input_ids[0]

    processor.tokenizer.add_tokens(["[CTX]"])
    model.config.CTX_token_id = processor.tokenizer("[CTX]", add_special_tokens=False).input_ids[0]
    state_dict = torch.load('/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/Qwen_joint.pth', map_location='cpu')
    # for key in state_dict:
    #     state_dict[key] = torch.nan_to_num(
    #         state_dict[key], 
    #         nan=0.0,          # nan替换为0
    #         posinf=1e4,       # 正无穷替换为1e4
    #         neginf=-1e4       # 负无穷替换为-1e4
    #     )
    # pdb.set_trace()
    model.load_state_dict(state_dict['model'], strict=True)
    
    
    
    return model, processor

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing double quotes may not be necessary
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = s + "." if s[-1] not in [".", "?", "!"] else s  # Add a period at the end of the string
    return s

def convert_qwen_format(processor, image_paths, prompt):
        content_list = []
        # if isinstance(image_paths, str):
        #     image_paths = [image_paths]

        prompt = format_string(prompt)
        query_txt_with_prompt = processor.tokenizer(prompt, truncation=True, max_length=450, padding=False, return_tensors=None, add_special_tokens=False)
        text = processor.tokenizer.decode(query_txt_with_prompt['input_ids'])
        text += '\nSummarize above image and sentence in one word: '
        
        for path in [image_paths]:
            if len(path) > 0:
                content_list.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": 4 * 28 * 28,
                        "max_pixels": 512 * 28 * 28,
                    },
                )
        content_list.append(
            {"type": "text", "text": text},
        )

        messages = [
            {
                "role": "user",
                "content": content_list
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[CTX][EMO]"}
                ]
            },
        ]

        return messages


def custom_collate_fn(batch):
    messages = [item['message'] for item in batch]
    labels = [item['emotion_label_idx'] for item in batch]
    return {'messages': messages, 'labels': labels}

def validate_ori(val_loader, model, processor, device, logger, accelerator):
    model.eval()
    
    # 初始化分布式指标
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        logger.info(f"Start distributed classification evaluation")
        for batch_idx, batch_data in enumerate(tqdm(val_loader, disable=not accelerator.is_main_process)):
            messages = batch_data['messages']
            label_id = batch_data['labels']
            
            # 处理输入数据
            texts = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            # 分布式前向传播
            outputs = model(**inputs)
            
            # 获取预测结果
            cls_fea = outputs.emo_pred_label
            pred_label = cls_fea.argmax(dim=1)
            label_id = torch.tensor(label_id).to(device)

            # 计算当前batch的准确率
            correct = (pred_label == label_id).float()
            
            # 跨设备聚合结果
            correct = accelerator.gather(correct)
            labels_gathered = accelerator.gather(label_id)
            batch_size = torch.tensor(len(label_id)).to(device)
            total_batch_size = accelerator.reduce(batch_size, reduction="sum")

            # 计算当前batch准确率
            batch_acc = correct.sum().item() / len(correct)
            
            # 只在主进程打印中间结果
            if accelerator.is_main_process:
                # 每10个batch打印一次中间精度
                if batch_idx != 0:
                    total_top1 = total_acc / total_samples
                    print(f'Total acc: {total_top1:.4f}')

                # 累加全局指标
                total_acc += correct.sum().item()
                total_samples += len(correct)

    # 主进程计算最终准确率
    if accelerator.is_main_process:
        final_acc = total_acc / total_samples if total_samples != 0 else 0
        logger.info(f'Final Validation Acc: {final_acc:.4f}')
        print(f'\nFinal Validation Accuracy: {final_acc:.4f}')
        return final_acc
    
    return None
@torch.no_grad()
def cal_emotion_space(val_loader, model, processor, device, logger, accelerator):
    # pdb.set_trace()
    model.eval()
    emotion_features = {emotion: {"cls_fea": [], "context": []} for emotion in Emotion}

    with torch.no_grad():
        logger.info(f"Start distributed emotion space calculation")
        for batch_idx, batch_data in enumerate(tqdm(val_loader, disable=not accelerator.is_main_process)):
            # 获取数据
            messages = batch_data['messages']
            label_id = torch.tensor(batch_data['labels']).to(device)
            # 处理输入数据
            texts = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(accelerator.device)

            # 分布式前向传播
            # pdb.set_trace()
            outputs = model(**inputs)
            cls_fea = outputs.emo_embed
            # pdb.set_trace()
            context_fea = outputs.ctx_embed
            
            # 收集所有设备的特征和标签
            gathered_cls_fea = accelerator.gather(cls_fea)
            gathered_context = accelerator.gather(context_fea)
         
            gathered_labels = accelerator.gather(label_id)

            # 只在主进程处理数据
            if accelerator.is_main_process:
                for i, emotion_idx in enumerate(gathered_labels):
                    emotion_name = Emotion[emotion_idx.item()]
                    emotion_features[emotion_name]["cls_fea"].append(gathered_cls_fea[i].cpu())
                    emotion_features[emotion_name]["context"].append(gathered_context[i].cpu())
                  

    # 主进程计算统计量
    if accelerator.is_main_process:
        logger.info(f"Aggregating features from {len(val_loader)*accelerator.num_processes} batches")
        for emotion, features in emotion_features.items():
            if features["cls_fea"]:
                # 计算CLS特征统计
                cls_fea_tensor = torch.stack(features["cls_fea"])
                print(cls_fea_tensor.shape)
                cls_mean, cls_std = cls_fea_tensor.mean(dim=0), cls_fea_tensor.std(dim=0)
                torch.save(cls_mean, f"emotion_space/{emotion}_cls_fea_mean.pt")
                torch.save(cls_std, f"emotion_space/{emotion}_cls_fea_std.pt")
                
                # 计算Context特征统计
                context_tensor = torch.stack(features["context"])
                context_mean, context_std = context_tensor.mean(dim=0), context_tensor.std(dim=0)
                torch.save(context_mean, f"emotion_space/{emotion}_context_mean.pt")
                torch.save(context_std, f"emotion_space/{emotion}_context_std.pt")
                
              

                logger.info(f"Saved {emotion} statistics | CLS shape: {cls_mean.shape}")
        
        print("All emotion spaces saved!")
        return 1
    
    # 其他设备返回空值
    return None
def main(args):
    # pdb.set_trace()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    # writer = SummaryWriter(log_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    
    logger = create_logger(output_dir=args.output_dir, accelerator=accelerator, name=f"Qwen_joint")
    
    logger.info(f"working dir: {args.output_dir}")
    # pdb.set_trace()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
 
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8
    # )
 
   
    
    args.device = accelerator.device
    encoder, processor = load_model_(args)
    encoder.to(accelerator.device)
    train_dataset = EmoSet(data_root='/zhu_yi_jie/Zyj_MM/EmoSet_aug_context', num_emotion_classes=8, phase='train', 
                              min_pixels=3136,
                                max_pixels=401408,
                                processor=processor, context=True, qwen_aug=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=8, collate_fn=custom_collate_fn
    )
    test_dataset = EmoSet(data_root='/zhu_yi_jie/Zyj_MM/EmoSet_aug_context', num_emotion_classes=8, phase='test', 
                              min_pixels=3136,
                                max_pixels=401408,
                                processor=processor, context=True, qwen_aug=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=8, collate_fn=custom_collate_fn
    )
    encoder, processor, test_dataloader, train_dataloader= accelerator.prepare(encoder, processor, test_dataloader, train_dataloader)
    
    # validate_ori(test_dataloader, encoder, processor, args.device, logger, accelerator)

    cal_emotion_space(train_dataloader, encoder, processor, args.device, logger, accelerator)

    
   
def eightemotion(Emo, Emo_num, label, correct):

        for i in range(label.shape[0]):
            emo_label = label[i].item()
            if correct[i] == True:
                Emo[emo_label] += 1
            Emo_num[emo_label] += 1
            # Emo_score[emo_label] += pre[i][emo_label]
        return Emo, Emo_num
Emotion = ["amusement", "awe", "contentment",
               "excitement",
               "anger",
               "disgust",
               "fear",
               "sadness"
               ]               
         
if __name__ == "__main__":

    def parameter(file_name):
        with open(file_name, 'r') as file:
            params = yaml.safe_load(file)
        
        args = parse_args(**params)
        params["project_name"] = os.path.basename(__file__)
        params_json = json.dumps(params)
        os.makedirs(f'{params["output_dir"]}',exist_ok=True)
        with open(f'{params["output_dir"]}/params.json', 'w') as f:
            f.write(params_json)
        return args

    # Choose your config file
    file_name = 'config/config.yaml'
    args = parameter(file_name)
    replace_qwen_training_modality_adaptive()
    main(args)