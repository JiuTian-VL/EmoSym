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
from EmoSet import EmoSet
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from utilss import *
# from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from qwen_vl_utils import process_vision_info

import torch.optim as optim
import clip

from transformers import AutoProcessor, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from training.model_file.qwen2_RL import *
import pdb  
import subprocess
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate.utils import gather_object
from torchvision import transforms
import pdb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import shutil
def load_model(args):
    device = args.device
    kwargs = {"device_map": args.device}
    kwargs['torch_dtype'] = torch.float16
    kwargs['_attn_implementation'] = 'flash_attention_2'
    
    if 'lora' in args.model_path.lower() and args.model_base is not None:
        # lora_cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
        # if hasattr(lora_cfg_pretrained, 'quantization_config'):
        #     del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(args.model_base)
        
        print('Loading Qwen2-VL from base model...')
        print(args.context, args.joint_train)
        model = Qwen2VLEmoRL.from_pretrained(args.model_base, low_cpu_mem_usage=True, joint_train=args.joint_train, **kwargs)
        
        print('Loading additional Qwen2-VL weights...')
        non_lora_trainables = torch.load(os.path.join(args.model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith('model.') else k): v
                for k, v in non_lora_trainables.items()
                if 'visual' not in k
            }
        # print(non_lora_trainables)
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, args.model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')
    else:
        raise NotImplementedError
    

    processor.tokenizer.add_tokens(["[EMO]"])
    model.config.EMO_token_id = processor.tokenizer("[EMO]", add_special_tokens=False).input_ids[0]

    processor.tokenizer.add_tokens(["[CTX]"])
    model.config.CTX_token_id = processor.tokenizer("[CTX]", add_special_tokens=False).input_ids[0]
    
    return model, processor




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
import pdb
def generate_attr(processor, model, attribute_total):
  
    data_pro = processor(text=attribute_total, return_tensors="pt", padding=True).to(model.device)
    if hasattr(model, 'module'):
        data_pro = model.module.get_text_features(**data_pro)
    else:
        data_pro = model.get_text_features(**data_pro)
    return data_pro


def get_coefficient():
    with open('dataset_balance/attr_coefficient_emotion_scene.pkl', 'rb') as f:
        coffeicient = pickle.load(f)
    with open('dataset_balance/attr_coefficient_emotion_object.pkl', 'rb') as f:
        tmp = pickle.load(f)
        coffeicient.update(tmp)
    # coffeicient = {key: 1 if value > 0 else 0 for key, value in coffeicient.items()}
    return coffeicient

def gudiance_attr(attribute, tokenizer, text_encoder, weight_dtype):
    # TODO
    ids = tokenizer(
        attribute,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    vec = text_encoder(ids)[1].to(dtype=weight_dtype)
    return vec


def read_attr():
    properties = ["object", "scene"]
    attribute_pro = {"object": [], "scene": []}
    attribute_total = []
    attribute_emo = {}
    for property in properties:
        with open(f'dataset_balance/{property}_attr.pkl', 'rb') as f:
            useful_attr = pickle.load(f)
            tmp = []
            for key in useful_attr:
                tmp.extend(useful_attr[key])
                try:
                    attribute_emo[key].extend(useful_attr[key])
                except:
                    attribute_emo[key] = []
                    attribute_emo[key].extend(useful_attr[key])
            attribute_pro[property].extend(tmp)
            attribute_total.extend(tmp)
    return attribute_total, attribute_emo


class image_encoder(nn.Module):
    def __init__(self):
        super(image_encoder, self).__init__()
        self.resnet = BackBone()
        state = torch.load("weights/image_encoder/2023-08-22-best.pth")
        self.resnet.load_state_dict(state)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[1:-1])

    def forward(self, x):
        out = self.resnet(x)
        return out


class Emotion_classifier(nn.Module):
    def __init__(self,vision_width ):
        super(Emotion_classifier, self).__init__()
        self.fc0 = nn.Linear(vision_width, 768)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(768, 8)


    def forward(self, x):
        x = self.fc0(x)
        x = self.drop_out(self.relu(x))
        x = self.fc1(x)

        return x
class TextualInversionDataset(Dataset):
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
    
    def __init__(
            self,
            data_root,
            tokenizer,
            learnable_property=None,  # [object, scene]
            emotion=None,
            size=512,
            repeats=1,
            flip_p=0.5,
            set="train",
            placeholder_token="*",
            center_crop=False,
    ):
        if learnable_property is None:
            learnable_property = ["scene"]
        # pdb.set_trace()
        # self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.image_paths = []
        
        self.data_root = "/zhu_yi_jie/Zyj_MM/EmoSet_aug_context"  # change it into your EmoSet file location
        self.info = self.get_info(self.data_root, 8)
        data_store = json.load(open(os.path.join(self.data_root, f'{set}.json')))
        self.data_store = [
            [
                self.info['emotion']['label2idx'][item[0]],
                item[1].split('/')[-1].rsplit('.', 1)[0],
                os.path.join(self.data_root, item[1]),
                os.path.join(self.data_root, item[2])
            ]
            for item in data_store
        ]
        # pdb.set_trace()
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.num_images = len(self.data_store)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.tfm = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
      


    def __len__(self):
        return len(self.data_store)
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
    # def get_all_attribute(self):
    #     return self.attribute_list
    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        # image = self.transform(image)
        return image

    def load_annotation_by_path(self, path):
        json_data = json.load(open(path))
        return json_data
    def __getitem__(self, i):
        
        emotion_label_idx, image_id, image_path, annotation_path = self.data_store[i % self.num_images]
        image = self.load_image_by_path(image_path)
        annotation_data = self.load_annotation_by_path(annotation_path)
        # data = {'image_id': image_id, 'image': image, 'emotion_label_idx': emotion_label_idx}

        example = {}
        # # pdb.set_trace()
        # path = self.image_paths[i % self.num_images]
        # image = Image.open(path)
        img_feat = self.tfm(image.copy())
        example["image"] = img_feat
        example['image_path'] = image_path
        example["emotion_label_idx"] = emotion_label_idx
        # if self.learnable_property != ["all"]:
        #     example["attribute"] = path.split('/')[-2].split(')')[-1].lower().replace(' ','_')
        # else:
        #     example["attribute"] = ' '
        # example["emotion"] = path.split('/')[-1].split('_')[0]
        for attribute in self.ATTRIBUTES_MULTI_CLASS:
            # if empty, set to -1, else set to label index
            attribute_label_idx = -1
            if attribute in annotation_data:
                # pdb.set_trace()
                if str(annotation_data[attribute]) in self.info[attribute]['label2idx']:
                    attribute_label_idx = self.info[attribute]['label2idx'][str(annotation_data[attribute])]
            example.update({f'{attribute}_label_idx': attribute_label_idx})
            if attribute == 'scene':
                if attribute in annotation_data:
                   
                    example.update({'scene_text': str(annotation_data[attribute])})
                else:
                    example.update({'scene_text': 'scene'})
        for attribute in self.ATTRIBUTES_MULTI_LABEL:
            assert attribute == 'object'
            num_classes = self.NUM_CLASSES[attribute]
            attribute_label_idx = torch.zeros(num_classes)
            if attribute in annotation_data:
                for label in annotation_data[attribute]:
                    example.update({'object_text': label})
                    
                    if label in self.info[attribute]['label2idx']:
                        attribute_label_idx[self.info[attribute]['label2idx'][label]] = 1
            else:
                example.update({'object_text': 'object'})
            example.update({f'{attribute}_label_idx': attribute_label_idx})
        example.update({
        'scene_aug': str(annotation_data['scene_aug']) if 'scene_aug' in annotation_data else 'scene',
        'object_aug': str(annotation_data['object_aug']) if 'object_aug' in annotation_data else 'object'
            })

        # pdb.set_trace()
        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token 
        text = placeholder_string

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
       

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size))

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def save_pic(img, path):
    if path is not None:
        os.makedirs(path, exist_ok=True)
    try:
        files = sorted([x for x in os.listdir(path) if x.endswith(".jpg")], key=lambda x: int(x.split(".")[0]))
        num = int(files[-1].split(".")[0])
        img.save(f"{path}/{num + 1}.jpg")
    except:
        img.save(f"{path}/0.jpg")
def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    # pdb.set_trace()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    # pdb.set_trace()
    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]    
 
def build_optimizer(model):
    
    model = model.module if hasattr(model, 'module') else model

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    learning_rate_prompts = 1.e-4  ##0.0001
    weight_decay_prompts = 1.e-2
    # learning_rate_prompts = 0  ##0.0001
    # weight_decay_prompts = 0
    prompts_parameters = set_weight_decay(model, skip, skip_keywords, 
    weight_decay=weight_decay_prompts, lr = learning_rate_prompts, 
    have=("EMO_embed", "CTX_embed", "cls_head"), not_have=()  )
    # prompts_parameters = set_weight_decay(model, skip, skip_keywords, 
    #     weight_decay=weight_decay_prompts, lr = learning_rate_prompts, 
    #     have=("cls_head"), not_have=()
    # )
   
 
    

    optimizer = optim.AdamW(prompts_parameters,
                        betas=(0.9, 0.999), eps=1e-8)
    # pdb.set_trace()
    return optimizer
class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        # 计算余弦相似度
        # pdb.set_trace()
        pos_sim = self.cos_sim(anchor, positive)
        neg_sim = self.cos_sim(anchor, negative)
        
        # 三元组损失
        loss = torch.clamp(self.margin + neg_sim - pos_sim, min=0.0)
        return loss.mean()
        
def find_min_feature_value(dict_list):
    all_feature_values = []
    
    # 遍历每个字典，收集所有特征值
    for d in dict_list:
        all_feature_values.append(d['num'])
    
    # 找到所有特征值中的最小值
    # pdb.set_trace()
    if all_feature_values:
        min_value = min(all_feature_values)
        return min_value
    else:
        return None  # 如果没有特征值，返回 None

def build_scheduler(optimizer, n_iter_per_epoch):

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max= 32
    )
    

    return lr_scheduler

def set_requires_grad(parameters, require_grad):
    for p in parameters:
        p.requires_grad = require_grad

def configure_llm(model):
    set_requires_grad(model.model.parameters(), False)
    set_requires_grad(model.visual.parameters(), False)
    set_requires_grad(model.visual.merger.parameters(), False)

    model.EMO_embed.requires_grad = True
    model.CTX_embed.requires_grad = True
    set_requires_grad(model.cls_head.parameters(), True)

def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (s or "").replace("\r", "").strip().strip('"')  # TODO: removing doub123789a?le quotes may not be necessary
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

class ConfidenceDataset(Dataset):
    """
    计算置信度并根据置信度值重新构建一个新的数据集。
    """

    def __init__(self, train_dataloader, generate_quality_image_loader, model, accelerator, output_base, processor):
        """
        初始化数据集，并根据给定的 train_dataloader 和 generate_quality_image_loader 计算每个样本的置信度值，
        然后根据置信度排序筛选数据集。

        参数:
            train_dataloader (DataLoader): 用于加载训练数据的 DataLoader。
            generate_quality_image_loader (DataLoader): 用于加载生成的高质量图像数据的 DataLoader。
            model (torch.nn.Module): 用于生成特征的模型。
            Emo_classifier (torch.nn.Module): 用于预测情感标签的分类器。
        """
        # 计算并获取两个dataloader中的置信度列表

        self.train_root  = '/zhu_yi_jie/Zyj_MM/EmoSet_aug_context/image'
        self.generate_root = os.path.join(output_base, 'image')
        self.transforms_dict = self.get_data_transforms()
        self.transform = self.transforms_dict['train']
        self.processor = processor
        self.train_confidence_list = self.calculate_confidence(generate_quality_image_loader, model,  accelerator, self.train_root, 'bank', self.processor)
        
        self.generate_confidence_list = self.calculate_confidence(generate_quality_image_loader, model,  accelerator, self.generate_root, 'generate', self.processor)
       
        
        
        
        # 从 generate_confidence_list 中获取前 20% 的高置信度样本
        generate_top_20_percent = self.generate_confidence_list[:max(1, int(len(self.generate_confidence_list) * 1))]
         
     
        self.updated_confidence_list = generate_top_20_percent + self.train_confidence_list[len(generate_top_20_percent):]
        

    def visualize_updated_distribution(self, updated_confidence_list, model, accelerator, method='pca+tsne', sample_size=10000, output_base=None):
        """
        可视化更新前后数据的特征分布

        参数:
            updated_confidence_list: 更新后的数据列表，每个样本包含 'source' 字段
            model: 用于特征提取的模型
            accelerator: Accelerate 对象
            epoch: 当前 epoch
            step: 当前 step
            method: 降维方法 ('pca+tsne', 'umap')
            sample_size: 每组数据抽样数量
            output_base: 可视化结果保存路径
        """
        
        model.eval()
        features = []
        sources = []  # 数据来源标记
        colors = {'ori': 'blue', 'generate': 'orange'}  # 可视化颜色映射

        # 抽样数据
        sampled_data = random.sample(updated_confidence_list, min(sample_size, len(updated_confidence_list)))

        with torch.no_grad():
            for item in tqdm(sampled_data):
                image_path = item['image_path']  # 假设 updated_confidence_list 中存储了图像路径
                image = self.load_image_by_path(image_path).to(accelerator.device)  # 自定义图像加载函数
                feature, _, _ = model(image.unsqueeze(0), scene_text=None, object_text=None)  # 提取特征
                features.append(feature.cpu())
                sources.append(item['source'])  # 保存数据来源

        # 将特征拼接为一个矩阵
        features = torch.cat(features, dim=0).numpy()
        # pdb.set_trace()

        # 特征标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
       
        # 降维
        if method == 'pca+tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            reduced_features = reducer.fit_transform(features_scaled)
        else:
            raise ValueError("Method must be 'pca+tsne' or 'umap'")
        
        # 可视化
        plt.figure(figsize=(12, 8))
        for source in set(sources):
            idx = [i for i, s in enumerate(sources) if s == source]
            plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=source, alpha=0.8, color=colors[source], edgecolors='w')
        
        plt.title(f'Feature Distribution')
        plt.legend()
        # plt.xlabel('Component 1')
        # plt.ylabel('Component 2')
        plt.tight_layout()
        
        # 保存图像
       
        plt.savefig(f"{output_base}/updated_feature_distribution.png")
      
        
    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image
    @classmethod
    def get_data_transforms(cls):
        transforms_dict = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
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

    def calculate_confidence(self, dataloader, model, accelerator, image_root, flag, processor):
        """
        计算每个样本的置信度，并将所有数据聚合到主设备上。
        
        参数:
            dataloader (DataLoader): 用于加载数据的 DataLoader。
            model (torch.nn.Module): 特征提取模型。
            Emo_classifier (torch.nn.Module): 用于计算置信度的分类器。
            accelerator (Accelerator): Accelerate 的加速器对象，支持多卡训练。
        
        返回:
            list or None: 在主进程上返回聚合并排序后的 confidence_list，其他进程返回 None。
        """
        #   messages = batch['messages']
        #         label_id = torch.tensor(batch['labels']).to(accelerator.device)
        #         texts = processor.apply_chat_template(
        #         messages, tokenize=False, add_generation_prompt=True
        #     )
        #         image_inputs, _ = process_vision_info(messages)
        #         inputs = processor(
        #                 text=texts,
        #                 images=image_inputs,
        #                 padding=True,
        #                 return_tensors="pt"
        #             ).to(accelerator.device)
        #         outputs = model(**inputs)

        #         ctx_embed, emo_v, emo_label = map(outputs.get, ['ctx_embed', 'emo_embed', 'emo_pred_label'])
        # 
        
        model.eval()
        confidence_list = []
        source = 'origin'
        # 遍历数据加载器中的每个批次
        if accelerator.is_main_process:
            dataloader = tqdm(dataloader, desc="Calculating confidence")
        for batch_data in dataloader:
            
            with torch.no_grad():
            # 将数据移动到当前设备
                messages = batch_data['messages']
                labels = batch_data['labels']
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
                outputs = model(**inputs)
                ctx_embed, emo_v, emo_label = map(outputs.get, ['ctx_embed', 'emo_embed', 'emo_pred_label'])
                if flag == 'generate':
                    source = 'generate'
    
        
                probabilities = emo_label.softmax(dim=-1)
                max_probs, _ = torch.max(probabilities, dim=-1)
            
            # 保存每个样本的置信度和相关信息
            # pdb.set_trace()
            for i in range(len(messages)):
                confidence = max_probs[i].item()
                confidence_list.append({
                    'message': messages[i],
                    'labels': labels[i],
                    'confidence_score': confidence
                })

        # 使用 accelerator.gather_object 聚合所有进程的 confidence_list
        gathered_confidence_lists = gather_object(confidence_list)
        # pdb.set_trace()
        
        gathered_confidence_lists.sort(key=lambda x: x['confidence_score'], reverse=True)

        return gathered_confidence_lists
    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.updated_confidence_list)

    def __getitem__(self, index):
        """
        获取指定索引的样本数据。

        参数:
            index (int): 要获取的样本的索引。

        返回:
            dict: 包含该样本的数据，包括图像、标签和置信度。
        """
        sample = self.updated_confidence_list[index]
        
        return {
            'message': sample['message'],
            'emotion_label_idx': sample['labels'],
            'confidence_score': sample['confidence_score']  # 置信度
        }
def custom_collate_fn(batch):
    messages = [item['message'] for item in batch]
    labels = [item['emotion_label_idx'] for item in batch]
    return {'messages': messages, 'labels': labels}
def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    # writer = SummaryWriter(log_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    logger = create_logger(output_dir=args.output_dir, accelerator=accelerator, name=f"{args.output_dir}")
    
    logger.info(f"working dir: {args.output_dir}")
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models
    args.device = accelerator.device
    model, processor = load_model(args)
    
    configure_llm(model)
  
    
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
    # pdb.set_trace()
    optimizer = build_optimizer(model)
    lr_scheduler = build_scheduler(optimizer, len(train_dataloader))
    # lr_scheduler = build_scheduler(optimizer, 500)
    optimizer, train_dataloader, lr_scheduler, model, processor= accelerator.prepare(
         optimizer, train_dataloader, lr_scheduler, model,  processor
    )

    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model.to(accelerator.device, dtype=weight_dtype)

    start_epoch, max_accuracy = 0, 0.0

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    label2idx = {
        "amusement": 0,
        "awe": 1,
        "contentment": 2,
        "excitement": 3,
        "anger": 4,
        "disgust": 5,
        "fear": 6,
        "sadness": 7
        }
    train_dataloader_container = [train_dataloader]
    prompt = 'What is the emotion of the image?'
    for epoch in range(start_epoch, args.num_train_epochs):
        model.train()
        num_steps = len(train_dataloader_container[0])
        train_iterator = iter(train_dataloader_container[0])
        step = 0
        while step < num_steps:
            
            batch = next(train_iterator)
            with accelerator.accumulate(model):
                model.train()
                messages = batch['messages']
                label_id = torch.tensor(batch['labels']).to(accelerator.device)
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
                outputs = model(**inputs)

                ctx_embed, emo_v, emo_label = map(outputs.get, ['ctx_embed', 'emo_embed', 'emo_pred_label'])        
                cls_loss = F.cross_entropy(emo_label, torch.tensor(batch['labels']).to(emo_label.device))
                loss_forward = cls_loss       
                accelerator.backward(loss_forward,  retain_graph=True)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if step % 2 ==0:
                    # 
                    lr = optimizer.param_groups[0]['lr']
                    
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                  
                    logger.info(
                    f'Train: [{epoch}/{args.num_train_epochs}][{step}/{num_steps}]\t'
                    f'total_loss {loss_forward:.4f}\t'
                    f'mem {memory_used:.0f}MB')
                if step % 195 ==0:
                    # cal_emotion_space(train_dataloader, model, processor, args.device, logger, accelerator)
                    output_base = f"/zhu_yi_jie/Zyj_MM/online_bank_generate/{epoch}_{step}"
                    os.makedirs(output_base, exist_ok=True)
                    if accelerator.is_main_process:  # 只在主进程执行
                        subprocess.run(['bash', '/zhu_yi_jie/onlinebank.sh', '--output_base', output_base], timeout=3600)
                    
                    torch.distributed.barrier() 
                    
                    # pdb.set_trace()
                    generate_quality_image = EmoSet(
                        data_root=output_base,
                        num_emotion_classes=8,
                        phase='train', min_pixels=3136,
                                max_pixels=401408,
                                processor=processor, context=True, qwen_aug=False
                                 )
                    # train_dataset = EmoSet(data_root='/zhu_yi_jie/Zyj_MM/EmoSet_aug_context', num_emotion_classes=8, phase='train', 
                    #           min_pixels=3136,
                    #             max_pixels=401408,
                    #             processor=processor, context=True, qwen_aug=False)
                    generate_quality_image_loader = torch.utils.data.DataLoader(
                        generate_quality_image, 32, shuffle=True, num_workers=2, collate_fn=custom_collate_fn
                    )
                    generate_quality_image_loader = accelerator.prepare(generate_quality_image_loader)
                    # 
                    train_new = ConfidenceDataset(train_dataloader_container[0], generate_quality_image_loader, model, accelerator, output_base, processor)
                    
                    new_train_dataloader  = torch.utils.data.DataLoader(
                            train_new, batch_size=128, shuffle=True, num_workers=8, collate_fn=custom_collate_fn
                        )
                    new_train_dataloader  = accelerator.prepare(new_train_dataloader)
                    # pdb.set_trace()

                    train_dataloader_container[0] = new_train_dataloader
                    train_iterator = iter(train_dataloader_container[0])
                    num_steps = len(train_dataloader_container[0])
                    # pdb.set_trace()
                    torch.distributed.barrier() 
            step += 1
        if accelerator.is_main_process:

            save_state = {'model': accelerator.unwrap_model(model).state_dict()
                            }
            torch.save(save_state,  os.path.join(output_base, "Qwen_joint.pth"))
            logger.info(f"training finish")


if __name__ == "__main__":
    import yaml

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
    main(args)