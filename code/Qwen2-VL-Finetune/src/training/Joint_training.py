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
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
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
        
        self.data_root = "/data4/EmoSym/EmoSet_aug_context"  # change it into your EmoSet file location
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
 
def build_optimizer(model,  mapper, mapper_context):
    
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
    mapper_parameters = set_weight_decay(mapper, skip, skip_keywords, weight_decay = 0.01, lr = 0.001)
    mapper_context_parameters = set_weight_decay(mapper_context, skip, skip_keywords, weight_decay = 0.01, lr = 0.001)
 
    

    optimizer = optim.AdamW(prompts_parameters + mapper_parameters + mapper_context_parameters,
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

def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    # writer = SummaryWriter(log_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

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
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    model = CLIPModel.from_pretrained("model/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("model/clip-vit-large-patch14")
    # pdb.set_trace()
    # encoder = image_encoder()
    
    # pdb.set_trace()
    
    # TODO：加载Qwen
    args.device = accelerator.device
    encoder, processor = load_model(args)
    
    configure_llm(encoder)
  

    model_dict = {
        "FC": lambda args: FC(),
        "MLP": lambda args: MLP(args.num_fc_layers, args.need_ReLU, args.need_LN, args.need_Dropout),
        "SimpleMLP": lambda args: SimpleMLP(args.need_ReLU, args.need_Dropout),
    }
    mapper = model_dict[args.model](args)
    mapper_context = model_dict[args.model](args)
    # mapper_scene = model_dict[args.model](args)
    # mapper_object = model_dict[args.model](args)
    ####各种mapper加载权重#######
    vae.requires_grad_(False)
    
   
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    
    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    # pdb.set_trace()
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # 
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            # unet_wrapper.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        emotion=args.emotion,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    test_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        emotion=args.emotion,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="test",
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    # pdb.set_trace()
    optimizer = build_optimizer(encoder,  mapper, mapper_context)
    lr_scheduler = build_scheduler(optimizer, len(train_dataloader))
    # lr_scheduler = build_scheduler(optimizer, 500)


    if args.validation_epochs is not None:
        args.validation_steps = args.validation_epochs * len(train_dataset) // accelerator.num_processes

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # 
    text_encoder, optimizer, train_dataloader, lr_scheduler, encoder, mapper, model,  mapper_context, processor= accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler, encoder, mapper, model,  mapper_context, processor
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("Emotion_generation")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            args.resume_from_checkpoint = None
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
   
    grad_pseudo = torch.tensor([0], requires_grad=False).to(accelerator.device)
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
    # pdb.set_trace()
    # total_attr, _ = read_attr()
    # total_attr_embed = generate_attr(processor, model, total_attr).detach()
    # attr_coefficient = get_coefficient()

    def change_grad(grad):
        # pdb.set_trace()
        return grad + grad_pseudo
    prompt = 'What is the emotion of the image?'
    for epoch in range(first_epoch, args.num_train_epochs):
        encoder.train()
        mapper.train()
        
        mapper_context.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # 
            messages = convert_qwen_format(processor, batch['image_path'][0], prompt)
            query_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            query_image_inputs, video_inputs = process_vision_info(messages)
            query_inputs = processor(
                text=query_text,
                images=query_image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(args.device)
            
           
            # qwen_outputs['ctx_embed'] qwen_outputs['emo_embed'] qwen_outputs['emo_pred_label'] 
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    num += total_batch_size
                continue

            with accelerator.accumulate(encoder):
                with accelerator.accumulate(mapper):
                    with accelerator.accumulate(mapper_context):
                        with torch.cuda.amp.autocast(enabled=False):
                                # pdb.set_trace()
                                qwen_outputs = encoder(**query_inputs)
                                ctx_embed, emo_v, emo_label = map(qwen_outputs.get, ['ctx_embed', 'emo_embed', 'emo_pred_label'])
                                
                                pred_emd_emo = mapper(emo_v)

                                pred_emd_emo.register_hook(change_grad)
                                pred_emd_context = mapper_context(ctx_embed)
                                # pred_emo_all = pred_emd_context + pred_emd_emo
                                pred_emo_all = pred_emd_emo + pred_emd_context
                                # pdb.set_trace()
                                cls_loss = F.cross_entropy(emo_label, torch.tensor(batch['emotion_label_idx']).to(emo_label.device))
                                # Change the embedding of new token
                                if hasattr(text_encoder, 'module'):
                                    token_embeds = text_encoder.module.get_input_embeddings().weight.data
                                else:
                                    token_embeds = text_encoder.get_input_embeddings().weight.data
                                # pdb.set_trace()
                                token_embeds[placeholder_token_ids] = pred_emo_all
                                # Convert images to latent space
                                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                                latents = latents * vae.config.scaling_factor

                                # Sample noise that we'll add to the latents



                                noise = torch.randn_like(latents)
                                bsz = latents.shape[0]
                                # Sample a random timestep for each image
                                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                                        device=latents.device)
                                timesteps = timesteps.long()

                                # Add noise to the latents according to the noise magnitude at each timestep
                                # (this is the forward diffusion process)
                                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                                
                                # Get the text embedding for conditioning
                                
                                output = text_encoder(batch["input_ids"])
                                encoder_hidden_states = output[0].to(dtype=weight_dtype)
                                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                     

                                # Get the target for loss depending on the prediction type
                                if noise_scheduler.config.prediction_type == "epsilon":
                                    target = noise
                                elif noise_scheduler.config.prediction_type == "v_prediction":
                                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                                else:
                                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                                loss_reconstruction = F.mse_loss(model_pred.float(), target.float(), reduction="mean")




                                # if batch["attribute"][0].lower() in total_attr:
                                #     score = F.cosine_similarity(total_attr_embed, project_semantic).unsqueeze(0)
                                #     fun_loss_attr = nn.CrossEntropyLoss()
                                #     index_attr = total_attr.index(batch["attribute"][0].lower())
                                #     index_attr = torch.tensor([index_attr]).detach().to(score.device)
                                #     loss_attr = fun_loss_attr(score, index_attr)
                                # else:
                                #     loss_attr = torch.tensor([0.0], requires_grad=True).to(mapper.device)
                                # if batch["attribute"][0] in attr_coefficient:
                                #     attr_rate = attr_coefficient[batch["attribute"][0]][label2idx[batch['emotion'][0]]].item()
                                #     if attr_rate < args.threshold:
                                #         attr_rate = 0
                                # else:
                                #     attr_rate = 0
                                
                                loss_forward = loss_reconstruction + cls_loss
                            
                                accelerator.backward(loss_forward,  retain_graph=True)
                                
                                
                                if hasattr(text_encoder, 'module'):
                                    grad_pseudo = text_encoder.module.get_input_embeddings().weight.grad[-1].detach().unsqueeze(0)
                                else:
                                    grad_pseudo = text_encoder.get_input_embeddings().weight.grad[-1].detach().unsqueeze(0)
                                
                                # fake loss in order to backward
                                loss_fake = torch.mean(pred_emd_emo)
                                loss = 0 * loss_fake
                                accelerator.backward(loss, retain_graph=True)
                                # pdb.set_trace()
                                if hasattr(text_encoder, 'module'):
                                    text_encoder.module.get_input_embeddings().weight.grad[-1] *= 0
                                else:
                                    text_encoder.get_input_embeddings().weight.grad[-1] *= 0

                                optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad()

                                # Let's make sure we don't update any embedding weights besides the newly added token
                                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                                index_no_updates[min(placeholder_token_ids): max(placeholder_token_ids) + 1] = False

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                    if global_step % args.validation_steps == 0:
                        tmp = os.path.join(args.output_dir, f"{global_step // len(train_dataloader)}")
                        os.makedirs(tmp, exist_ok=True)
                        torch.save(accelerator.unwrap_model(mapper).state_dict(),
                                   os.path.join(
                                       os.path.join(args.output_dir, f"{global_step // len(train_dataloader)}"),
                                       "mapper.pth"))
                        torch.save(accelerator.unwrap_model(mapper_context).state_dict(),
                                   os.path.join(
                                       os.path.join(args.output_dir, f"{global_step // len(train_dataloader)}"),
                                       "mapper_context.pth"))
                        save_state = {'model': accelerator.unwrap_model(encoder).state_dict(),
                            }
                        torch.save(save_state, os.path.join(
                                       os.path.join(args.output_dir, f"{global_step // len(train_dataloader)}"),
                                       "Qwen_joint.pth"))
                      
                    for tracker in accelerator.trackers:
                        tracker.writer.add_scalar("Loss", loss_forward, global_step)
                        tracker.writer.add_scalar("loss_reconstruction", loss_reconstruction, global_step)
                        # tracker.writer.add_scalar("loss_attribute", loss_attr, global_step)
                        # tracker.writer.add_scalar("loss_emo", loss_emo, global_step)
            logs = {"loss": loss_forward.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
    
    # Create the pipeline using the trained modules and save it.
    # validate
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        torch.save(accelerator.unwrap_model(mapper).state_dict(), os.path.join(args.output_dir, "mapper.pth"))
        torch.save(accelerator.unwrap_model(mapper_context).state_dict(), os.path.join(args.output_dir, "mapper_context.pth"))
       
        # torch.save(accelerator.unwrap_model(encoder).state_dict(), os.path.join(args.output_dir, "encoder.pth"))
        save_state = {'model': accelerator.unwrap_model(encoder).state_dict()
                            }
        torch.save(save_state,  os.path.join(args.output_dir, "Qwen_joint.pth"))
        # torch.save(accelerator.unwrap_model(obj_coefficient).state_dict(), os.path.join(args.output_dir, "obj_coefficient.pth"))
        # torch.save(accelerator.unwrap_model(sce_coefficient).state_dict(), os.path.join(args.output_dir, "sce_coefficient.pth"))
        # torch.save(accelerator.unwrap_model(CTI).state_dict(), os.path.join(args.output_dir, "CTI.pth"))
    accelerator.end_training()


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