import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json

from PIL import Image
import pdb
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
from utils import format_string
from qwen_vl_utils import process_vision_info

class FI(Dataset):
    def __init__(self, root_dir, min_pixels, max_pixels, processor, context, transform=None, split='train'):
        """ 
        Args:
            root_dir (string): 数据集根目录，包含各个情感标签的子文件夹。
            transform (callable, optional): 用于对图像进行转换的函数（如归一化、数据增强等）。
            split (string, optional): 数据集的划分类型，'train'或'test'，默认为'train'。
        """
        self.root_dir = root_dir
        self.split = split  # 'train' or 'test'

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = processor
        self.context = context
        
        transforms_dict = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.emotion_labels = [
            "amusement", "awe", "contentment", "excitement", "anger", 
            "disgust", "fear", "sadness"
        ]
        self.emotion_label_map = {label: idx for idx, label in enumerate(self.emotion_labels)}
        self.transform = transforms_dict
        self.emotions = os.listdir(root_dir)  # 获取所有情感标签（文件夹）
        self.samples = self._load_samples()
        
        self.train_samples, self.test_samples = self._split_samples()

    def _load_samples(self):
        """
        加载所有样本，遍历每个情感标签文件夹中的文件。
        """
        samples = []
        for emotion in self.emotion_labels:
            emotion_dir = os.path.join(self.root_dir, emotion)
            if os.path.isdir(emotion_dir):
                for filename in os.listdir(emotion_dir):
                    if filename.endswith(".jpg") or filename.endswith(".png"):  # 假设样本是图片文件
                        sample_path = os.path.join(emotion_dir, filename)
                        samples.append((sample_path, emotion))
        return samples

    def _split_samples(self):
        """
        划分训练集和测试集，按80%的训练集和15%的测试集比例。
        """
        train_samples, test_samples = train_test_split(self.samples, test_size=0.15, random_state=42)
        return train_samples, test_samples

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        if self.split == 'train':
            return len(self.train_samples)
        elif self.split == 'test':
            return len(self.test_samples)
        else:
            raise ValueError("split must be either 'train' or 'test'")
        
    def process_qwen_format(self, image_paths, prompt):
        content_list = []
        # if isinstance(image_paths, str):
        #     image_paths = [image_paths]

        prompt = format_string(prompt)
        query_txt_with_prompt = self.processor.tokenizer(prompt, truncation=True, max_length=450, padding=False, return_tensors=None, add_special_tokens=False)
        text = self.processor.tokenizer.decode(query_txt_with_prompt['input_ids'])
        text += '\nSummarize above image and sentence in one word: '
        # if len(text) > 0 and len(image_paths) > 0:
        #     text += '\nSummarize above image and sentence in one word: '
        # elif len(image_paths) == 0:
        #     text += '\nSummarize above sentence in one word: '
        # else:
        #     text += '\nSummarize above image and sentence in one word: '
    
        # if self.img_token and image_paths != '':
        #     text = '[IMG] ' + text
        
        # print(text)
        
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




    def __getitem__(self, idx):
        """
        获取指定索引的样本，返回图像和字典，包括情感标签索引。
        """
        # pdb.set_trace()
        if self.split == 'train':
            sample_path, emotion = self.train_samples[idx]
        elif self.split == 'test':
            sample_path, emotion = self.test_samples[idx]
        else:
            raise ValueError("split must be either 'train' or 'test'")
        
        image = Image.open(sample_path)

        if self.transform:
            image = self.transform(image)

        # 获取情感标签的索引
        emotion_label_idx = self.emotion_label_map[emotion]
        annotation_path = sample_path.replace('jpg', 'json').replace('image', 'annotation')
        qwen_aug = ''
        if self.split == 'train':
            with open(annotation_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            qwen_aug = data['qwen_aug']

        prompt = 'What is the emotion of the image?'

        message = self.process_qwen_format(sample_path, prompt)

        # 返回图像和情感标签的字典
        return {
            "message": message,
            "emotion_label_idx": emotion_label_idx,
            'qwen_aug': qwen_aug
        }

# 使用示例
# 训练集
# train_dataset = FI(root_dir='./emotion_dataset', transform=None, split='train')

# 测试集
# test_dataset = FI(root_dir='./emotion_dataset', transform=None, split='test')

class DataCollatorForPretrainDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        batch_inputs = []
        batch_emo_labels = []

        batch_sup_text = []
        batch_qwen_aug = []
        if 'scene_aug' in examples[0]:
            batch_sup_text = [
                example['scene_aug'] + ' '+ example['object_aug']
                for example in examples
            ]
        if 'qwen_aug' in examples[0]:
            batch_qwen_aug = [
                example['qwen_aug']
                for example in examples
            ]
        
        batch_emo_labels = [
            example['emotion_label_idx']
            for example in examples
        ]
        
        batch_inputs = [
            example['message']
            for example in examples
        ]
        query_inputs = {
            'pixel_values': None,
            'image_grid_thw': None
        }

        query_text = self.processor.apply_chat_template(
            batch_inputs, tokenize=False, add_generation_prompt=True
        )
       
        query_image_inputs, video_inputs = process_vision_info(batch_inputs)
        query_inputs = self.processor(
            text=query_text,
            images=query_image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if 'pixel_values' not in query_inputs:
            query_inputs['pixel_values'] = None
            query_inputs['image_grid_thw'] = None

            
        data_dict = {
            'input_ids': query_inputs['input_ids'],
            'attention_mask': query_inputs['attention_mask'],
            'pixel_values': query_inputs['pixel_values'],
            'image_grid_thw': query_inputs['image_grid_thw'],
            'emotion_label_idx': batch_emo_labels,
            'supervise_text': batch_sup_text,
            'qwen_augment': batch_qwen_aug
        }
            
        return data_dict


def make_pretrain_data_module_FI(processor, data_args):
    sft_dataset = FI(
        root_dir='/zhu_yi_jie/Zyj_MM/FI/image', min_pixels=data_args.min_pixels, max_pixels=data_args.max_pixels, processor=processor, context=data_args.context
    )
    data_collator = DataCollatorForPretrainDataset(processor)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

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
    

def make_pretrain_data_module_EmoSet(processor, data_args):
    sft_dataset = EmoSet(
        data_root='/data4/EmoSym/EmoSet_aug_context', num_emotion_classes=8, phase='train', 
        min_pixels=data_args.min_pixels, max_pixels=data_args.max_pixels, processor=processor, context=data_args.context, qwen_aug=data_args.qwen_aug
    )
    
    data_collator = DataCollatorForPretrainDataset(processor)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

import pdb
if __name__ == '__main__':
    data_root = '/data/Users/zyj/EmoGen/data/EmoSet'
    num_emotion_classes = 8
    phase = 'test'

    dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=num_emotion_classes,
        phase=phase,
    )
    

    # print(dataset.info)
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

    for i, data in enumerate(dataloader):
        pdb.set_trace()
        print(data['emotion_label_idx'])
        print(data['scene_label_idx'])
        print(data['facial_expression_label_idx'])
        print(data['human_action_label_idx'])
        print(data['brightness_label_idx'])
        print(data['colorfulness_label_idx'])
        print(data['object_label_idx'])
        # pdb.set_trace()
        break

