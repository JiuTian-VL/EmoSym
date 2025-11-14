import os
import sys
sys.path.append('/zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src')

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration
import argparse
from training.augment_dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset



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

def load_model(args):
    device = args.device
    kwargs = {"device_map": args.device}
    kwargs['torch_dtype'] = torch.float16
    kwargs['_attn_implementation'] = 'flash_attention_2'

   
    processor = AutoProcessor.from_pretrained(args.model_base)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, low_cpu_mem_usage=True, **kwargs).to(device)
    
    return model, processor



def validate(val_loader, model, processor, device):
    model.eval()
    
    accuracy = 0
    total = 0
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            messages = batch_data['messages']
            label_id = batch_data['labels']
            annotation_path = batch_data['annotation_path']
            # label_id = label_id.reshape(-1)
            # _image = _image.view(b, n, t, c, h, w)
            texts = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            #  image_inputs = [img_path if img_path is not None else None for img_path in img_paths]
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # cls_fea, patch_token, visual_query_list = model(**inputs)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=77)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # print(output_texts)
            for idx, file_path in enumerate(annotation_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)  # 解析 JSON

                data['qwen_aug'] = output_texts[idx]

                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=4)
    
          

def custom_collate_fn(batch):
    messages = [item['message'] for item in batch]
    labels = [item['emotion_label_idx'] for item in batch]
    annotation_path = [item['annotation_path'] for item in batch]
    return {'messages': messages, 'labels': labels, 'annotation_path': annotation_path}

def main(args):
    model, processor = load_model(args)

    if args.dataset == 'FI':
        test_dataset = FI(
            root_dir=args.data_path, 
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            processor=processor,
            split='train'
        )
    else:
        test_dataset = EmoSet(data_root=args.data_root, num_emotion_classes=8, phase='train', 
                              min_pixels=args.min_pixels,
                                max_pixels=args.max_pixels,
                                processor=processor, context=args.context)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn
    )
    validate(test_dataloader, model, processor, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/zhu_yi_jie/Zyj_MM/Qwen2-VL-2B-Instruct')
    parser.add_argument("--model-base", type=str, default="/zhu_yi_jie/Zyj_MM/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data-path", type=str, default='/zhu_yi_jie/Zyj_MM/FI/image')
    parser.add_argument("--data-root", type=str, default='/data4/Zyj_MM/EmoSet_aug_context')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--dataset", type=str, default='FI')
    parser.add_argument("--context", type=bool, default=False)
    args = parser.parse_args()
    # args.data_type = 'query' if args.infer_query else 'cand'

    main(args)