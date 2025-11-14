import os
import sys
sys.path.append('/zhu_yi_jie/Zyj_MM/code/Qwen2-VL-Finetune/src')

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from training.model_file.qwen2_RL import *
import argparse
from training.EmoSet import *
from tqdm import tqdm
import logger
from torch.utils.data import DataLoader, Dataset

logger = logging.get_logger("/zhu_yi_jie/Zyj_MM/code/logger/debug")


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
    
    if 'lora' in args.model_path.lower() and args.model_base is not None:
        # lora_cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
        # if hasattr(lora_cfg_pretrained, 'quantization_config'):
        #     del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(args.model_base)
        
        print('Loading Qwen2-VL from base model...')
        print(args.context, args.joint_train)
        model = Qwen2VLEmoRL.from_pretrained(args.model_base, low_cpu_mem_usage=True, **kwargs)
        
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



def validate(val_loader, model, processor, device, logger):
    model.eval()
    
    accuracy = 0
    total = 0
    with torch.no_grad():
        logger.info(f"start classification evaluation")
        for batch_data in tqdm(val_loader):
            messages = batch_data['messages']
            label_id = batch_data['labels']
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
                outputs = model(**inputs)
            ########calcaulate_mean std
            # pdb.set_trace()
            cls_fea = outputs.emo_pred_label
            pred_label = cls_fea.argmax(dim=1)
            label_id = torch.tensor(label_id).to(pred_label.device)
            acc = (pred_label == label_id).sum().item()
            accuracy += acc
            total += pred_label.shape[0]
            print(accuracy/total)
    total_acc = accuracy / total
    print(f'Total acc: {total_acc:.4f}')



def custom_collate_fn(batch):
    messages = [item['message'] for item in batch]
    labels = [item['emotion_label_idx'] for item in batch]
    return {'messages': messages, 'labels': labels}

def main(args):
    model, processor = load_model(args)

    if args.dataset == 'FI':
        test_dataset = FI(
            root_dir=args.data_path, 
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            processor=processor,
            context=args.context,
            split='test'
        )
    else:
        test_dataset = EmoSet(data_root=args.data_root, num_emotion_classes=8, phase='test', 
                              min_pixels=args.min_pixels,
                                max_pixels=args.max_pixels,
                                processor=processor, context=args.context, qwen_aug=False)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn
    )
    validate(test_dataloader, model, processor, args.device, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/zhu_yi_jie/Zyj_MM/Training_RL/EmoSet_aug_lora_context_equal_1_RL_pretrain_FI/checkpoint-39')
    parser.add_argument("--model-base", type=str, default="/zhu_yi_jie/Zyj_MM/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data-path", type=str, default='/zhu_yi_jie/Zyj_MM/FI/image')
    parser.add_argument("--data-root", type=str, default='/zhu_yi_jie/Zyj_MM/EmoSet_aug_context')
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--dataset", type=str, default='FI')
    parser.add_argument("--context", type=bool, default=True)
    parser.add_argument("--joint_train", type=bool, default=False)
    
    args = parser.parse_args()
    print(args.context)
    # args.data_type = 'query' if args.infer_query else 'cand'
    # replace_qwen_training_modality_adaptive()
    main(args)