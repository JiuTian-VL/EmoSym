import inspect
import math
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import AutoProcessor
import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import BatchEncoding
from peft import PeftModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import *
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers import CLIPTextModel, CLIPTokenizer
import pdb
from .qwen2_emo_class import *
import os

@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    emo_pred_label: Optional[torch.LongTensor] = None
    emo_embed: Optional[torch.FloatTensor] = None
    ctx_embed: Optional[torch.FloatTensor] = None

def load_model(rf_model_base, rf_model_path):
    kwargs = {"device_map": 'cpu'}
    # kwargs = {}
    kwargs['torch_dtype'] = torch.float16
    kwargs['_attn_implementation'] = 'flash_attention_2'
    # replace_qwen_training_modality_adaptive()
    if 'lora' in rf_model_path.lower() and rf_model_base is not None:
        processor = AutoProcessor.from_pretrained(rf_model_base)
        print('Loading Qwen2-VL from base model...')
        model = Qwen2VLEmo.from_pretrained(rf_model_base, low_cpu_mem_usage=True, context=True, joint_train=False, **kwargs)
        print('Loading additional Qwen2-VL weights...')
        non_lora_trainables = torch.load(os.path.join(rf_model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith('model.') else k): v
                for k, v in non_lora_trainables.items()
                if 'visual' not in k
            }
        model.load_state_dict(non_lora_trainables, strict=False)
        model = PeftModel.from_pretrained(model, rf_model_path)
        model = model.merge_and_unload()
    else:
        raise NotImplementedError
    processor.tokenizer.add_tokens(["[EMO]"])
    model.config.EMO_token_id = processor.tokenizer("[EMO]", add_special_tokens=False).input_ids[0]
    processor.tokenizer.add_tokens(["[CTX]"])
    model.config.CTX_token_id = processor.tokenizer("[CTX]", add_special_tokens=False).input_ids[0]
    return model, processor


class Qwen2VLEmoRL(Qwen2VLForConditionalGeneration):
    def __init__(self, config, rf_model_base='', rf_model_path='', joint_train=False):
        super(Qwen2VLForConditionalGeneration, self).__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.EMO_embed = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.EMO_embed.data.normal_(mean=0.0, std=0.02)

        self.CTX_embed = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.CTX_embed.data.normal_(mean=0.0, std=0.02)
     
        self.rf_model_base = rf_model_base
        self.rf_model_path = rf_model_path
        self.joint_train = joint_train
        # pdb.set_trace()
        if self.rf_model_base != '':
            self.rf_model, _ = load_model(self.rf_model_base, self.rf_model_path)
            if 'CTX_embed' and 'EMO_embed' in self.rf_model.state_dict():
                with torch.no_grad():  # 禁用梯度计算
                    self.CTX_embed.copy_(self.rf_model.state_dict()['CTX_embed'])
        self.cls_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_output_embeddings(self):
        return
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        emotion_label_idx: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        supervise_text: Optional[str] = None,
        qwen_augment: Optional[str] = None
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # pdb.set_trace()
        emo_mask = input_ids == self.config.EMO_token_id
        # pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds = torch.where(emo_mask.unsqueeze(-1), self.EMO_embed.data, inputs_embeds)
            # print(self.context)
            
            context_mask = input_ids == self.config.CTX_token_id
            # inputs_embeds = torch.where(context_mask.unsqueeze(-1), self.CTX_embed.data, inputs_embeds)
            inputs_embeds[context_mask] = self.CTX_embed.data.expand(inputs_embeds.shape[0], -1, -1).reshape(-1, inputs_embeds.shape[-1])
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        emo_hidden_states = hidden_states[emo_mask]
        emo_pred_label = self.cls_head(emo_hidden_states)
        ctx_embed = None
        if context_mask is not None:
            ctx_embed = hidden_states[context_mask]
        # emotion_label_idx
        loss = None
        if self.training and not self.joint_train:
            cls_loss = F.cross_entropy(emo_pred_label, torch.tensor(emotion_label_idx).to(emo_pred_label.device))
            with torch.no_grad():
                self.rf_model.eval()
                rf_outputs = self.rf_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            ref_probs = F.softmax(rf_outputs.emo_pred_label, dim=-1)
            current_logits = emo_pred_label
            current_probs = F.softmax(current_logits, dim=-1)
            target_probs = current_probs.gather(1, torch.tensor(emotion_label_idx).to(emo_pred_label.device).unsqueeze(1)).squeeze() 
            emo_reward = (current_logits.argmax(-1) == torch.tensor(emotion_label_idx).to(emo_pred_label.device)).float().detach() 
            dense_reward = 0.3 * emo_reward.detach() + 0.7 * target_probs
            differentiable_reward = dense_reward * target_probs 
            kl_penalty = F.kl_div(
                ref_probs.log(),
                current_probs,
                reduction='batchmean'
            )
            
            # 自适应KL系数（示例）
            kl_coeff = 0.1
            if kl_penalty > 0.5:    # 当KL过大时降低约束
                kl_coeff = 0.05
            elif kl_penalty < 0.1:  # 当KL过小时增强约束
                kl_coeff = 0.2
            rl_loss = -(differentiable_reward.mean() - kl_coeff * kl_penalty)
            # print()
            # loss = cls_loss + rl_loss
            loss = cls_loss +rl_loss
        
        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
            emo_pred_label=emo_pred_label,
            emo_embed=emo_hidden_states,
            ctx_embed=ctx_embed
        )
