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


def gen_init(self, config, context=False, joint_train=False, training_context=False):
    super(Qwen2VLForConditionalGeneration, self).__init__(config)
    self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
    self.model = Qwen2VLModel(config)
    self.EMO_embed = nn.Parameter(torch.zeros(1, config.hidden_size))
    self.EMO_embed.data.normal_(mean=0.0, std=0.02)

    self.CTX_embed = nn.Parameter(torch.zeros(1, config.hidden_size))
    self.CTX_embed.data.normal_(mean=0.0, std=0.02)
    self.context = context
    self.joint_train = joint_train
    
    if training_context:
        self.text_encoder = CLIPTextModel.from_pretrained('/data4/EmoSym/model/clip-vit-large-patch14')
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('/data4/EmoSym/model/clip-vit-large-patch14')
        self.con_head = nn.Sequential(
            nn.Linear(768, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    # self.cls_head = nn.Linear(config.hidden_size, 8)
    self.cls_head = nn.Sequential(
        nn.Linear(config.hidden_size, 256),
        nn.ReLU(),
        nn.Linear(256, 8)
    )

    self.vocab_size = config.vocab_size
    # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

    # Initialize weights and apply final processing
    self.post_init()

def genforward(
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

        emo_mask = input_ids == self.config.EMO_token_id
        # pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds = torch.where(emo_mask.unsqueeze(-1), self.EMO_embed.data, inputs_embeds)
            # print(self.context)
            context_mask = None
            if getattr(self, "context", False):
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
        # pdb.set_trace()
        loss = None
        if self.training and not self.joint_train:
            cls_loss = F.cross_entropy(emo_pred_label, torch.tensor(emotion_label_idx).to(emo_pred_label.device))
            con_loss = 0.
            if self.context:
                with torch.no_grad():
                    inputs = self.clip_tokenizer(qwen_augment if self.config.qwen_aug else supervise_text, truncation=True, padding=True, return_tensors="pt").to(self.device)
                    text_outputs = self.text_encoder(**inputs)
                # cls_token = text_outputs.last_hidden_state[:, 0, :]
                cls_token = text_outputs[1]
                cls_token = self.con_head(cls_token)

                # ctx_embed = torch.mean(ctx_embed, dim=0)

                cls_token = F.normalize(cls_token, p=2, dim=-1)
                ctx_embed_norm = F.normalize(ctx_embed, p=2, dim=-1)
                scores = torch.matmul(ctx_embed_norm, cls_token.transpose(0, 1)) / 0.07
                scores = (1 + scores) / 2
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                con_loss = F.cross_entropy(scores, target, reduction='mean') # 4.7
            loss = cls_loss * 0.5 + con_loss * 0.5
            # loss = cls_loss 

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

# def get_input_embeddings(self):
#     return self.fusion.embed_tokens

# def set_input_embeddings(self, value):
#     self.fusion.embed_tokens = value

def get_output_embeddings(self):
    return 


def replace_qwen_training_modality_adaptive():
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.__init__ = gen_init
    # transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = flashforward
    # transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = preforward
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = genforward
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.get_output_embeddings = get_output_embeddings
    # transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.get_input_embeddings = get_input_embeddings
    # transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.set_input_embeddings = set_input_embeddings
    

    

