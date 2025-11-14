import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser
from training.trainer import QwenTrainer

from training.EmoSet import make_pretrain_data_module_FI, make_pretrain_data_module_EmoSet
from training.params import DataArguments, ModelArguments, TrainingArguments

from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
# from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
# from training.model_file.qwen2_RL import *
# from training.model_file.qwen2_emo import *
from training.model_file.qwen2_RL import *

from peft import PeftModel

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def count_params(model):
    total_param = 0
    trainable_param = 0
    trainable_param_list = []
    for name, param in model.named_parameters():
        total_param += param.numel()
        if param.requires_grad == True:
            
            trainable_param_list.append(name)
            trainable_param += param.numel()

    rank0_print(trainable_param_list)
    rank0_print(f'total params: {total_param / 1e6} M\ntrainable params: {trainable_param / 1e6} M')
    rank0_print("--- NOTE: When using deepspeed zero3, the params count maybe inaccurate ---")

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, training_args.tune_merger)

def configure_llm(model, training_args):
    # lm_head = model.lm_head.parameters()
    # set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    # apply_liger_kernel_to_qwen2_vl()
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert not (training_args.lora_enable and training_args.freeze_llm), 'When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA.'

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    # replace_qwen_training_modality_adaptive_RL()
    # replace_qwen_training_modality_adaptive()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    training_args.remove_unused_columns = False

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))


    model = Qwen2VLEmoRL.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        rf_model_base=model_args.rf_model_base, rf_model_path=model_args.rf_model_path,
        **bnb_model_from_pretrained_args
    )

    # TODO: video_token_id
    world_size = dist.get_world_size()
    process_rank = dist.get_rank()

    training_args.process_rank = process_rank

    model.config.use_cache = False
    model.config.qwen_aug = training_args.qwen_aug

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_id,
                                              padding_side="right",
                                              min_pixels=data_args.min_pixels,
                                              max_pixels=data_args.max_pixels,)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    processor.tokenizer.add_tokens(["[EMO]"])
    model.config.EMO_token_id = processor.tokenizer("[EMO]", add_special_tokens=False).input_ids[0]

    processor.tokenizer.add_tokens(["[CTX]"])
    model.config.CTX_token_id = processor.tokenizer("[CTX]", add_special_tokens=False).input_ids[0]

    # model.config.context = training_args.context
    
    
    # When using LoRA, the model is rapped once more.
    if training_args.lora_enable:
        model_to_configure = model.model

        set_requires_grad(model.cls_head.parameters(), True)
        model.EMO_embed.requires_grad = True
        model.CTX_embed.requires_grad = True
        if training_args.context:
            set_requires_grad(model.rf_model.parameters(), False)
            model.rf_model.CTX_embed.requires_grad = False
            model.rf_model.CTX_embed.requires_grad = False
    else:
        model_to_configure = model
        configure_llm(model_to_configure, training_args)
    # print(model)
    # for name, param in model.lm_head.named_parameters():
    #     print(f"Name: {name}, Requires Grad: {param.requires_grad}")
    # print(model.lm_head.training)
    
    if not training_args.vision_lora:
        configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    model.config.vision_lr = training_args.vision_lr

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.clip_path = model_args.clip_path
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_args.context = training_args.context
    data_args.qwen_aug = training_args.qwen_aug
    if training_args.training_dataset == 'FI':
        data_module = make_pretrain_data_module_FI(processor=processor,
                                                    data_args=data_args)
    else:
        data_module = make_pretrain_data_module_EmoSet(processor=processor,
                                                    data_args=data_args)
    count_params(model)
    trainer = QwenTrainer(
        model=model,
        processor=processor,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
                model.config.save_pretrained(training_args.output_dir)
                model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                processor.tokenizer.save_pretrained(training_args.output_dir)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
    



if __name__ == "__main__":
    train()