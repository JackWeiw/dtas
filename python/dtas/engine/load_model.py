import json
from typing import List
import os

import mlc_llm
from mlc_llm.relax_model import (
    chatglm,
    gpt_bigcode,
    gpt_neox,
    gptj,
    llama,
    llama_batched_vllm,
    minigpt,
    mistral,
    param_manager,
    rwkv,
    stablelm_3b,
)
from mlc_llm.relax_model.param_manager import (
    chain_parameter_transforms,
    transform_params_for_each_rank,
)
from mlc_llm.core import mod_transform_before_build
import tvm
from tvm.ir.module import IRModule

model_generators = {
    "llama": llama,
    "mistral": mistral,
    "stablelm_epoch": stablelm_3b,
    "gpt_neox": gpt_neox,
    "gpt_bigcode": gpt_bigcode,
    "minigpt": minigpt,
    "gptj": gptj,
    "rwkv": rwkv,
    "rwkv_world": rwkv,
    "chatglm": chatglm,
}


def load_model(args) -> IRModule:
    if args.model_category == "minigpt":
        # Special case for minigpt, which neither provides nor requires a configuration.
        config = {}
    else:
        with open(
            os.path.join(args.model_path, "config.json"), encoding="utf-8"
        ) as i_f:
            config = json.load(i_f)

    assert args.model_category in model_generators, f"Model {args.model} not supported"
    mod, param_manager, params, model_config = model_generators[
        args.model_category
    ].get_model(args, config)
    for qspec_updater_class in param_manager.qspec_updater_classes:
        qspec_updater = qspec_updater_class(param_manager)
        qspec_updater.visit_module(mod)
    mod = mod_transform_before_build(mod, param_manager, args, model_config)
    return mod


def load_model_2(model_category, model_path, model) -> IRModule:
    pass


__all__ = ["load_model"]
