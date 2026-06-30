
from __future__ import annotations


import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from trt.compile import save_trt_engine_module
from trt.io_spec import ComponentIOSpec, VLA_VISION_IO
from trt.plugin_utils import patch_vision_attention, restore_attention
from trt.utils import free_cuda_memory
from trt.profile import VLAProfile

logger = logging.getLogger(__name__)

class VLAExportEdgeLLMPipeline:
    def __init__(self, profile: VLAProfile):
        self.profile = profile
        self.vision_pipeline = PixelsToImageEmbeddingsPipeline(profile)
    
    #public
    def __call__(self, *args, **kwargs):
        self.forward(args.dataset_id, args.engine_root)

    def forward(self, 
        dataset_id, 
        engine_root):

        # load test data
        data = load_test_data(
            dataset_id=args.dataset_id, 
            episode_index=0,
            frame_index=0
        )
        # take test data and produce model specific inputs using the preprocessor
        model_inputs = self._prepare_model_inputs(data, True)

        # save engines
        # load engines with plugins
        # create eager models
        # benchmark loaded engines vs eager
        # compile in mem engines with plugins
        # benchmark loaded engines vs eager

        # create local engine path
        engine_root = str(pathlib.Path(args.engine_root))

        # every model input has the following
        tokenized_data = model_inputs['tokenized_data']
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']

        # --------- pre process for action input ---------
        #self.pre_process_action_input()

        

        # -------------------------
        # Vision engine
        # -------------------------
        print("compiling vision")
        engine_dir = str(pathlib.Path(engine_root) / "visual")

        vis_params = self.build_vision_export_params(
            model,
            pixel_values,
            device,
            io=io,
            trt_settings=VISION_TRT_SETTINGS,
            input_dtype=torch.float16,
        )

        self.vision_pipeline(pixel_values,
            engine_dir,
            vis_params,
            device=device)
            
        '''# VitRunner visual.engine output (flat): image_embed_flat_shape == [B*S, H]
        image_embs = self.create_image_embs()

        # --------- pack text + image embs together to create 1 tensor of multimodal embeds ---------

        # llm_inference requires tokenizer + embedding table + chat template in language/.
        print("saving tokenizer")
        language_engine_dir = str(pathlib.Path(engine_root) / "language")
        language_model = model.backbone.eagle_model.language_model
        save_embedding_table(language_model, language_engine_dir)
        save_tokenizer_for_edge_llm(
            language_engine_dir,
            tokenizer=tokenizer,
            chat_template=build_groot_vitrunner_chat_template(tokenizer),
        )

        # -------------------------
        # Language engine
        # -------------------------
        print("compiling language")

        spec = build_groot_language_export_params(
            model,
            input_ids,
            image_token_id=int(vis_params.image_token_id),
            seq_len_per_image=int(vis_params.config_seq_len),
            device=torch.device(device),
            io=io,
            dtype=torch.float16,
        )
        save_language_engine_for_edge_llm(language_engine_dir, spec)

        mtmdl_embds = pack_groot_language_inputs(
            model,
            trt_image_embs,
            input_ids,
            attention_mask,
        )'''

# trt/pipeline.py

from pathlib import Path

from trt.config.stage_config import PipelineConfig
from trt.hooks.resolve import resolve
from trt.runner.base import StageContext


class VLAExportPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, profile, policy, device, model_inputs, engine_root):
        ctx = StageContext(
            profile=profile,
            policy=policy,
            model=policy.model,
            device=device,
            model_inputs=model_inputs,
            engine_root=Path(engine_root),
        )

        hooks = self.config.hooks
        if hooks.preprocess:
            resolve(hooks.preprocess)(ctx)

        for stage_cfg in self.config.stages:
            runner = resolve(stage_cfg.runner)(stage_cfg)
            result = runner.run(ctx)
            ctx.artifacts[f"stage_{stage_cfg.stage_id}"] = result

        if hooks.postprocess:
            resolve(hooks.postprocess)(ctx)

        return ctx