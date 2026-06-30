# trt/executor/models/groot/glue.py

def vision_to_language(ctx, upstream, stage_inputs):
    vision = upstream[0]  # StageResult from stage 0
    return {
        **stage_inputs,
        "image_embs": vision.tensors["image_embs"],
        "config_seq_len": vision.metadata["config_seq_len"],
        "image_token_id": vision.metadata["image_token_id"],
    }

def language_to_action(ctx, upstream, stage_inputs):
    language, vision = upstream[0], upstream[1]
    return {
        **stage_inputs,
        "lm_hidden_states": language.tensors["hidden_states"],
        "image_token_id": vision.metadata["image_token_id"],
        ...
    }