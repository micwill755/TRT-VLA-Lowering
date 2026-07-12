
import torch

MODEL_ID = "nvidia/Cosmos3-Nano"

# we can load in cosmos through individual components
def load_cosmos_components(device):
    from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from transformers import Qwen2TokenizerFast, Qwen3VLVisionModel

    dtype = torch.bfloat16

    transformer = Cosmos3OmniTransformer.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device).eval()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device).eval()
    vision_encoder = Qwen3VLVisionModel.from_pretrained(
        MODEL_ID,
        subfolder="vision_encoder",
        torch_dtype=dtype,
    ).to(device).eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        MODEL_ID,
        subfolder="text_tokenizer",
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler",
    )
    return {
        "transformer": transformer,
        "vae": vae,
        "vision_encoder": vision_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }

def load_cosmos_hf_pipeline(device):
    from diffusers import Cosmos3OmniPipeline

    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=str(device)
    )

    # then pull indiviual components

    return {
        "transformer": pipe.transformer,
        "vae": pipe.vae,
        "vision_encoder": pipe.vision_encoder,
        "tokenizer": pipe.text_tokenizer,
        "scheduler": pipe.scheduler,
    }



def load_policy():
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cosmos_componens = load_cosmos_components(device)
    #cosmos_componens = load_cosmos_hf_pipeline(device)
    
    return 0

if __name__ == "__main__":
    SystemExit(main())