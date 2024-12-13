
from diffusers import FluxPipeline
import torch
from huggingface_hub import hf_hub_download, login

base_model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", token="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")
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors", use_auth_token="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"))
pipe.fuse_lora(lora_scale=0.125)
pipe.to("cuda", dtype=torch.float16)
