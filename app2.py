from flask import Flask, request, jsonify,render_template
import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download
from diffusers import FluxPipeline
from datetime import datetime
from huggingface_hub import hf_hub_download, login
import os

app = Flask(__name__)

# HUGGINGFACE_TOKEN = "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"
# login(token=HUGGINGFACE_TOKEN)

# Load the model and weights
# base_model_id = "black-forest-labs/FLUX.1-dev"
# repo_name = "ByteDance/Hyper-SD"
# ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
# pipe = FluxPipeline.from_pretrained(base_model_id, token="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")
# pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
# pipe.fuse_lora(lora_scale=0.125)
# pipe.to("cuda", dtype=torch.float16)

# @app.route('/')
# def index():
#     return render_template('index.html')


# def hugging_login():
#     base_model_id = "black-forest-labs/FLUX.1-dev"
#     repo_name = "ByteDance/Hyper-SD"
#     ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
#     pipe = FluxPipeline.from_pretrained(base_model_id, token="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")
#     pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
#     pipe.fuse_lora(lora_scale=0.125)
#     pipe.to("cuda", dtype=torch.float16)


# @app.route('/generate-image', methods=['POST'])
# def generate_image():
#     data = request.get_json()
#     ip_address = request.host
#     prompt = data.get('prompt', '')

#     if not prompt:
#         return jsonify({"error": "Prompt is required"}), 400

#     # Generate the image
#     image = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=3.5).images[0]

#     # Create the static/images directory if it doesn't exist
#     if not os.path.exists('static/images'):
#         os.makedirs('static/images')

#     # Save the image with a timestamp
#     timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
#     filename = f"{timestamp}.png"
#     filepath = os.path.join('static', 'images', filename)
#     image.save(filepath)

#     return jsonify({"message": "Image generated successfully", "image_url": ip_address+f'/static/images/{filename}'}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




from flask import Flask, request, jsonify, render_template
import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download, login
from datetime import datetime
import os

app = Flask(__name__)

# Login to Hugging Face
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
login(token=HUGGINGFACE_TOKEN)

# Load the model and weights
base_model_id = "black-forest-labs/FLUX.1-dev"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
pipe = FluxPipeline.from_pretrained(base_model_id, token=HUGGINGFACE_TOKEN)
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name, use_auth_token=HUGGINGFACE_TOKEN))
pipe.fuse_lora(lora_scale=0.125)
pipe.to("cuda", dtype=torch.float16)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    ip_address = request.host
    prompt = data.get('prompt', '')
    width = data.get('width', 512)  # default width
    height = data.get('height', 512)  # default height
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Generate the image with specified width and height
    image = pipe(prompt=prompt, width=width, height=height, num_inference_steps=8, guidance_scale=3.5).images[0]
    
    # Create the static/images directory if it doesn't exist
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    
    # Save the image with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"{timestamp}.png"
    filepath = os.path.join('static', 'images', filename)
    image.save(filepath)
    
    return jsonify({"message": "Image generated successfully", "image_url": ip_address + f'/static/images/{filename}'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
