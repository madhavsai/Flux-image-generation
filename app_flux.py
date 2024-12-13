from flask import Flask, request, jsonify, render_template
import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)

# Ensure the static/images directory exists
os.makedirs('static/images', exist_ok=True)

# Load model outside the endpoint
base_model_id = "black-forest-labs/FLUX.1-dev"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
pipe = FluxPipeline.from_pretrained(base_model_id, token="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")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora(lora_scale=0.125)
pipe.to("cuda", dtype=torch.float16)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Generate image
    image = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=3.5).images[0]
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"static/images/image_{timestamp}.png"
    
    # Save image to the static/images folder
    image.save(filename)
    
    return jsonify({"message": "Image generated successfully", "filename": filename}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
