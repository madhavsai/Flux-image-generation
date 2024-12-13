from flask import Flask, request, jsonify, render_template
import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download, login
from datetime import datetime
import os

app = Flask(__name__)

# Login to Hugging Face
HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

# Initialize the global pipe variable
pipe = None

def load_pipeline():
    global pipe
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", token=HUGGINGFACE_TOKEN)
    pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors", use_auth_token=HUGGINGFACE_TOKEN))
    pipe.fuse_lora(lora_scale=0.125)
    pipe.to("cuda", dtype=torch.float16)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image-single', methods=['POST'])
def generate_image_single():
    global pipe
    if pipe is None:
        load_pipeline()
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
    print(ip_address + f'/static/images/{filename}')


    return jsonify({"message": "Image generated successfully", "image_url": "http://"+ip_address + f'/static/images/{filename}'}), 200

@app.route('/generate-image', methods=['POST'])
def generate_image():
    global pipe
    if pipe is None:
        load_pipeline()
    data = request.get_json()
    ip_address = request.host
    prompt = data.get('prompt', '')
    width = data.get('width', 512)  # default width
    height = data.get('height', 512)  # default height

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    image_urls = []
    for _ in range(4):
        # Generate the image with specified width and height
        image = pipe(prompt=prompt, width=width, height=height, num_inference_steps=8, guidance_scale=3.5).images[0]

        # Create the static/images directory if it doesn't exist
        if not os.path.exists('static/images'):
            os.makedirs('static/images')

        # Save the image with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')  # Add microseconds to ensure uniqueness
        filename = f"{timestamp}.png"
        filepath = os.path.join('static', 'images', filename)
        image.save(filepath)

        # Append the image URL to the list
        image_urls.append(ip_address + f'/static/images/{filename}')

    return jsonify({"message": "Images generated successfully", "image_urls": image_urls}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
