from flask import Flask, request, render_template, send_file
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import io
import base64

app = Flask(__name__, template_folder='templates')

DEVICE = "cpu"
if torch.cuda.is_available() and True:  
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and False:  # Set to True to enable MPS
    DEVICE = "mps"

print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '')
    uncond_prompt = request.form.get('negative_prompt', '')
    cfg_scale = float(request.form.get('cfg_scale', 7.5))
    num_inference_steps = int(request.form.get('steps', 50))
    seed = int(request.form.get('seed', 42))
    sampler = request.form.get('sampler', 'ddpm')
    
    input_image = None
    strength = float(request.form.get('strength', 0.9))
    if 'input_image' in request.files:
        file = request.files['input_image']
        if file.filename != '':
            input_image = Image.open(file).convert('RGB')

    # Generate image
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=True,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    img = Image.fromarray(output_image)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode()

    return {'image': f'data:image/png;base64,{img_base64}'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)