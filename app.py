###########################################################################################
# Code based on the Hugging Face Space of Depth Anything v2
# https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/app.py
###########################################################################################

import gradio as gr
import cv2
import matplotlib
import numpy as np
import os
from PIL import Image
import spaces
import torch
import tempfile
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from Marigold.marigold import MarigoldPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder='vae')
scheduler = DDIMScheduler.from_pretrained(checkpoint_path, timestep_spacing="trailing", subfolder='scheduler')
image_encoder = CLIPVisionModelWithProjection.from_pretrained(checkpoint_path, subfolder="image_encoder")
feature_extractor = CLIPImageProcessor.from_pretrained(checkpoint_path, subfolder="feature_extractor")
unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
pipe = DepthNormalEstimationPipeline(vae=vae,
                            image_encoder=image_encoder,
                            feature_extractor=feature_extractor,
                            unet=unet,
                            scheduler=scheduler)
pipe = pipe.to(DEVICE)
pipe.unet.eval()

title = "# End-to-End Fine-Tuned GeoWizard"
description = """ Please refer to our [paper](https://arxiv.org/abs/2409.11355) and [GitHub](https://vision.rwth-aachen.de/diffusion-e2e-ft) for more details."""
    
@spaces.GPU
def predict_depth(image, processing_res_choice):
    with torch.no_grad():
        pipe_out = pipe(image, denoising_steps=1, ensemble_size=1, noise="zeros", normals=False, processing_res=processing_res_choice, match_input_res=True)
    pred = pipe_out.depth_np
    pred_colored = pipe_out.depth_colored
    return pred, pred_colored

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)

    with gr.Row():
        submit = gr.Button(value="Compute Depth")
        processing_res_choice = gr.Radio(
                [
                    ("Recommended (768)", 768),
                    ("Native", 0),
                ],
                label="Processing resolution",
                value=768,
            )

    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
    raw_file = gr.File(label="Raw Depth Data (.npy)", elem_id="download")

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image, processing_res_choice):

        if image is None:
            print("No image uploaded.")
            return None
    
        pil_image = Image.fromarray(image.astype('uint8'))
        depth_npy, depth_colored = predict_depth(pil_image, processing_res_choice)
    
        # Save the npy data (raw depth map)
        tmp_npy_depth = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(tmp_npy_depth.name, depth_npy)
    
        # Save the grayscale depth map
        depth_gray = (depth_npy * 65535.0).astype(np.uint16)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        Image.fromarray(depth_gray).save(tmp_gray_depth.name, mode="I;16")
    
        # Save the colored depth map
        tmp_colored_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        depth_colored.save(tmp_colored_depth.name)
   
        return [(image, depth_colored),  tmp_gray_depth.name, tmp_npy_depth.name]

    submit.click(on_submit, inputs=[input_image, processing_res_choice], outputs=[depth_image_slider, gray_depth_file, raw_file])

    example_files = os.listdir('assets/examples')
    example_files.sort()
    example_files = [os.path.join('assets/examples', filename) for filename in example_files]
    example_files = [[image, 768] for image in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_image, processing_res_choice], outputs=[depth_image_slider, gray_depth_file, raw_file], fn=on_submit)


if __name__ == '__main__':
    demo.queue().launch(share=True)