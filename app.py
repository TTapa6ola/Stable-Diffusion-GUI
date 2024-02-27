import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

st.title("Stable Diffusion GUI")
text_input = st.text_input("Введите prompt:")
negative_input = st.text_input("Введите negative prompt:")
start_button = st.button("Сгенерировать изображение")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

if torch.cuda.is_available():
    pipe = pipe.to(torch.device("cuda"))

if start_button:
    prompt = text_input
    negative_prompt = negative_input

    image = pipe(prompt, negative_prompt=negative_prompt).images[0]

    st.image(image)
    