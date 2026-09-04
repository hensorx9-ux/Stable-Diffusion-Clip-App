import os
import streamlit as st
import torch
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor

# Load local environment variables if running locally
from dotenv import load_dotenv
load_dotenv()

# NEW (Works locally and on Streamlit Cloud)
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Stable Diffusion + CLIP Scorer", layout="wide")
st.title("🎨 AI Image Generator & CLIP Evaluator")

if not HF_TOKEN:
    st.warning("⚠️ Hugging Face token not found. Please add HF_TOKEN to your Streamlit Secrets or local .env file.")

@st.cache_resource
def load_clip():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(
        model_name, 
        image_processor=CLIPImageProcessor.from_pretrained(model_name)
    )
    return model, processor

# User UI Controls
st.sidebar.header("Prompt Settings")
prompt_1 = st.sidebar.text_input("Prompt 1", "A hyper-realistic photo of a friendly lion")
prompt_2 = st.sidebar.text_input("Prompt 2", "A stylized oil painting of a NYC Brownstone")

if st.sidebar.button("Generate & Evaluate"):
    if not HF_TOKEN:
        st.error("Cannot generate images without a valid HF_TOKEN.")
    else:
        with st.spinner("Generating images via Hugging Face Serverless API..."):
            client = InferenceClient(api_key=HF_TOKEN)
            prompts = [prompt_1, prompt_2]
            
            # Generate images via serverless API using FLUX.1-schnell
            images = []
            for p in prompts:
                img = client.text_to_image(
                    p, 
                    model="black-forest-labs/FLUX.1-schnell"
                )
                images.append(img)

            # Evaluate zero-shot matching probability using CLIP
            clip_model, clip_processor = load_clip()
            inputs = clip_processor(text=prompts, images=images, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.nn.functional.softmax(logits / 10, dim=-1)

        # Display Generated Images & Scores
        col1, col2 = st.columns(2)
        for idx, (col, img, prompt) in enumerate(zip([col1, col2], images, prompts)):
            with col:
                st.image(img, caption=prompt, use_container_width=True)
                st.write("*CLIP Matching Probabilities:*")
                st.write(f"- {prompts[0]}: {probs[idx][0].item():.2%}")
                st.write(f"- {prompts[1]}: {probs[idx][1].item():.2%}")