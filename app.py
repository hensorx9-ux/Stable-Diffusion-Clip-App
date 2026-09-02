import streamlit as st
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor

st.set_page_config(page_title="Stable Diffusion + CLIP Scorer", layout="wide")
st.title("🎨 Stable Diffusion Image Generator & CLIP Evaluator")

# Cache models to prevent re-loading on every interaction
@st.cache_resource
def load_sd_pipeline():
    return StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        revision="fp16" if torch.cuda.is_available() else None,
        use_safetensors=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")

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
    with st.spinner("Loading models and generating images..."):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = load_sd_pipeline()
        clip_model, clip_processor = load_clip()

        prompts = [prompt_1, prompt_2]
        
        # Image Generation
        images = pipe(prompts).images

        # CLIP Evaluation
        inputs = clip_processor(text=prompts, images=images, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image
        probs = torch.nn.functional.softmax(logits / 10, dim=-1)

    # Display Results
    col1, col2 = st.columns(2)
    for idx, (col, img, prompt) in enumerate(zip([col1, col2], images, prompts)):
        with col:
            st.image(img, caption=prompt, use_column_width=True)
            st.write(f"*CLIP Probabilities:*")
            st.write(f"- {prompts[0]}: {probs[idx][0].item():.2%}")
            st.write(f"- {prompts[1]}: {probs[idx][1].item():.2%}")