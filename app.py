import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
gen_model = genai.GenerativeModel("models/gemini-2.5-pro")

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def get_caption_and_nutrition(image: Image.Image):
    inputs = caption_processor(images=image, return_tensors="pt")
    output = caption_model.generate(**inputs)
    caption = caption_processor.decode(output[0], skip_special_tokens=True)

    prompt = f"""You are a certified nutritionist.
Estimate nutrition values for: {caption}.
Respond in a markdown table with columns: Nutrient | Amount | % Daily Value."""

    try:
        response = gen_model.generate_content(prompt)
        return caption, response.text
    except Exception as e:
        return caption, f"⚠️ Gemini Error: {e}"

st.set_page_config(page_title="NutriScan AI", page_icon="🥕")
st.title("🥗 NutriScan - AI-Powered Nutrition Estimator")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image and estimating nutrition..."):
        caption, nutrition = get_caption_and_nutrition(image)

    st.subheader("🖼️ Image Caption")
    st.markdown(f"> {caption}")

    st.subheader("🍎 Estimated Nutrition")
    st.markdown(nutrition)
