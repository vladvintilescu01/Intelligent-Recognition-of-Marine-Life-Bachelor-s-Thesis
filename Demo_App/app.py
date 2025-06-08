import streamlit as st
from PIL import Image
from utils import load_model_by_name, predict_image
import random

st.set_page_config(page_title="Intelligent Fish Recognition", layout="centered")

# Model metadata
MODEL_META = {
        "Normal Eye": {
        "file": "FishImgDataset_augmented_balancedV2_InceptionV3_full",
        "short": "Best for clean, well-lit images. Very balanced, standard use-case.",
        "details": """
<span style="color:#23c1ff"><b>Normal Eye</b></span> is the model that is balanced, recognizes fish in most cases but is not very attentive to details, 
and if the image is distorted or the fish is not very well seen, then the models have a high chance of making mistakes. This model is pre-trained using InceptionV3. 
"""
    },
        "Crystal Vision": {
            "file": "FishImgDataset_augmented_balancedV2_DenseNet121_full",
            "short": "Excels on high-quality, sharp images. Maintains very high detail sensitivity.",
            "details": """
<span style="color:#23c1ff"><b>Crystal Vision</b></span> is the model that pays attention to details, especially when the focus is very good on fish. It offers 
good results even with lower resolutions, thanks to pretraining on DenseNet121.
    """
        },
        "Rugged Eye": {
            "file": "FishImgDataset_augmented_balancedV2_ResNet50_full",
            "short": "Very robust to distortions. Works well even on noisy, low-quality images.",
            "details": """
<span style="color:#23c1ff"><b>Rugged Eye</b></span> is robust on distorted images, including if the focus is not the best. Based on ResNet50 pre-training, it is perfect for real life where 
images may not always come out perfect and may still have various imperfections.
    """
        }
}

LOADING_MESSAGES = [
    "Thinking hard... I'll tell you the fish type in just a moment!",
    "Analyzing scales and fins... Please hold on, your answer is almost ready!",
    "Just a little more patience, I decoding the mysteries of the sea...",
    "Almost done! You're about to discover your fish.",
    "AI magic in progress—making sure you get the right answer!"
]

# --- CSS for buttons and details ---
st.markdown("""
    <style>
        .desc-box {
            text-align: justify;
            background: #20394d;
            color: #fff;
            width: 100%;
            border-radius: 15px;
            padding: 20px 32px;
            font-size: 1.12em;
            font-weight: 400;
            margin-top: 0.1em;
            margin-bottom: 1.3em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color:#1f77b4; font-size:2.5em; white-space:nowrap; margin-bottom:1.3em;'>Intelligent Recognition of Marine Life</h1>",
    unsafe_allow_html=True
)

st.markdown("""
#### Fish that can be recognized by the models:
</div>
<div style='text-align:center; font-size:1.07em; color:#ddefff; font-weight:600;'>
    🐟 Tilapia &nbsp; &nbsp; 🐠 Gourami &nbsp; &nbsp; 🐡 Catfish &nbsp; &nbsp; 🦈 Grass Carp &nbsp; &nbsp;
    🔪 Knifefish &nbsp; &nbsp; 🔍 Glass Perchlet &nbsp; &nbsp;&nbsp; ✨ Silver Barb &nbsp; &nbsp; 🦑 Goby
</div>
""", unsafe_allow_html=True)

st.markdown("#### Choose a model to get started: ")

# --- State for which model is selected ---
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Normal Eye"

# --- The buttons that represents the models ---
cols = st.columns([2, 2, 2.2, 2, 2])
model_names = list(MODEL_META.keys())
for idx, name in enumerate(model_names):
    with cols[idx+1]:
        btn = st.button(
            name,
            key=f"btn_{name}",
            help=MODEL_META[name]["short"],
            use_container_width=True
        )
        # Handle selection
        if btn:
            st.session_state.selected_model = name

# --- Show details about the selected model ---
model_selected = st.session_state.selected_model
st.markdown(f"<div class='desc-box'>{MODEL_META[model_selected]['details']}</div>", unsafe_allow_html=True)

# --- File uploader and predict ---
uploaded_file = st.file_uploader("Upload a fish image (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width="auto")
    if st.button("Predict"):
        with st.spinner(random.choice(LOADING_MESSAGES)):
            model = load_model_by_name(MODEL_META[model_selected]["file"])
            predicted_class, confidence = predict_image(model, image)
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                margin-top: 0.3em;
            ">
            <div style="
                background: #20394d;
                border-radius: 16px;
                box-shadow: 0 4px 30px 0 rgba(70,180,255,0.13);
                padding: 38px 44px;
                display: flex;
                flex-direction: row;
                gap: 40px;
                min-width: 620px;
            ">
                <div style="
                    background: #223045;
                    border-radius: 10px;
                    border: 2px solid #1f77b4;
                    padding: 14px 28px;
                    font-size: 1.15em;
                    font-weight: 500;
                    color: #fff;
                    min-width: 275px;
                    text-align: center;
                ">
                🧠 I <span style="color: #23c1ff">think</span> it's a <b><span style="color: #23c1ff">{predicted_class}</span></b>
                </div>
                <div style="
                    background: #223045;
                    border-radius: 10px;
                    border: 2px solid #1f77b4;
                    padding: 14px 28px;
                    font-size: 1.15em;
                    color: #23c1ff;
                    min-width: 275px;
                    text-align: center;
                ">
                📈 I am sure <b>{confidence:.2%}</b>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True
        )
else:
    st.info("Please upload an image to start prediction.")
