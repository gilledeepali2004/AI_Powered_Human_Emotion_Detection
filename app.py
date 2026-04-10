import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from collections import deque
from datetime import datetime
import json
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="MoodMirror",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset and Font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Base App Background: Deep dark gradient */
.stApp {
    background: linear-gradient(-45deg, #020617, #0f172a, #111827, #0b1020);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #f8fafc;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Hide main menu hamburger and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeader"] {display: none;}

/* =============== TOP NAVBAR =============== */
/* We disguise the st.radio as a floating top navigation bar */
div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 30px;
    background: #0b1020;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding: 15px 20px;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    z-index: 999999;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}

/* Hide Sidebar Toggle */
[data-testid="collapsedControl"] {display: none !important;}
section[data-testid="stSidebar"] {display: none !important;}

/* Make entire label clickable and style it */
div[data-testid="stRadio"] label {
    cursor: pointer !important;
    padding: 5px 10px !important;
    margin: 0 !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Hide the radio circles entirely across Streamlit versions */
div[data-testid="stRadio"] label > div:first-child,
div[data-testid="stRadio"] label span[data-baseweb="radio"],
div[data-testid="stRadio"] label div[role="radio"],
div[data-testid="stRadio"] label input[type="radio"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Remove side margins that Streamlit leaves next to the radio circle */
div[data-testid="stRadio"] label > div:nth-child(2),
div[data-testid="stRadio"] label > div:last-child {
    margin-left: 0 !important;
    padding: 0 !important;
}


/* Style the text of the radio buttons (Nav Links) */
div[data-testid="stRadio"] span[data-testid="stMarkdownContainer"] p {
    font-size: 16px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex;
    align-items: center;
    transition: all 0.3s ease !important;
}

/* Inject Icons */
div[data-testid="stRadio"] label:nth-child(1) span[data-testid="stMarkdownContainer"] p::before { content: "🏠 "; margin-right: 6px; font-size: 16px; }
div[data-testid="stRadio"] label:nth-child(2) span[data-testid="stMarkdownContainer"] p::before { content: "📷 "; margin-right: 6px; font-size: 16px; }
div[data-testid="stRadio"] label:nth-child(3) span[data-testid="stMarkdownContainer"] p::before { content: "📊 "; margin-right: 6px; font-size: 16px; }
div[data-testid="stRadio"] label:nth-child(4) span[data-testid="stMarkdownContainer"] p::before { content: "ℹ️ "; margin-right: 6px; font-size: 16px; }

/* Hover State */
div[data-testid="stRadio"] label:hover {
    background: transparent !important;
    transform: translateY(-2px);
}
div[data-testid="stRadio"] label:hover span[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    
    text-shadow:
        0 0 5px rgba(168, 85, 247, 0.9),
        0 0 10px rgba(168, 85, 247, 0.9),
        0 0 20px rgba(168, 85, 247, 0.9),
        0 0 40px rgba(139, 92, 246, 0.8),
        0 0 60px rgba(99, 102, 241, 0.7);

    transition: all 0.3s ease;
}
            @keyframes glowPulse {
    0% {
        text-shadow:
            0 0 5px rgba(168, 85, 247, 0.6),
            0 0 10px rgba(168, 85, 247, 0.6);
    }
    50% {
        text-shadow:
            0 0 20px rgba(168, 85, 247, 1),
            0 0 40px rgba(139, 92, 246, 1),
            0 0 60px rgba(99, 102, 241, 1);
    }
    100% {
        text-shadow:
            0 0 5px rgba(168, 85, 247, 0.6),
            0 0 10px rgba(168, 85, 247, 0.6);
    }
}

div[data-testid="stRadio"] label:hover span[data-testid="stMarkdownContainer"] p {
    animation: glowPulse 1.5s infinite;
}
}

/* Active State indicator */
div[data-testid="stRadio"] label[data-checked="true"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stRadio"] label[data-checked="true"] span[data-testid="stMarkdownContainer"] p {
    color: #fff !important;
    font-weight: 600 !important;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.8) !important;
}

/* Shift main container down */
.block-container {
    padding-top: 100px !important;
}

/* Responsive collapse to icon-only */
@media (max-width: 768px) {
    div[data-testid="stRadio"] > div {
        padding: 8px 15px;
        right: 15px;
    }
    div[data-testid="stRadio"] label {
        padding: 10px 10px !important;
    }
    div[data-testid="stRadio"] span[data-testid="stMarkdownContainer"] p {
        font-size: 0px !important; /* hide text */
    }
    div[data-testid="stRadio"] span[data-testid="stMarkdownContainer"] p::before {
        font-size: 20px !important; /* enlarge icon */
        margin-right: 0px;
        display: block;
    }
}

/* Hide the main widget label entirely to prevent 'Navigation' from showing up */
div[data-testid="stRadio"] > label {
    display: none !important;
}


/* =============== BRAND TITLE & HERO =============== */
.brand-title {
    font-size: 200px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
    letter-spacing: -2px;
    margin-top: 0px;
             transform: translateY(-50px);
    line-height:1;
            font-size: clamp(100px, 14vw, 220px);
             font-family: 'Montserrat', sans-serif !important;
            letter-spacing: 4px;  
            
    text-shadow: 
        0 0 30px rgba(56, 189, 248, 0.8),
        0 0 60px rgba(99, 102, 241, 1),
        0 0 90px rgba(139, 92, 246, 0.8);
            
}
            
}
            @keyframes titleGlow {
    0% {
        text-shadow:
            0 0 20px rgba(56, 189, 248, 0.6),
            0 0 40px rgba(99, 102, 241, 0.6);
    }
    50% {
        text-shadow:
            0 0 40px rgba(56, 189, 248, 1),
            0 0 80px rgba(139, 92, 246, 1),
            0 0 120px rgba(99, 102, 241, 1);
    }
    100% {
        text-shadow:
            0 0 20px rgba(56, 189, 248, 0.6),
            0 0 40px rgba(99, 102, 241, 0.6);
    }
}

.brand-title {
    animation: titleGlow 3s infinite ease-in-out;
}

.subtitle-text {
    font-size: 20px;
    font-weight: 400;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 40px;
             
}

.hero-subtext {
    font-size: 18px;
    color: #cbd5e1;
    font-weight: 300;
    max-width: 600px;
    margin: 0;
    text-align: left;
    line-height: 1.6;
}

/* =============== CARDS & GLASSMORPHISM =============== */
.card {
    background: rgba(20, 25, 40, 0.4);
    border-radius: 16px;
    padding: 30px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 24px;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.card h3 {
    margin-top: 0;
    font-size: 22px;
    font-weight: 600;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding-bottom: 12px;
    margin-bottom: 16px;
}

.card p, .card li {
    font-size: 15px;
    color: #94a3b8;
    line-height: 1.6;
}

/* =============== BUTTONS =============== */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    padding: 16px 32px;
    font-size: 18px;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
}
.stButton > button:active {
    transform: translateY(0);
}

/* =============== METRICS OVERRIDE =============== */
.metric-box {
    background: rgba(20, 25, 40, 0.6);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease;
}
.metric-box:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.3);
}
.metric-box h3 {
    font-size: 36px;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-box p {
    font-size: 14px;
    color: #94a3b8;
    margin: 0;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* =============== EMOTION BADGE (Dynamic) =============== */
.emotion-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 16px;
    color: white;
    text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.badge-happy { background: linear-gradient(135deg, #10b981, #059669); }
.badge-sad { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.badge-angry { background: linear-gradient(135deg, #ef4444, #dc2626); }
.badge-surprise { background: linear-gradient(135deg, #f59e0b, #d97706); }
.badge-neutral { background: linear-gradient(135deg, #64748b, #475569); }
.badge-fear { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
.badge-disgust { background: linear-gradient(135deg, #84cc16, #65a30d); }

/* Progress bar container */
.confidence-bar-container {
    width: 100%;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    height: 12px;
    margin-top: 10px;
    overflow: hidden;
}

/* Progress bar fill */
.confidence-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    border-radius: 10px;
    transition: width 0.5s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(
            128, (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.01)
        ),
        tf.keras.layers.Conv2D(
            256, (3, 3),
            padding="valid",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.01)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation="softmax")
    ])
    import os
import gdown

FILE_ID = "1_vagA5VYZll2PNxwvCFUl9z-2NsS9U91"

if not os.path.exists("fer2.h5"):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, "fer2.h5", quiet=False)

    model.build((None, 48, 48, 1))
    model.load_weights("fer2.h5")
    
    # ⚡ WARM UP THE ENGINE
    import numpy as np
    model(np.zeros((1, 48, 48, 1), dtype=np.float32), training=False)
    
    return model

@st.cache_data
def load_labels():
    with open("class_labels.json", "r") as f:
        return json.load(f)

# =========================
# Face Detector
# =========================
@st.cache_resource
def load_face_detector():
    import cv2
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# =========================
# Session State
# =========================
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []
# =========================
# Deferred Loading Strategy Applied 🚀
# =========================

# =========================
# Top-Right Navbar Replacement
# =========================
# Handle page selection natively via session_state binding
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

st.markdown("<div style='margin-bottom: -15px;'></div>", unsafe_allow_html=True)
page = st.radio(
    "",
    options=["Home", "Live Detection", "Dashboard", "About"],
    key="current_page",
    horizontal=True,
    label_visibility="collapsed"
)


# =========================
# Home Page
# =========================
if page == "Home":
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Spacer
    
    def go_to_page(page_name):
        st.session_state.current_page = page_name

    # CENTERED BRAND TITLE AND SUBHEADING
    st.markdown("""
    <div style='text-align: center; padding-top: 10px; padding-bottom: 30px;'>
        <h1 class='brand-title'>MoodMirror</h1>
        <p class='hero-subtext' style='font-size: 24px; color: #f8fafc; font-weight: 500; margin: 0 auto 15px auto; text-align: center; max-width: 800px;'>
            Real Time AI-Powered Human Emotion Detection System 
        </p>
    </div>
    """, unsafe_allow_html=True)

    # HERO SECTION (Two Columns)
    col_text, col_img = st.columns([1.2, 1])
    
    with col_text:
        st.markdown("""
        <div style='padding-top: 0px; padding-bottom: 20px; padding-right: 20px;'>
            <p class='hero-subtext'>
                Instantly decode human expressions with high-precision deep learning.
                Experience the next generation of visual emotional analysis through our immersive dark-mode interface.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True) # Spacer

        # CTA Buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.button("Start Detection 🚀", on_click=go_to_page, args=("Live Detection",), use_container_width=True)
        with btn_col2:
            st.button("View Dashboard 📊", on_click=go_to_page, args=("Dashboard",), use_container_width=True)

    with col_img:
        try:
            st.markdown("<div style='border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-top: 20px;'>", unsafe_allow_html=True)
            st.image("home_banner.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.info("💡 Tip: Save your image as `home_banner.png` in the application folder to display it here!")
            
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True) # Spacer
    
    # CARDS SECTION
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='card' style='height: 100%;'>
            <h3>🎯 High Precision</h3>
            <p>
                MoodMirror leverages an advanced Convolutional Neural Network (CNN) 
                trained on vast datasets to distinguish subtle micro-expressions across 7 primary emotions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card' style='height: 100%;'>
            <h3>⚡ Real-Time Processing</h3>
            <p>
                Experience instantaneous visual feedback whether you're uploading static images 
                or utilizing a live webcam feed for fluid expression tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card' style='height: 100%;'>
            <h3>📊 Advanced Analytics</h3>
            <p>
                Dive deep into historical emotion data. Track confidence distributions and 
                dominant mood swings in our aesthetically pleasing Dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Live Detection Page
# =========================
elif page == "Live Detection":
    # Defer loading to drastically speed up Home and Dashboard navigation times
    with st.spinner("Initializing Local Engine..."):
        model = load_model()
        class_labels = load_labels()
        face_cascade = load_face_detector()
    
    emotion_dict = {v: k.capitalize() for k, v in class_labels.items()}

    st.markdown(
        "<h2 style='text-align: center; color: #f8fafc; font-weight: 800; font-size: 38px; margin-top: 10px; margin-bottom: 5px; "
        "text-shadow: 0 0 15px rgba(56, 189, 248, 0.8), 0 0 30px rgba(99, 102, 241, 0.8);'>"
        "Let MoodMirror Discover How You Feel"
        "</h2>", 
        unsafe_allow_html=True
    )

    # Centered layout using columns
    _, center_col, _ = st.columns([0.2, 3, 0.2])

    with center_col:
        # st.markdown(?
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px; color: #38bdf8;'>Emotion Intelligence Studio</h3>", unsafe_allow_html=True)
        
        if "input_type" not in st.session_state:
            st.session_state.input_type = None
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Upload Image", use_container_width=True):
                st.session_state.input_type = "Upload Image"
                st.rerun()
        with col_btn2:
            if st.button("Use Live Webcam", use_container_width=True):
                st.session_state.input_type = "Use Live Webcam"
                st.rerun()
                
                
        option = st.session_state.input_type
        
        if option is not None:
            st.markdown("<hr style='border: 0; height: 1px; background-image: linear-gradient(to right, transparent, rgba(56, 189, 248, 0.4), transparent); margin-top: 15px; margin-bottom: 25px;'>", unsafe_allow_html=True)

        if option == "Upload Image":
            uploaded_file = st.file_uploader("Upload a high-quality human face image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
                    )

                    if len(faces) == 0:
                        st.warning("⚠️ No face detected in the image. Please try again with a clear face.")
                    else:
                        emotion_window = deque(maxlen=10)
                        main_emotion_text = "Neutral"
                        main_confidence = 0.0

                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]
                            face = cv2.resize(face, (48, 48))
                            face = face.astype("float32") / 255.0
                            face = np.reshape(face, (1, 48, 48, 1))

                            prediction = model(face, training=False).numpy()
                            emotion_index = int(np.argmax(prediction))
                            confidence = float(np.max(prediction) * 100)

                            emotion_window.append(emotion_index)
                            smooth_emotion_index = max(set(emotion_window), key=emotion_window.count)
                            emotion_text = emotion_dict[smooth_emotion_index]

                            st.session_state.emotion_history.append(emotion_text)
                            st.session_state.confidence_history.append(confidence)
                            st.session_state.timestamps.append(datetime.now().strftime("%H:%M:%S"))

                            # Draw bounding box on image
                            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
                            
                            if main_confidence == 0.0:
                                main_emotion_text = emotion_text
                                main_confidence = confidence

                        st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
                        
                        # Determine Badge Color Class
                        badge_class = f"badge-{main_emotion_text.lower()}"
                        
                        # Result UI HTML
                        st.markdown(f"""
                        <div style='text-align: center; margin-bottom: 24px;'>
                            <h3 style='color: #cbd5e1; font-size: 16px; font-weight: 500; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;'>Primary Emotion</h3>
                            <div class='emotion-badge {badge_class}' style='font-size: 20px; padding: 10px 24px;'>{main_emotion_text}</div>
                        </div>
                        
                        <div style='margin-bottom: 20px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px;'>
                            <div style='display: flex; justify-content: space-between; color: #94a3b8; font-size: 14px; margin-bottom: 8px;'>
                                <span style='font-weight: 500;'>AI Confidence Level</span>
                                <span style='color: #f8fafc; font-weight: 600;'>{main_confidence:.1f}%</span>
                            </div>
                            <div class='confidence-bar-container'>
                                <div class='confidence-bar-fill' style='width: {main_confidence}%;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # st.image(
                        #     cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        #     caption="Neural Network Analysis",
                        #     use_container_width=True
                        # )
                        
                        preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])

                        with preview_col2:
                             st.image(
                                   cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                              caption="Neural Network Analysis",
                               width=350
                                       )
                        st.markdown("</div>", unsafe_allow_html=True)

        elif option == "Use Live Webcam":
            st.toast("Warming up WebCamera... Please allow a few seconds to connect.", icon="⏳")
            st.info("💡 Grant browser camera permissions to activate real-time detection.")
            
            # Keep references to the session state lists so the thread can append to them
            em_hist = st.session_state.emotion_history
            conf_hist = st.session_state.confidence_history
            time_hist = st.session_state.timestamps
            
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()

            class EmotionProcessor(VideoTransformerBase):
                def __init__(self):
                    self.frame_count = 0
                    self.emotion_window = deque(maxlen=5) # Kept for primary face
                    self.last_predictions = [] # Support multiple faces
                    self.ctx = ctx
                    self.last_toast_time = 0
                    self.last_toast_emotion = ""
                    self.em_hist = em_hist
                    self.conf_hist = conf_hist
                    self.time_hist = time_hist

                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    self.frame_count += 1

                    # Frame Skipping: Optmized processing every 3rd frame
                    if self.frame_count % 3 == 0:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Downscale for much faster face detection
                        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                        faces = face_cascade.detectMultiScale(
                            small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                        )

                        current_preds = []
                        if len(faces) > 0:
                            # Find the largest face by area for tracking the primary user
                            areas = [w*h for (x,y,w,h) in faces]
                            largest_idx = np.argmax(areas)
                            
                            face_imgs = []
                            face_boxes = []
                            # Process all faces
                            for idx, (x, y, w, h) in enumerate(faces):
                                x, y, w, h = x*2, y*2, w*2, h*2 # Scale matching back to original size
                                
                                # Extract and preprocess face
                                face_img = gray[y:y+h, x:x+w]
                                if face_img.size == 0:
                                    continue
                                face_img = cv2.resize(face_img, (48, 48))
                                face_img = face_img.astype("float32") / 255.0
                                face_img = np.reshape(face_img, (48, 48, 1))
                                
                                face_imgs.append(face_img)
                                face_boxes.append((x, y, w, h))

                            if face_imgs:
                                face_batch = np.array(face_imgs)
                                predictions = model(face_batch, training=False).numpy()
                                
                                for i, pred in enumerate(predictions):
                                    emotion_index = int(np.argmax(pred))
                                    confidence = float(np.max(pred) * 100)
                                    emotion_text = emotion_dict[emotion_index]
                                    
                                    if i == largest_idx:
                                        self.emotion_window.append(emotion_index)
                                        smooth_emotion_index = max(set(self.emotion_window), key=self.emotion_window.count)
                                        emotion_text = emotion_dict[smooth_emotion_index]
                                        
                                        # Active Popup System asynchronously
                                        import time
                                        current_time = time.time()
                                        if current_time - getattr(self, 'last_toast_time', 0) > 4.0:
                                            if emotion_text != getattr(self, 'last_toast_emotion', "") and emotion_text != "Detecting...":
                                                try:
                                                    import threading
                                                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                                                    add_script_run_ctx(threading.current_thread(), self.ctx)
                                                    emojis = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Surprise": "😲", "Neutral": "😐", "Fear": "😨", "Disgust": "🤢"}
                                                    st.toast(f"Dominant Emotion Shift: **{emotion_text}** {emojis.get(emotion_text, '')}", icon="🌟")
                                                    self.last_toast_time = current_time
                                                    self.last_toast_emotion = emotion_text
                                                except Exception:
                                                    pass
                                        
                                        # Add to analytics history roughly every second (assuming 30 FPS)
                                        if self.frame_count % 30 == 0:
                                            self.em_hist.append(emotion_text)
                                            self.conf_hist.append(confidence)
                                            self.time_hist.append(datetime.now().strftime("%H:%M:%S"))
                                            
                                    current_preds.append((emotion_text, confidence, face_boxes[i]))
                        else:
                            current_preds = [("Detecting...", 0.0, None)]
                            
                        self.last_predictions = current_preds

                    # Overlays
                    for pred in self.last_predictions:
                        emotion_text, confidence, face_coords = pred

                        if face_coords is not None:
                            (x, y, w, h) = face_coords
                            
                            # Make the bounding box slightly broader around the face
                            pad_x = int(w * 0.15)
                            pad_y = int(h * 0.15)
                            x = max(0, x - pad_x)
                            y = max(0, y - pad_y)
                            w = w + (pad_x * 2)
                            h = h + (pad_y * 2)

                            # Futuristic Bounding Box Styling
                            colors = {
                                "Happy": (81, 185, 16), "Sad": (235, 99, 37),
                                "Angry": (38, 38, 220), "Surprise": (11, 158, 245),
                                "Neutral": (139, 116, 100), "Fear": (237, 58, 124),
                                "Disgust": (22, 204, 132), "Detecting...": (150, 150, 150)
                            }
                            color = colors.get(emotion_text, (255, 255, 255))
                            thickness = 4
                            length = 30

                            # Thicker Full Box
                            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)

                            # Glowing Corners
                            cv2.line(img, (x, y), (x + length, y), color, thickness)
                            cv2.line(img, (x, y), (x, y + length), color, thickness)
                            cv2.line(img, (x+w, y), (x+w - length, y), color, thickness)
                            cv2.line(img, (x+w, y), (x+w, y + length), color, thickness)
                            cv2.line(img, (x, y+h), (x + length, y+h), color, thickness)
                            cv2.line(img, (x, y+h), (x, y+h - length), color, thickness)
                            cv2.line(img, (x+w, y+h), (x+w - length, y+h), color, thickness)
                            cv2.line(img, (x+w, y+h), (x+w, y+h - length), color, thickness)

                            # Dynamic Overlay Plate
                            label = f"{emotion_text} | {confidence:.1f}%"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                            
                            # Background label plate
                            cv2.rectangle(img, (x, y - th - 12), (x + tw + 10, y), color, -1)
                            # Text
                            cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    return img

            # WebRTC Component
            _, cam_col, _ = st.columns([1, 4, 1])
            with cam_col:
                webrtc_streamer(
                    key="moodmirror-live",
                    mode=WebRtcMode.SENDRECV,
                    video_transformer_factory=EmotionProcessor,
                    rtc_configuration={"iceServers": []}, # Bypass external STUN to load instantly constraint-free
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 1280},
                            "height": {"ideal": 720},
                            "frameRate": {"ideal": 15, "max": 20}
                        }, 
                        "audio": False
                    },
                    async_processing=True,
                    desired_playing_state=True,
                    video_html_attrs={
                        "autoPlay": True, 
                        "controls": False, 
                        "style": {"width": "100%", "border-radius": "16px"}, 
                        "muted": True
                    },
                )
            
            st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
            _, stop_col, _ = st.columns([1, 2, 1])
            with stop_col:
                if st.button("Stop Webcam", use_container_width=True):
                    st.session_state.input_type = None
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Unconditionally close the master card wrapper
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Dashboard Page
# =========================
elif page == "Dashboard":
    st.markdown("<div class='subtitle-text'><b>MoodMirror</b> | Emotion Analytics Dashboard</div>", unsafe_allow_html=True)

    if len(st.session_state.emotion_history) == 0:
        st.info("No emotion data available yet. Please detect emotions first.")
    else:
        df = pd.DataFrame({
            "Time": st.session_state.timestamps,
            "Emotion": st.session_state.emotion_history,
            "Confidence": st.session_state.confidence_history
        })

        total_scans = len(df)
        top_emotion = df["Emotion"].mode()[0]
        avg_conf = round(df["Confidence"].mean(), 2)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='metric-box'><h3>{total_scans}</h3><p>Total Detections</p></div>",
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"<div class='metric-box'><h3>{top_emotion}</h3><p>Most Frequent Emotion</p></div>",
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"<div class='metric-box'><h3>{avg_conf}%</h3><p>Average Confidence</p></div>",
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        emotion_counts = df["Emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]

        # Premium Bar Chart
        bar_chart = alt.Chart(emotion_counts).mark_bar(
            cornerRadiusTopLeft=8,
            cornerRadiusTopRight=8,
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#38bdf8', offset=0),
                       alt.GradientStop(color='#8b5cf6', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X("Emotion:N", sort="-y", axis=alt.Axis(labelAngle=0, labelColor='#94a3b8', titleColor='#94a3b8')),
            y=alt.Y("Count:Q", axis=alt.Axis(labelColor='#94a3b8', titleColor='#94a3b8')),
            tooltip=["Emotion", "Count"]
        ).properties(height=350).configure_view(strokeWidth=0).configure_axis(gridColor='rgba(255,255,255,0.05)', domain=False)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Emotion Frequency Distribution")
        st.altair_chart(bar_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Premium Line Chart
        line_chart = alt.Chart(df).mark_line(
            point=alt.OverlayMarkDef(color="#38bdf8", size=60),
            color="#8b5cf6",
            strokeWidth=3,
            tension=0.4 # smooth interpolation
        ).encode(
            x=alt.X("Time:N", axis=alt.Axis(labelColor='#94a3b8', titleColor='#94a3b8')),
            y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor='#94a3b8', titleColor='#94a3b8')),
            color=alt.Color("Emotion:N", legend=alt.Legend(labelColor='#94a3b8', titleColor='#94a3b8')),
            tooltip=["Time", "Emotion", "Confidence"]
        ).properties(height=350).configure_view(strokeWidth=0).configure_axis(gridColor='rgba(255,255,255,0.05)', domain=False)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Confidence Trend Over Time")
        st.altair_chart(line_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Detailed Detection Logs")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Clear Dashboard Data"):
            st.session_state.emotion_history = []
            st.session_state.confidence_history = []
            st.session_state.timestamps = []
            st.success("Dashboard data cleared successfully.")
            st.rerun()

# =========================
# About Page
# =========================
elif page == "About":
    st.markdown(
        "<h2 style='text-align: center; color: #f8fafc; font-weight: 800; font-size: 38px; margin-top: 10px; margin-bottom: 30px; "
        "text-shadow: 0 0 15px rgba(56, 189, 248, 0.8), 0 0 30px rgba(99, 102, 241, 0.8);'>"
        "About MoodMirror"
        "</h2>", 
        unsafe_allow_html=True
    )

    # 1. Project Overview & Objective
    st.markdown("""
    <div class='card' style='margin-bottom: 30px; text-align: center; padding: 40px;'>
        <h3 style='font-size: 28px; margin-bottom: 15px; background: linear-gradient(90deg, #38bdf8, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none;'>Our Mission</h3>
        <p style='font-size: 18px; color: #cbd5e1; max-width: 800px; margin: 0 auto; line-height: 1.8;'>
            MoodMirror bridges the gap between human emotion and artificial intelligence. 
            Our objective is to deliver a frictionless, real-time emotional intelligence engine 
            capable of analyzing micro-expressions with state-of-the-art precision. We envision 
            a future where technology adapts empathetically to human emotional states.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Technologies Used (Cards)
    st.markdown("<h3 style='color: #f8fafc; font-weight: 700; margin-top: 20px; margin-bottom: 20px;'>Core Technologies</h3>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    tek_style = "text-align: center; padding: 25px 15px; transition: all 0.3s ease; height: 100%; border-radius: 16px; background: rgba(20, 25, 40, 0.5); border: 1px solid rgba(255,255,255,0.05);"
    with t1:
        st.markdown(f"<div class='card' style='{tek_style}'><h1 style='font-size: 40px; margin:0;'>🧠</h1><h4 style='color: #e2e8f0; margin-top: 15px;'>Deep Learning</h4><p style='font-size: 13px; color: #94a3b8;'>TensorFlow & Keras CNN Architecture</p></div>", unsafe_allow_html=True)
    with t2:
        st.markdown(f"<div class='card' style='{tek_style}'><h1 style='font-size: 40px; margin:0;'>👁️</h1><h4 style='color: #e2e8f0; margin-top: 15px;'>Computer Vision</h4><p style='font-size: 13px; color: #94a3b8;'>OpenCV Haar Cascades & Image Processing</p></div>", unsafe_allow_html=True)
    with t3:
        st.markdown(f"<div class='card' style='{tek_style}'><h1 style='font-size: 40px; margin:0;'>⚡</h1><h4 style='color: #e2e8f0; margin-top: 15px;'>Real-Time Streaming</h4><p style='font-size: 13px; color: #94a3b8;'>WebRTC asynchronous video pipelining</p></div>", unsafe_allow_html=True)
    with t4:
        st.markdown(f"<div class='card' style='{tek_style}'><h1 style='font-size: 40px; margin:0;'>📊</h1><h4 style='color: #e2e8f0; margin-top: 15px;'>Data Analytics</h4><p style='font-size: 13px; color: #94a3b8;'>Pandas & Altair interactive visualizations</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # 3. Team Members Section
    st.markdown("<h3 style='color: #f8fafc; font-weight: 700; margin-bottom: 20px;'>The Engineers Behind MoodMirror</h3>", unsafe_allow_html=True)
    
    # 5 members
    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    team_style = "text-align: center; padding: 20px 10px; background: rgba(20, 25, 40, 0.4); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); transition: transform 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"
    
    with tm1:
        st.markdown(f"<div class='card' style='{team_style}'> \
            <div style='width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #38bdf8, #6366f1); margin: 0 auto 15px auto; display: flex; align-items: center; justify-content: center; font-size: 30px;'>🧑‍💻</div> \
            <h4 style='color: #f8fafc; font-size: 14px; margin: 0; padding-bottom: 5px; border:none;'>Prachi Urgunde</h4> \
            <p style='color: #38bdf8; font-size: 12px; font-weight: 600; margin: 0;'>Backend Architect</p> \
            </div>", unsafe_allow_html=True)
    with tm2:
        st.markdown(f"<div class='card' style='{team_style}'> \
            <div style='width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #8b5cf6, #d946ef); margin: 0 auto 15px auto; display: flex; align-items: center; justify-content: center; font-size: 30px;'>👩‍💻</div> \
            <h4 style='color: #f8fafc; font-size: 14px; margin: 0; padding-bottom: 5px; border:none;'>Deepali Gille</h4> \
            <p style='color: #a855f7; font-size: 12px; font-weight: 600; margin: 0;'>Frontend and UI</p> \
            </div>", unsafe_allow_html=True)
    with tm3:
        st.markdown(f"<div class='card' style='{team_style}'> \
            <div style='width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #10b981, #059669); margin: 0 auto 15px auto; display: flex; align-items: center; justify-content: center; font-size: 30px;'>🧔</div> \
            <h4 style='color: #f8fafc; font-size: 14px; margin: 0; padding-bottom: 5px; border:none;'>Gauri Hushangabadkar</h4> \
            <p style='color: #10b981; font-size: 12px; font-weight: 600; margin: 0;'>Backend Architect</p> \
            </div>", unsafe_allow_html=True)
    with tm4:
        st.markdown(f"<div class='card' style='{team_style}'> \
            <div style='width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #f59e0b, #d97706); margin: 0 auto 15px auto; display: flex; align-items: center; justify-content: center; font-size: 30px;'>🧑‍🎨</div> \
            <h4 style='color: #f8fafc; font-size: 14px; margin: 0; padding-bottom: 5px; border:none;'>Neha Bokad</h4> \
            <p style='color: #f59e0b; font-size: 12px; font-weight: 600; margin: 0;'>Frontend and UI</p> \
            </div>", unsafe_allow_html=True)
    with tm5:
        st.markdown(f"<div class='card' style='{team_style}'> \
            <div style='width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #ef4444, #dc2626); margin: 0 auto 15px auto; display: flex; align-items: center; justify-content: center; font-size: 30px;'>👩‍🔬</div> \
            <h4 style='color: #f8fafc; font-size: 14px; margin: 0; padding-bottom: 5px; border:none;'>Mohini Shrikhande</h4> \
            <p style='color: #ef4444; font-size: 12px; font-weight: 600; margin: 0;'>Documentation and Testing</p> \
            </div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # 4. Future Scope
    st.markdown("""
    <div class='card' style='background: linear-gradient(145deg, rgba(20,25,40,0.8), rgba(15,23,42,0.9)); border-left: 4px solid #6366f1;'>
        <h3 style='font-size: 22px; color: #e2e8f0; margin-bottom: 15px; border:none;'>🚀 Future Scope & Roadmap</h3>
        <ul style='color: #94a3b8; font-size: 16px; line-height: 1.8; margin-left: 20px;'>
            <li><b>Multimodal Emotion Detection:</b> Integrating vocal tone and speech sentiment analysis for comprehensive profiling.</li>
            <li><b>API & Enterprise SDK:</b> Releasing developer endpoints to allow third-party apps to embed MoodMirror's intelligence.</li>
            <li><b>Continuous Learning Integration:</b> Expanding the 7-emotion constraint into micro-expression spectrums using federated learning.</li>
            <li><b>Mental Health Dashboards:</b> Partnering with tele-health services for therapeutic analytics tracking.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer' style='text-align: center; color: #475569; margin-top: 50px; padding: 20px; font-weight: 500; letter-spacing: 1px;'>© 2026 MoodMirror AI. All rights reserved.</div>", unsafe_allow_html=True)
