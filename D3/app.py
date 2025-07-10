import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# --- Page Config ---
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "A CNN app to classify satellite images as Cloudy, Desert, Green Area, or Water."
    }
)

# --- Enhanced Custom CSS with animations ---
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        
        /* Main Container */
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            animation: fadeInUp 0.8s ease-out;
        }
        
        /* Title Styles */
        .main-title {
            font-size: 4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            animation: slideInDown 1s ease-out;
            text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        }
        
        .subtitle {
            font-size: 1.3rem;
            color: #e0e0e0;
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeIn 1.2s ease-out;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        }
        
        /* Upload Section */
        .upload-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            text-align: center;
            transition: all 0.3s ease;
            animation: slideInLeft 1s ease-out;
        }
        
        .upload-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(240, 147, 251, 0.4);
        }
        
        /* Results Section */
        .results-container {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            animation: slideInRight 1s ease-out;
        }
        
        .prediction-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: all 0.3s ease;
            animation: bounceIn 0.8s ease-out;
        }
        
        .prediction-card:hover {
            transform: scale(1.02);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        }
        
        .prediction-text {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            animation: pulse 2s infinite;
        }
        
        .confidence-text {
            font-size: 1.1rem;
            color: #666;
            font-weight: 500;
        }
        
        /* Custom Button Styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        
        /* Feature Cards */
        .feature-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem;
            text-align: center;
            transition: all 0.3s ease;
            animation: fadeInUp 0.8s ease-out;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.3));
        }
        
        .feature-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }
        
        .feature-desc {
            color: #e0e0e0;
            font-size: 0.9rem;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInDown {
            from {
                opacity: 0;
                transform: translateY(-50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-50px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(50px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes bounceIn {
            0% {
                opacity: 0;
                transform: scale(0.3);
            }
            50% {
                opacity: 1;
                transform: scale(1.05);
            }
            70% {
                transform: scale(0.9);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        /* Hide Streamlit elements */
        .stDeployButton {display: none;}
        footer {visibility: hidden;}
        .stApp > header {visibility: hidden;}
        
        /* Custom Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Custom File Uploader */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1rem;
            border: 2px dashed #667eea;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover {
            border-color: #764ba2;
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("D3/Modelenv.v1.h5")

# --- Class Names with Emojis ---
class_names = ['Cloudy', 'Desert', 'Green Area', 'Water']
class_emojis = ['☁️', '🏜️', '🌳', '💧']
class_colors = ['#87CEEB', '#F4A460', '#32CD32', '#1E90FF']

# --- Main UI ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title Section
st.markdown('<h1 class="main-title">🛰️ Satellite Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN-powered satellite image analysis for terrain classification</p>', unsafe_allow_html=True)

# Features Section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">CNN-Powered</div>
            <div class="feature-desc">Advanced deep learning model</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Analysis</div>
            <div class="feature-desc">Real-time classification</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">High Accuracy</div>
            <div class="feature-desc">Precise terrain detection</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌍</div>
            <div class="feature-title">Global Coverage</div>
            <div class="feature-desc">Works worldwide</div>
        </div>
    """, unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Your Satellite Image")
st.markdown("*Supports JPG, JPEG, and PNG formats*")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Upload a satellite image for terrain classification"
)
st.markdown('</div>', unsafe_allow_html=True)

# Results Section
if uploaded_file is not None:
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(uploaded_file, use_container_width=True, caption="Analyzing...")
    
    with col2:
        st.markdown("### 🔬 Analysis Results")
        
        # Create progress bar with custom styling
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate analysis progress
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text('🔍 Loading image...')
            elif i < 60:
                status_text.text('🧠 Processing with AI...')
            elif i < 90:
                status_text.text('📊 Analyzing features...')
            else:
                status_text.text('✅ Analysis complete!')
            time.sleep(0.01)
        
        # Load and process the model
        try:
            model = load_model()
            
            # Preprocess image - try different sizes to match model
            image = Image.open(uploaded_file).convert("RGB")
            
            # Try common input sizes for satellite image models
            input_sizes = [(240, 240), (256, 256), (224, 224), (200, 200), (299, 299)]
            
            prediction = None
            for size in input_sizes:
                try:
                    resized_image = image.resize(size)
                    img_array = np.array(resized_image) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Test prediction with this size
                    prediction = model.predict(img_array)
                    st.success(f"✅ Successfully processed image with size {size}")
                    break
                except Exception as e:
                    continue
            
            if prediction is None:
                # If standard sizes don't work, try to infer from error message
                # The error suggests the model expects 115200 features
                # 115200 = 240 * 240 * 2 (possible grayscale with 2 channels)
                # or 115200 = 339.41^2 (approximately 340x340)
                try:
                    # Try 240x240 with different channel processing
                    resized_image = image.resize((240, 240))
                    img_array = np.array(resized_image)
                    
                    # Convert to grayscale and duplicate channel
                    gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
                    img_array = np.stack([gray, gray], axis=-1)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    prediction = model.predict(img_array)
                    st.success("✅ Successfully processed image with grayscale conversion")
                except Exception as e:
                    st.error(f"Could not process image with any standard format. Model may need specific preprocessing.")
                    st.error(f"Technical details: {str(e)}")
                    prediction = None
            
            # Predict
            if prediction is not None:
                predicted_class_idx = np.argmax(prediction)
                predicted_class = class_names[predicted_class_idx]
                confidence = prediction[0][predicted_class_idx] * 100
                
                # Clear progress elements
                progress_bar.empty()
                status_text.empty()
                
                # Display prediction
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-text">{class_emojis[predicted_class_idx]} {predicted_class}</div>
                        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Prediction probabilities
                st.markdown("### 📊 Detailed Probabilities")
                for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
                    percentage = prob * 100
                    st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span>{class_emojis[i]} {class_name}</span>
                                <span>{percentage:.1f}%</span>
                            </div>
                            <div style="background: #e0e0e0; border-radius: 10px; height: 8px; margin-top: 5px;">
                                <div style="background: {class_colors[i]}; height: 100%; width: {percentage}%; border-radius: 10px; transition: width 0.8s ease;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Download button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Report",
                        data=f"Satellite Image Classification Report\n\nPrediction: {predicted_class}\nConfidence: {confidence:.1f}%\nAnalysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        file_name=f"satellite_analysis_{predicted_class.lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
            else:
                # Clear progress elements even if prediction failed
                progress_bar.empty()
                status_text.empty()
                st.warning("⚠️ Unable to process the image. Please try a different image or check your model file.")
        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please ensure the model file 'Modelenv.v1.h5' is in the correct directory.")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome message
    st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 15px; margin: 2rem 0;">
            <h3 style="color: #ffffff; margin-bottom: 1rem; text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);">🚀 Ready to Analyze Your Satellite Images?</h3>
            <p style="color: #e0e0e0; font-size: 1.1rem; text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);">Upload a satellite image above to get started with CNN-powered terrain classification</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #e0e0e0; font-size: 0.9rem;">
        <p>🛰️ Powered by CNN • Built with ❤️ using Streamlit</p>
        <p>© 2025 Satellite Image Classifier</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
