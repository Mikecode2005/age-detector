"""
Age & Sex Detection Application
===============================
A streamlined application for age and gender detection
using camera feed or image upload.
"""

import streamlit as st
from PIL import Image
import numpy as np
from utils import load_model, predict_age_sex

# Page configuration
st.set_page_config(
    page_title="Age Detector",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Dark theme, minimal, professional
st.markdown("""
<style>
    /* Reset and base styles */
    .stApp {
        background: #0e1117;
    }
    
    /* Main header */
    .header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid #262730;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header p {
        color: #8b949e;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #161b22;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #ffffff !important;
    }
    
    /* Upload area */
    .upload-section {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .upload-text {
        color: #8b949e;
        font-size: 0.9rem;
    }
    
    /* Image display */
    .image-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 300px;
    }
    
    /* Results card */
    .results-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
    }
    
    .result-label {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
    
    .result-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .result-value-small {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 500;
        margin: 0;
    }
    
    .gender-tag {
        display: inline-block;
        background: #238636;
        color: #ffffff;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .gender-tag.female {
        background: #8b5cf6;
    }
    
    /* Loading spinner text */
    .loading-text {
        color: #8b949e;
        text-align: center;
        padding: 2rem;
    }
    
    /* Error message */
    .error-box {
        background: #2d1f1f;
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 1rem;
        color: #f85149;
    }
    
    /* Info text */
    .info-box {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 1rem;
        color: #9ca3af;
        font-size: 0.85rem;
    }
    
    /* Two column layout */
    .main-content {
        display: flex;
        gap: 1.5rem;
    }
    
    @media (max-width: 768px) {
        .main-content {
            flex-direction: column;
        }
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Streamlit overrides */
    .stCameraInput > div {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    .stFileUploader > div {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #2ea043;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'process_trigger' not in st.session_state:
        st.session_state.process_trigger = None
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>Age Detector</h1>
        <p>Detect age and gender from photos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Camera", "Upload"])
    
    image_source = None
    
    with tab1:
        st.markdown('<p class="upload-text">Capture your face using the camera</p>', 
                   unsafe_allow_html=True)
        
        camera_image = st.camera_input("Capture", label_visibility="collapsed")
        
        if camera_image:
            image_source = camera_image
            st.session_state.current_image = camera_image
            st.session_state.prediction_result = None
    
    with tab2:
        st.markdown('<p class="upload-text">Upload an image file</p>', 
                   unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help=None,
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image_source = uploaded_file
            st.session_state.current_image = uploaded_file
            st.session_state.prediction_result = None
    
    # Auto-process when image is captured/uploaded
    if st.session_state.current_image and not st.session_state.prediction_result:
        if image_source == st.session_state.current_image:
            with st.spinner(""):
                st.markdown('<p class="loading-text">Analyzing image...</p>', 
                           unsafe_allow_html=True)
                try:
                    # Load model
                    load_model()
                    
                    # Read and process image
                    image = Image.open(st.session_state.current_image)
                    
                    # Make prediction
                    result = predict_age_sex(image)
                    
                    st.session_state.prediction_result = result
                    st.rerun()
                    
                except Exception as e:
                    st.markdown(f'''
                    <div class="error-box">
                        Error: {str(e)}
                    </div>
                    ''', unsafe_allow_html=True)
    
    # Main content area
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two column layout: Image | Results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.session_state.current_image:
            st.image(st.session_state.current_image, 
                    caption="Input Image", 
                    use_container_width=True)
        else:
            st.markdown('''
            <div class="info-box">
                No image selected. Use the Camera or Upload tab to get started.
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            gender_class = "" if result['sex'] == "Male" else " female"
            
            st.markdown(f'''
            <div class="results-card">
                <p class="result-label">Estimated Age</p>
                <p class="result-value">{result['age']} years</p>
                
                <br>
                
                <p class="result-label">Gender</p>
                <span class="gender-tag{gender_class}">{result['sex']}</span>
                
                <br><br>
                
                <p class="result-label">Confidence</p>
                <p class="result-value-small">{result['sex_confidence']:.0%}</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            if st.session_state.current_image:
                st.markdown('''
                <div class="results-card">
                    <p class="loading-text">Processing...</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="info-box">
                    Results will appear here after image analysis.
                </div>
                ''', unsafe_allow_html=True)
    
    # Clear button (only show if there's an image)
    if st.session_state.current_image or st.session_state.prediction_result:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear & Start Over", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.current_image = None
            st.rerun()


if __name__ == "__main__":
    main()
