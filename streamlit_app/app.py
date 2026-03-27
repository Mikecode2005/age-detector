"""
Age & Sex Detection Application
===============================
A streamlined application for age and gender detection
using camera feed or image upload.
"""

import streamlit as st
from PIL import Image
from utils import predict_age_sex

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
    
    /* Upload text */
    .upload-text {
        color: #8b949e;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Results card */
    .results-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .result-label {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
        text-align: center;
    }
    
    .result-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        text-align: center;
    }
    
    .result-value-small {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 500;
        margin: 0;
        text-align: center;
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
    
    /* Info text */
    .info-text {
        color: #9ca3af;
        font-size: 0.9rem;
        text-align: center;
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
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>Age Detector</h1>
        <p>Detect age and gender from photos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Camera", "Upload"])
    
    current_image = None
    
    with tab1:
        st.markdown('<p class="upload-text">Capture your face using the camera</p>', 
                   unsafe_allow_html=True)
        
        camera_image = st.camera_input("Capture", label_visibility="collapsed")
        
        if camera_image:
            current_image = camera_image
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
            current_image = uploaded_file
            st.session_state.prediction_result = None
    
    # Main content area
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two column layout: Image | Results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if current_image:
            st.image(current_image, caption="Input Image", width='stretch')
        else:
            st.markdown('<p class="info-text">No image selected. Use the Camera or Upload tab to get started.</p>', 
                       unsafe_allow_html=True)
    
    with col2:
        if current_image:
            # Process the image
            if st.button("Analyze Image", key="analyze_btn", width='stretch'):
                with st.spinner("Analyzing..."):
                    try:
                        image = Image.open(current_image)
                        result = predict_age_sex(image)
                        st.session_state.prediction_result = result
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Show results if available
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                
                # Create a container for results
                st.markdown('<div class="results-card">', unsafe_allow_html=True)
                
                st.markdown('<p class="result-label">Estimated Age</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["age"]} years</p>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown('<p class="result-label">Gender</p>', unsafe_allow_html=True)
                
                if result['sex'] == "Male":
                    st.markdown(f'<p class="gender-tag">{result["sex"]}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="gender-tag female">{result["sex"]}</p>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown('<p class="result-label">Confidence</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value-small">{result["sex_confidence"]:.0%}</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="info-text">Click "Analyze Image" to detect age and gender.</p>', 
                           unsafe_allow_html=True)
        else:
            st.markdown('<p class="info-text">Results will appear here after image analysis.</p>', 
                       unsafe_allow_html=True)
    
    # Clear button (only show if there's an image)
    if current_image:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear & Start Over", key="clear_btn", width='stretch'):
            st.session_state.prediction_result = None
            st.rerun()


if __name__ == "__main__":
    main()
