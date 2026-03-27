# JABU DIgiTech 2,0 - Age & Sex Detection

A modern Streamlit application for real-time age and gender detection using AI.

![JABU DIgiTech 2,0](https://img.shields.io/badge/JABU%20DIgiTech-2,0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)

## Features

-  **Camera Feed Detection** - Use your webcam for real-time face analysis
-  **Image Upload** - Upload photos for age and gender prediction
-  **AI-Powered** - Powered by deep learning with face detection preprocessing
-  **Modern UI** - Clean, professional blue and white theme
-  **Responsive Design** - Works on desktop and mobile

## Installation

1. **Install dependencies:**
```bash
cd streamlit_app
pip install -r requirements.txt
```

2. **Run the application:**
```bash
streamlit run app.py
```

3. **Open in browser:**
The app will automatically open at `http://localhost:8501`

## Usage

### Camera Mode
1. Click the "📷 Camera" tab
2. Grant camera permissions if prompted
3. Click "Take a photo" to capture
4. Click "🔍 Detect Age & Gender" to analyze

### Upload Mode
1. Click the "📁 Upload Image" tab
2. Select an image file (JPG, PNG, or WebP)
3. Click "🔍 Detect Age & Gender" to analyze

## Project Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── utils.py                # Model loading and prediction utilities
├── requirements.txt        # Python dependencies
├── models/
│   └── Age_Sex_Detection.h5  # Trained Keras model
└── README.md               # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- Streamlit 1.28+
- OpenCV
- MTCNN (for face detection)
- Pillow

## Model Information

The application uses a pre-trained Keras model (`Age_Sex_Detection.h5`) that outputs:
- **Age**: Estimated age (1-100 years)
- **Sex**: Gender classification (Male/Female)

The model includes MTCNN face detection preprocessing for improved accuracy.

## Troubleshooting

### Camera not working?
- Ensure your browser has camera permissions
- Try using a different browser (Chrome recommended)
- Check if another application is using the camera

### Model prediction errors?
- Make sure your face is clearly visible in the image
- Ensure good lighting conditions
- Try uploading a different image

### Installation issues?
- For TensorFlow, you may need to install Microsoft Visual C++ Redistributable
- Consider using `tensorflow-cpu` if GPU support is not needed

## License

This project is developed by JABU DIgiTech 2,0

## Acknowledgments

- TensorFlow/Keras for the deep learning framework
- Streamlit for the web interface
- MTCNN for face detection
