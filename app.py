import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pickle
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import queue
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

st.set_page_config(page_title="EyeDentify", layout="centered")

# Load the face detector model
@st.cache_resource
def load_face_detector():
    protoPath = os.path.sep.join(["detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    return detector

# Load the liveness detection model
@st.cache_resource
def load_liveness_model():
    model = load_model("liveness_model_v3.keras")
    le = pickle.loads(open("le_model_v3.pickle", "rb").read())
    return model, le

# Initialize models
try:
    face_detector = load_face_detector()
    liveness_model, label_encoder = load_liveness_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading models: {e}")

# Sidebar
st.sidebar.title("EyeDentify")
st.sidebar.markdown("*Face Spoof Detection Demo*")
st.sidebar.markdown("Built with OpenCV + Streamlit + TensorFlow")

# About Model section in sidebar
with st.sidebar.expander("About Model"):
    st.markdown("""
    #### Liveness Detection Model
    
    **Architecture:** Custom CNN (LivenessNet)
    
    **Input Size:** 32x32 pixels
    
    **Output Classes:** Real, Fake
    
    **Training:**
    - Trained on a dataset of real faces and presentation attacks
    - Uses data augmentation for robustness
    - Achieves high accuracy in distinguishing real faces from spoofed ones
    
    **Face Detection:**
    - Uses OpenCV's DNN module with a SSD face detector
    - Model: res10_300x300_ssd
    
    **Capabilities:**
    - Detects photo-based spoofing
    - Works with different camera angles
    - Real-time inference
    """)

# Title
st.title(":eyes: EyeDentify - Face Spoof Detection")
st.markdown("<hr style='border:1px solid gray'/>", unsafe_allow_html=True)

# Description
with st.expander("What is EyeDentify?"):
    st.write("""
    EyeDentify is a face spoof detection app designed to distinguish real faces from spoofed ones (like photos, videos, or masks). 
    It uses deep learning to analyze facial texture patterns that differ between real faces and printed/digital reproductions.
    
    The system performs two key tasks:
    1. **Face Detection**: Locates faces in the image
    2. **Liveness Detection**: Determines if each detected face is real or fake
    """)

# Process function for liveness detection
def process_frame(frame):
    if not model_loaded:
        return frame, []
        
    # Convert to OpenCV format
    image_cv = frame.copy()
    
    # Resize if too big
    if image_cv.shape[1] > 600:
        image_cv = cv2.resize(image_cv, (600, int(600 * image_cv.shape[0] / image_cv.shape[1])))
    
    (h, w) = image_cv.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image_cv, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network to get detections
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    # Initialize results
    results = []
    result_image = image_cv.copy()
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter weak detections
        if confidence > 0.5:
            # Calculate bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the detected face is within the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Extract the face ROI
            face = image_cv[startY:endY, startX:endX]
            
            # Check if the face is valid
            if face.size == 0:
                continue
                
            # Preprocess the face for the liveness model
            try:
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                
                # Make prediction
                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_name = label_encoder.classes_[j]
                probability = preds[j] * 100
                
                # Create result
                result = {
                    "label": label_name,
                    "confidence": probability,
                    "position": (startX, startY, endX, endY)
                }
                results.append(result)
                
                # Draw bounding box and label
                label_text = f"{label_name}: {probability:.2f}%"
                if label_name == "real":
                    color = (0, 255, 0)  # Green for real
                else:
                    color = (0, 0, 255)  # Red for fake
                    
                cv2.putText(result_image, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(result_image, (startX, startY), (endX, endY), color, 2)
            except Exception as e:
                # If there's an error processing this face, just skip it
                continue
    
    return result_image, results

# WebRTC Video Processor class for real-time webcam
class LivenessDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.results_queue = queue.Queue(maxsize=1)
        self.frame_skip = 0  # Process every frame
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Skip frames to reduce CPU load if needed
        if self.frame_skip > 0:
            self.frame_skip -= 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            self.frame_skip = 0  # Process every frame
        
        # Process the frame
        result_img, results = process_frame(img)
        
        # Update the results (keep only the most recent)
        try:
            # Clear the queue first to ensure we only have the latest result
            while not self.results_queue.empty():
                self.results_queue.get_nowait()
            self.results_queue.put(results, block=False)
        except queue.Full:
            pass
        
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")
    
    def get_latest_results(self):
        try:
            if not self.results_queue.empty():
                return self.results_queue.get_nowait()
            return []
        except:
            return []

# Input Method
input_type = st.radio("Choose input method:", ["Real-time Webcam", "Upload Image"])

# Real-time Webcam processing
if input_type == "Real-time Webcam":
    st.info("Click 'Start' to begin real-time liveness detection through your webcam.")
    
    # Use webrtc_streamer for real-time processing
    webrtc_ctx = webrtc_streamer(
        key="liveness-detection",
        video_processor_factory=LivenessDetectionProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Create a placeholder for live results display
    results_placeholder = st.empty()
    
    # Check if streamer is running and display results
    if webrtc_ctx.video_processor:
        # Display results in a loop as long as the streamer is active
        while webrtc_ctx.state.playing:
            results = webrtc_ctx.video_processor.get_latest_results()
            if results:
                with results_placeholder.container():
                    col1, col2 = st.columns(2)
                    for idx, result in enumerate(results):
                        col = col1 if idx % 2 == 0 else col2
                        with col:
                            if result['label'] == 'real':
                                st.success(f"REAL FACE detected with {result['confidence']:.2f}% confidence")
                            else:
                                st.error(f"FAKE FACE detected with {result['confidence']:.2f}% confidence")
            time.sleep(0.1)  # Short sleep to prevent high CPU usage

# Upload Image Flow
elif input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGBA to RGB if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    result_image, results = process_frame(image_np)
                    
                    if results:
                        st.image(result_image, caption="Analysis Result", use_column_width=True)
                        
                        for idx, result in enumerate(results):
                            with st.expander(f"Face #{idx+1} - {result['label'].upper()}"):
                                if result['label'] == 'real':
                                    st.success(f"REAL FACE detected with {result['confidence']:.2f}% confidence")
                                else:
                                    st.error(f"FAKE FACE detected with {result['confidence']:.2f}% confidence")
                    else:
                        st.warning("No faces detected in the image. Please try another image.")
    else:
        st.info("Please upload an image to proceed.")

# Tips for better results
with st.expander("Tips for Better Results"):
    st.markdown("""
    - Ensure good, even lighting on your face
    - Position your face clearly in view of the camera
    - For testing fake detection, try using a printed photo or display a face on another screen
    - If using webcam, allow a moment for the camera to adjust focus and lighting
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤ by Diyana and Nalin")