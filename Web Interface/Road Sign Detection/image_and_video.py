import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 1. Load the Model once
# Check if 'best.pt' is in the same folder!
try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Page Layout
st.title("🚦 Road Sign Detection System")
st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option:", ["Upload Image", "Live Camera"])

# --- MODE 1: UPLOAD IMAGE ---
if mode == "Upload Image":
    st.header("Upload Image Mode")
    st.write("Upload an image to detect road signs.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Detecting...")

        # Run detection
        results = model(image)

        # Show results
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Processed Image', use_column_width=True)

        # Show Class Names
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.warning("No signs detected.")
        else:
            with st.expander("See detected signs details"):
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    st.write(f"**{name}** (Confidence: {conf:.2f})")

# --- MODE 2: LIVE CAMERA ---
elif mode == "Live Camera":
    st.header("Live Camera Mode")
    st.write("Allow camera access to start detecting in real-time.")


    # Define the callback function for video processing
    def video_frame_callback(frame):
        # Convert frame to format YOLO understands (numpy array)
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection on this single frame
        results = model(img)

        # Draw the boxes on the frame
        annotated_frame = results[0].plot()

        # Return the processed frame back to the video stream
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


    # Start the webcam streamer
    webrtc_streamer(
        key="road-sign-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},  # Video only, no audio
        async_processing=True
    )