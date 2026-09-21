import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Title and Instructions
st.title("🚦 Road Sign Detection System")
st.write("Upload an image of a road to detect traffic signs.")

# 2. Load your custom trained model
# Make sure 'best.pt' is in the same folder as this script!
try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"Error loading model: {e}. Did you download 'best.pt'?")

# 3. Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Detecting...")

    # 4. Run detection
    results = model(image)

    # 5. Display the Result Image (with boxes drawn)
    # Plot the results on the image
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='Processed Image', use_column_width=True)

    # 6. Tell the user "Which class is this"
    st.write("### Detection Results:")

    # Loop through detections to get class names
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.warning("No road signs detected.")
    else:
        for box in boxes:
            # Get the class ID (number)
            class_id = int(box.cls[0])
            # Get the class name (text) from the model's dictionary
            class_name = model.names[class_id]
            # Get confidence score
            confidence = float(box.conf[0])

            # Print it out for the user
            st.success(f"Found: **{class_name}** (Confidence: {confidence:.2f})")