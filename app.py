import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- STREAMLIT PAGE ----------------

st.set_page_config(page_title="Smart Traffic AI", layout="wide")

st.title("🚦 AI Smart Traffic Management System")

# ---------------- LOAD MODEL (CACHED) ----------------

@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # your trained ambulance model
    return model

model = load_model()

# ---------------- VEHICLE CLASSES ----------------

VEHICLE_CLASSES = ["car","bus","truck","motorcycle","bicycle"]
AMBULANCE_CLASS = "ambulance"

# ---------------- IMAGE PROCESS FUNCTION ----------------

def process_image(image):

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(frame)

    vehicle_count = 0
    ambulance_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf < 0.40:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            # Ambulance detection
            if label == AMBULANCE_CLASS:
                ambulance_detected = True

            # Vehicle counting
            if label in VEHICLE_CLASSES:
                vehicle_count += 1

            # Draw bounding box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.putText(frame,f"{label} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(255,0,255),2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame, vehicle_count, ambulance_detected


# ---------------- LAYOUT ----------------

col1,col2 = st.columns(2)

# ---------------- LANE 1 ----------------

with col1:

    st.subheader("Lane 1 Camera")

    img1 = Image.open("lane1.jpg")

    frame1,vehicles1,ambulance1 = process_image(img1)

    st.image(frame1,use_container_width=True)

    st.write(f"Vehicles Detected: {vehicles1}")

# ---------------- LANE 2 ----------------

with col2:

    st.subheader("Lane 2 Camera")

    img2 = Image.open("lane2.jpg")

    frame2,vehicles2,ambulance2 = process_image(img2)

    st.image(frame2,use_container_width=True)

    st.write(f"Vehicles Detected: {vehicles2}")


# ---------------- AMBULANCE PRIORITY ----------------

if ambulance1 or ambulance2:

    st.error("🚑 Ambulance Detected! Priority Signal Activated")

    if ambulance1:
        st.success("Lane 1 → GREEN Signal")

    if ambulance2:
        st.success("Lane 2 → GREEN Signal")

else:

    if vehicles1 > vehicles2:
        st.success("Lane 1 → GREEN Signal")

    elif vehicles2 > vehicles1:
        st.success("Lane 2 → GREEN Signal")

    else:
        st.warning("Equal Traffic → Default Timer Running")