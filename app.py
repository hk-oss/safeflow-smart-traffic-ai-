import streamlit as st
import cv2
from ultralytics import YOLO
import streamlit.components.v1 as components

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Smart Traffic Signal", layout="wide")

st.title("🚦 AI Smart Traffic Signal System")

st.warning(
"NOTE: This system currently uses sample images for demonstration purposes. "
"In the final implementation, live traffic footage will be captured through the system using external cameras."
)

# ---------------- LOAD MODEL ----------------
model = YOLO("best.pt")

VEHICLE_CLASSES = {"car","bus","truck","motorcycle","motorbike"}
AMBULANCE_CLASS = "ambulance"

# ---------------- TRAFFIC LIGHT UI ----------------
def traffic_light(signal):

    if signal == "RED":
        red = "red"
        yellow = "#222"
        green = "#222"

    elif signal == "YELLOW":
        red = "#222"
        yellow = "yellow"
        green = "#222"

    else:
        red = "#222"
        yellow = "#222"
        green = "lime"

    html = f"""
    <div style="display:flex;justify-content:center">
        <div style="
            background:black;
            width:80px;
            padding:15px;
            border-radius:15px;
            box-shadow:0px 0px 15px grey;
        ">

            <div style="
                width:40px;
                height:40px;
                border-radius:50%;
                background:{red};
                margin:10px auto;
            "></div>

            <div style="
                width:40px;
                height:40px;
                border-radius:50%;
                background:{yellow};
                margin:10px auto;
            "></div>

            <div style="
                width:40px;
                height:40px;
                border-radius:50%;
                background:{green};
                margin:10px auto;
            "></div>

        </div>
    </div>
    """

    return html


# ---------------- DETECTION FUNCTION ----------------
def detect(frame):

    results = model(frame)

    vehicle_count = 0
    ambulance_detected = False

    for r in results:

        for box in r.boxes:

            label = r.names[int(box.cls)]
            conf = float(box.conf)

            if label in VEHICLE_CLASSES:
                vehicle_count += 1

            if label == AMBULANCE_CLASS and conf > 0.4:
                ambulance_detected = True

        frame = r.plot()

    return vehicle_count, ambulance_detected, frame


# ---------------- LOAD IMAGES ----------------
lane1_path = "C:/Users/hkuma/Desktop/smart-traffic-ai/lane1.jpg"
lane2_path = "C:/Users/hkuma/Desktop/smart-traffic-ai/lane2.jpg"

frame1 = cv2.imread(lane1_path)
frame2 = cv2.imread(lane2_path)

if frame1 is None or frame2 is None:
    st.error("Image files not found. Check lane1.jpg and lane2.jpg")
    st.stop()

# ---------------- RUN DETECTION ----------------
vehicles1, ambulance1, frame1 = detect(frame1)
vehicles2, ambulance2, frame2 = detect(frame2)

# ---------------- TRAFFIC DECISION ----------------
if ambulance1:
    signal1 = "GREEN"
    signal2 = "RED"
    decision = "🚑 Ambulance Priority → Lane 1"

elif ambulance2:
    signal1 = "RED"
    signal2 = "GREEN"
    decision = "🚑 Ambulance Priority → Lane 2"

else:
    if vehicles1 >= vehicles2:
        signal1 = "GREEN"
        signal2 = "RED"
        decision = "🚦 Lane 1 Cleared"
    else:
        signal1 = "RED"
        signal2 = "GREEN"
        decision = "🚦 Lane 2 Cleared"

# ---------------- AMBULANCE ALERT ----------------
if ambulance1 or ambulance2:
    st.error("🚑 Ambulance Detected! Priority Signal Activated")


# ---------------- CAMERA DISPLAY ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Lane 1 Camera")
    st.image(frame1, channels="BGR")
    st.write(f"Vehicles Detected: {vehicles1}")

with col2:
    st.subheader("Lane 2 Camera")
    st.image(frame2, channels="BGR")
    st.write(f"Vehicles Detected: {vehicles2}")


# ---------------- TRAFFIC SIGNAL UI ----------------
st.markdown("## 🚦 Traffic Signal Status")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Lane 1 Signal")
    components.html(traffic_light(signal1), height=200)

with col4:
    st.subheader("Lane 2 Signal")
    components.html(traffic_light(signal2), height=200)


# ---------------- TRAFFIC DECISION ----------------
st.markdown("## 🚦 Traffic Decision")
st.info(decision)