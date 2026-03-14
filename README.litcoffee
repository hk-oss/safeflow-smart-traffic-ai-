AI Smart Traffic Signal System

This project implements an AI-based smart traffic signal system using YOLO object detection.

Features
- Vehicle detection using YOLO
- Ambulance priority signal control
- Smart traffic decision logic
- Web dashboard built using Streamlit
- Real traffic signal replica interface

Demo Note
This demo version uses static traffic images. In the final implementation,
live traffic footage will be captured through external cameras connected to the system.

Technologies Used
- Python
- YOLO (Ultralytics)
- Streamlit
- OpenCV

How to Run

1. Install dependencies
pip install -r requirements.txt

2. Run the application
streamlit run app.py

Project Structure
app.py          → Streamlit dashboard
best.pt         → Trained YOLO model
lane1.jpg       → Sample traffic image
lane2.jpg       → Sample traffic image
requirements.txt → Python dependencies

Author
Team WHAT IF