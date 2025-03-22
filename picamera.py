from picamera2 import Picamera2
from flask import Flask, Response, render_template
import cv2
import numpy as np
import serial
import threading
import time

# UART Setup
ser = serial.Serial('/dev/serial0', 9600, timeout=1)

# Flask Setup
app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Known object parameters (calibrate these!)
KNOWN_WIDTH = 5.0  # Real width of the object in cm (e.g., a 5 cm red square)
FOCAL_LENGTH = 500  # Focal length in pixels (calibrate with known distance)

# Global variables for status
status = {
    "object_detected": "No Object",
    "distance": "N/A",
    "message_status": "Not Sent"
}

# Distance Estimation Function
def estimate_distance(apparent_width):
    if apparent_width > 0:
        distance = (FOCAL_LENGTH * KNOWN_WIDTH) / apparent_width
        return round(distance, 2)
    return float('inf')

# Object Detection and Processing
def detect_object_and_distance(frame):
    global status
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define red color range (adjust these values based on your object)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the object)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw rectangle and distance on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        distance = estimate_distance(w)
        cv2.putText(frame, f"Distance: {distance} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update status
        status["object_detected"] = "Object Detected"
        status["distance"] = f"{distance} cm"
        # Send "stop" only if distance is between 15 and 25 cm
        if 10 <= distance <= 15:
            message = f"{distance},Object Detected,stop\n"
            status["message_status"] = "Stop Sent"
        else:
            message = f"{distance},Object Detected\n"
            status["message_status"] = "Not Sent"
    else:
        status["object_detected"] = "No Object"
        status["distance"] = "N/A"
        message = "N/A,No Object\n"
        status["message_status"] = "Not Sent"
    
    # Send message to ESP32
    ser.write(message.encode())
    
    return frame

# Video Streaming Generator
def gen_frames():
    while True:
        frame = picam2.capture_array()
        frame = detect_object_and_distance(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', status=status)

# Background task to update status
def update_status():
    while True:
        frame = picam2.capture_array()
        detect_object_and_distance(frame)
        time.sleep(0.1)

if __name__ == '__main__':
    # Start status update in a separate thread
    threading.Thread(target=update_status, daemon=True).start()
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)
