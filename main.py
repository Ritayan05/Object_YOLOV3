import cv2
import pyttsx3
import numpy as np
import threading  # Add this line

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak text
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Video capture
url = 'http://192.168.137.89:81/stream'
cap = cv2.VideoCapture(0)  # Use 0 for webcam
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
cap.set(10, 70)   # Brightness

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to calculate distance from camera
def calculate_distance(box_width, focal_length, known_width):
    return (known_width * focal_length) / box_width

# Real-world width of the object being detected (in centimeters)
known_width = 50  # Example: Width of a standard credit card

# Focal length of the camera (experimentally determined or obtained from camera specifications)
focal_length = 400  # Example value, needs to be adjusted based on your camera setup

while True:
    # Read frame
    success, img = cap.read()
    height, width, _ = img.shape  # Get the dimensions of the image

    # Detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialization for NMS
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Update for NMS
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)

    # Draw bounding boxes and labels on the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Calculate distance from camera
            distance_cm = calculate_distance(w, focal_length, known_width)

            # Speak out the detected object and its distance in a separate thread
            threading.Thread(target=speak_text, args=(f"Obstacle detected: {label}, at a distance of {distance_cm:.2f} centimeters",)).start()

            # Add class name, confidence score, and distance to the image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"Distance: {distance_cm:.2f} cm", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("Output", img)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
