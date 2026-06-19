
# Alert Eye - Jupyter Ready
# YOLOv8n ONNX + Ultrasonic + Bluetooth Audio

import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import os
from gtts import gTTS
import RPi.GPIO as GPIO
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# -----------------------------
# Bluetooth Audio
# -----------------------------
def speak(text):
    print("VOICE:", text)

    def _play():
        try:
            tts = gTTS(text=text, lang="en")
            tts.save("alert.mp3")
            os.system("mpg123 -q alert.mp3")
        except Exception as e:
            print("Audio Error:", e)

    threading.Thread(target=_play, daemon=True).start()

# -----------------------------
# Ultrasonic Sensor
# -----------------------------
TRIG = 4
ECHO = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    timeout = pulse_start + 0.1

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return -1

    pulse_end = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return -1

    duration = pulse_end - pulse_start
    return round(duration * 17150, 1)

# -----------------------------
# Load ONNX Model
# -----------------------------
session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

classes = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

priority_objects = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]

cap = cv2.VideoCapture(0)

last_spoken = ""
last_voice_time = 0
frame_count = 0

speak("Alert Eye Started")

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        original = frame.copy()

        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        start = time.time()
        outputs = session.run(None, {input_name: img})
        fps = 1 / max((time.time() - start), 1e-6)

        predictions = outputs[0][0].T

        h, w = original.shape[:2]

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            cls_scores = pred[4:]
            class_id = np.argmax(cls_scores)
            confidence = float(cls_scores[class_id])

            if confidence < 0.5:
                continue

            x_center, y_center, bw, bh = pred[:4]

            x1 = int((x_center - bw/2) * w / 640)
            y1 = int((y_center - bh/2) * h / 640)
            x2 = int((x_center + bw/2) * w / 640)
            y2 = int((y_center + bh/2) * h / 640)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)

        best_label = None
        best_direction = None
        best_conf = 0

        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, bw, bh = boxes[idx]
                label = classes[class_ids[idx]]
                conf = scores[idx]

                center_x = x + bw // 2

                if center_x < w * 0.33:
                    direction = "left"
                elif center_x > w * 0.66:
                    direction = "right"
                else:
                    direction = "center"

                cv2.rectangle(original, (x, y), (x+bw, y+bh), (0,255,0), 2)
                cv2.putText(original, f"{label} {conf:.2f}",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

                score_bonus = 1 if label in priority_objects else 0
                if conf + score_bonus > best_conf:
                    best_conf = conf + score_bonus
                    best_label = label
                    best_direction = direction

        distance = get_distance()

        current_time = time.time()

        if best_label and current_time - last_voice_time > 3:

            if distance > 0:
                speech = (
                    f"{best_label} detected on "
                    f"{best_direction} side "
                    f"distance {int(distance)} centimeters"
                )
            else:
                speech = (
                    f"{best_label} detected on "
                    f"{best_direction} side"
                )

            if speech != last_spoken:
                speak(speech)
                last_spoken = speech
                last_voice_time = current_time

        cv2.putText(original, f"FPS: {fps:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

        cv2.putText(original, f"Distance: {distance} cm",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,0), 2)

        if frame_count % 5 == 0:
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            clear_output(wait=True)
            plt.figure(figsize=(8,5))
            plt.imshow(rgb)
            plt.axis("off")
            display(plt.gcf())
            plt.close()

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    GPIO.cleanup()
    speak("Alert Eye Stopped")
