import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import os
from gtts import gTTS
import RPi.GPIO as GPIO

# =========================
# BLUETOOTH AUDIO
# =========================

def speak(text):

    def _play():

        try:

            print("VOICE:", text)

            tts = gTTS(
                text=text,
                lang="en"
            )

            tts.save("alert.mp3")

            os.system(
                "mpg123 -q alert.mp3"
            )

        except Exception as e:

            print("Audio Error:", e)

    threading.Thread(
        target=_play,
        daemon=True
    ).start()

# =========================
# ULTRASONIC SENSOR
# =========================

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

    distance = duration * 17150

    return round(distance, 1)

# =========================
# YOLO ONNX
# =========================

session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# =========================
# COCO CLASSES
# =========================

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

# =========================
# PRIORITY OBJECTS
# =========================

priority_objects = [
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
]

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

if not cap.isOpened():

    print("Camera Not Found")
    exit()

speak("Alert Eye Started")

last_spoken = ""
last_voice_time = 0

# =========================
# MAIN LOOP
# =========================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        start_time = time.time()

        original = frame.copy()

        h, w = frame.shape[:2]

        # =====================
        # PREPROCESS
        # =====================

        img = cv2.resize(
            frame,
            (640, 640)
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = img.astype(
            np.float32
        ) / 255.0

        img = np.transpose(
            img,
            (2,0,1)
        )

        img = np.expand_dims(
            img,
            axis=0
        )

        # =====================
        # INFERENCE
        # =====================

        outputs = session.run(
            None,
            {input_name: img}
        )

        predictions = outputs[0][0].T

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:

            score_array = pred[4:]

            class_id = np.argmax(
                score_array
            )

            confidence = score_array[
                class_id
            ]

            if confidence < 0.50:
                continue

            x_center,y_center,bw,bh = pred[:4]

            x1 = int(
                (x_center - bw/2)
                * w / 640
            )

            y1 = int(
                (y_center - bh/2)
                * h / 640
            )

            width = int(
                bw * w / 640
            )

            height = int(
                bh * h / 640
            )

            boxes.append(
                [x1,y1,width,height]
            )

            scores.append(
                float(confidence)
            )

            class_ids.append(
                class_id
            )

        # =====================
        # NMS
        # =====================

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            0.5,
            0.45
        )

        best_object = None

        if len(indexes) > 0:

            for i in indexes.flatten():

                x,y,w_box,h_box = boxes[i]

                label = classes[
                    class_ids[i]
                ]

                conf = scores[i]

                cv2.rectangle(
                    original,
                    (x,y),
                    (x+w_box,y+h_box),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    original,
                    f"{label} {conf:.2f}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

                if best_object is None:

                    best_object = (
                        label,
                        x,
                        w_box
                    )

                elif (
                    label in priority_objects
                ):

                    best_object = (
                        label,
                        x,
                        w_box
                    )

        # =====================
        # AUDIO ALERT
        # =====================

        if best_object:

            label,x,w_box = best_object

            center_x = x + w_box//2

            if center_x < original.shape[1]*0.33:

                direction = "left"

            elif center_x > original.shape[1]*0.66:

                direction = "right"

            else:

                direction = "center"

            distance = get_distance()

            current_time = time.time()

            speech = (
                f"{label} detected on "
                f"{direction} side "
                f"distance "
                f"{int(distance)} "
                f"centimeters"
            )

            if (
                speech != last_spoken
                and current_time - last_voice_time > 3
            ):

                speak(speech)

                last_spoken = speech

                last_voice_time = current_time

        # =====================
        # FPS
        # =====================

        fps = 1 / (
            time.time() -
            start_time
        )

        cv2.putText(
            original,
            f"FPS: {fps:.1f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        cv2.imshow(
            "ALERT EYE ONNX",
            original
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord('q')
        ):
            break

finally:

    cap.release()

    cv2.destroyAllWindows()

    GPIO.cleanup()

    speak("Alert Eye Stopped")
