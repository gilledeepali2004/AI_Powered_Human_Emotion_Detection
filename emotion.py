import cv2
import numpy as np
import json
import tensorflow as tf
from collections import deque

# ===============================
# 1️⃣ Load Trained Model
# ===============================
model = tf.keras.models.load_model("fer2.h5", compile=False)

# ===============================
# 2️⃣ Load Emotion Labels
# ===============================
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Reverse mapping: index → emotion
emotion_dict = {v: k for k, v in class_labels.items()}

# ===============================
# 3️⃣ Face Detection Model
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# 4️⃣ Webcam Start
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("🎥 Webcam Started - Press 'Q' to Exit")

# For smoothing predictions
emotion_window = deque(maxlen=10)

# ===============================
# 5️⃣ Real-Time Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize to 48x48
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        # Reshape for CNN
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict
        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Smooth predictions
        emotion_window.append(emotion_index)
        smooth_emotion_index = max(set(emotion_window),
                                   key=emotion_window.count)

        emotion_text = emotion_dict[smooth_emotion_index]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

        # Display Emotion + Confidence
        cv2.putText(frame,
                    f"{emotion_text} ({confidence:.1f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# 6️⃣ Release Everything
# ===============================
cap.release()
cv2.destroyAllWindows()

