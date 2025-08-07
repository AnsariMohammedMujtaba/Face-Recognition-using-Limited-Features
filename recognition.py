import cv2
import os

# Path to datasets directory containing subfolders named "<Name>_<ID>"
DATASETS_DIR = "datasets"

# Build dynamic mapping from numeric label to person name
person_dirs = sorted(os.listdir(DATASETS_DIR))
names = ["Unknown"]  # label 0 reserved for unknown
for label, folder in enumerate(person_dirs, start=1):
    names.append(folder.split("_")[0])  # extract Name portion

# Load the pre-trained eye recognizer
eye_recognizer = cv2.face.LBPHFaceRecognizer_create()
eye_recognizer.read('eye_recognizer.yml')

# Initialize Haar cascade for eye detection
cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(cascade_path)

# Ask user for input source
source_option = input("Select input source ('camera' or 'video'): ").strip().lower()

if source_option == "camera":
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
elif source_option == "video":
    video_path = input("Enter path to video file: ").strip()
    if not os.path.isfile(video_path):
        print("Invalid video path. Exiting.")
        exit()
    cam = cv2.VideoCapture(video_path)
else:
    print("Invalid option. Please select 'camera' or 'video'. Exiting.")
    exit()

# Setup window and keep it on top
window_name = 'Eye Recognition'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

# Fixed display size
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

font = cv2.FONT_HERSHEY_SIMPLEX

print("Starting real-time eye-based recognition. Press ESC to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        # Loop video from start if video input
        if source_option == "video":
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cam.read()
            if not ret:
                break
        else:
            break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2,
                                        minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        eye_roi = gray[y:y + h, x:x + w]
        eye_resized = cv2.resize(eye_roi, (100, 100))

        label, conf = eye_recognizer.predict(eye_resized)

        if conf < 100:
            name = names[label] if label < len(names) else "Unknown"
            confidence_pct = 100 - int(conf)
            display_text = f"{name} ({confidence_pct}%)"
            color = (0, 255, 0)  # Green for recognized
        else:
            display_text = "Unknown"
            name = "Unknown"
            confidence_pct = None
            color = (0, 0, 255)  # Red for unknown

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text_x = x
        text_y = max(y - 10, 15)
        cv2.putText(frame, display_text, (text_x, text_y),
                    font, 0.7, color, 2, cv2.LINE_AA)

        # Print recognition result to terminal
        if confidence_pct is not None:
            print(f"Recognized: {name} with confidence {confidence_pct}%")
        else:
            print("Recognized: Unknown")

    resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)

    cv2.imshow(window_name, resized_frame)

    key = cv2.waitKey(33) & 0xFF  # ~30 FPS
    if key == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
