import cv2
import os

# Base datasets directory
base_dir = "datasets"
os.makedirs(base_dir, exist_ok=True)

# Load Haar cascades once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                   'haarcascade_eye.xml')

# Ask user for input source
source_option = input("Select input source ('camera' or 'video'): ").strip().lower()

if source_option == "camera":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
elif source_option == "video":
    video_path = input("Enter path to video file: ").strip()
    if not os.path.isfile(video_path):
        print("Invalid video path. Exiting.")
        exit()
    cap = cv2.VideoCapture(video_path)
else:
    print("Invalid option. Please select 'camera' or 'video'. Exiting.")
    exit()

# Input ID and Name (for saving dataset)
user_id = input("ID: ").strip()
user_name = input("Name: ").strip()

# Create person-specific directory under datasets
person_dir = os.path.join(base_dir, f"{user_name}_{user_id}")
os.makedirs(person_dir, exist_ok=True)

sample_num = 0
max_samples = 60

print(f"Capturing up to {max_samples} samples. Press 'q' to quit early.")

while sample_num < max_samples:
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read or cannot access input.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes within face ROI
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1,
                                            minNeighbors=5)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

            # Resize to fixed size for smoother appearance
            eye_resized = cv2.resize(eye_roi, (100, 100),
                                     interpolation=cv2.INTER_CUBIC)

            # Display the eye ROI
            window_name = "Left Eye" if ex < w/2 else "Right Eye"
            cv2.imshow(window_name, eye_resized)

            # Build filename and save in person-specific folder
            side = "left" if ex < w/2 else "right"
            filename = (f"{person_dir}/{side}_eye_{user_name}_"
                        f"{user_id}_{sample_num:03d}.jpg")
            cv2.imwrite(filename, eye_resized)

            sample_num += 1
            if sample_num >= max_samples:
                break
        if sample_num >= max_samples:
            break

    # Show main frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection finished.")
