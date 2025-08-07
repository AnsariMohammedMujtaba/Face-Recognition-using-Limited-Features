import cv2
import os
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# CLAHE for contrast enhancement per-channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_eye_color(eye_bgr):
    # Contrast enhancement in YCrCb color space
    ycrcb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = clahe.apply(y)
    merged = cv2.merge([y, cr, cb])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    # Fast color denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return denoised

def extract_eye_regions(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None, []

    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm])

    left_idxs  = [33,133,160,159,158,157,173,144,145,153,154,155]
    right_idxs = [362,263,387,386,385,384,398,373,374,380,381,382]

    def get_crop_and_box(idxs):
        c = pts[idxs]
        x1,y1 = c.min(axis=0)
        x2,y2 = c.max(axis=0)
        pad_x = int((x2 - x1) * 0.5)
        pad_y = int((y2 - y1) * 0.8)
        x1p, y1p = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2p, y2p = min(w, x2 + pad_x), min(h, y2 + pad_y)
        crop = frame[y1p:y2p, x1p:x2p].copy()
        return crop, (x1p, y1p, x2p, y2p)

    left_crop, left_box = get_crop_and_box(left_idxs)
    right_crop, right_box = get_crop_and_box(right_idxs)
    return left_crop, right_crop, [left_box, right_box]

def collect_eye_dataset():
    # Ensure base datasets directory exists
    base_dir = "datasets"
    os.makedirs(base_dir, exist_ok=True)

    source_option = input("Select input source ('camera' or 'video'): ").strip().lower()
    if source_option == "camera":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    elif source_option == "video":
        video_path = input("Enter path to video file: ").strip()
        if not os.path.isfile(video_path):
            print("Invalid video path. Exiting.")
            return
        cap = cv2.VideoCapture(video_path)
    else:
        print("Invalid option. Please select 'camera' or 'video'. Exiting.")
        return

    id_num = input("Enter ID: ").strip()
    name = input("Enter Name: ").strip()
    person_dir = os.path.join(base_dir, f"{name}_{id_num}")
    os.makedirs(person_dir, exist_ok=True)

    samples = 0
    target = 50
    interval = 1.0  # seconds between captures
    last_capture = time.time() - interval

    print(f"Collecting up to {target} samples. Automatic capture every {interval} second(s).")
    print("Press ESC to quit early.")

    cv2.namedWindow("Eye Capture", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Capture", cv2.WND_PROP_TOPMOST, 1)

    while samples < target:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or cannot read input. Exiting.")
            break

        left_eye, right_eye, boxes = extract_eye_regions(frame)

        # Draw green rectangles on a copy exclusively for display
        display_frame = frame.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Eye Capture", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key pressed
            print("Capture interrupted by user.")
            break

        now = time.time()
        if (now - last_capture >= interval and left_eye is not None and right_eye is not None):
            for eye_type, eye_bgr in (("left", left_eye), ("right", right_eye)):
                enhanced = enhance_eye_color(eye_bgr)
                final = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                filename = f"{person_dir}/{name}_{id_num}_{eye_type}_{samples:02d}.jpg"
                cv2.imwrite(filename, final)
            samples += 1
            last_capture = now
            print(f"Saved sample {samples}/{target}")

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")

if __name__ == "__main__":
    collect_eye_dataset()
