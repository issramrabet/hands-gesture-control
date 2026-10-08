import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import pyautogui
import math
from pycaw.pycaw import AudioUtilities

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen_width, screen_height = pyautogui.size()

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

volume = AudioUtilities.GetSpeakers().EndpointVolume
vol_min, vol_max = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]

def draw_landmarks(img, lm):
    h, w, _ = img.shape
    for s, e in HAND_CONNECTIONS:
        a, b = lm[s], lm[e]
        cv2.line(img, (int(a.x*w), int(a.y*h)), (int(b.x*w), int(b.y*h)), (0,255,0), 2)
    for p in lm:
        cv2.circle(img, (int(p.x*w), int(p.y*h)), 5, (0,0,255), -1)

def dist(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def is_click(lm):
    return dist(lm[8], lm[12]) < 0.05

def is_hand_open(lm):
    thumb_open = lm[4].x < lm[3].x if lm[4].x < lm[0].x else lm[4].x > lm[3].x
    index_open = lm[8].y < lm[6].y
    middle_open = lm[12].y < lm[10].y
    ring_open = lm[16].y < lm[14].y
    pinky_open = lm[20].y < lm[18].y

    return thumb_open and index_open and middle_open and ring_open and pinky_open

def is_hand_closed(lm):
    thumb_closed = lm[4].x > lm[3].x if lm[4].x < lm[0].x else lm[4].x < lm[3].x
    index_closed = lm[8].y > lm[6].y
    middle_closed = lm[12].y > lm[10].y
    ring_closed = lm[16].y > lm[14].y
    pinky_closed = lm[20].y > lm[18].y

    return thumb_closed and index_closed and middle_closed and ring_closed and pinky_closed

def volume_value(lm):
    d = dist(lm[4], lm[8])
    d = max(0.02, min(0.18, d))
    return (d - 0.02) / (0.18 - 0.02)

sx, sy = 0, 0
smooth = 0.5
was_click = False
current_vol = 0.0
show_volume_bar = False

# --- FPS measurement ---
prev_time = time.time()
frame_count = 0
fps_sum = 0.0
fps = 0.0
min_fps = float("inf")
max_fps = 0.0
WARMUP_FRAMES = 30   # ignore the first frames (camera and model warm-up)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_img)

    if res.hand_landmarks:
        lm = res.hand_landmarks[0]
        draw_landmarks(frame, lm)

        if is_hand_open(lm):
            show_volume_bar = True
        elif is_hand_closed(lm):
            show_volume_bar = False

        idx = lm[8]
        x = int(idx.x * screen_width)
        y = int(idx.y * screen_height)

        sx = sx * (1 - smooth) + x * smooth
        sy = sy * (1 - smooth) + y * smooth
        pyautogui.moveTo(sx, sy)

        c = is_click(lm)
        if c and not was_click:
            pyautogui.click()
            cv2.putText(frame, "CLICK", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        was_click = c

        if show_volume_bar:
            current_vol = volume_value(lm)
            level = vol_min + current_vol * (vol_max - vol_min)
            volume.SetMasterVolumeLevel(level, None)
            cv2.putText(frame, "VOLUME", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    if show_volume_bar:
        bar_x, bar_y = 30, 150
        bar_h, bar_w = 200, 20
        filled = int(bar_h * current_vol)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255,255,255), 2)
        cv2.rectangle(frame, (bar_x, bar_y + bar_h - filled),
                      (bar_x + bar_w, bar_y + bar_h), (0,0,255), -1)

        cv2.putText(frame, f"{int(current_vol*100)}%", (20, bar_y + bar_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # --- FPS calculation (placed at the end of the loop, right before display) ---
    now = time.time()
    elapsed = now - prev_time
    prev_time = now
    if elapsed > 0:
        fps = 1.0 / elapsed
        frame_count += 1
        if frame_count > WARMUP_FRAMES:
            fps_sum += fps
            min_fps = min(min_fps, fps)
            max_fps = max(max_fps, fps)

    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
detector.close()

# --- FPS report ---
measured = frame_count - WARMUP_FRAMES
if measured > 0:
    print(f"Frames measured : {measured}")
    print(f"Average FPS     : {fps_sum / measured:.1f}")
    print(f"Min / Max FPS   : {min_fps:.1f} / {max_fps:.1f}")
else:
    print("Run the program a bit longer to get an FPS measurement.")
