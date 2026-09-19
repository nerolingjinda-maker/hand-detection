import cv2
import numpy as np
import mediapipe as mp
import winsound
import time
import threading

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# --- Camera setup with fallback ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("WARNING: External webcam (index 1) not found, trying default camera (index 0)")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: No webcam could be opened")
    exit()

# This is a fixed monitoring camera, not a selfie app -- keep the real
# orientation so your actual left hand stays on the left side of the frame.
# Set to True only if you specifically want a mirror-style flip.
MIRROR_VIEW = False

LINE_Y = 230
LINE_MARGIN = 4           # small dead-zone: still absorbs camera noise but
                          # triggers almost as soon as the tip touches the line

# Direction of travel that counts as a "crossing":
# hand normally sits BELOW the line (safe/normal) and crossing UP into the
# area ABOVE the line is what should be detected/alerted.
ALERT_ON_ENTER_ABOVE = True   # below -> above triggers alert (what you asked for)
ALERT_ON_ENTER_BELOW = False  # above -> below triggers alert too, if you also want that

# --- Motion/object based detection settings ---
# MediaPipe only detects an actual hand skeleton. If someone crosses the line
# holding a tool/object that hides most of their fingers, MediaPipe finds
# nothing at all. This second detector watches for ANY foreground object
# (hand, gloved hand, held tool, etc.) moving up across the line, using
# background subtraction instead of skeleton tracking.
USE_OBJECT_DETECTION = True
OBJECT_BAND_ABOVE = 220   # how far above LINE_Y to scan for motion (pixels)
OBJECT_BAND_BELOW = 260   # how far below LINE_Y to scan for motion (pixels)
MIN_CONTOUR_AREA = 1200   # ignore blobs smaller than this (noise/shadow flecks)

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=40, detectShadows=True
)
object_side = "below"

hand_side = {}
cross_message = ""
last_sound_time = 0
SOUND_COOLDOWN = 0.7


def play_sound():
    """Non-blocking beep so the capture/processing loop never stalls."""
    global last_sound_time
    current_time = time.time()
    if current_time - last_sound_time >= SOUND_COOLDOWN:
        last_sound_time = current_time
        threading.Thread(
            target=lambda: winsound.Beep(1200, 300),
            daemon=True
        ).start()


def get_side(y):
    if y < LINE_Y - LINE_MARGIN:
        return "above"
    elif y > LINE_Y + LINE_MARGIN:
        return "below"
    else:
        return "line"


FINGER_NAMES = {
    4: "THUMB CROSSED",
    8: "INDEX FINGER CROSSED",
    12: "MIDDLE FINGER CROSSED",
    16: "RING FINGER CROSSED",
    20: "PINKY FINGER CROSSED",
}
# Track ALL 21 landmarks, not just fingertips -- when a hand grips an object,
# the tips are often the first thing hidden, but the wrist/knuckles usually
# stay visible. Any visible landmark crossing is enough to count as a hand
# crossing; tip landmarks just get a more specific message.

morph_kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read webcam")
        break

    frame = cv2.flip(frame, 1) if MIRROR_VIEW else frame
    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (255, 0, 0), 4)
    cv2.putText(
        frame, "VIRTUAL SAFETY LINE", (20, LINE_Y - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
    )

    current_hand_is_inside = False
    hand_is_in_upper_area = False
    seen_ids = set()
    triggered_this_frame = False

    # ---------------- Hand skeleton based detection ----------------
    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Use hand_index (stable within a frame's ordering) rather than the
            # handedness label, which can flicker between "Left"/"Right" and
            # was silently dropping crossing events when it changed.
            hand_key = hand_index

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                point_id = (hand_key, landmark_id)
                seen_ids.add(point_id)
                current_side = get_side(y)

                if current_side == "above":
                    hand_is_in_upper_area = True
                if current_side == "below":
                    current_hand_is_inside = True

                if current_side == "line":
                    cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
                    continue

                if point_id not in hand_side:
                    # First time we've seen this point -- record state, don't
                    # treat it as a crossing (avoids false alarm when a hand
                    # first enters frame already below the line).
                    hand_side[point_id] = current_side
                else:
                    previous_side = hand_side[point_id]

                    if previous_side != current_side:
                        entered_above = (previous_side == "below" and current_side == "above")
                        entered_below = (previous_side == "above" and current_side == "below")

                        if (entered_above and ALERT_ON_ENTER_ABOVE) or (entered_below and ALERT_ON_ENTER_BELOW):
                            cross_message = FINGER_NAMES.get(landmark_id, "HAND CROSSING DETECTED")
                            play_sound()
                            triggered_this_frame = True
                            cv2.circle(frame, (x, y), 18, (0, 0, 255), 4)

                    hand_side[point_id] = current_side

                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

        # Drop tracking state for any point_ids not seen this frame
        # (e.g. a hand left the frame) so stale entries don't linger.
        stale = [pid for pid in hand_side if pid not in seen_ids]
        for pid in stale:
            del hand_side[pid]
    else:
        hand_side.clear()

    # ---------------- Motion/object based detection ----------------
    # Catches hands holding tools/objects that hide the finger skeleton,
    # which MediaPipe alone cannot see.
    if USE_OBJECT_DETECTION:
        band_top = max(0, LINE_Y - OBJECT_BAND_ABOVE)
        band_bottom = min(height, LINE_Y + OBJECT_BAND_BELOW)
        roi = frame[band_top:band_bottom, :]

        fg_mask = back_sub.apply(roi)
        # Drop MOG2's gray "shadow" pixels (value 127), keep only solid foreground
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, morph_kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_object_side = "below"
        breach_box = None

        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            top_y = y + band_top
            bottom_y = y + h + band_top

            # Only count objects that are actually straddling/crossing the
            # line (so something sitting statically above the line, like a
            # mouse in the background, doesn't get flagged).
            straddles_line = top_y < LINE_Y + LINE_MARGIN and bottom_y > LINE_Y - LINE_MARGIN
            if straddles_line and top_y < LINE_Y - LINE_MARGIN:
                current_object_side = "above"
                breach_box = (x, top_y - band_top, w, h)
                break

        if current_object_side == "above":
            hand_is_in_upper_area = True

        if object_side == "below" and current_object_side == "above" and ALERT_ON_ENTER_ABOVE:
            cross_message = "OBJECT CROSSING DETECTED"
            play_sound()
            triggered_this_frame = True
        elif object_side == "above" and current_object_side == "below" and ALERT_ON_ENTER_BELOW:
            cross_message = "OBJECT CROSSING DETECTED"
            play_sound()
            triggered_this_frame = True

        object_side = current_object_side

    # ---------------- Status display ----------------
    # "Inside the virtual line" = any part of a hand or object has crossed
    # above it. No hand/object, or fully below the line, = normal.
    if not hand_is_in_upper_area:
        cv2.putText(
            frame, "NORMAL", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    if hand_is_in_upper_area:
        cv2.putText(
            frame, "INTRUSION DETECTED", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3
        )
        if cross_message:
            cv2.putText(
                frame, cross_message, (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
            )

    cv2.imshow("Hand Line Crossing Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()