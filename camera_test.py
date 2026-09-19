import cv2

for camera_index in range(5):

    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera {camera_index}: WORKING")
        else:
            print(f"Camera {camera_index}: detected but cannot read")

        cap.release()

    else:
        print(f"Camera {camera_index}: not available")