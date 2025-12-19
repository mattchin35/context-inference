import cv2
import numpy as np

# ---------------------------------------------------
# ROI drawing state
# ---------------------------------------------------
roi_start = None
roi_end = None
drawing = False
roi_box = None


def roi_mouse_handler(event, x, y, flags, param):
    """
    Mouse callback for drawing a new ROI.
    """
    global roi_start, roi_end, drawing, roi_box
    frame = param["frame"]
    disp = frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)
        cv2.rectangle(disp, roi_start, roi_end, (0, 255, 0), 2)
        cv2.imshow("Draw ROI", disp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)

        x1, y1 = roi_start
        x2, y2 = roi_end
        roi_box = (min(x1, x2), min(y1, y2),
                   abs(x2 - x1), abs(y2 - y1))

        print(f"ROI selected: {roi_box}")

        cv2.rectangle(disp, roi_start, roi_end, (0, 255, 0), 2)
        cv2.imshow("Draw ROI", disp)


# ---------------------------------------------------
# FRAME SCROLLER WITH ROI OVERLAY + ACCEPT/REPLACE
# ---------------------------------------------------
def choose_frame_and_roi(video_path):
    global roi_box

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0

    print("\n=== Controls ===")
    print(" ← / →   : move ±1 frame")
    print(" A / D   : move ±10 frames")
    print(" W / S   : move ±100 frames")
    print(" HOME    : go to first frame")
    print(" END     : go to last frame")
    print(" R       : replace ROI")
    print(" A       : accept existing ROI")
    print(" Q       : quit without ROI\n")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        disp = frame.copy()

        # Draw ROI overlay if available
        if roi_box is not None:
            x, y, w, h = roi_box
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(disp, f"Frame {idx+1}/{total_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Video Viewer", disp)
        key = cv2.waitKey(0) & 0xFF

        # Quit
        if key == ord('q'):
            roi_box = None
            break

        # Accept ROI
        if key == ord('a') and roi_box is not None:
            print("ROI accepted.")
            cap.release()
            cv2.destroyAllWindows()
            return roi_box

        # Replace ROI
        if key == ord('r'):
            print("Draw a new ROI. Press ENTER when done.")

            clone = frame.copy()
            roi_box = None

            cv2.namedWindow("Draw ROI")
            cv2.setMouseCallback("Draw ROI",
                                 roi_mouse_handler,
                                 param={"frame": clone})

            while True:
                temp = clone.copy()
                if roi_box is not None:
                    x, y, w, h = roi_box
                    cv2.rectangle(temp, (x, y),
                                  (x + w, y + h),
                                  (0, 255, 0), 2)

                cv2.imshow("Draw ROI", temp)
                k2 = cv2.waitKey(10) & 0xFF

                if k2 == 13 and roi_box is not None:  # ENTER
                    cv2.destroyWindow("Draw ROI")
                    break

                if k2 == 27:  # ESC
                    roi_box = None
                    cv2.destroyWindow("Draw ROI")
                    break

        # Navigation
        if key == 81:      # ←
            idx = max(idx - 1, 0)
        elif key == 83:    # →
            idx = min(idx + 1, total_frames - 1)
        elif key == ord('d'):
            idx = min(idx + 10, total_frames - 1)
        elif key == ord('a'):
            pass
        elif key == ord('w'):
            idx = min(idx + 100, total_frames - 1)
        elif key == ord('s'):
            idx = max(idx - 100, 0)
        elif key == 36:    # HOME
            idx = 0
        elif key == 35:    # END
            idx = total_frames - 1

    cap.release()
    cv2.destroyAllWindows()
    return None


# ---------------------------------------------------
# PROCESS VIDEO USING ROI
# ---------------------------------------------------
def process_video(video_path, roi_box):
    x, y, w, h = roi_box
    sums = []

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sums.append(np.sum(gray))

    cap.release()
    return np.array(sums)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    video_file = "input.h264"

    roi_box = choose_frame_and_roi(video_file)

    if roi_box is None:
        print("No ROI selected. Exiting.")
    else:
        print(f"Processing with ROI = {roi_box}")
        values = process_video(video_file, roi_box)
        print("Pixel sums shape:", values.shape)

        # ----------------------------
        # Save to .npy file
        # ----------------------------
        np.save("roi_sums.npy", values)
        print("Saved ROI sums to roi_sums.npy")
