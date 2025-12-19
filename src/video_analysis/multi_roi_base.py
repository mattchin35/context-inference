import cv2
import numpy as np

# -----------------------------------------------------------------
# ROI drawing
# -----------------------------------------------------------------
drawing = False
roi_start = None
roi_end = None
current_roi = None

def roi_mouse_handler(event, x, y, flags, param):
    """
    Mouse callback for drawing ROI with click & drag.
    """
    global drawing, roi_start, roi_end, current_roi
    frame = param["frame"]
    disp = frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)
        cv2.rectangle(disp, roi_start, roi_end, (0, 255, 0), 2)
        cv2.imshow("Draw ROI", disp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)

        x1, y1 = roi_start
        x2, y2 = roi_end
        current_roi = (min(x1, x2), min(y1, y2),
                       abs(x2 - x1), abs(y2 - y1))

        cv2.rectangle(disp, roi_start, roi_end, (0, 255, 0), 2)
        cv2.imshow("Draw ROI", disp)
        print(f"ROI selected: {current_roi}")


# -----------------------------------------------------------------
# FRAME SCROLLER – supports multiple ROIs
# -----------------------------------------------------------------
def choose_rois(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0

    rois = []   # store all ROI boxes
    colors = [(0,255,0), (0,128,255), (255,0,0), (255,255,0),
              (255,0,255), (0,255,255), (128,0,255)]  # repeated cycle

    print("\n=== Controls ===")
    print(" ← / →   : move ±1 frame")
    print(" A / D   : move ±10 frames")
    print(" W / S   : move ±100 frames")
    print(" HOME    : go to first frame")
    print(" END     : go to last frame")
    print(" R       : add a new ROI on current frame")
    print(" A       : accept all ROIs")
    print(" Q       : quit without saving\n")

    while True:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        disp = frame.copy()

        # Draw all existing ROIs
        for i, (x, y, w, h) in enumerate(rois):
            c = colors[i % len(colors)]
            cv2.rectangle(disp, (x, y), (x+w, y+h), c, 2)
            cv2.putText(disp, f"ROI {i+1}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        # Draw frame index
        cv2.putText(disp, f"Frame {idx+1}/{total_frames}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Video Viewer", disp)
        key = cv2.waitKey(0) & 0xFF

        # Quit
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

        # Accept existing ROIs
        if key == ord('a') and len(rois) > 0:
            cap.release()
            cv2.destroyAllWindows()
            return rois

        # Add a new ROI
        if key == ord('r'):
            print("Draw ROI on this frame. Press ENTER when done.")

            clone = frame.copy()
            global current_roi
            current_roi = None

            cv2.namedWindow("Draw ROI")
            cv2.setMouseCallback("Draw ROI", roi_mouse_handler,
                                 param={"frame": clone})

            while True:
                tmp = clone.copy()
                if current_roi is not None:
                    x, y, w, h = current_roi
                    cv2.rectangle(tmp, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.imshow("Draw ROI", tmp)

                k2 = cv2.waitKey(10) & 0xFF
                if k2 == 13 and current_roi is not None:  # ENTER
                    print(f"Added ROI #{len(rois)+1}: {current_roi}")
                    rois.append(current_roi)
                    cv2.destroyWindow("Draw ROI")
                    break
                if k2 == 27:  # ESC
                    current_roi = None
                    cv2.destroyWindow("Draw ROI")
                    break

        # Navigation controls
        if key == 81:
            idx = max(idx - 1, 0)
        elif key == 83:
            idx = min(idx + 1, total_frames - 1)
        elif key == ord('d'):
            idx = min(idx + 10, total_frames - 1)
        elif key == ord('a'):
            pass
        elif key == ord('w'):
            idx = min(idx + 100, total_frames - 1)
        elif key == ord('s'):
            idx = max(idx - 100, 0)
        elif key == 36:  # HOME
            idx = 0
        elif key == 35:  # END
            idx = total_frames - 1

    cap.release()
    cv2.destroyAllWindows()
    return rois


# -----------------------------------------------------------------
# PROCESS VIDEO WITH MULTIPLE ROIs
# -----------------------------------------------------------------
def process_video(video_path, rois):
    """
    Returns array of shape: (num_rois, num_frames)
    """
    cap = cv2.VideoCapture(video_path)
    sums = [[] for _ in rois]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i, (x, y, w, h) in enumerate(rois):
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            sums[i].append(np.sum(gray))

    cap.release()
    return np.array(sums)  # shape: (R, F)


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
if __name__ == "__main__":
    video_file = "input.h264"

    rois = choose_rois(video_file)

    if rois is None or len(rois) == 0:
        print("No ROIs selected. Exiting.")
    else:
        print(f"Processing {len(rois)} ROIs...")
        values = process_video(video_file, rois)

        print("Output array shape (ROIs × Frames):", values.shape)

        np.save("roi_sums.npy", values)
        np.save("rois.npy", np.array(rois, dtype=int))

        print("\nSaved:")
        print("  roi_sums.npy  (ROI × Frame pixel sums)")
        print("  rois.npy      (ROI coordinate list)")
