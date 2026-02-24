import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

try:
    # Preferred: robust, fast, well-tested
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception as e:
    lowess = None

# ---------------------------------------------------
# ROI drawing state
# ---------------------------------------------------
roi_start = None
roi_end = None
drawing = False
roi_box = None


def _sanitize_roi(roi_box, frame_shape):
    """
    Clamp ROI to frame bounds and return (x, y, w, h), or None if invalid/empty.
    """
    x, y, w, h = [int(v) for v in roi_box]
    if w <= 0 or h <= 0:
        return None

    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(x, frame_w))
    y1 = max(0, min(y, frame_h))
    x2 = max(0, min(x + w, frame_w))
    y2 = max(0, min(y + h, frame_h))

    w2 = x2 - x1
    h2 = y2 - y1
    if w2 <= 0 or h2 <= 0:
        return None
    return x1, y1, w2, h2


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
    # Some codecs (especially raw h264) may report 0/1 frames inaccurately.
    # Treat such counts as unknown and learn the true end from failed reads.
    has_reliable_count = total_frames > 1
    max_idx = total_frames - 1 if has_reliable_count else None
    idx = 0
    last_idx = None

    # Small LRU cache keeps recently visited frames for smooth back/forward stepping.
    frame_cache = OrderedDict()
    max_cache_frames = 300

    def cache_put(frame_idx, frame):
        frame_cache[frame_idx] = frame.copy()
        frame_cache.move_to_end(frame_idx)
        if len(frame_cache) > max_cache_frames:
            frame_cache.popitem(last=False)

    def reopen_capture():
        nonlocal cap
        cap.release()
        cap = cv2.VideoCapture(video_path)
        return cap.isOpened()

    def decode_to_frame(target_idx):
        """
        Deterministically read frame target_idx by decoding from the start.
        Used as a fallback when direct seek is unreliable.
        """
        if not reopen_capture():
            return False, None
        frame_local = None
        for _ in range(target_idx + 1):
            ok, frame_local = cap.read()
            if not ok:
                return False, None
        return True, frame_local

    def discover_last_frame_index():
        """
        Determine last readable frame index by a one-time sequential decode.
        Needed when CAP_PROP_FRAME_COUNT is unreliable/unknown.
        """
        nonlocal max_idx, last_idx
        if max_idx is not None:
            return True
        if not reopen_capture():
            return False

        n = 0
        while True:
            ok, frame_local = cap.read()
            if not ok:
                break
            # Keep a rolling tail cache so navigation is responsive right after END.
            cache_put(n, frame_local)
            n += 1

        if n <= 0:
            return False

        max_idx = n - 1
        last_idx = None
        # Reset decoder state after EOF scan; some backends misbehave if we keep
        # using the same capture immediately after reading to the end.
        if not reopen_capture():
            return False
        return True

    print("\n=== Controls ===")
    print(" ← / → or J / L : move ±1 frame")
    print(" A / D   : move ±10 frames")
    print(" W / S   : move ±100 frames")
    print(" HOME    : go to first frame")
    print(" END     : go to last frame")
    print(" R       : replace ROI")
    print(" ENTER   : accept existing ROI")
    print(" Q       : quit without ROI\n")

    # Arrow/Home/End keycodes across OpenCV backends.
    left_keys = {81, 2424832, 65361}
    right_keys = {83, 2555904, 65363}
    home_keys = {36, 2359296, 65360}
    end_keys = {35, 2293760, 65367}

    while True:
        if max_idx is None:
            idx = max(0, idx)
        else:
            idx = max(0, min(idx, max_idx))

        if idx in frame_cache:
            ret, frame = True, frame_cache[idx]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # Fallback for codecs where random seek occasionally fails.
                ret, frame = decode_to_frame(idx)

        if not ret:
            # Recover from failed seeks by shrinking bounds instead of exiting.
            # This handles both backward and forward overscroll on videos where
            # reported frame count is slightly inaccurate.
            if idx > 0:
                if max_idx is None:
                    max_idx = idx - 1
                else:
                    max_idx = min(max_idx, idx - 1)
                idx = max_idx
                continue
            print("Error: Could not read frame 0.")
            break
        cache_put(idx, frame)
        last_idx = idx

        disp = frame.copy()

        # Draw ROI overlay if available
        if roi_box is not None:
            x, y, w, h = roi_box
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if max_idx is None:
            frame_label = f"Frame {idx+1}"
        else:
            frame_label = f"Frame {idx+1}/{max_idx + 1}"
        cv2.putText(disp, frame_label,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Video Viewer", disp)
        key = cv2.waitKeyEx(0)
        if key < 0:
            continue
        is_ascii = 0 <= key <= 255
        key8 = key & 0xFF
        key_hi = key & 0xFFFF00

        # Some OpenCV backends encode special keys with extra low-byte data.
        # Match by high-byte pattern as a fallback.
        is_left = key in left_keys or key_hi == 0x250000
        is_right = key in right_keys or key_hi == 0x270000
        is_home = key in home_keys or key_hi == 0x240000
        is_end = key in end_keys or key_hi == 0x230000

        # Quit
        if is_ascii and key8 == ord('q'):
            roi_box = None
            break

        # Accept ROI
        if key in (10, 13) and roi_box is not None:
            print("ROI accepted.")
            cap.release()
            cv2.destroyAllWindows()
            return roi_box

        # Replace ROI
        if is_ascii and key8 in (ord('r'), ord('R')):
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
        if is_left or (is_ascii and key8 in (ord('j'), ord('J'))):  # ← / J
            idx = max(idx - 1, 0)
        elif is_right or (is_ascii and key8 in (ord('l'), ord('L'))):  # → / L
            idx = idx + 1 if max_idx is None else min(idx + 1, max_idx)
        elif is_ascii and key8 in (ord('a'), ord('A')):
            idx = max(idx - 10, 0)
        elif is_ascii and key8 in (ord('d'), ord('D')):
            idx = idx + 10 if max_idx is None else min(idx + 10, max_idx)
        elif is_ascii and key8 in (ord('w'), ord('W')):
            idx = idx + 100 if max_idx is None else min(idx + 100, max_idx)
        elif is_ascii and key8 in (ord('s'), ord('S')):
            idx = max(idx - 100, 0)
        elif is_home:    # HOME
            idx = 0
        elif is_end:     # END
            if max_idx is None:
                print("Finding last frame...")
                if not discover_last_frame_index():
                    print("Could not determine last frame.")
                    continue
            idx = max_idx

    cap.release()
    cv2.destroyAllWindows()
    return None


# ---------------------------------------------------
# PROCESS VIDEO USING ROI
# ---------------------------------------------------
def process_video(video_path, roi_box):
    sums = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    dropped = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame is None or frame.size == 0:
            dropped += 1
            continue

        safe_roi = _sanitize_roi(roi_box, frame.shape)
        if safe_roi is None:
            cap.release()
            raise ValueError(
                f"Invalid ROI {roi_box} for frame shape {frame.shape[:2]}. "
                "Draw a non-zero ROI fully inside the frame."
            )
        x, y, w, h = safe_roi
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            dropped += 1
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sums.append(np.sum(gray))

    cap.release()
    if dropped > 0:
        print(f"Warning: skipped {dropped} empty/unreadable frame(s).")
    return np.array(sums)


def extract_signal_from_video(
    video_path: str,
    roi_box: tuple[int, int, int, int],
    method: str = "sums",
    centroid_debug_viz: bool = False,
) -> np.ndarray:
    """
    Extract a 1D signal from video using either:
      - method='sums': fixed ROI pixel sums
      - method='centroid': LED centroid tracking initialized from selected ROI
    """
    if method not in {"sums", "centroid"}:
        raise ValueError("method must be 'sums' or 'centroid'")

    if method == "sums":
        return process_video(video_path, roi_box)

    # Centroid mode: use chosen ROI to initialize tracking strategy.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame0 = cap.read()
    cap.release()
    if not ok or frame0 is None or frame0.size == 0:
        raise RuntimeError(f"Could not read first frame from video: {video_path}")

    safe_roi = _sanitize_roi(roi_box, frame0.shape)
    if safe_roi is None:
        raise ValueError(
            f"Invalid ROI {roi_box} for frame shape {frame0.shape[:2]}. "
            "Draw a non-zero ROI fully inside the frame."
        )
    x, y, w, h = safe_roi
    start_xy = (x + 0.5 * w, y + 0.5 * h)
    search_halfsize = max(8, int(round(0.5 * max(w, h))))
    local_halfsize = max(2, int(round(0.5 * min(w, h))))

    _, _, trace = track_led_in_video(
        video_path,
        start_xy=start_xy,
        search_halfsize=search_halfsize,
        roi_halfsize=local_halfsize,
        position_smoothing=0.6,
        led_stat_mode="sum",
        max_missed=10,
        debug_viz=centroid_debug_viz,
    )
    return trace


def reload_roi_sums(npy_path="roi_sums.npy"):
    """
    Reload previously saved ROI sums for debugging/inspection.
    """
    npy_file = Path(npy_path)
    if not npy_file.exists():
        raise FileNotFoundError(f"ROI sums file not found: {npy_file}")

    values = np.load(npy_file)
    if values.ndim != 1:
        values = np.asarray(values).reshape(-1)

    print(f"Reloaded {npy_file} with shape {values.shape}")
    print(f"min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}, std={values.std():.3f}")
    return values


def plot_roi_sums(values: np.ndarray, title: str = "ROI Sum Over Frames"):
    """
    Shared plotting for ROI sums (frame index on x-axis).
    """
    frame_idx = np.arange(values.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(frame_idx, values, linewidth=1)
    plt.xlabel("Frame index")
    plt.ylabel("ROI pixel sum")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()


def running_median(x: np.ndarray, win: int) -> np.ndarray:
    """
    Compute running median with edge padding.
    win must be odd and >= 3.
    """
    if win < 3 or win % 2 == 0:
        raise ValueError("win must be an odd integer >= 3")
    x = np.asarray(x, dtype=float)

    half = win // 2
    xp = np.pad(x, (half, half), mode="edge")

    # Simple (readable) implementation. For very long signals, consider faster methods.
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i + win])
    return out


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    Simple moving average with edge padding.
    win >= 1. If win == 1, returns x.
    """
    if win <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    half = win // 2
    xp = np.pad(x, (half, half), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(xp, kernel, mode="valid")


def first_difference(x: np.ndarray) -> np.ndarray:
    """
    First-difference detrending: y[t] = x[t] - x[t-1].
    Keeps output length equal to input by prepending 0 at t=0.
    """
    x = np.asarray(x, dtype=float)
    return np.diff(x, prepend=x[0])


def hysteresis_threshold(y: np.ndarray, thr_on: float, thr_off: float) -> np.ndarray:
    """
    Hysteresis thresholding:
      - state turns ON when y >= thr_on
      - state turns OFF when y <= thr_off
    Returns uint8 array of 0/1.
    """
    if thr_off > thr_on:
        raise ValueError("thr_off must be <= thr_on")

    y = np.asarray(y, dtype=float)
    out = np.zeros(len(y), dtype=np.uint8)

    state = 0
    for i, v in enumerate(y):
        if state == 0:
            if v >= thr_on:
                state = 1
        else:
            if v <= thr_off:
                state = 0
        out[i] = state
    return out


def remove_short_runs(binary: np.ndarray, min_len: int, value: int = 1) -> np.ndarray:
    """
    Remove runs of `value` shorter than min_len by flipping them to 1-value.
    """
    b = np.asarray(binary, dtype=np.uint8).copy()
    if min_len <= 1:
        return b

    n = len(b)
    i = 0
    while i < n:
        if b[i] != value:
            i += 1
            continue
        j = i
        while j < n and b[j] == value:
            j += 1
        if (j - i) < min_len:
            b[i:j] = 1 - value
        i = j
    return b


def fill_short_gaps(binary: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Fill 0-runs (gaps) of length <= max_gap that are surrounded by 1s.
    """
    b = np.asarray(binary, dtype=np.uint8).copy()
    if max_gap <= 0:
        return b

    n = len(b)
    i = 0
    while i < n:
        if b[i] != 0:
            i += 1
            continue
        j = i
        while j < n and b[j] == 0:
            j += 1

        gap_len = j - i
        left_is_one = (i - 1 >= 0) and (b[i - 1] == 1)
        right_is_one = (j < n) and (b[j] == 1)

        if left_is_one and right_is_one and gap_len <= max_gap:
            b[i:j] = 1
        i = j
    return b


def extract_ttl(
    X: np.ndarray,
    fps: float | None = None,
    baseline_win_s: float = 1.0,
    smooth_win_s: float = 0.0,
    thr_on_sigma: float = 3.0,
    thr_off_sigma: float = 1.5,
    min_pulse_s: float = 0.01,
    max_gap_s: float = 0.005,
) -> dict[str, np.ndarray | float]:
    """
    Main pipeline.

    If fps is provided, *_s parameters are converted to samples.
    If fps is None, windows are treated as samples by rounding.

    Returns a dict with:
      - baseline, detrended, smoothed, binary, thr_on, thr_off
    """
    x = np.asarray(X, dtype=float)

    def to_samples(seconds: float, default: int) -> int:
        if fps is None:
            return max(default, int(round(seconds)))
        return max(default, int(round(seconds * fps)))

    # Choose sample-based windows
    baseline_win = to_samples(baseline_win_s, default=101)
    if baseline_win % 2 == 0:
        baseline_win += 1  # make odd for median

    smooth_win = to_samples(smooth_win_s, default=1)
    if smooth_win < 1:
        smooth_win = 1

    min_pulse = to_samples(min_pulse_s, default=1)
    max_gap = to_samples(max_gap_s, default=0)

    # 1) Baseline removal (robust)
    base = running_median(x, baseline_win)
    y = x - base

    # 2) Optional smoothing (disabled by default)
    y_s = moving_average(y, smooth_win)

    # Robust scale estimate via MAD (works even with pulses)
    mad = np.median(np.abs(y_s - np.median(y_s)))
    sigma = 1.4826 * mad + 1e-12  # avoid divide-by-zero

    thr_on = thr_on_sigma * sigma
    thr_off = thr_off_sigma * sigma

    # 3) Hysteresis threshold
    b = hysteresis_threshold(y_s, thr_on=thr_on, thr_off=thr_off)

    # 4) Cleanup: remove very short pulses, fill tiny gaps, remove short pulses again
    b = remove_short_runs(b, min_len=min_pulse, value=1)
    b = fill_short_gaps(b, max_gap=max_gap)
    b = remove_short_runs(b, min_len=min_pulse, value=1)

    return {
        "baseline": base,
        "detrended": y,
        "smoothed": y_s,
        "binary": b,
        "thr_on": float(thr_on),
        "thr_off": float(thr_off),
    }


#!/usr/bin/env python3
"""
Simple LED tracking (bright spot) with a small ROI that follows the LED.

Approach:
- Read frames from a video (OpenCV).
- In each frame, search for the LED only inside a window around the last position.
- Detect the LED as the brightest blob using thresholding + connected components.
- Update the LED position (optionally smoothed).
- Extract a local ROI sum/mean around the tracked position.

Works well when:
- The LED is one of the brightest things in the search window.
- Motion between frames isn’t huge (or you increase search_window).

Requires: opencv-python, numpy
"""

# from __future__ import annotations
import cv2
import numpy as np


def clamp_box(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    return x0, y0, x1, y1


def find_led_centroid(gray: np.ndarray,
                      last_xy: tuple[float, float] | None,
                      search_halfsize: int = 60,
                      blur_ksize: int = 5,
                      top_percentile: float = 99.5,
                      min_area: int = 5) -> tuple[float, float] | None:
    """
    Locate the LED centroid in a grayscale frame.
    - Searches in a box around last_xy (or full frame if None).
    - Threshold at a high percentile within the search region.
    - Picks the largest bright component; returns centroid in full-frame coordinates.
    """
    h, w = gray.shape

    if last_xy is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        cx, cy = last_xy
        x0 = int(round(cx - search_halfsize))
        x1 = int(round(cx + search_halfsize))
        y0 = int(round(cy - search_halfsize))
        y1 = int(round(cy + search_halfsize))
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, w, h)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    # Denoise a bit
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        roi_blur = roi

    # High threshold based on percentile (robust against overall intensity drift)
    thr = np.percentile(roi_blur, top_percentile)
    mask = (roi_blur >= thr).astype(np.uint8) * 255

    # Clean up specks / holes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Connected components: choose the largest bright blob
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None

    # Skip label 0 (background). Pick max area
    best = None
    best_area = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_area = area
            best = lab

    if best is None:
        return None

    cx_roi, cy_roi = centroids[best]
    return (x0 + float(cx_roi), y0 + float(cy_roi))


def extract_local_roi_stat(gray: np.ndarray,
                           center_xy: tuple[float, float],
                           roi_halfsize: int = 10,
                           mode: str = "mean") -> float:
    """
    Extract a statistic (mean/sum/top_percentile_mean) from a small ROI around the tracked LED.
    """
    h, w = gray.shape
    cx, cy = center_xy
    x0 = int(round(cx - roi_halfsize))
    x1 = int(round(cx + roi_halfsize + 1))
    y0 = int(round(cy - roi_halfsize))
    y1 = int(round(cy + roi_halfsize + 1))
    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, w, h)

    roi = gray[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return float("nan")

    if mode == "mean":
        return float(roi.mean())
    if mode == "sum":
        return float(roi.sum())
    if mode == "top1pct_mean":
        # Average of the brightest 1% pixels, robust to small motion/partial coverage
        flat = roi.ravel()
        k = max(1, int(0.01 * flat.size))
        # partial sort to avoid full sort for large ROIs
        idx = np.argpartition(flat, -k)[-k:]
        return float(flat[idx].mean())

    raise ValueError("mode must be one of: mean, sum, top1pct_mean")


def track_led_in_video(video_path: str,
                       start_xy: tuple[float, float] | None = None,
                       search_halfsize: int = 60,
                       roi_halfsize: int = 10,
                       position_smoothing: float = 0.6,
                       led_stat_mode: str = "top1pct_mean",
                       max_missed: int = 10,
                       debug_viz: bool = False):
    """
    Tracks an LED across video frames and extracts a 1D trace from a moving ROI.

    position_smoothing in [0..1]:
      - 0   = no smoothing (use new detection)
      - 0.6 = common value (EMA)
      - 0.9 = very smooth / laggy

    Returns:
      xs, ys (tracked positions; NaN when missing), trace (ROI stat)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    last_xy = start_xy
    missed = 0

    xs, ys, trace = [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        xy = find_led_centroid(
            gray,
            last_xy=last_xy,
            search_halfsize=search_halfsize,
            blur_ksize=5,
            top_percentile=99.5,
            min_area=5,
        )

        if xy is None:
            missed += 1
            if missed > max_missed:
                # give up and search full frame again
                last_xy = None
                missed = 0
            xs.append(np.nan)
            ys.append(np.nan)
            trace.append(np.nan)
        else:
            missed = 0
            # Exponential smoothing on position
            if last_xy is None:
                smoothed = xy
            else:
                a = float(position_smoothing)
                smoothed = (a * last_xy[0] + (1 - a) * xy[0],
                            a * last_xy[1] + (1 - a) * xy[1])

            last_xy = smoothed
            xs.append(smoothed[0])
            ys.append(smoothed[1])
            trace.append(extract_local_roi_stat(gray, smoothed, roi_halfsize, mode=led_stat_mode))

            if debug_viz:
                vis = frame.copy()
                cx, cy = int(round(smoothed[0])), int(round(smoothed[1]))
                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), 2)

                # draw the search window
                x0 = cx - search_halfsize
                y0 = cy - search_halfsize
                x1 = cx + search_halfsize
                y1 = cy + search_halfsize
                h, w = gray.shape
                x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, w, h)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

                # draw the ROI box
                r = roi_halfsize
                x0, y0, x1, y1 = clamp_box(cx - r, cy - r, cx + r + 1, cy + r + 1, w, h)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)

                cv2.imshow("LED tracking", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    cap.release()
    if debug_viz:
        cv2.destroyAllWindows()

    return np.array(xs, float), np.array(ys, float), np.array(trace, float)


def lowess_detrend(x, frac=0.02, it=1, return_baseline=False):
    """
    LOWESS detrend for a 1D signal.

    Parameters
    ----------
    x : array-like, shape (n,)
        Input trace.
    frac : float
        Fraction of points used for each local regression (controls smoothness).
        Larger = smoother baseline. Typical: 0.01–0.2 depending on drift timescale.
    it : int
        Robustifying iterations. 0 = standard LOWESS, 1–3 helps ignore pulses/outliers.
    return_baseline : bool
        If True, return (detrended, baseline). Else return detrended.

    Returns
    -------
    detrended : np.ndarray
        x - baseline
    baseline : np.ndarray (optional)
        LOWESS fit
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    t = np.arange(n)

    if lowess is None:
        raise ImportError(
            "statsmodels is required for LOWESS.\n"
            "Install with: pip install statsmodels"
        )

    # statsmodels.lowess returns an (n,2) array: [t, fit]
    fit = lowess(endog=x, exog=t, frac=frac, it=it, return_sorted=True)
    baseline = fit[:, 1]

    detrended = x - baseline
    return (detrended, baseline) if return_baseline else detrended



# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    # Set True to skip video processing and reload saved ROI sums.
    reload_saved_debug = False
    extraction_method = "centroid"  # "sums" or "centroid"
    video_file = "/home/matt/Documents/EXPERIMENTS/test_runs/test-mouse_2026-02-23_164141/test-mouse_2026-02-23_164141_cam0_output.h264"
    fps = 30

    values = None
    output_npy = "roi_sums.npy" if extraction_method == "sums" else "roi_signal_centroid.npy"

    # Load or compute values in one branch, then run shared analysis below.
    if reload_saved_debug:
        values = reload_roi_sums(output_npy)
    else:
        roi_box = choose_frame_and_roi(video_file)

        # Example:
        # xs, ys, led_trace = track_led_in_video("your_video.mp4", start_xy=(320, 240), debug_viz=True)
        # print("Trace length:", len(led_trace), "NaNs:", np.isnan(led_trace).sum())

        if roi_box is None:
            print("No ROI selected. Exiting.")
            raise SystemExit(0)
        if roi_box[2] <= 0 or roi_box[3] <= 0:
            raise ValueError(f"ROI must have non-zero width/height, got {roi_box}")

        print(f"Processing with ROI = {roi_box} using method='{extraction_method}'")
        values = extract_signal_from_video(
            video_file,
            roi_box,
            method=extraction_method,
            centroid_debug_viz=False,
        )
        print("Extracted signal shape:", values.shape)
        np.save(output_npy, values)
        print(f"Saved ROI signal to {output_npy}")

    # Shared downstream analysis regardless of source (reload vs fresh extraction).
    plot_roi_sums(values, title="ROI Sum Over Frames")
    ttl = extract_ttl(values, fps=fps)
    y, base = lowess_detrend(values, frac=0.005, it=2, return_baseline=True)
    f = plt.figure(figsize=(10, 4))
    plt.plot(y, label="Detrended")
    plt.plot(base, label="Base")
    plt.title("LOWESS Detrending")
    plt.xlabel("Frame index")
    plt.ylabel("Pixel sum")

    print(y[:5], base[:5])

    print(f"TTL thresholds: on={ttl['thr_on']:.3f}, off={ttl['thr_off']:.3f}")

    # Plot extract_ttl output for debugging.
    frame_idx = np.arange(values.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(frame_idx, ttl["detrended"], linewidth=1, label="Detrended")
    plt.xlabel("Frame index")
    plt.ylabel("Detrended ROI signal")
    plt.title("extract_ttl Output: Detrended Signal")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # Alternative detrending strategy: first differencing.
    values_diff = first_difference(values)
    plt.figure(figsize=(10, 4))
    plt.plot(frame_idx, values_diff, linewidth=1, color="tab:orange", label="First difference")
    plt.xlabel("Frame index")
    plt.ylabel("First-differenced ROI signal")
    plt.title("Alternative Detrending: First Difference")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
