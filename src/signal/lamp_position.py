"""Read a 3-lamp round signal head by WHICH lamp is lit, not by its hue (owner, 18/11/2026: yellow is always the middle lamp).

VN yellow is orange and VN red often looks orange on camera, so hue alone confuses them; the lamp position does not.
The ROI must cover the head housing (lamps only, no countdown) and the layout must be known:
  vertical3   = red top, yellow middle, green bottom
  horizontal3 = red left, yellow middle, green right
Not for arrow heads, countdown displays or heads with more than three lamps (return None -> caller falls back).
"""
import cv2
import numpy as np

LAYOUTS = ("vertical3", "horizontal3")
COLOURS = ("red", "yellow", "green")


def lit_mask(hsv):
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return ((s >= 80) & (v >= 100)) | (v >= 235)  # coloured lit pixels, or a blown-out lamp centre


def _greenish(hsv_pixels):
    sat = hsv_pixels[hsv_pixels[:, 1] >= 80]
    return len(sat) > 0 and 35 <= float(np.median(sat[:, 0])) <= 100  # OpenCV hue 0-179


def read_by_position(patch_bgr, layout, min_px=3, dominance=2.0):
    """Return (colour, share of lit pixels in the winning zone) or None when no single lamp is clearly lit."""
    if patch_bgr is None or layout not in LAYOUTS or min(patch_bgr.shape[:2]) < 3:
        return None
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    if layout == "horizontal3":
        hsv = hsv.transpose(1, 0, 2)
    n = hsv.shape[0]
    zones = [hsv[k * n // 3:(k + 1) * n // 3] for k in range(3)]
    lit = [z[lit_mask(z)] for z in zones]
    counts = [len(p) for p in lit]
    order = sorted(range(3), key=lambda k: -counts[k])
    win, second = order[0], counts[order[1]]
    if counts[win] < min_px or counts[win] < dominance * second:
        return None
    # sanity: a green-hued blob in the red/yellow zone (or a non-green one in the green zone) is a reflection or a misplaced ROI
    if _greenish(lit[win]) != (win == 2) and any(lit[win][:, 1] >= 80):
        return None
    return COLOURS[win], counts[win] / max(1, sum(counts))
