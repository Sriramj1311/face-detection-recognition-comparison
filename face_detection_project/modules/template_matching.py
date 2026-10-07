"""OpenCV template matching (searches for a pixel pattern, NOT face recognition)."""
import cv2

THRESHOLD = 0.6  # normalised correlation score above which we call it a match


def run(img_rgb, template_rgb):
    if template_rgb is None:
        return {"ok": False, "error": "A template image is required. Upload one in the sidebar "
                                      "(e.g. a cropped face or object from the main image)."}
    H, W = img_rgb.shape[:2]
    h, w = template_rgb.shape[:2]
    if h > H or w > W:
        return {"ok": False, "error": f"Template ({w}x{h}) is larger than the main image ({W}x{H})."}

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    tmpl = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)

    # Slide the template over the image; each position gets a similarity score (-1..1)
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)  # best score and its top-left corner

    ann = img_rgb.copy()
    x, y = max_loc
    found = max_val >= THRESHOLD
    cv2.rectangle(ann, (x, y), (x + w, y + h), (255, 0, 0) if found else (255, 165, 0), 3)

    msg = (f"{'Match found' if found else 'No reliable match'} at (x={x}, y={y}), "
           f"size {w}x{h}, score={max_val:.3f} (threshold {THRESHOLD}).")
    return {"ok": True, "image": ann, "template": template_rgb, "score": float(max_val),
            "location": (int(x), int(y)), "found": bool(found), "message": msg, "summary": msg,
            "note": "Template matching compares raw pixel patterns at one scale/rotation. "
                    "It does not understand faces, so it is not modern face recognition."}
