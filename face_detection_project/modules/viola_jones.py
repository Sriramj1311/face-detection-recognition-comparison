"""Viola-Jones face detection using OpenCV's Haar Cascade."""
import os
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_XML = os.path.join(HERE, "models", "haarcascade_frontalface_default.xml")


def run(img_rgb):
    # 1. Find the cascade file: local models/ folder first, then OpenCV's built-in copy
    xml = LOCAL_XML if os.path.exists(LOCAL_XML) else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml):
        return {"ok": False, "error": "haarcascade_frontalface_default.xml not found."}
    cascade = cv2.CascadeClassifier(xml)
    if cascade.empty():
        return {"ok": False, "error": f"Could not load cascade file: {xml}"}

    # 2. Haar cascades work on grayscale images
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 3. Slide windows at many scales; minNeighbors filters weak detections
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 4. Draw boxes
    ann = img_rgb.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 200, 0), 3)

    n = len(faces)
    msg = "No face detected." if n == 0 else f"{n} face(s) detected."
    return {"ok": True, "image": ann, "count": n, "boxes": [list(map(int, f)) for f in faces],
            "message": msg, "summary": msg}
