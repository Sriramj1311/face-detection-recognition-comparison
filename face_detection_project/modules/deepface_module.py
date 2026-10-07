"""DeepFace: detection, verification against references, and attribute analysis."""
import cv2

MODEL = "VGG-Face"        # recognition model used by DeepFace.verify
DETECTOR = "opencv"       # fast detector backend


def run(img_rgb, references):
    from deepface import DeepFace  # imported here so a broken install only affects this section

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # DeepFace expects BGR arrays

    # 1. Detect faces
    faces = DeepFace.extract_faces(img_path=bgr, detector_backend=DETECTOR, enforce_detection=True)
    ann = img_rgb.copy()
    for f in faces:
        a = f["facial_area"]
        cv2.rectangle(ann, (a["x"], a["y"]), (a["x"] + a["w"], a["y"] + a["h"]), (255, 0, 255), 3)
    main = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
    a = main["facial_area"]
    crop = img_rgb[max(0, a["y"]):a["y"] + a["h"], max(0, a["x"]):a["x"] + a["w"]]

    # 2. Optional attribute analysis (estimates only, can be wrong)
    try:
        r = DeepFace.analyze(img_path=bgr, actions=["age", "gender", "emotion"],
                             detector_backend=DETECTOR, enforce_detection=False, silent=True)
        r = r[0] if isinstance(r, list) else r
        attrs = {"Estimated age": r.get("age"), "Dominant gender": r.get("dominant_gender"),
                 "Dominant emotion": r.get("dominant_emotion")}
    except Exception as e:
        attrs = {"Attribute analysis failed": str(e)}

    # 3. Verify against each reference image
    matches = []
    for name, ref in references:
        try:
            v = DeepFace.verify(img1_path=bgr, img2_path=cv2.cvtColor(ref, cv2.COLOR_RGB2BGR),
                                model_name=MODEL, detector_backend=DETECTOR, enforce_detection=True)
            matches.append({"Reference": name, "Verified": "YES" if v["verified"] else "NO",
                            "Distance": round(v["distance"], 4), "Threshold": round(v["threshold"], 4)})
        except Exception as e:
            matches.append({"Reference": name, "Verified": f"Error: {e}", "Distance": None, "Threshold": None})

    if not references:
        msg = f"{len(faces)} face(s) detected. No reference faces provided, so no verification was performed."
    else:
        hits = [m for m in matches if m["Verified"] == "YES"]
        msg = (f"Verified as same person as: {min(hits, key=lambda m: m['Distance'])['Reference']}."
               if hits else "No reference face verified as the same person.")
    if len(faces) > 1:
        msg += f" ({len(faces)} faces found; the largest was analysed.)"

    return {"ok": True, "image": ann, "face": crop, "attributes": attrs, "matches": matches,
            "model": MODEL, "message": msg, "summary": msg}
