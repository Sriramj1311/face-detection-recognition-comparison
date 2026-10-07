"""FaceNet (facenet-pytorch): MTCNN detects faces, InceptionResnetV1 makes 512-d embeddings."""
import numpy as np
import cv2

COSINE_THRESHOLD = 0.70  # similarity >= this => "same person" (adjustable)
_cache = {}


def _load():
    """Load models once (first run downloads pretrained weights)."""
    if not _cache:
        from facenet_pytorch import MTCNN, InceptionResnetV1
        _cache["mtcnn"] = MTCNN(keep_all=True, device="cpu")
        _cache["model"] = InceptionResnetV1(pretrained="vggface2").eval()
    return _cache["mtcnn"], _cache["model"]


def _detect(img_rgb):
    from PIL import Image
    mtcnn, _ = _load()
    boxes, probs = mtcnn.detect(Image.fromarray(img_rgb))
    return boxes, probs


def _embed(img_rgb, box):
    """Crop face -> resize 160x160 -> normalise -> 512-d L2-normalised embedding."""
    import torch
    _, model = _load()
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    crop = img_rgb[y1:y2, x1:x2]
    face = cv2.resize(crop, (160, 160))
    t = torch.from_numpy(((face.astype(np.float32) - 127.5) / 128.0)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        e = model(t)[0].numpy()
    return e / np.linalg.norm(e), crop


def run(img_rgb, references):
    boxes, probs = _detect(img_rgb)
    if boxes is None or len(boxes) == 0:
        return {"ok": False, "error": "No face detected in the uploaded image."}

    ann = img_rgb.copy()
    for b in boxes:
        cv2.rectangle(ann, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 120, 255), 3)

    best = int(np.argmax(probs))  # use the most confident face
    emb, crop = _embed(img_rgb, boxes[best])

    matches = []
    for name, ref_img in references:
        try:
            rb, rp = _detect(ref_img)
            if rb is None or len(rb) == 0:
                matches.append({"Reference": name, "Cosine similarity": None, "Euclidean": None, "Match": "No face in reference"})
                continue
            r_emb, _ = _embed(ref_img, rb[int(np.argmax(rp))])
            cos = float(np.dot(emb, r_emb))
            euc = float(np.linalg.norm(emb - r_emb))
            matches.append({"Reference": name, "Cosine similarity": round(cos, 3),
                            "Euclidean": round(euc, 3), "Match": "YES" if cos >= COSINE_THRESHOLD else "NO"})
        except Exception as e:
            matches.append({"Reference": name, "Cosine similarity": None, "Euclidean": None, "Match": f"Error: {e}"})

    if not references:
        msg = "Embedding generated. No reference faces provided, so no recognition was performed."
    else:
        hits = [m for m in matches if m["Match"] == "YES"]
        if hits:
            top = max(hits, key=lambda m: m["Cosine similarity"])
            msg = f"Matching face found: {top['Reference']} (cosine {top['Cosine similarity']})."
        else:
            msg = "No matching face found in the reference database."
    if len(boxes) > 1:
        msg += f" ({len(boxes)} faces found; the most confident one was used.)"

    return {"ok": True, "image": ann, "face": crop, "embedding": emb, "matches": matches,
            "message": msg, "summary": msg}
