# Comparative Face Detection and Recognition
FaceNet • DeepFace • Template Matching • Viola-Jones (Streamlit app)

## Install and run (Windows, Python 3.10 or 3.11 recommended)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
# Comparative Face Detection and Recognition
FaceNet • DeepFace • Template Matching • Viola-Jones (Streamlit app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-detection-recognition-comparison-f2fwckzyzkqrc5z754vbez.streamlit.app/)

**Live demo:** https://face-detection-recognition-comparison-f2fwckzyzkqrc5z754vbez.streamlit.app/

> Note: the first upload may take a while because the FaceNet and DeepFace models download on first use. The free hosted version can be slow or restart under memory limits, so for best results run it locally (see below).
## How to use
1. (Optional) In the sidebar upload reference faces (and click "Save to database folder") - needed for recognition/verification.
2. (Optional) Upload a template image - needed for Template Matching (e.g. a cropped face from the main image).
3. Upload the main image. All four methods run automatically; each section shows its own result or its own error.

The first run downloads model weights (FaceNet ~100 MB, DeepFace VGG-Face ~500 MB, so an internet connection is needed once).

## Structure
app.py (UI) | content.py (text) | modules/ (one file per method) | models/ (optional Haar XML) | database/reference_faces/ | results/ (saved outputs)

## How each technique works
- **FaceNet**: MTCNN finds the face, InceptionResnetV1 turns it into a 512-d embedding, cosine similarity (threshold 0.70) to reference embeddings decides match.
- **DeepFace**: detector finds the face, DeepFace.verify (VGG-Face) gives distance vs threshold, DeepFace.analyze estimates age/gender/emotion.
- **Template Matching**: cv2.matchTemplate (TM_CCOEFF_NORMED) slides the template; best score gives the location. Not face recognition.
- **Viola-Jones**: grayscale + Haar cascade + detectMultiScale gives face boxes.
Full explanations and 18 viva Q&As are inside the app (tabs "About the Methods" and "Viva Questions", source in content.py).

## Troubleshooting
- **Haar XML missing**: the app uses OpenCV's built-in copy. To copy it into models/: `python -c "import cv2,shutil;shutil.copy(cv2.data.haarcascades+'haarcascade_frontalface_default.xml','models/')"`
- **ModuleNotFoundError**: activate the venv, then `pip install -r requirements.txt` again.
- **NumPy 2 errors / "numpy.core.multiarray failed to import"**: `pip install "numpy<2" --force-reinstall`
- **Keras 3 error in DeepFace** ("use tf-keras"): `pip install tf-keras`
- **facenet-pytorch installs an old torch / conflicts**: `pip install torch torchvision` then `pip install facenet-pytorch --no-deps`
- **TensorFlow fails on Python 3.12+**: use Python 3.10 or 3.11.
- **Model download fails**: check internet/proxy and retry; DeepFace weights go to `C:\Users\<you>\.deepface\weights`, FaceNet to `~\.cache\torch\checkpoints`.
- **"Face could not be detected"**: use a clear, front-facing, well-lit photo.
- **Template score is low**: template must be cut from the same image/scale; it cannot handle size or rotation changes.
