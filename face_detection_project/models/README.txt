Optional: put haarcascade_frontalface_default.xml here.
If missing, the app automatically uses OpenCV's built-in copy (cv2.data.haarcascades).
Copy it with (Windows):
python -c "import cv2,shutil;shutil.copy(cv2.data.haarcascades+'haarcascade_frontalface_default.xml','models/')"
