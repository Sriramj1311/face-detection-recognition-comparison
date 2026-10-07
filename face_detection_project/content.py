"""Static text for the 'About the Methods', comparison table and 'Viva Questions' sections."""

METHODS = {
    "FaceNet": dict(
        what="A deep CNN (Google, 2015) that converts a face into a 512-number vector called an embedding.",
        how="Face is detected (MTCNN), cropped to 160x160 and passed through the network. Trained with triplet loss so photos of the same person give nearby vectors. Two faces are compared by cosine similarity or Euclidean distance.",
        input="Face image (cropped).", output="Embedding vector + similarity score.",
        adv="Very accurate; one embedding can be compared to thousands quickly.",
        lim="Needs a pretrained model (large download), heavy on CPU, needs reference images to 'recognise', can be biased by training data.",
        apps="Phone unlock, attendance systems, photo-album grouping."),
    "DeepFace": dict(
        what="A Python library that wraps several detectors (OpenCV, MTCNN, RetinaFace...) and recognition models (VGG-Face, Facenet, ArcFace...).",
        how="Detects the face, aligns it, creates an embedding with the chosen model and measures the distance between two embeddings. Verified = distance below the model's threshold. Can also estimate age, gender and emotion.",
        input="Image(s).", output="Face box, verified yes/no, distance, optional age/gender/emotion estimates.",
        adv="Easy API, many models, extra facial analysis.",
        lim="Large dependencies (TensorFlow); attribute estimates are only approximate and can be wrong; slower on first run.",
        apps="Identity verification, customer analytics, research prototypes."),
    "Template Matching": dict(
        what="A classical OpenCV technique that looks for a small template image inside a bigger image.",
        how="cv2.matchTemplate slides the template over every position and computes a similarity score (here normalised cross-correlation). The highest score is the best match location.",
        input="Main image + template image.", output="Best location (x, y), rectangle, score.",
        adv="Simple, fast, no training.",
        lim="Not scale, rotation or lighting invariant; does not understand faces, so it is NOT face recognition.",
        apps="Finding logos/icons, industrial part inspection, game bots."),
    "Viola-Jones": dict(
        what="A classical real-time face detector (2001) using Haar-like features and a cascade of classifiers.",
        how="Integral images make Haar features fast to compute. AdaBoost selects the best features. A cascade rejects non-face windows early; detectMultiScale slides windows at many scales.",
        input="Grayscale image.", output="Bounding boxes of faces (no identity).",
        adv="Very fast, runs on CPU, no extra download in OpenCV.",
        lim="Only mostly frontal faces; sensitive to lighting/angle; more false positives than deep detectors; detection only.",
        apps="Camera autofocus, basic face counting, early webcam apps."),
}

COMPARISON = {
    "FaceNet": ("Deep Learning", "Face recognition", "Accurate embeddings, easy similarity search", "Heavy model, needs reference faces"),
    "DeepFace": ("Deep Learning", "Face verification/analysis", "Many models, age/gender/emotion", "Large dependencies, estimates not always reliable"),
    "Template Matching": ("Image Processing", "Pattern matching", "Simple and fast", "Not scale/rotation invariant, not face recognition"),
    "Viola-Jones": ("Classical Computer Vision", "Face detection", "Very fast, CPU only", "Frontal faces only, no identification"),
}

VIVA = [
    ("What is the difference between face detection and face recognition?",
     "Detection finds WHERE a face is (a box). Recognition decides WHO it is by comparing it with known faces."),
    ("What is FaceNet?", "A CNN that maps a face to a 128/512-d embedding trained with triplet loss so the same person's faces are close together."),
    ("What is a face embedding?", "A fixed-length numeric vector that represents the identity features of a face."),
    ("What is triplet loss?", "It trains on (anchor, positive, negative) faces so anchor-positive distance is smaller than anchor-negative by a margin."),
    ("What is cosine similarity?", "cos(a,b) = a.b / (|a||b|). Range -1 to 1; closer to 1 means more similar direction, i.e. more likely the same person."),
    ("Euclidean distance vs cosine similarity?", "Euclidean measures straight-line distance (smaller = more similar); cosine measures angle (larger = more similar). On normalised vectors they are related."),
    ("What is DeepFace?", "A Python framework that wraps several detectors and recognition models and offers verify, find and analyze functions."),
    ("What does DeepFace.verify return?", "A dict with verified (True/False), distance, threshold, model and detector used."),
    ("How does template matching work?", "It slides a template over the image and computes a similarity score at every position; the maximum marks the best match."),
    ("Why is template matching not face recognition?", "It compares raw pixels at one size and angle, so changes in scale, pose, or lighting break it, and it has no concept of identity."),
    ("What is Viola-Jones?", "A real-time face detector using Haar-like features, integral images, AdaBoost and a cascade of classifiers."),
    ("What is a Haar Cascade?", "An XML file holding trained Viola-Jones classifier stages, loaded in OpenCV with cv2.CascadeClassifier."),
    ("What do scaleFactor and minNeighbors do in detectMultiScale?", "scaleFactor sets how much the image is shrunk per pass; minNeighbors sets how many overlapping detections are needed to accept a face."),
    ("Why convert to grayscale for Viola-Jones?", "Haar features use brightness differences, so colour is unnecessary and grayscale is faster."),
    ("What is a CNN?", "A neural network using convolution filters to learn visual features (edges, textures, parts) automatically from images."),
    ("What is OpenCV?", "An open-source computer vision library with image processing, detection and ML functions, used here for I/O, drawing, Haar cascades and template matching."),
    ("Which method here is a deep-learning method and which is classical?", "FaceNet and DeepFace are deep learning; Template Matching and Viola-Jones are classical."),
    ("What is a threshold in recognition?", "A cut-off on similarity/distance deciding match vs non-match; raising it reduces false accepts but increases false rejects."),
]
