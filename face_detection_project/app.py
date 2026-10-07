"""Comparative Face Detection and Recognition - Streamlit app.
Run with:  streamlit run app.py
"""
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules import facenet_module, deepface_module, template_matching, viola_jones
from content import METHODS, COMPARISON, VIVA

BASE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(BASE, "database", "reference_faces")
RESULT_DIR = os.path.join(BASE, "results")
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

st.set_page_config(page_title="Face Detection & Recognition Comparison", page_icon="🧑", layout="wide")


def decode(uploaded):
    """Uploaded file -> RGB numpy array, or None if it is not a valid image."""
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def safe_run(fn, *args):
    """Run one method; if it crashes, return the error instead of stopping the app."""
    try:
        return fn(*args)
    except Exception as e:  # dependency, download or model errors land here
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def load_references():
    refs = []
    for f in sorted(os.listdir(REF_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(os.path.join(REF_DIR, f))
            if img is not None:
                refs.append((f, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return refs


def show_error(res):
    st.error(res["error"])


# ---------------- Header ----------------
st.title("Comparative Face Detection and Recognition")
st.caption("FaceNet • DeepFace • Template Matching • Viola-Jones")
st.write("Upload an image and the app runs all four techniques independently so you can compare their real outputs.")

# ---------------- Sidebar: reference database + template ----------------
with st.sidebar:
    st.header("Reference faces (database)")
    ref_files = st.file_uploader("Upload reference face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if ref_files and st.button("Save to database folder"):
        for rf in ref_files:
            with open(os.path.join(REF_DIR, rf.name), "wb") as fh:
                fh.write(rf.getvalue())
        st.success(f"Saved {len(ref_files)} file(s).")
    st.header("Template image")
    tmpl_file = st.file_uploader("Upload template (for Template Matching)", type=["jpg", "jpeg", "png"], key="tmpl")

references = load_references()
for rf in ref_files or []:  # uploaded refs are used immediately, even if not saved
    img = decode(rf)
    if img is not None and rf.name not in [n for n, _ in references]:
        references.append((rf.name, img))
st.sidebar.write(f"Reference faces in use: **{len(references)}**")

# ---------------- Main upload ----------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
tab_main, tab_about, tab_viva = st.tabs(["Results", "About the Methods", "Viva Questions"])

with tab_main:
    if uploaded is None:
        st.info("Please upload an image to begin.")
    else:
        image = decode(uploaded)
        if image is None:
            st.error("Invalid image file. Please upload a valid JPG or PNG.")
        else:
            st.subheader("Original Image")
            st.image(image, width=450)

            template = None
            if tmpl_file is not None:
                template = decode(tmpl_file)
                if template is None:
                    st.warning("Template file is not a valid image.")

            with st.spinner("Running all four methods (first run downloads models, please wait)..."):
                r_fn = safe_run(facenet_module.run, image, references)
                r_df = safe_run(deepface_module.run, image, references)
                r_tm = safe_run(template_matching.run, image, template)
                r_vj = safe_run(viola_jones.run, image)

            # ----- FaceNet -----
            with st.container(border=True):
                st.subheader("1. FaceNet Result")
                if not r_fn["ok"]:
                    show_error(r_fn)
                else:
                    c1, c2 = st.columns(2)
                    c1.image(r_fn["image"], caption="Detected face(s)")
                    c2.image(r_fn["face"], caption="Cropped face", width=200)
                    e = r_fn["embedding"]
                    st.write(f"Embedding: {e.shape[0]}-dimensional vector. First 8 values: {np.round(e[:8], 3).tolist()}")
                    if r_fn["matches"]:
                        st.dataframe(pd.DataFrame(r_fn["matches"]), use_container_width=True)
                    st.success(r_fn["message"])

            # ----- DeepFace -----
            with st.container(border=True):
                st.subheader("2. DeepFace Result")
                if not r_df["ok"]:
                    show_error(r_df)
                else:
                    c1, c2 = st.columns(2)
                    c1.image(r_df["image"], caption="Detected face(s)")
                    c2.image(r_df["face"], caption="Cropped face", width=200)
                    st.write(f"Recognition model: {r_df['model']}")
                    if r_df["matches"]:
                        st.dataframe(pd.DataFrame(r_df["matches"]), use_container_width=True)
                    st.write("Attribute estimates (approximate, may be inaccurate):", r_df["attributes"])
                    st.success(r_df["message"])

            # ----- Template Matching -----
            with st.container(border=True):
                st.subheader("3. Template Matching Result")
                st.caption("Template matching searches for a similar image pattern. It is not the same as modern face recognition.")
                if not r_tm["ok"]:
                    show_error(r_tm)
                else:
                    c1, c2 = st.columns(2)
                    c1.image(r_tm["template"], caption="Template", width=200)
                    c2.image(r_tm["image"], caption="Best matching area")
                    st.metric("Matching score (TM_CCOEFF_NORMED)", f"{r_tm['score']:.3f}")
                    st.write(f"Location (top-left): {r_tm['location']}")
                    (st.success if r_tm["found"] else st.warning)(r_tm["message"])

            # ----- Viola-Jones -----
            with st.container(border=True):
                st.subheader("4. Viola-Jones Result")
                if not r_vj["ok"]:
                    show_error(r_vj)
                else:
                    st.image(r_vj["image"], caption="Haar Cascade detections")
                    st.metric("Faces detected", r_vj["count"])
                    st.write("Bounding boxes (x, y, w, h):", r_vj["boxes"])

            # save a copy of annotated outputs
            for key, r in (("facenet", r_fn), ("deepface", r_df), ("template", r_tm), ("viola_jones", r_vj)):
                if r["ok"]:
                    cv2.imwrite(os.path.join(RESULT_DIR, f"{key}_result.png"), cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR))

            # ----- Comparison table (Result column filled from real outputs) -----
            st.subheader("Comparison Table")
            rows = []
            for name, r in (("FaceNet", r_fn), ("DeepFace", r_df), ("Template Matching", r_tm), ("Viola-Jones", r_vj)):
                t, p, a, l = COMPARISON[name]
                rows.append({"Method": name, "Type": t, "Main Purpose": p,
                             "Result": r["summary"] if r["ok"] else f"FAILED: {r['error']}",
                             "Advantages": a, "Limitations": l})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_about:
    for name, m in METHODS.items():
        with st.expander(name, expanded=False):
            st.markdown(f"**What it is:** {m['what']}\n\n**How it works:** {m['how']}\n\n"
                        f"**Input:** {m['input']}\n\n**Output:** {m['output']}\n\n"
                        f"**Advantages:** {m['adv']}\n\n**Limitations:** {m['lim']}\n\n"
                        f"**Real-world applications:** {m['apps']}")

with tab_viva:
    for i, (q, a) in enumerate(VIVA, 1):
        with st.expander(f"Q{i}. {q}"):
            st.write(a)
