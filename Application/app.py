import os
import streamlit as st
from PIL import Image
from detect import GarbageDetector
from rag import RecyclingRAG
from llm_formatter import LLMFormatter

st.set_page_config(page_title="Garbage Detection + Recycling RAG", layout="centered")

st.title("Multi-Class Garbage Detection with Recycling Recommendation")
st.write("Upload a waste image to detect waste classes and get recycling guidance.")

@st.cache_resource
def load_detector():
    return GarbageDetector("Garbage-detection.pt")

@st.cache_resource
def load_rag():
    return RecyclingRAG("knowledge_base")

detector = load_detector()
rag = load_rag()

os.makedirs("uploads", exist_ok=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
confidence = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    save_path = os.path.join("uploads", uploaded_file.name)
    image.save(save_path)

    st.subheader("Uploaded Image")
    st.image(image, width=500)

    detected, annotated_image = detector.detect(save_path, conf=confidence)

    if detected:
        st.subheader("Detected Waste Objects")

        unique_classes = {}
        for cls, score in detected:
            if cls not in unique_classes or score > unique_classes[cls]:
                unique_classes[cls] = score

        for cls, score in unique_classes.items():
            st.write(f"**{cls}** — Confidence: {score:.3f}")

        st.subheader("Detected Image with Bounding Boxes")
        st.image(annotated_image, width=500)

        st.subheader("Recycling Recommendations")

        # load formatter only here, after UI is already visible
        try:
            formatter = LLMFormatter()

            for cls in unique_classes.keys():
                context = rag.retrieve_context(cls, top_k=3)
                result = formatter.format_recycling_response(cls, context)

                st.markdown(f"### {cls}")
                st.markdown("**Why recycling this material is useful:**")
                st.write(result["usefulness"])

                st.markdown("**Steps:**")
                for step in result["steps"]:
                    st.write(step)

        except Exception as e:
            st.error(f"LLM formatting error: {e}")
            st.info("Showing retrieved context instead.")
            for cls in unique_classes.keys():
                context = rag.retrieve_context(cls, top_k=3)
                st.markdown(f"### {cls}")
                st.write(context)

    else:
        st.warning("No waste objects detected.")