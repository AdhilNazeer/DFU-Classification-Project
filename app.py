
import streamlit as st
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DFU Classification System",
    page_icon="🩺",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "dfu_grade_model_best.keras"

OUTPUT_DIR = PROJECT_DIR / "outputs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
REPORT_DIR = OUTPUT_DIR / "patient_reports"

GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (300, 300)

CLASS_NAMES = [
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
]


# ============================================================
# 2. MODEL LOADING
# ============================================================

@st.cache_resource
def load_dfu_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# 3. IMAGE PREPROCESSING AND PREDICTION
# ============================================================

def preprocess_image(image):
    """
    Resize an RGB image to the model's expected input size.

    This assumes training used RGB pixel values in the 0-255
    range and that preprocessing is handled inside the model.
    Verify this against the original training notebook.
    """

    resized = image.convert("RGB").resize(IMG_SIZE)

    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_dfu(image):
    model = load_dfu_model()
    image_batch = preprocess_image(image)

    scores = model.predict(image_batch, verbose=0)[0]

    predicted_index = int(np.argmax(scores))

    return (
        CLASS_NAMES[predicted_index],
        float(scores[predicted_index]),
        scores,
    )


# ============================================================
# 4. GRAD-CAM FOR NESTED EFFICIENTNETB3
# ============================================================

def generate_gradcam(model, image, class_index=None):
    """
    Generate Grad-CAM from the nested EfficientNetB3 backbone.

    The feature map from 'top_conv' is passed through the
    remaining outer-model classification layers.
    """

    backbone = model.get_layer("efficientnetb3")
    last_conv_layer = backbone.get_layer("top_conv")

    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            last_conv_layer.output,
            backbone.output,
        ],
    )

    backbone_index = model.layers.index(backbone)
    head_layers = model.layers[backbone_index + 1:]

    image_batch = preprocess_image(image)

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(
            image_batch,
            training=False,
        )

        predictions = features

        for layer in head_layers:
            predictions = layer(
                predictions,
                training=False,
            )

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients are unavailable. "
            "Check the model architecture and classifier head."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights,
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.cond(
        maximum > 0,
        lambda: heatmap / maximum,
        lambda: heatmap,
    )

    return heatmap.numpy()


def create_gradcam_overlay(image, heatmap):
    """Blend the Grad-CAM heatmap with the original RGB image."""

    original = image.convert("RGB").resize(IMG_SIZE)

    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize(
        original.size,
        Image.Resampling.BILINEAR,
    )

    heatmap_array = (
        np.asarray(heatmap_image, dtype=np.float32) / 255.0
    )

    colored_heatmap = cm.get_cmap("jet")(
        heatmap_array
    )[..., :3]

    original_array = (
        np.asarray(original, dtype=np.float32) / 255.0
    )

    overlay = np.clip(
        0.6 * original_array + 0.4 * colored_heatmap,
        0,
        1,
    )

    return Image.fromarray(
        np.uint8(overlay * 255)
    )


def image_to_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# 5. PATIENT PDF GENERATION
# ============================================================

def generate_patient_pdf(
    patient_details,
    foot_image,
    predicted_grade,
    confidence,
    scores=None,
    gradcam_image=None,
):
    """Generate an in-memory patient assessment PDF."""

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph(
            "DFU Patient Assessment Report",
            styles["Title"],
        )
    )

    content.append(Spacer(1, 10))

    content.append(
        Paragraph(
            f"Generated: {datetime.now():%d-%m-%Y %H:%M}",
            styles["Normal"],
        )
    )

    content.append(Spacer(1, 16))

    content.append(
        Paragraph("Patient Details", styles["Heading2"])
    )

    patient_data = [
        ["Field", "Details"],
        ["Patient ID", str(patient_details.get("id", "N/A"))],
        ["Patient Name", str(patient_details.get("name", "N/A"))],
        ["Age", str(patient_details.get("age", "N/A"))],
        ["Gender", str(patient_details.get("gender", "N/A"))],
    ]

    patient_table = Table(
        patient_data,
        colWidths=[130, 330],
    )

    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    content.append(patient_table)
    content.append(Spacer(1, 18))

    content.append(
        Paragraph("AI Model Output", styles["Heading2"])
    )

    content.append(
        Paragraph(
            f"Predicted grade: <b>{predicted_grade}</b>",
            styles["Normal"],
        )
    )

    content.append(
        Paragraph(
            f"Model output score: <b>{confidence:.2%}</b>",
            styles["Normal"],
        )
    )

    content.append(Spacer(1, 12))

    if scores is not None:
        content.append(
            Paragraph("Scores by Grade", styles["Heading3"])
        )

        score_data = [["Grade", "Model output score"]]

        for grade, score in zip(CLASS_NAMES, scores):
            score_data.append([grade, f"{float(score):.2%}"])

        score_table = Table(
            score_data,
            colWidths=[130, 330],
        )

        score_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 7),
            ])
        )

        content.append(score_table)
        content.append(Spacer(1, 18))

    content.append(
        Paragraph("Uploaded Foot Image", styles["Heading2"])
    )

    foot_buffer = BytesIO()
    foot_image.save(foot_buffer, format="JPEG")
    foot_buffer.seek(0)

    content.append(
        PDFImage(
            foot_buffer,
            width=3.0 * inch,
            height=3.0 * inch,
            kind="proportional",
        )
    )

    if gradcam_image is not None:
        content.append(Spacer(1, 14))

        content.append(
            Paragraph("Grad-CAM Visualization", styles["Heading2"])
        )

        heatmap_buffer = BytesIO()
        gradcam_image.save(heatmap_buffer, format="PNG")
        heatmap_buffer.seek(0)

        content.append(
            PDFImage(
                heatmap_buffer,
                width=3.0 * inch,
                height=3.0 * inch,
                kind="proportional",
            )
        )

        content.append(Spacer(1, 8))

        content.append(
            Paragraph(
                "The heatmap visualizes image regions influencing "
                "the model output. It does not identify a confirmed "
                "ulcer boundary or establish a diagnosis.",
                styles["Normal"],
            )
        )

    content.append(Spacer(1, 18))

    content.append(
        Paragraph(
            "Important Medical Disclaimer",
            styles["Heading2"],
        )
    )

    content.append(
        Paragraph(
            "This is an experimental AI research prototype. "
            "The predicted grade and score are not a confirmed "
            "medical diagnosis and must not be used alone to make "
            "treatment decisions. Consult a qualified healthcare "
            "professional for assessment and care.",
            styles["Normal"],
        )
    )

    document.build(content)

    return pdf_buffer.getvalue()


# ============================================================
# 6. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "patient_details": None,
    "predicted_grade": None,
    "confidence": None,
    "class_probabilities": None,
    "prediction_image": None,
    "gradcam_image": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🩺 DFU Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Evaluation", "Grad-CAM", "Patient PDF"],
)

st.sidebar.divider()

st.sidebar.caption(
    "AI-Powered Diabetic Foot Ulcer Classification"
)


# ============================================================
# 8. HOME
# ============================================================

if page == "Home":

    st.title("AI-Powered DFU Classification System")

    st.write(
        "A deep-learning research prototype for classifying "
        "diabetic foot ulcer images into four grades."
    )

    st.subheader("Project Overview")

    st.write(
        "The system uses an EfficientNetB3 image classification "
        "model to predict a grade from an uploaded foot image. "
        "The dashboard provides image evaluation, Grad-CAM "
        "visualization, and patient assessment PDF generation."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "EfficientNetB3")
    col2.metric("Input Size", "300 × 300")
    col3.metric("Output Classes", "4")

    st.subheader("Dashboard Modules")

    st.markdown("""
    1. **Evaluation:** Enter patient details, upload an image, and run the model.
    2. **Grad-CAM:** Visualize model-related activation regions.
    3. **Patient PDF:** Generate a downloadable assessment report.
    """)

    st.info(
        "This project is for research and demonstration, not "
        "for independent medical diagnosis or treatment decisions."
    )


# ============================================================
# 9. EVALUATION
# ============================================================

elif page == "Evaluation":

    st.title("AI Foot Image Evaluation")

    st.subheader("1. Patient Details")

    with st.form("patient_form"):

        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Patient Name")

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
        )

        gender = st.selectbox(
            "Gender",
            ["Prefer not to say", "Female", "Male", "Other"],
        )

        save_patient = st.form_submit_button(
            "Save Patient Details"
        )

        if save_patient:
            if not patient_id.strip() or not patient_name.strip():
                st.error("Please enter the patient ID and name.")
            else:
                st.session_state["patient_details"] = {
                    "id": patient_id.strip(),
                    "name": patient_name.strip(),
                    "age": age,
                    "gender": gender,
                }

                st.success("Patient details saved.")

    st.subheader("2. Upload Foot Image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="evaluation_image",
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded foot image",
                width=400,
            )

            if st.button("Run AI Model", type="primary"):

                if st.session_state["patient_details"] is None:
                    st.warning(
                        "Save patient details before running the model."
                    )

                else:
                    try:
                        with st.spinner("Analyzing foot image..."):
                            grade, score, scores = predict_dfu(image)

                        st.session_state["predicted_grade"] = grade
                        st.session_state["confidence"] = score
                        st.session_state["class_probabilities"] = [
                            float(x) for x in scores
                        ]
                        st.session_state["prediction_image"] = image.copy()
                        st.session_state["gradcam_image"] = None

                        st.success("Prediction completed.")

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

        except Exception as error:
            st.error(f"Could not open the uploaded image: {error}")

    if st.session_state["predicted_grade"] is not None:

        st.divider()
        st.subheader("3. Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted DFU Grade",
            st.session_state["predicted_grade"],
        )

        col2.metric(
            "Model Output Score",
            f"{st.session_state['confidence']:.2%}",
        )

        st.caption(
            "This score is the model's output for its selected class; "
            "it is not a calibrated probability of clinical correctness."
        )

        st.subheader("Scores for Each Grade")

        for grade, score in zip(
            CLASS_NAMES,
            st.session_state["class_probabilities"],
        ):
            st.write(f"**{grade}: {score:.2%}**")
            st.progress(float(score))

        st.warning(
            "This experimental model is not a medical diagnostic tool. "
            "Seek assessment from a qualified healthcare professional."
        )


# ============================================================
# 10. GRAD-CAM
# ============================================================

elif page == "Grad-CAM":

    st.title("Grad-CAM Visualization")

    st.write(
        "Visualize regions that influence the model's predicted class."
    )

    prediction_image = st.session_state["prediction_image"]
    predicted_grade = st.session_state["predicted_grade"]
    confidence = st.session_state["confidence"]

    if prediction_image is None or predicted_grade is None:

        st.info(
            "Go to Evaluation, upload an image, and run the AI model first."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                prediction_image,
                use_container_width=True,
            )

        with col2:
            st.subheader("Prediction")
            st.metric("Predicted Grade", predicted_grade)

            if confidence is not None:
                st.metric(
                    "Model Output Score",
                    f"{confidence:.2%}",
                )

        if st.button(
            "Generate Grad-CAM Heatmap",
            type="primary",
        ):

            try:
                with st.spinner("Generating Grad-CAM..."):

                    model = load_dfu_model()

                    heatmap = generate_gradcam(
                        model,
                        prediction_image,
                    )

                    overlay = create_gradcam_overlay(
                        prediction_image,
                        heatmap,
                    )

                    st.session_state["gradcam_image"] = overlay

                    output_path = (
                        GRADCAM_DIR / "latest_gradcam.png"
                    )

                    overlay.save(output_path)

                st.success("Grad-CAM generated successfully.")

            except Exception as error:
                st.error(f"Grad-CAM generation failed: {error}")

        gradcam_image = st.session_state["gradcam_image"]

        if gradcam_image is not None:

            st.subheader("Grad-CAM Overlay")

            st.image(
                gradcam_image,
                caption="Original image with Grad-CAM overlay",
                use_container_width=True,
            )

            st.download_button(
                label="Download Grad-CAM Image",
                data=image_to_png_bytes(gradcam_image),
                file_name="dfu_gradcam.png",
                mime="image/png",
            )

            st.caption(
                "The heatmap visualizes model-related activation. "
                "It does not establish an ulcer boundary or diagnosis."
            )


# ============================================================
# 11. PATIENT PDF
# ============================================================

elif page == "Patient PDF":

    st.title("Patient Assessment PDF")

    patient_details = st.session_state["patient_details"]
    predicted_grade = st.session_state["predicted_grade"]
    confidence = st.session_state["confidence"]
    prediction_image = st.session_state["prediction_image"]
    scores = st.session_state["class_probabilities"]
    gradcam_image = st.session_state["gradcam_image"]

    if patient_details is None:

        st.info(
            "Save patient details on the Evaluation page first."
        )

    elif predicted_grade is None or prediction_image is None:

        st.info(
            "Run a model prediction on the Evaluation page first."
        )

    else:

        st.subheader("Report Preview")

        st.write(f"**Patient ID:** {patient_details['id']}")
        st.write(f"**Patient Name:** {patient_details['name']}")
        st.write(f"**Age:** {patient_details['age']}")
        st.write(f"**Predicted Grade:** {predicted_grade}")
        st.write(f"**Model Output Score:** {confidence:.2%}")

        st.image(
            prediction_image,
            caption="Foot image for the report",
            width=300,
        )

        if gradcam_image is not None:
            st.image(
                gradcam_image,
                caption="Grad-CAM image for the report",
                width=300,
            )

        if st.button(
            "Generate Patient PDF",
            type="primary",
        ):

            try:
                pdf_bytes = generate_patient_pdf(
                    patient_details=patient_details,
                    foot_image=prediction_image,
                    predicted_grade=predicted_grade,
                    confidence=confidence,
                    scores=scores,
                    gradcam_image=gradcam_image,
                )

                safe_patient_id = "".join(
                    char for char in patient_details["id"]
                    if char.isalnum() or char in "-_"
                )

                filename = (
                    f"DFU_Report_{safe_patient_id}_"
                    f"{datetime.now():%Y%m%d_%H%M%S}.pdf"
                )

                # Save a copy in the project output directory.
                saved_report_path = REPORT_DIR / filename
                saved_report_path.write_bytes(pdf_bytes)

                st.success("Patient assessment report generated.")

                st.download_button(
                    label="Download Patient Assessment PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                )

                st.caption(
                    f"A copy was saved to: {saved_report_path}"
                )

            except Exception as error:
                st.error(f"PDF generation failed: {error}")
