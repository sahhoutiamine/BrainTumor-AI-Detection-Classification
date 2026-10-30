
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO


CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.set_page_config(page_title="Brain Tumor AI", layout="wide")
st.title("Brain Tumor AI — MRI Analysis")
st.caption("Classification and detection using Deep Learning")


@st.cache_resource
def charger_modeles():
    """Load saved models only once per session."""

    df = pd.read_csv("reports/comparaison_step4.csv")
    meilleur = df.sort_values("f1_macro", ascending=False).iloc[0]

    nom = meilleur["modele"]
    config = meilleur["configuration"]
    phase = meilleur["phase"].lower().replace("-", "")

    chemin = f"models/classification/{nom}_{config}_{phase}.keras"

    classification = tf.keras.models.load_model(chemin)
    detection = YOLO("models/detection/yolo_best.pt")

    return classification, detection, nom


def classifier(image, modele, nom):
    """Prepare MRI and predict its class."""

    image = image.resize((224, 224))
    image = np.array(image).astype("float32")

    if nom == "MobileNetV2":
        image = image / 127.5 - 1.0
    else:
        image = tf.keras.applications.resnet50.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return modele.predict(image, verbose=0)[0]


classification, detection, nom = charger_modeles()

st.write("Classification model:", nom)

onglet1, onglet2 = st.tabs(["MRI Prediction", "Model Results"])

with onglet1:
    fichier = st.file_uploader(
        "Upload an MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if fichier:
        image = Image.open(fichier).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original MRI")
            st.image(image, use_container_width=True)

        # Classification
        probabilites = classifier(image, classification, nom)
        classe = CLASSES[np.argmax(probabilites)]

        with col2:
            st.subheader("Classification")
            st.success(f"Predicted class: {classe}")

            for nom_classe, proba in zip(CLASSES, probabilites):
                st.write(f"{nom_classe}: {proba:.1%}")
                st.progress(float(proba))

        # YOLO detection
        st.subheader("YOLOv8 Detection")

        image_cv = cv2.cvtColor(
            np.array(image), cv2.COLOR_RGB2BGR
        )

        resultat = detection.predict(
            image_cv, conf=0.25, verbose=False
        )[0]

        image_predite = cv2.cvtColor(
            resultat.plot(), cv2.COLOR_BGR2RGB
        )

        st.image(image_predite, use_container_width=True)

        if len(resultat.boxes) == 0:
            st.info("No tumor region detected.")
        else:
            for box in resultat.boxes:
                classe_id = int(box.cls.item())
                confiance = float(box.conf.item())

                st.write(
                    f"{resultat.names[classe_id]}: "
                    f"{confiance:.1%} confidence"
                )


with onglet2:
    st.subheader("Classification Metrics")

    fichier = Path("reports/classification_metrics.csv")

    if fichier.exists():
        st.dataframe(pd.read_csv(fichier), hide_index=True)

    matrice = Path("reports/classification_confusion.png")

    if matrice.exists():
        st.image(str(matrice), caption="Classification Confusion Matrix")

    st.subheader("YOLOv8 Results")

    for nom_image in ["confusion_matrix.png", "PR_curve.png"]:
        chemin = Path("reports/detection/metrics") / nom_image

        if chemin.exists():
            st.image(str(chemin), caption=nom_image)

    erreurs = Path("reports/detection/errors.csv")

    if erreurs.exists():
        st.write("Detection errors")
        st.dataframe(pd.read_csv(erreurs), hide_index=True)

st.caption("Educational project — not intended for clinical diagnosis.")
