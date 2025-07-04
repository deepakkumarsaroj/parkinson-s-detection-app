import streamlit as st
import numpy as np
import joblib
import cv2
from skimage.feature import hog
import os
import gdown

# Google Drive model link and local filename
MODEL_FILE = "parkinson_knn_model.pkl"
FILE_ID = "1--rgHv5E4qMLeK_0kCODr2jhlCHeDo9b"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model file if not already present
if not os.path.exists(MODEL_FILE):
    st.info("Downloading model file from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

# Load all models
svm_model = joblib.load("parkinsons_svm_model.pkl")
dct_model = joblib.load("parkinson_dct_model.pkl")
knn_image_model = joblib.load("parkinson_knn_model.pkl")
knn_form_model = joblib.load("knn_model.pkl")

st.title("🧠 Parkinson's Disease Detection App")

tab1, tab2 = st.tabs(["Combined Prediction", "KNN Form Only"])

with tab1:
    st.header("Upload Image and Enter Features")

    with st.form("prediction_form"):
        st.subheader("SVM Features")
        svm_features = [
            'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT',
            'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]
        svm_values = [st.number_input(label, value=0.0) for label in svm_features]

        st.subheader("DCT Features")
        dct_features = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]
        dct_values = [st.number_input(label, value=0.0) for label in dct_features]

        image = st.file_uploader("Upload Drawing Image for KNN Model", type=["jpg", "jpeg", "png"])

        submitted = st.form_submit_button("Predict")

    if submitted and image is not None:
        file_bytes = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        img_resized = cv2.resize(img, (80, 80))

        if len(img_resized.shape) == 3:
            gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img_resized

        features = hog(gray_img)
        features = np.array(features).reshape(1, -1)

        result_svm = svm_model.predict([svm_values])[0]
        result_dct = dct_model.predict([dct_values])[0]
        result_knn_image = knn_image_model.predict(features)[0]

        combined_score = 0.3 * result_svm + 0.3 * result_dct + 0.4 * result_knn_image
        final_output = 1 if combined_score >= 0.6 else 0

        st.success(f"🎯 Final Output: {'Parkinson Detected' if final_output == 1 else 'No Parkinson'}")
        st.info(f"SVM: {result_svm}, DCT: {result_dct}, KNN (Image): {result_knn_image}")

with tab2:
    st.header("KNN Form-based Prediction")

    knn_form_features = [
        'Speech', 'Facial Expression', 'Rest Tremor (Face/Lips)', 'Rest Tremor (Right Hand)',
        'Rest Tremor (Left Hand)', 'Rest Tremor (Right Leg)', 'Rest Tremor (Left Leg)',
        'Action/Postural Tremor (Right Hand)', 'Action/Postural Tremor (Left Hand)', 'Rigidity (Neck)',
        'Rigidity (Right Arm)', 'Rigidity (Left Arm)', 'Rigidity (Right Leg)', 'Rigidity (Left Leg)',
        'Finger Tapping (Right)', 'Finger Tapping (Left)', 'Hand Movements (Right)', 'Hand Movements (Left)',
        'Pronation-Supination (Right)', 'Pronation-Supination (Left)', 'Toe Tapping (Right)',
        'Toe Tapping (Left)', 'Leg Agility (Right)', 'Leg Agility (Left)', 'Arising from Chair',
        'Posture', 'Gait', 'Postural Stability', 'Body Bradykinesia'
    ]

    with st.form("knn_form"):
        values = [st.slider(label, 0, 4, 0) for label in knn_form_features]
        submit_knn = st.form_submit_button("Predict with KNN Form")

    if submit_knn:
        total_sum = sum(values)
        knn_result = knn_form_model.predict(np.array([[total_sum]]))[0]
        st.success(f"KNN Form Prediction: {'Parkinson Detected' if knn_result == 1 else 'No Parkinson'}")
