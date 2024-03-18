import base64
import streamlit as st
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "growpro-413910-a513a74d23fa.json"

st.title('Image Classification')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

project_id = '546899236073'
endpoint = '2716121375071797248'

@st.cache
def get_prediction_client():
    return aiplatform.gapic.PredictionServiceClient()

def predict_image_classification_sample(project: str, endpoint_id: str, image_content):
    client = get_prediction_client()

    with image_content as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]

    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=5,
    ).to_value()

    endpoint_path = client.endpoint_path(
        project=project, location="us-central1", endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint_path, instances=instances, parameters=parameters
    )

    return response.predictions

if uploaded_file is not None:
    if st.button('Predict'):
        predictions = predict_image_classification_sample(project_id, endpoint, uploaded_file)
        st.write("Predictions:")
        for prediction in predictions:
            st.write(prediction)
else:
    st.write("Please upload an image.")
