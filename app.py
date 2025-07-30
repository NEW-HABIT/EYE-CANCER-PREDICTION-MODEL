import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile
from APIKEYS import value


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=value  
)


st.title(" Eye Cancer Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)


    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)

       
        with st.spinner("Running inference..."):
            result = CLIENT.infer(temp_file.name, model_id="my-first-project-t099v/1")
        
        st.success("Inference Complete!")

        
        st.subheader("Prediction Results")
        st.json(result)
