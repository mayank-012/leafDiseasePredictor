import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Set page title and favicon
st.set_page_config(page_title="AnnVriddhi", page_icon="🌿")

# Set page background color
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AnnVridhhi")
st.subheader("Plant Disease Detection")

test_image = st.file_uploader("Upload an Image:")

if st.button("Show Image"):
    if test_image is not None:
        st.image(test_image, width=128, use_column_width=True)
    else:
        st.warning("Please upload an image first.")

if st.button("Predict"):
    if test_image is not None:
        st.subheader("Prediction:")
        result_index = model_prediction(test_image)
        class_name = ['Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple',
                    'Healthy Blueberry', 'Cherry Powdery Mildew', 'Healthy Cherry',
                    'Corn Cercospora Leaf Spot Gray Leaf Spot', 'Corn Common Rust',
                    'Corn Northern Leaf Blight', 'Healthy Corn', 'Grape Black Rot',
                    'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)',
                    'Healthy Grape', 'Orange Haunglongbing (Citrus greening)',
                    'Peach Bacterial spot', 'Healthy Peach', 'Pepper, bell Bacterial spot',
                    'Healthy Pepper, bell', 'Potato Early blight', 'Potato Late blight',
                    'Healthy Potato', 'Healthy Raspberry', 'Healthy Soybean',
                    'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Healthy Strawberry',
                    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
                    'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
                    'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot',
                    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Healthy Tomato']
        remedy = [
            "Prune infected branches, apply fungicides",  # Apple Scab
            "Prune infected branches, apply fungicides",  # Black Rot
            "Remove cedar trees nearby, apply fungicides",  # Cedar Apple Rust
            "No specific remedy needed (healthy plant)",  # Apple healthy
            "No specific remedy needed (healthy plant)",  # Blueberry healthy
            "Improve air circulation, apply fungicides",  # Cherry Powdery Mildew
            "No specific remedy needed (healthy plant)",  # Cherry healthy
            "Apply fungicides, practice crop rotation",  # Corn Cercospora Leaf Spot Gray Leaf Spot
            "Apply fungicides, practice crop rotation",  # Corn Common Rust
            "Apply fungicides, practice crop rotation",  # Corn Northern Leaf Blight
            "No specific remedy needed (healthy plant)",  # Corn healthy
            "Prune infected branches, apply fungicides",  # Grape Black Rot
            "No specific remedy needed (healthy plant)",  # Grape Esca (Black Measles)
            "Apply fungicides, practice crop rotation",  # Grape Leaf blight (Isariopsis Leaf Spot)
            "No specific remedy needed (healthy plant)",  # Grape healthy
            "Remove infected trees, control insect vectors",  # Orange Haunglongbing (Citrus greening)
            "Apply copper-based fungicides, practice crop rotation",  # Peach Bacterial spot
            "No specific remedy needed (healthy plant)",  # Peach healthy
            "Apply copper-based fungicides, practice crop rotation",  # Pepper, bell Bacterial spot
            "No specific remedy needed (healthy plant)",  # Pepper, bell healthy
            "Remove infected plant parts, apply fungicides",  # Potato Early blight
            "Remove infected plant parts, apply fungicides",  # Potato Late blight
            "No specific remedy needed (healthy plant)",  # Potato healthy
            "No specific remedy needed (healthy plant)",  # Raspberry healthy
            "No specific remedy needed (healthy plant)",  # Soybean healthy
            "Improve air circulation, apply fungicides",  # Squash Powdery mildew
            "Control aphids and mites, ensure proper watering",  # Strawberry Leaf scorch
            "No specific remedy needed (healthy plant)",  # Strawberry healthy
            "Apply copper-based fungicides, control insect vectors",  # Tomato Bacterial spot
            "Remove infected plant parts, apply fungicides",  # Tomato Early blight
            "Remove infected plant parts, apply fungicides",  # Tomato Late blight
            "Apply fungicides, improve air circulation",  # Tomato Leaf Mold
            "Apply fungicides, practice crop rotation",  # Tomato Septoria leaf spot
            "Apply miticides, practice crop rotation",  # Tomato Spider mites Two-spotted spider mite
            "Apply fungicides, practice crop rotation",  # Tomato Target Spot
            "Remove infected plants, control whiteflies",  # Tomato Tomato Yellow Leaf Curl Virus
            "Remove infected plants, control insect vectors",  # Tomato Tomato mosaic virus
            "No specific remedy needed (healthy plant)"  # Tomato healthy
        ]
        st.write(f"Plant Disease: **{class_name[result_index]}**")
        st.write(f"Recommended Remedy: **{remedy[result_index]}**")
    else:
        st.warning("Please upload an image first.")
