import streamlit as st
import tensorflow as tf
import numpy as np


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


st.header("Disease Recognition")
test_image = st.file_uploader("Choose an Image:")
if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        remedies = [
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

        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        st.success("Remedy: {}".format(remedies[result_index]))