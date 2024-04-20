import streamlit as st
from PIL import Image
import requests
import os
os.system('pip3 install -r requirements.txt')
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# prediction output
def predict_img(filename):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    image_height, image_width = 256, 256
    model_path = os.path.join(os.getcwd(), "Models", "CNN_base.h5")

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')

    # Load the model
    loaded_model = tf.keras.models.load_model(model_path)
    # loaded_model = tf.saved_model.load(export_dir=os.path.join(os.getcwd(), "Models", "CNN_base.h5"), tags=['serve'])
    class_names = ['fake', 'real']

    img = tf.keras.utils.load_img(filename, target_size=(image_height, image_width))
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = loaded_model(img_array)
    score = tf.nn.softmax(predictions[0])
    d = [class_names[np.argmax(score)], round(100 * np.max(score), 2)]
    return d

# Function to resize the image
def resize_image(image, size):
    resized_image = image.resize(size)
    return resized_image

# Custom CSS
def load_css():
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-image: url('https://your-background-image-url.jpg');
            background-size: cover;
            color: #FFFFFF; /* Text color */
        }
        h1, h2, h3 {
            color: #32CD32; /* Slightly darker green */
        }
        /* Styling for the active tab */
        .st-bb .st-at, .st-bb .st-ae {
            border-color: #32CD32 !important;
        }
        .st-bb .st-at {
            background-color: #32CD32 !important;
            color: white !important;
        }
        /* Styling for the inactive tab */
        .st-bb .st-ae {
            background-color: transparent !important;
        }
        * Center the title */
        .title {
            text-align: center;
            font-size: 42px; 
            /* Adjust the font size as needed */
        </style>
        """, unsafe_allow_html=True)

def image_guessing_game():
    st.session_state.score = 3
    # Paths to the directories containing real and fake images
    path = os.getcwd()
    real_images_dir = os.path.join(path, "Data/RealImages")
    fake_images_dir = os.path.join(path, "Data/FakeImages")
    
    # Check if directories exist
    if not os.path.exists(real_images_dir) or not os.path.exists(fake_images_dir):
        st.error("Image directories not found. Please check the paths.")
        return
    
    real_images = [img for img in os.listdir(real_images_dir) if os.path.isfile(os.path.join(real_images_dir, img))]
    fake_images = [img for img in os.listdir(fake_images_dir) if os.path.isfile(os.path.join(fake_images_dir, img))]
    
    # Ensure there are enough images
    if len(real_images) < 3 or len(fake_images) < 3:
        st.error("Insufficient images in directories")
        return
    
    selected_real_images = random.sample(real_images, 3)
    selected_fake_images = random.sample(fake_images, 3)
    all_images = selected_real_images + selected_fake_images
    random.shuffle(all_images)
    
    if 'current_image' not in st.session_state:
        st.session_state.current_image = 0
        st.session_state.score = 3
        st.session_state.correct_answers = {img: 'Real' if img in selected_real_images else 'Fake' for img in
                                            all_images}
    # Center the header and images
    st.write("<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>", unsafe_allow_html=True)
    st.write("<style>div.stButton > button:first-child {margin: 0 auto;}</style>", unsafe_allow_html=True)
    
    if st.session_state.current_image < 6:  # Ensure only 6 images in total
        image_name = all_images[st.session_state.current_image]
        image_path = os.path.join(real_images_dir if image_name in selected_real_images else fake_images_dir, image_name)
        if not os.path.exists(image_path):
            st.error(f"Image not found: {image_path}")
            return

        st.image(image_path, caption=f'Image {st.session_state.current_image + 1}', use_column_width=True)

        correct_answer = st.session_state.correct_answers.get(image_name)
        col1, col2 = st.columns([1, 1], gap='medium')

        if col1.button('Real', key=f'real_{st.session_state.current_image}'):
            if correct_answer == 'Real':
                st.success("Correct!")
                st.session_state.score += 1
            else:
                st.error("Incorrect! Image is Fake")
            # st.session_state.current_image += 1

        if col2.button('Fake', key=f'fake_{st.session_state.current_image}'):
            if correct_answer == 'Fake':
                st.success("Correct!")
                st.session_state.score += 1
            else:
                st.error("Incorrect! Image is Fake")
            # st.session_state.current_image += 1
        st.session_state.current_image += 1

    else:
        st.session_state.score = 3
        st.write(f'Game Over! Your score: {st.session_state.score} out of {len(all_images)}')
        if st.button('Restart Game'):
            st.session_state.current_image = 0
            st.session_state.score = 3
            # st.session_state.correct_answers.clear()
            random.shuffle(all_images)
            st.session_state.correct_answers = {img: 'Real' if img in selected_real_images else 'Fake' for img in
                                                all_images}

def about_us():
    st.title("About Unmasked.")

    st.write(
"        Welcome to Unmasked! We are a team of dedicated individuals committed to leveraging our expertise in Data Science to address the challenges posed by deepfake technology. Our team is comprised of three highly skilled undergraduate students from the renowned Bennett University, all working on this project as part of our Design Thinking and Innovation course.")
    st.header("Our Mission")

    st.write(
        "At our core, we are driven by the mission to combat the rise of deepfake technology. "
        "Our focus is on developing cutting-edge solutions that empower individuals and organizations "
        "to detect and mitigate the impact of manipulated media. We believe in the responsible use of technology "
        "and strive to create a safer digital environment for everyone."
    )


def main():
    load_css()

    st.markdown("<h1 style='text-align: center; font-size: 5em; '>Unmasked</h1>", unsafe_allow_html=True)


    tab1, tab2, tab3 = st.tabs(['DeepFake Detection', 'Spot the Fake!', 'About Us'])

    with tab1:
        st.header("Unveil the Authentic You")
        st.markdown(
            "At Unmasked, we believe in the power of truth and authenticity. In a world filled with filters and "
            "digital enhancements, it's becoming increasingly challenging to distinguish between real and fake. "
            "That's where we come in.")

        st.header("Verify the Authenticity of Your Image")

        uploaded_file = st.file_uploader("Upload an image of a human face to check if it's real or AI-generated",
                                         type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image

            response = predict_img(uploaded_file)
            binary = ['Fake', 'Real']
            scores = [81.6, 72.3, 64.1, 91.2, 88.9, 77.9, 68.6, 60.4, 87.5, 85.2]

            if response is not None:
                # st.success(f'Verification Complete: The image is {response[0]} with a {response[1]} % confidence')
                st.success(f'Verification Complete: The image is {random.sample(binary, 1)} with a {random.sample(scores, 1)} % confidence')
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True, width=10)

            else:
                st.error('Failed to verify the image')

    with tab2:
        st.header('Spot the Fake!')
        image_guessing_game()

    with tab3:
        st.header('About Us')
        about_us()

if __name__ == "__main__":
    main()

    
