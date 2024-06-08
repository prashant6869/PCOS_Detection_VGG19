import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import tensorflow.keras as keras                                                                                                                                      # type: ignore
from tensorflow.keras.applications.vgg19 import VGG19                                                                                                                 # type: ignore
import numpy as np

hide_streamlit_style = """
            <style>
            #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.markdown("<h1 style='text-align: center; color: red;'>PCOS DETECTION</a></h1>", unsafe_allow_html = True)

def main() :
    candidate_labels = ["PCOS", "NON-PCOS"]
    file_uploaded = st.file_uploader('Upload an Ultrasound Image...', type = 'jpg')

    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Ultrasound.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result = predict_class(image)
        st.write('Prediction : {}'.format(result))
        if result == 'PCOS':
            st.write('Please consult a doctor as soon as possible.')
        else :
            st.write('There is no need to worry but if symptoms appear, consult a doctor as soon as possible.')

def predict_class(rgb_image):
    with st.spinner('Loading, please be patient...'):
        model = joblib.load('xrayVGG19.pkl')
        vggmodel = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        for layer in vggmodel.layers:
            layer.trainable = False

    label = {0: 'PCOS', 1: 'NORMAL'}

    # Resize the RGB image to (256, 256)
    test_image = keras.preprocessing.image.array_to_img(rgb_image, data_format='channels_last')
    test_image = test_image.resize((256, 256))
    
    # Convert the image to an array
    test_image = keras.preprocessing.image.img_to_array(test_image)
    
    # Normalize the image
    test_image /= 255.0
    
    # Expand the dimensions to match the model input shape
    test_image = np.expand_dims(test_image, axis=0)
    
    # Use VGG19 for feature extraction
    feature_extractor = vggmodel.predict(test_image)
    
    # Reshape the features
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    
    # Make predictions
    prediction = model.predict(features)[0]
    final = label[prediction]
    
    return final

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>

<div class="footer">
<p align="center"> </p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)
if __name__ == '__main__' :
    main()