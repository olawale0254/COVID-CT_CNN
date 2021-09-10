import streamlit as st
import numpy as np
import pandas as pd
import tensorflow.keras
from PIL import Image, ImageOps
import seaborn as sns
import cv2

img1 = Image.open("img.jpg").convert('RGB')
st.image([img1], width=100)
# st.write( """
# ## Covid Testing based on Chest CT Scans
# # """)
st.markdown("<h2 style='text-align: center; color: green;'>A CONVOLUTION NEURAL NETWORKS APPROACH IN THE CLASSIFICATION OF COVID-19 CT SCAN IMAGES</h2>", unsafe_allow_html=True)


st.subheader("Overview")
st.markdown("<p style='text-align: justify; color: white;'>In the last few years, deep learning has increasingly shown the potential to improve healthcare by aiding medical professionals with diagnostic processes and patient interactions. In particular, Convolutional Neural Networks (CNNs), a class of deep learning algorithms, have successfully been applied to classify images of biological features that are often used to help monitor overall health and detect disorders in patients. As the rate of COVID-19 increases the use of the traditional method of diagnosing becomes too inefficient and time consuming therefore, a computational model that processes a CT scan sample Image and to classify if there exist COVID-19 or not in the image, have significant practical applications in the field of medicine. This project proposed to take a step towards creating this model by using Deep CNNs to build a model that can classify an CT scan image that contains COVID-19 or not. Three models (Xception, VGG-16, GoogleNet) were proposed and will be trained using a dataset of 2492 CT scan images</p>", unsafe_allow_html=True)

with st.sidebar.header( "WELCOME!!!"):
    imageF = st.sidebar.file_uploader("Please Upload a CT-Scan Image",type = ['jpg',"jpeg",'png'])
    # img22 = np.array(Image.open(imageF))



res_model = {1: "VGG16", 2: "GoogleNet", 3:"Xception"}
mode_selection = st.sidebar.radio("Please choose a CNN-Model", (res_model[1], res_model[2],res_model[3]))
    
    
if imageF is not None:
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    if mode_selection == res_model[1]:
        model = tensorflow.keras.models.load_model('vgg_ct.h5', compile = False)
    if mode_selection == res_model[2]:
        model = tensorflow.keras.models.load_model('ggnet_ct.h5', compile = False)
    if mode_selection == res_model[3]:
        model = tensorflow.keras.models.load_model('xception_ct.h5', compile = False)
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(imageF).convert('RGB')
    img22 = np.array(image)
    gray_img = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    st.image([imageF,heatmap_img])

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
    image_array = np.asarray(image)
    
    # display the resized image
    #image.show()
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # run the inference
    prediction = model.predict(data)
    
    st.markdown("There's {:.2f} percent probability that the scan shows Covid ".format(prediction[0][0]*100))
    st.markdown("There's {:.2f} percent probability that the scan does not show Covid ".format(prediction[0][1]*100))

    df = [[prediction[0][0],prediction[0][1]]]
    
    data = pd.DataFrame(df, columns = ['Covid','Non-Covid']).T
    # st.dataframe(data)
   
    # st.bar_chart(data)
    sns.barplot(x=data.index, y=0, data=data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

else:
    st.markdown("Happy Testing!!! 🙂")