import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os,json

# Load the trained model
model =tf.keras.models.load_model('model/mnist.h5')
classes = ['zero','one','two','three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


# Function to make predictions
def predict(image_path):
    image = plt.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image=gray_image/255.
    # gray_image.resize((28,28))
    image = tf.expand_dims(gray_image,0)  # Add batch dimension

    output=model.predict(image)
    index=tf.argmax(output[0])
    return int(index)

def update_data(name,predicted_label,original_label):
    # updating the file containing the data of incorrect predictions
    with open("wrong/image_data.json",'r') as f:
        data=f.read()
        data=json.loads(data)
        f.close()
    
    data[name]=[predicted_label,original_label]
    print(data)

    with open("wrong/image_data.json","w") as f:
        json.dump(data,f)
        f.close()
    

def callback():
    st.session_state.button_clicked=True

# Streamlit app
st.title("MNIST Digit Classifier")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked=False
    st.session_state.text_input=False

if uploaded_file is not None :
    st.image(uploaded_file, caption="Uploaded Image.")
    st.write("")

    if st.button('Classify',on_click=callback) or st.session_state.button_clicked:
        class_index = predict(uploaded_file)
        st.write(f"Prediction: {classes[class_index]}")

        # add feedback mechanism 
        st.write("Is the prediction correct ?")
        col1, col2= st.columns([1,1])
        with col1:
            if st.button('Yes'):
                st.write("Thanks for your feedback !")
        with col2:
            if st.button('No') or st.session_state.text_input:
                user_input = st.text_area("Please enter correct label (in words):")
                st.session_state.text_input = True
                if user_input:
                    img=plt.imread(uploaded_file)
                    k=os.listdir('wrong/')
                    name=f"image{len(k)}"
                    plt.imsave(f"wrong/{name}.jpg",img)
                    true_label=classes.index(user_input.lower())
                    update_data(name,class_index,true_label)
                    st.write("Thanks for your feedback !")
