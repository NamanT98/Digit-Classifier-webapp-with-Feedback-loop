import tensorflow as tf
import json
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np

# loading dataset for evaluation
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# normalizing images
x_test=x_test/255.

# converting labels to categorical form or one hot encoded format
y_test=to_categorical(y_test)

xtest2=x_test.reshape(10000,28,28,1)

# loading data for wrong predictions images
with open("wrong/image_data.json",'r') as f:
    data=f.read()
    data=json.loads(data)
    f.close()


x=[]
y=[]
for i in list(data.keys()):
    k=np.zeros((10),dtype=float)     # for making label vectors
    img=plt.imread("wrong/"+i+".jpg")
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  # converting rgb image to gray
    gray=gray/255.  # normalizing image
    x.append(gray)
    k[data[i][1]]=1.0
    y.append(k)

x=np.array(x)
y=np.array(y)

x=x.reshape((-1,28,28,1))
print(y.shape,x.shape,y[0])

model =tf.keras.models.load_model('model/mnist.h5')   # loading originally trained model

for layer in model.layers[:-2]:   # freezing the weights of layers except last two layers for finetuning 
    # print(layer.name)
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # compiling model 
model.fit(x,y,epochs=5,batch_size=1000,verbose=True,validation_data=(xtest2,y_test))  # training model
print(model.evaluate(xtest2,y_test))    # evaluating fine tuned model

model.save("model/mnist_retrained.h5")   # saving fine tuned model
print("Model saved successfully !")
