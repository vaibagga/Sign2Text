from tensorflow.keras.models import load_model
model=load_model('Mnist_Hand_sign.h5')
print(model.summary())
