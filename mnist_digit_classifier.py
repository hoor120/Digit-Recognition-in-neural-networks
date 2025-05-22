# ANN (Artificial Neurol Network)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
print(x_train[7])
print(x_train.shape)

plt.imshow(x_train[7])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,epochs=10)

predict = model.predict(x_test[4].reshape(1,28,28))
label = np.argmax(predict)
print(label)
print(predict)

plt.imshow(x_test[4])
plt.show()