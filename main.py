import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

(x_treniranje, y_treniranje), (x_testiranje, y_testiranje) = cifar10.load_data()

print("X_treniranje izvorni oblik", x_treniranje.shape)
print("y_train izvorni oblik", y_treniranje.shape)
print("X_test izvorni oblik", x_testiranje.shape)
print("y_test izvorni oblik", y_testiranje.shape)

y_kat_treniranje = to_categorical(y_treniranje,10)
y_kat_testiranje = to_categorical(y_testiranje,10)

print("y_train izvorni oblik", y_kat_treniranje.shape)
print("y_test izvorni oblik", y_kat_testiranje.shape)

x_treniranje.max()
X_train_norm = x_treniranje/225
X_test_norm = x_testiranje/255

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary() 

early_stop = EarlyStopping(monitor='val_loss',patience=3)
model.fit(X_train_norm,y_kat_treniranje,epochs=10,validation_data=(X_test_norm,y_kat_testiranje),callbacks=[early_stop])

print(model.metrics_names)
print(model.evaluate(X_test_norm,y_kat_testiranje,verbose=0))

predictions = model.predict(X_test_norm)
predict_classes = np.argmax(predictions, axis=1)

print(classification_report(y_testiranje,predict_classes))

confusion_matrix(y_testiranje,predict_classes)
y_testiranje[7]
predict_classes[7] 
plt.imshow(x_testiranje[7])