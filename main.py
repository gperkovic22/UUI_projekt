from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

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
X_treniranje_norm = x_treniranje/225
X_testiranje_norm = x_testiranje/255

bazni_model = MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet'
)

x = bazni_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(128, activation='relu')(x)  
output = Dense(10, activation='softmax')(x)  

model = Model(inputs=bazni_model.input, outputs=output)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary() 

early_stop = EarlyStopping(monitor='val_loss',patience=3)
model.fit(X_treniranje_norm,y_kat_treniranje,epochs=2,validation_data=(X_testiranje_norm,y_kat_testiranje),callbacks=[early_stop])

print(model.metrics_names)
print(model.evaluate(X_testiranje_norm,y_kat_testiranje,verbose=0))

predvidanja = model.predict(X_testiranje_norm)
predvidi_klase = np.argmax(predvidanja, axis=1)

print(classification_report(y_testiranje,predvidi_klase))
print(confusion_matrix(y_testiranje, predvidi_klase))