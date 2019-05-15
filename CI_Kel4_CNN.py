import numpy as np
import pandas as pd

import os
from glob import glob
from PIL import Image
%matplotlib inline
import matplotlib.pyplot as plt
import cv2
import fnmatch
import keras
from time import sleep

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation
from keras.optimizers import RMSprop,Adam
# from tensorflow.keras.callbacks import EarlyStopping

train_imagePatches = glob('~/Chest X-Ray Pneumonia/chest_xray/train/**/*.jpeg', recursive=True)
test_imagePatches = glob('~/Chest X-Ray Pneumonia/chest_xray/test/**/*.jpeg', recursive=True)
val_imagePatches = glob('~/Chest X-Ray Pneumonia/chest_xray/val/**/*.jpeg', recursive=True)
print(len(train_imagePatches))
print(len(test_imagePatches))
print(len(val_imagePatches))

pattern_normal = '*NORMAL*'
pattern_bacteria = '*_bacteria_*'
pattern_virus = '*_virus_*'

normal = fnmatch.filter(train_imagePatches, pattern_normal)
bacteria = fnmatch.filter(train_imagePatches, pattern_bacteria)
virus = fnmatch.filter(train_imagePatches, pattern_virus)
x = []
y = []
for img in train_imagePatches:
    full_size_image = cv2.imread(img)
    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)
    x.append(im)
    if img in normal:
        y.append(0)
    elif img in bacteria:
        y.append(1)
    elif img in virus:
        y.append(1)
    else:
        #break
        print('no class')
x = np.array(x)
y = np.array(y)

x_train = x
y_train = to_categorical(y, num_classes = 2)
del x,y

pattern_normal = '*NORMAL*'
pattern_bacteria = '*_bacteria_*'
pattern_virus = '*_virus_*'

normal = fnmatch.filter(test_imagePatches, pattern_normal)
bacteria = fnmatch.filter(test_imagePatches, pattern_bacteria)
virus = fnmatch.filter(test_imagePatches, pattern_virus)
x = []
y = []
for img in test_imagePatches:
    full_size_image = cv2.imread(img)
    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)
    x.append(im)
    if img in normal:
        y.append(0)
    elif img in bacteria:
        y.append(1)
    elif img in virus:
        y.append(1)
    else:
        #break
        print('no class')
x = np.array(x)
y = np.array(y)

x_test = x
y_test = to_categorical(y, num_classes = 2)
del x,y

pattern_normal = '*NORMAL*'
pattern_bacteria = '*_bacteria_*'
pattern_virus = '*_virus_*'

normal = fnmatch.filter(val_imagePatches, pattern_normal)
bacteria = fnmatch.filter(val_imagePatches, pattern_bacteria)
virus = fnmatch.filter(val_imagePatches, pattern_virus)
x = []
y = []
for img in val_imagePatches:
    full_size_image = cv2.imread(img)
    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)
    x.append(im)
    if img in normal:
        y.append(0)
    elif img in bacteria:
        y.append(1)
    elif img in virus:
        y.append(1)
    else:
        #break
        print('no class')
x = np.array(x)
y = np.array(y)

x_val = x
y_val = to_categorical(y, num_classes = 2)
del x,y

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU

model = Sequential()

model.add(Conv2D(32,(7,7),activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
# model.add(Dropout(0.15))

model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
# model.add(Dropout(0.15))

model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
# model.add(Dropout(0.15))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
# model.add(Dropout(0.15))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
# model.add(Dropout(0.15))

model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# print(model.summary())

from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(filepath='percobaan_2.hdf5',monitor="val_acc", save_best_only=True, save_weights_only=False)

hist = model.fit(
    x_train, y_train, batch_size=32,
    epochs = 40, validation_data=(x_val, y_val),
    callbacks=[mcp])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('w')
ax.grid(b=False)
ax.plot(hist.history['acc'], color='red')
ax.plot(hist.history['val_acc'], color ='green')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('w')
ax.grid(b=False)
ax.plot(hist.history['loss'], color='red')
ax.plot(hist.history['val_loss'], color ='green')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

model.load_weights('~/percobaan_2.hdf5')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.grid(b=False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_test, axis = 1),np.argmax(pred, axis = 1))
plot_confusion_matrix(cm = cm,
                      normalize    = False,
                      cmap ='Reds',
                      target_names = ['Normal','Pneumonia'],
                      title        = "Confusion Matrix")

# # Visualize Prediction
df = pd.DataFrame(pred)
df.columns = [ 'Normal' , 'Pneumonia' ]
df.index = y_test[:,1]
print(df)

#Receiver Operating Characteristic (ROC)
import numpy as np
import matplotlib.pyplot as plt
#from itertools import cycle
from sklearn.metrics import roc_curve, auc
#from scipy import interp
fpr, tpr, thresholds = roc_curve(y_test[:,1], pred[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Class Activation Map

import matplotlib.image as mpimg
from keras.preprocessing import image
from keras import backend as K

# img_path = '~/chest_xray/test/PNEUMONIA/person2_bacteria_4.jpeg'
img_path = '~/chest_xray/train/NORMAL/IM-0578-0001.jpeg'
img=mpimg.imread(img_path)
plt.imshow(img)

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
argmax = np.argmax(preds[0])
output = model.output[:, argmax]
last_conv_layer = model.get_layer('conv2d_5')
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(64):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
hif = .8
superimposed_img = heatmap * hif + img

output = 'D:/output.jpeg'
cv2.imwrite(output, superimposed_img)

img=mpimg.imread(output)

dfx = pd.DataFrame(preds[:,1], index=list(range(1)), columns=['probability'])

plt.imshow(img)
plt.axis('off')
pneum_prob = preds[0,1]*100
plt.title('Pneumonia Probability: ' + str(round(pneum_prob,2)) + '%')