##import necessary libraries
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
import time
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib

matplotlib.use('TkAgg')

print("Device: \n", tf.config.experimental.list_physical_devices())
print(tf.__version__)
print(tf.test.is_built_with_cuda())

#First We need to save the images as numpy array then we load the numpy array

images = np.load('E:/MN/Research/MD/All CXR Save File/CX8SF/CXR8_X124.npy')    #Images
y = np.load('E:/MN/Research/MD/All CXR Save File/CX8SF/CXR8_y124.npy')         #Corresponding level of the images

#Split into train and test

X_train, X_test, y_train, y_test =  train_test_split(images, y, test_size=0.20, stratify= y, random_state =2)

#Custom CNN Model

model = tf.keras.Sequential()

#1st conv layer
model.add(tf.keras.layers.Conv2D(64, 5, padding="same", input_shape=(124, 124, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(2))

#2nd conv layer
model.add(tf.keras.layers.Conv2D(32, 3, padding="valid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(2))

#3rd conv layer (from here model gives good result)
model.add(tf.keras.layers.Conv2D(16, 3, padding="valid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(2))
model.add(tf.keras.layers.Dropout(0.3))      #Use dropout to reduce overfitting

#Flatten
model.add(tf.keras.layers.Flatten())

#Fully connected1
model.add(tf.keras.layers.Dense(1024))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.5))

#Fully connected2
model.add(tf.keras.layers.Dense(512, name ='CXR3BVfeature'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.5))

#Output layer
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Activation('softmax'))
adam = tf.keras.optimizers.Adam(lr=0.0001)

model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor= 'val_acc', save_best_only=True, mode='max', verbose=1)

#Run using GPU
with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, batch_size= 128, epochs=50, verbose=1, validation_data=(X_test, y_test), callbacks=[mc])

#Loss curve

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.pause(1)

#Accuracy Curve

plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
plt.pause(1)

#Load the best model for feature extraction
model = tf.keras.models.load_model('./best_model.h5')
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('CXR3BVfeature').output)
intermediate_layer_model.summary()

#Extract the features
feature_engg_data = intermediate_layer_model.predict(images)
feature_engg_data = pd.DataFrame(feature_engg_data)

'''PCA Apply'''
from sklearn.preprocessing import StandardScaler
x = feature_engg_data.loc[:, feature_engg_data.columns].values
x = StandardScaler().fit_transform(x)

y = tf.keras.utils.to_categorical(y, num_classes=8)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify= y, random_state =2)

""" Finding Correlation"""
from sklearn.preprocessing import Normalizer
normalizeddf_train = Normalizer().fit_transform(X_train)
normalizeddf_test = Normalizer().fit_transform(X_test)
print (normalizeddf_train)
print(pd.DataFrame(normalizeddf_train).corr(method='pearson'))

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = pd.DataFrame(dataset).corr(method='pearson')
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
corr_features = correlation(normalizeddf_train, 0.82)  #Set the correlation value
print(len(set(corr_features)))


#Drop the correlated features
X_train = pd.DataFrame(normalizeddf_train).drop(corr_features,axis=1)
X_test = pd.DataFrame(normalizeddf_test).drop(corr_features,axis=1)

#Custom Extreme Learning Machine
input_size = X_train.shape[1]
hidden_size = 1500

input_weights = np.random.normal(size=[input_size, hidden_size])
biases = np.random.normal(size=[hidden_size])

def relu(x):
   return np.maximum(x, 0, x)

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

output_weights = np.dot(scipy.linalg.pinv(hidden_nodes(X_train)), y_train)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(X_test)


#Calculate the classification results
rounded_labels=np.argmax(prediction, axis=1)
real = np.argmax(y_test, axis=1)
print(confusion_matrix(real, rounded_labels))
print(classification_report(real, rounded_labels))
print(metrics.accuracy_score(real, rounded_labels))
rounded_labels=np.argmax(prediction, axis=1)
real = np.argmax(y_train, axis=1)


#ROC
import matplotlib.pyplot as plt
from itertools import cycle

prediction = predict(X_test)
lw = 1
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(8):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prediction.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


colors = cycle([ 'red', 'blue', 'green', 'yellow', 'cyan', 'red', 'blue', 'green',])
for i, color in zip(range(8), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.4f})'
                                                        ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Classification of CXR8 using CNN-PCC-ELM ')
plt.legend(loc="lower right")
plt.show()
plt.pause(1)
c = roc_auc_score(y_test, prediction, multi_class='ovo')
print("AUC:", c)



"""ResNet50-PCC-ELM"""

X_train, X_test, y_train, y_test =  train_test_split(images, y, test_size=0.20, stratify= y, random_state =2)
Res = ResNet50(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in Res.layers:
    layer.trainable = False
x = Flatten()(Res.output)

x = Dense(512, activation='relu', name='LastDenseRes')(x)
prediction = Dense(8, activation='softmax')(x)

# create a model object
model_res = Model(inputs=Res.input, outputs=prediction)
model_res.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model_res.summary()
start = time.time()

history = model_res.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), shuffle=1, callbacks=[es])

end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

model_res.save("E:/MN/Research/MD/ALL CXR Model/Resnet50cx8.h5")


"""PCC_ELM"""
start = time.time()
model_res = tf.keras.models.load_model('E:/MN/Research/MD/ALL CXR Model/Resnet50cx8.h5')

intermediate_layer_model = tf.keras.Model(inputs=model_res.input, outputs=model_res.get_layer('LastDenseRes').output)
intermediate_layer_model.summary()

feature_engg_data1 = intermediate_layer_model.predict(images)
feature_engg_data1 = pd.DataFrame(feature_engg_data1)



'''PCC Apply'''
from sklearn.preprocessing import StandardScaler
x1 = feature_engg_data1.loc[:, feature_engg_data1.columns].values
x1 = StandardScaler().fit_transform(x1)

y1 = tf.keras.utils.to_categorical(y, num_classes=8)
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.20, stratify= y, random_state =2)

""" Finding Correlation"""
from sklearn.preprocessing import Normalizer
normalizeddf_train = Normalizer().fit_transform(X_train)
normalizeddf_test = Normalizer().fit_transform(X_test)
print (normalizeddf_train)
print(pd.DataFrame(normalizeddf_train).corr(method='pearson'))

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = pd.DataFrame(dataset).corr(method='pearson')
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(normalizeddf_train, 0.82)

print(len(set(corr_features)))

X_1 = pd.DataFrame(normalizeddf_train).drop(corr_features, axis=1)
X_2 = pd.DataFrame(normalizeddf_test).drop(corr_features, axis=1)
del X_test
del X_train

from sklearn.model_selection import KFold
#y = tf.keras.utils.to_categorical(y, num_classes=8)
cv = KFold(n_splits=5, shuffle=True, random_state=2)
for train_index, test_index in cv.split(x1):
    X_train, X_test, y_train, y_test = x1[train_index], x1[test_index], y1[train_index], y1[test_index]
    input_size = X_train.shape[1]
    hidden_size = 1500

    input_weights = np.random.normal(size=[input_size, hidden_size])
    biases = np.random.normal(size=[hidden_size])

    def relu(x):
       return np.maximum(x, 0, x)

    def hidden_nodes(X):
        G = np.dot(X, input_weights)
        G = G + biases
        H = relu(G)
        return H

    output_weights = np.dot(scipy.linalg.pinv(hidden_nodes(X_train)), y_train)

    def predict(X):
        out = hidden_nodes(X)
        out = np.dot(out, output_weights)
        return out

    start = time.time()
    prediction = predict(X_test)
    rounded_labels = np.argmax(prediction, axis=1)
    real = np.argmax(y_test, axis=1)
    start = time.time()
    print(confusion_matrix(real, rounded_labels))
    print(classification_report(real, rounded_labels))
    print(metrics.accuracy_score(real, rounded_labels))

    end = time.time()
    elapsed = end - start
    print("Total Time:", elapsed)

    import matplotlib.pyplot as plt
    from itertools import cycle

    prediction = predict(X_test)
    lw = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prediction.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    colors = cycle([ 'red', 'blue', 'green', 'yellow', 'cyan', 'red', 'blue', 'green',])
    for i, color in zip(range(8), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.4f})'
                                                           ''.format(i, roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Classification of CXR8 using ResNet50-PCC-ELM ')
    plt.legend(loc="lower right")
    plt.show()
    plt.pause(1)


    c = roc_auc_score(y_test, prediction, multi_class='ovo')
    print("AUC:", c)




""""""""""""""VGG19-PCC-ELM""""""
X_train, X_test, y_train, y_test =  train_test_split(images, y, test_size=0.20, stratify= y, random_state =2)


Vgg = VGG19(input_shape=(124, 124, 3), weights='imagenet', include_top=False)

for layer in Vgg.layers:
    layer.trainable = False
x = Flatten()(Vgg.output)

x = Dense(512, activation='relu', name='LastDenseVGG')(x)
prediction = Dense(8, activation='softmax')(x)

# create a model object
model_vgg = Model(inputs=Vgg.input, outputs=prediction)
model_vgg.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

#model.summary()
start = time.time()

history = model_vgg.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), shuffle=1, callbacks=[es])

end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

model_vgg.save("E:/MN/Research/MD/ALL CXR Model/VGG19_pcc.h5")

"""PCC_ELM"""
start = time.time()
model_vgg = tf.keras.models.load_model('E:/MN/Research/MD/ALL CXR Model/VGG19_pcc.h5')
intermediate_layer_model = tf.keras.Model(inputs=model_vgg.input, outputs=model_vgg.get_layer('LastDenseVGG').output)
intermediate_layer_model.summary()

feature_engg_data2 = intermediate_layer_model.predict(images)
feature_engg_data2 = pd.DataFrame(feature_engg_data2)



'''PCC Apply'''
from sklearn.preprocessing import StandardScaler
x2 = feature_engg_data2.loc[:, feature_engg_data2.columns].values
x2 = StandardScaler().fit_transform(x2)

y2 = tf.keras.utils.to_categorical(y, num_classes=8)
X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.20, stratify= y2, random_state =2)

""" Finding Correlation"""
from sklearn.preprocessing import Normalizer
normalizeddf = Normalizer().fit_transform(x2)
print (normalizeddf)


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = pd.DataFrame(dataset).corr(method='pearson')
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(normalizeddf, 0.82)

print(len(set(corr_features)))

x = pd.DataFrame(normalizeddf).drop(corr_features, axis=1)

from sklearn.model_selection import KFold
#y = tf.keras.utils.to_categorical(y, num_classes=8)
cv = KFold(n_splits=5, shuffle=True)
for train_index, test_index in cv.split(x2):
    X_train, X_test, y_train, y_test = x2[train_index], x2[test_index], y2[train_index], y2[test_index]
    #input_size = X_train1.shape[1]
    input_size = X_train.shape[1]
    hidden_size = 1500

    input_weights = np.random.normal(size=[input_size, hidden_size])
    biases = np.random.normal(size=[hidden_size])

    def relu(x):
       return np.maximum(x, 0, x)

    def hidden_nodes(X):
        G = np.dot(X, input_weights)
        G = G + biases
        H = relu(G)
        return H

    output_weights = np.dot(scipy.linalg.pinv(hidden_nodes(X_train)), y_train)

    def predict(X):
        out = hidden_nodes(X)
        out = np.dot(out, output_weights)
        return out

    prediction = predict(X_test)
    rounded_labels = np.argmax(prediction, axis=1)
    real = np.argmax(y_test, axis=1)

    print(confusion_matrix(real, rounded_labels))
    print(classification_report(real, rounded_labels))
    print(metrics.accuracy_score(real, rounded_labels))

    end = time.time()
    elapsed = end - start
    print("Total Time:", elapsed)

    import matplotlib.pyplot as plt
    from itertools import cycle

    prediction = predict(X_test)
    lw = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prediction.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    colors = cycle([ 'red', 'blue', 'green', 'yellow', 'cyan', 'red', 'blue', 'green',])
    for i, color in zip(range(8), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.4f})'
                                                           ''.format(i, roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Classification of CXR8 using VGG19-PCC-ELM ')
    plt.legend(loc="lower right")
    plt.show()
    plt.pause(1)


    c = roc_auc_score(y_test, prediction, multi_class='ovo')
    print("AUC:", c)




