#%% Read csv

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from sklearn import datasets, linear_model, preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from preprocessing import default_preprocessing, get_poly_coeffs,get_poly_coeffs_lg10, default_normalize_r, default_normalize_c
from keras.models import Sequential
from keras.layers import Dense

plt.style.use('ggplot')
#matplotlib inline

SENSOR = 'R1'
#DATASET_PATH = './preprocessed data'
DATASET_PATH = './real data'

#####################################Introducción del dataset: csv#####################################

#se obtienen los valores de los excel, para resistencia y concentracion, entrada y salida
X_train_df = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_X_train.csv'), header=None)
X_val_df = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_X_val.csv'), header=None)
X_test_df = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_X_test.csv'), header=None)

Y_train_df = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_Y_train.csv'), header=None)
Y_val_df = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_Y_val.csv'), header=None)
Y_test_df = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_Y_test.csv'), header=None)

#se obtienen la descripcion para cada dato
train_log = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_log_train.csv'),index_col=0)
val_log = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_log_val.csv'),index_col=0)
#test_log = pd.read_csv(os.path.join(DATASET_PATH, SENSOR+'_log_test.csv'),index_col=0)

#filtra las primeras 5 observaciones de resistencia
#X_train = np.array(X_train_df.loc[train_log.loc[train_log['curr obs']>=0].index])
#X_val = np.array(X_val_df.loc[val_log.loc[val_log['curr obs']>=0].index])
#X_test = np.array(X_test_df.loc[test_log.loc[test_log['curr obs']>=0].index])

#se pasa a np
X_train = np.array(X_train_df)
X_val = np.array(X_val_df)
X_test = np.array(X_test_df)

Y_train = np.array(Y_train_df)
Y_val = np.array(Y_val_df)
Y_test = np.array(Y_test_df)

#se pasa a escala logaritmica
X_train_lg10 = default_preprocessing(X_train)
X_val_lg10 = default_preprocessing(X_val)
X_test_lg10 = default_preprocessing(X_test)

#se normaliza por filas, tiempo:
X_train_lg10_n = tf.keras.utils.normalize(X_train_lg10)
X_val_lg10_n = tf.keras.utils.normalize(X_val_lg10)
X_test_lg10_n = tf.keras.utils.normalize(X_test_lg10)

row_number = np.random.randint(X_train_df.shape[0])
#%% Plot Raw & Log
#########################Muestreo de un dato en el tiempo, lineal y log##############################
# grafico de una muestra alatoriamente
plt.rcParams['figure.figsize'] = (12,5)
plt.plot(np.linspace(0,55, X_test_df.shape[1]), X_test_df.iloc[5])
plt.title("{gas} at concentration {concentration}ppm observed at {experiment}".format(\
                  gas=train_log.loc[row_number]['gas'],\
                  concentration=int(train_log.loc[row_number]['concentration']),\
                  experiment= train_log.loc[row_number]['experiment']).split('_')[0],\
                  fontsize = 18)
plt.ylabel('Sensor response', fontsize = 16), plt.xlabel('Time, s', fontsize = 16)
plt.show()

# grafico de una muestra alatoriamente en logaritmo
plt.rcParams['figure.figsize'] = (12,5)
plt.plot(np.linspace(0,55, X_test_lg10_n.shape[1]), X_test_lg10_n[5])
plt.title("{gas} at concentration {concentration}ppm observed at {experiment}".format(\
                  gas=train_log.loc[row_number]['gas'],\
                  concentration=int(train_log.loc[row_number]['concentration']),\
                  experiment= train_log.iloc[row_number]['experiment']).split('_')[0],\
                  fontsize = 18)
plt.ylabel('Log10(sensor response)', fontsize = 16), plt.xlabel('Time, s', fontsize = 16)
plt.show()


#%% ANN Binary
###########################Modelo binario de la ANN:################################

#Se pasa a binario las salidas, en trains y validation
Y_train_binary=np.zeros([Y_train.shape[0], 1])
Y_train_binary[Y_train[:,1]!=0] = 1

Y_val_binary=np.zeros([Y_val.shape[0], 1])
Y_val_binary[Y_val[:,1]!=0] = 1

Y_test_binary=np.zeros([Y_test.shape[0], 1])
Y_test_binary[Y_test[:,1]!=0] = 1

#Se comienza a hacer la red neuronal
model_binary = tf.keras.models.Sequential() #modelo sequencial
model_binary.add(tf.keras.layers.Flatten()) #las capas de los datos se dan como una linea

model_binary.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #1 hidden layer con 160 neurons
model_binary.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #2 hidden layer con 160 neurons

model_binary.add(tf.keras.layers.Dense(1, activation='sigmoid')) #salida binaria

model_binary.compile(optimizer='adam', #optimizacion estandar
              loss='binary_crossentropy',
              metrics=['accuracy']) 


history_binary=model_binary.fit(X_train_lg10_n, Y_train_binary, epochs=30, validation_data=(X_val_lg10_n, Y_val_binary)) #se ingresa X y Y, se prueba con varias iteraciones

test_loss_binary, test_acc_binary = model_binary.evaluate(X_test_lg10_n, Y_test_binary) #se calcula la perdida y la precision del modelo

Y_predict_binary = model_binary.predict(X_val_lg10_n) #Se obtiene el vector de resultados calculados por la red
Y_predict_binary= np.heaviside(Y_predict_binary-0.5*np.ones(Y_predict_binary.shape), 1)

loss_train_binary = history_binary.history['loss']
loss_val_binary = history_binary.history['val_loss']
epochs_binary = range(1, len(loss_train_binary) + 1)
plt.plot(epochs_binary, loss_train_binary, 'y', label='Training loss')
plt.plot(epochs_binary, loss_val_binary, 'r', label='Validation loss')
plt.title('Binary Model: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train_binary = history_binary.history['accuracy']
acc_val_binary = history_binary.history['val_accuracy']
plt.plot(epochs_binary, acc_train_binary, 'y', label='Training Accuracy')
plt.plot(epochs_binary, acc_val_binary, 'r', label='Validation Accuracy')
plt.title('Binary Model: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Modelos binarios: https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6
#%% ANN Norm
###########################Modelo no binario de la ANN:################################
Y_train_sequential = Y_train[:,1]
Y_val_sequential = Y_val[:,1]
Y_test_sequential = Y_test[:,1]

model_sequential = tf.keras.models.Sequential() #modelo sequencial
model_sequential.add(tf.keras.layers.Flatten()) #las capas de los datos se dan como una linea

model_sequential.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #1 hidden layer con 160 neurons
model_sequential.add(tf.keras.layers.Dropout(0.05))
model_sequential.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #2 hidden layer con 160 neurons
#model_sequential.add(tf.keras.layers.Dropout(0.07))
model_sequential.add(tf.keras.layers.Dense(1601, activation=tf.nn.softmax)) #salida binaria

model_sequential.compile(optimizer='adam', #optimizacion estandar
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


history_sequential=model_sequential.fit(X_train_lg10_n, Y_train_sequential, epochs=30, validation_data=(X_val_lg10_n, Y_val_sequential)) #se ingresa X y Y, se prueba con varias iteraciones

test_loss_sequential, test_acc_sequential = model_sequential.evaluate(X_test_lg10_n, Y_test_sequential) #se calcula la perdida y la precision del modelo

Y_predict_sequential = model_sequential.predict(X_val_lg10_n)
Y_predict_sequential = np.argmax(Y_predict_sequential, axis=1)

loss_train_sequential = history_sequential.history['loss']
loss_val_sequential = history_sequential.history['val_loss']
epochs_sequential = range(1, len(loss_train_sequential) + 1)
plt.plot(epochs_sequential, loss_train_sequential, 'y', label='Training loss')
plt.plot(epochs_sequential, loss_val_sequential, 'r', label='Validation loss')
plt.title('Sequential Model: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train_sequential = history_sequential.history['accuracy']
acc_val_sequential = history_sequential.history['val_accuracy']
plt.plot(epochs_sequential, acc_train_sequential, 'y', label='Training Accuracy')
plt.plot(epochs_sequential, acc_val_sequential, 'r', label='Validation Accuracy')
plt.title('Sequential Model: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#%% PCF
##############################Aproximación polinomial PCF:###################################

#Polynomial approximation, se utiliza para aproximar el orden, por eso es que se utiliza la misma resistencia verificar si el modelo se aproxima


'''AICc = n * log(SSE/n) + (n + p) / (1 - (p + 2) / n)
SSE = Sum of Squared Errors for the training set
n = Number of training cases
p = Number of parameters (weights and biases)'''
NUM_HIDDEN = 160

results_list = []
for poly_coef in range(2, 18):
    
    
    X_train_mult = np.array(list(map(lambda x: get_poly_coeffs(x/np.sum(x), poly_coef), X_train_lg10)))
    X_train_coefs = np.array([list(i[0]) for i in X_train_mult])
    X_resid = np.array([list(i[1]) for i in X_train_mult])

    X_train_lg10_mult = np.array(list(map(lambda x: get_poly_coeffs_lg10(x, poly_coef), X_train_lg10)))
    X_train_lg10_coefs = np.array([list(i[0]) for i in X_train_lg10_mult])
    
    X_val_lg10_mult = np.array(list(map(lambda x: get_poly_coeffs_lg10(x, poly_coef), X_val_lg10)))
    X_val_lg10_coefs = np.array([list(i[0]) for i in X_val_lg10_mult])   

    X_test_lg10_mult = np.array(list(map(lambda x: get_poly_coeffs_lg10(x, poly_coef), X_test_lg10)))
    X_test_lg10_coefs = np.array([list(i[0]) for i in X_test_lg10_mult]) 
    
    SSE = np.mean(X_resid)
    n = X_train.shape[1]
    p = np.copy(poly_coef)*NUM_HIDDEN

    results_list.append([p,  n *(np.log(2*np.pi*SSE/n)+1) + 2*p])
    
#plot de la red neuronal con 160 neurones
plt.plot(np.arange(2,18), np.array(results_list)[:,1])
plt.title('AIC value for neural net with {neurons} neurons'.format(neurons=NUM_HIDDEN), fontsize=18)
plt.xlabel('Polynomial order', fontsize=16)
plt.show()

p = np.poly1d(X_train_coefs[row_number])
t = np.arange(0, X_train.shape[1])

plt.plot(X_train_lg10[row_number]/sum(X_train_lg10[row_number]))
plt.plot(t, p(t)-0.0001,'--',  linewidth = 4)
plt.title("{gas} at concentration {concentration}ppm observed at {experiment}".format(\
                  gas=train_log.loc[row_number]['gas'],\
                  concentration=int(train_log.loc[row_number]['concentration']),\
                  experiment= train_log.loc[row_number]['experiment']).split('_')[0],\
                  fontsize = 18)
leg = plt.legend(['Observed signal', 'Approximated\n signal'], prop={'size': 16}, loc='upper right', bbox_to_anchor=(0.9,0.98))
leg.get_frame().set_linewidth(0.0)
plt.show()


#%% ANN PCF
###########################Modelo PCF:################################
X_train_pcf=[]
X_val_pcf=[]
X_test_pcf=[]
tt= np.arange(0, X_train_lg10.shape[1])

for i in np.arange(0, X_train_lg10.shape[0]):
    p=np.poly1d(X_train_lg10_coefs[i])
    X_train_pcf.append(p(tt))
   
for i in np.arange(0, X_val_lg10.shape[0]):
    p=np.poly1d(X_val_lg10_coefs[i])
    X_val_pcf.append(p(tt))

for i in np.arange(0, X_test_lg10.shape[0]):
    p=np.poly1d(X_test_lg10_coefs[i])
    X_test_pcf.append(p(tt))
    
#X_train_lg10_pcf=np.array(X_train_lg10_coefs)
#X_val_lg10_pcf=np.array(X_val_lg10_coefs)

X_train_lg10_pcf=np.array(X_train_pcf)
X_val_lg10_pcf=np.array(X_val_pcf)
X_test_lg10_pcf=np.array(X_test_pcf)

Y_train_lg10_pcf = Y_train[:,1]
Y_val_lg10_pcf = Y_val[:,1]
Y_test_lg10_pcf = Y_test[:,1]

model_lg10_pcf = tf.keras.models.Sequential() #modelo sequencial
model_lg10_pcf.add(tf.keras.layers.Flatten()) #las capas de los datos se dan como una linea

model_lg10_pcf.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #1 hidden layer con 160 neurons
#model_lg10_pcf.add(tf.keras.layers.Dropout(0.9))
model_lg10_pcf.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #2 hidden layer con 160 neurons
#model_lg10_pcf.add(tf.keras.layers.Dropout(0.8))
model_lg10_pcf.add(tf.keras.layers.Dense(1601, activation=tf.nn.softmax)) #salida binaria


model_lg10_pcf.compile(optimizer='adam', #optimizacion estandar
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


history_lg10_pcf=model_lg10_pcf.fit(X_train_lg10_pcf, Y_train_lg10_pcf, epochs=30, validation_data=(X_val_lg10_pcf, Y_val_lg10_pcf)) #se ingresa X y Y, se prueba con varias iteraciones

#val_loss_lg10_pcf, val_acc_lg10_pcf = model_lg10_pcf.evaluate(X_train_lg10_pcf, Y_train_lg10_pcf) #se calcula la perdida y la precision del modelo

test_loss_lg10_pcf, test_acc_lg10_pcf = model_lg10_pcf.evaluate(X_test_lg10_pcf, Y_test_lg10_pcf) #se calcula la perdida y la precision del modelo


Y_predict_lg10_pcf = model_lg10_pcf.predict(X_val_lg10_pcf)
Y_predict_lg10_pcf = np.argmax(Y_predict_lg10_pcf, axis=1)

loss_train_lg10_pcf = history_lg10_pcf.history['loss']
loss_val_lg10_pcf = history_lg10_pcf.history['val_loss']
epochs_lg10_pcf = range(1, len(loss_train_lg10_pcf) + 1)
plt.plot(epochs_lg10_pcf, loss_train_lg10_pcf, 'y', label='Training loss')
plt.plot(epochs_lg10_pcf, loss_val_lg10_pcf, 'r', label='Validation loss')
plt.title('PCF Model: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train_lg10_pcf = history_lg10_pcf.history['accuracy']
acc_val_lg10_pcf = history_lg10_pcf.history['val_accuracy']
plt.plot(epochs_lg10_pcf, acc_train_lg10_pcf, 'y', label='Training Accuracy')
plt.plot(epochs_lg10_pcf, acc_val_lg10_pcf, 'r', label='Validation Accuracy')
plt.title('PCF Model: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#%% PCA
################################PCA decomposition####################################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler_x = StandardScaler()
pca_model = PCA(n_components=100)

X_train_lg10_pca = scaler_x.fit_transform(X_train_lg10/np.sum(X_train_lg10, axis=1).reshape([-1,1]))
X_val_lg10_pca = scaler_x.fit_transform(X_val_lg10/np.sum(X_val_lg10, axis=1).reshape([-1,1]))
X_test_lg10_pca = scaler_x.fit_transform(X_test_lg10/np.sum(X_test_lg10, axis=1).reshape([-1,1]))

pca_model.fit(X_train_lg10_pca)

expl_var_cumulative = np.cumsum(pca_model.explained_variance_ratio_)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1,101), expl_var_cumulative, marker='o')

ax.set_ylim(0,1.1)
ax.set_xlim(-0.2,10)
ax.set_xticks(np.arange(1,11), minor=False)

for i,j in zip(range(10),expl_var_cumulative):
    ax.annotate("{:.1f}%".format(j*100),xy=(i,j), fontsize=16)

plt.xlabel("Principal component number", fontsize=16)
plt.title('Cumulative explained variance', fontsize=18)
plt.show()

#%% PCA ANN
###########################Modelo PCA:################################
Y_train_lg10_pca = Y_train[:,1]
Y_val_lg10_pca = Y_val[:,1]
Y_test_lg10_pca = Y_test[:,1]

model_lg10_pca = tf.keras.models.Sequential() #modelo sequencial
model_lg10_pca.add(tf.keras.layers.Flatten()) #las capas de los datos se dan como una linea

model_lg10_pca.add(tf.keras.layers.Dense(180, activation=tf.nn.relu)) #1 hidden layer con 160 neurons
#model_lg10_pca.add(tf.keras.layers.Dropout(0.5))
model_lg10_pca.add(tf.keras.layers.Dense(180, activation=tf.nn.relu)) #2 hidden layer con 160 neurons
#model_lg10_pca.add(tf.keras.layers.Dropout(0.05))
model_lg10_pca.add(tf.keras.layers.Dense(1601, activation=tf.nn.softmax)) #salida binaria

model_lg10_pca.compile(optimizer='adam', #optimizacion estandar
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


history_lg10_pca=model_lg10_pca.fit(X_train_lg10_pca, Y_train_lg10_pca, epochs=30, validation_data=(X_val_lg10_pca, Y_val_lg10_pca)) #se ingresa X y Y, se prueba con varias iteraciones

test_loss_lg10_pca, test_acc_lg10_pca = model_lg10_pca.evaluate(X_test_lg10_pca, Y_test_lg10_pca) #se calcula la perdida y la precision del modelo

Y_predict_lg10_pca = model_lg10_pca.predict(X_val_lg10_pca)
Y_predict_lg10_pca = np.argmax(Y_predict_lg10_pca, axis=1)

loss_train_lg10_pca = history_lg10_pca.history['loss']
loss_val_lg10_pca = history_lg10_pca.history['val_loss']
epochs_lg10_pca = range(1, len(loss_train_lg10_pca) + 1)
plt.plot(epochs_lg10_pca, loss_train_lg10_pca, 'y', label='Training loss')
plt.plot(epochs_lg10_pca, loss_val_lg10_pca, 'r', label='Validation loss')
plt.title('PCA Model: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train_lg10_pca = history_lg10_pca.history['accuracy']
acc_val_lg10_pca = history_lg10_pca.history['val_accuracy']
plt.plot(epochs_lg10_pca, acc_train_lg10_pca, 'y', label='Training Accuracy')
plt.plot(epochs_lg10_pca, acc_val_lg10_pca, 'r', label='Validation Accuracy')
plt.title('PCA Model: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% DWT
################################Discrete Wavelet Transform####################################
import pywt
from pywt import wavedec
from pywt import downcoef

X_train_lg10_dwt_list=list()
X_val_lg10_dwt_list=list()
X_test_lg10_dwt_list=list()

for lvl in range(2,7):
    dwt_sample = wavedec(X_train_lg10[row_number]/np.sum(X_train_lg10[row_number]), wavelet = 'db4', mode = 'zero', level=lvl)[0][7:-2]
    plt.plot(dwt_sample)
    
    dwt_train = wavedec(X_train_lg10, wavelet = 'db4', mode = 'zero', level=lvl)[0]
    dwt_val= wavedec(X_val_lg10, wavelet = 'db4', mode = 'zero', level=lvl)[0]
    dwt_test= wavedec(X_test_lg10, wavelet = 'db4', mode = 'zero', level=lvl)[0]
    
    #dwt_train = downcoef('a', X_train_lg10, wavelet = 'gaus', mode = 'zero', level=lvl)[0]
    #dwt_val= downcoef('a', X_val_lg10, wavelet = 'gaus', mode = 'zero', level=lvl)[0] 
    
    X_train_lg10_dwt_list.append(dwt_train)
    X_val_lg10_dwt_list.append(dwt_val)
    X_test_lg10_dwt_list.append(dwt_test)
plt.legend(['level {}'.format(i) for i in range(2,7)], fontsize=14)
plt.xlabel("Number of features", fontsize=16)
plt.title("Approximation coefficients for Daubechies 4 wavelet", fontsize=18)
plt.show()


#%% DWT ANN
###########################Modelo DWT:################################
X_train_lg10_dwt = X_train_lg10_dwt_list[2] #0 equivale a lvl=2, hasta 4 que es lvl=6
X_val_lg10_dwt = X_val_lg10_dwt_list[2]
X_test_lg10_dwt = X_test_lg10_dwt_list[2]

Y_train_lg10_dwt = Y_train[:,1]
Y_val_lg10_dwt = Y_val[:,1]
Y_test_lg10_dwt = Y_test[:,1]

model_lg10_dwt = tf.keras.models.Sequential() #modelo sequencial
model_lg10_dwt.add(tf.keras.layers.Flatten()) #las capas de los datos se dan como una linea

model_lg10_dwt.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #1 hidden layer con 160 neurons
#model_lg10_dwt.add(tf.keras.layers.Dropout(0.5))
model_lg10_dwt.add(tf.keras.layers.Dense(160, activation=tf.nn.relu)) #2 hidden layer con 160 neurons
#model_lg10_dwt.add(tf.keras.layers.Dropout(0.05))
model_lg10_dwt.add(tf.keras.layers.Dense(1601, activation=tf.nn.softmax)) #salida binaria

model_lg10_dwt.compile(optimizer='adam', #optimizacion estandar
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


history_lg10_dwt=model_lg10_dwt.fit(X_train_lg10_dwt, Y_train_lg10_dwt, epochs=30, validation_data=(X_val_lg10_dwt, Y_val_lg10_dwt)) #se ingresa X y Y, se prueba con varias iteraciones

test_loss_lg10_dwt, test_acc_lg10_dwt = model_lg10_dwt.evaluate(X_test_lg10_dwt, Y_test_lg10_dwt) #se calcula la perdida y la precision del modelo

Y_predict_lg10_dwt = model_lg10_dwt.predict(X_val_lg10_dwt)
Y_predict_lg10_dwt = np.argmax(Y_predict_lg10_dwt, axis=1)

loss_train_lg10_dwt = history_lg10_dwt.history['loss']
loss_val_lg10_dwt = history_lg10_dwt.history['val_loss']
epochs_lg10_dwt = range(1, len(loss_train_lg10_dwt) + 1)
plt.plot(epochs_lg10_dwt, loss_train_lg10_dwt, 'y', label='Training loss')
plt.plot(epochs_lg10_dwt, loss_val_lg10_dwt, 'r', label='Validation loss')
plt.title('DWT Model: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train_lg10_dwt = history_lg10_dwt.history['accuracy']
acc_val_lg10_dwt = history_lg10_dwt.history['val_accuracy']
plt.plot(epochs_lg10_dwt, acc_train_lg10_dwt, 'y', label='Training Accuracy')
plt.plot(epochs_lg10_dwt, acc_val_lg10_dwt, 'r', label='Validation Accuracy')
plt.title('DWT Model: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% Misc: Regretion models
###########################Creación de modelos de regresión:################################
X_train_aprox=X_train_lg10
X_val_aprox=X_val_lg10 

var=1000000000000000000000000000000000
for i in range(X_train_aprox.shape[1]): 
    var_loop=np.var(X_train_aprox[:,i])
    if var_loop<var:
        column_number=i
        var=var_loop
        
#column_number=np.random.randint(X_train_df.shape[1]) # se utiliza una columna random del tiempo

X_train_col=X_train_aprox[:,column_number].reshape(-1, 1) #una columna del array
X_val_col=X_val_aprox[:,column_number].reshape(-1, 1) #una columna del array

#se hacen los modelos
regr = linear_model.LinearRegression() 
degree=2
polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())

#se utilizan los modelos para crear la salida
regr.fit(X_train_col, Y_train_df) #
polyreg.fit(X_train_col, Y_train_df) 
degree_fit=3
polyfit = np.polyfit(X_train_aprox[:,column_number], Y_train_df, degree_fit)

#se utilizan los modelos para crear la salida
Y_pred_regr = regr.predict(X_val_col) #
Y_pred_polynom = polyreg.predict(X_val_col) 
Y_pred_polyfit = np.polyval(polyfit, X_val_col)

# Plot outputs
plt.scatter(X_train_col, Y_train_df, color='black', s = 5, label = 'Data points')
plt.plot(X_val_col, Y_pred_regr, color='blue', alpha = 0.5,label = 'Linear Regression')
plt.plot(X_val_col, Y_pred_polynom, color='red', alpha = 0.5,label = 'Polynomial Regression order: {}'.format(degree))
plt.plot(X_val_col, Y_pred_polyfit, color='yellow', alpha = 0.5,label = 'Polynomial fit order: {}'.format(degree_fit))
plt.title("Hydrogen concentration vs Resistance, Regression Models: Expected & Actual")
plt.ylabel('Concentration', fontsize = 16), plt.xlabel('Resistance', fontsize = 16)
plt.legend()
plt.show()
