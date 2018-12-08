# Author: Luv Patel (201501459@daiict.ac.in)




import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipy
import keras
from keras.datasets import mnist as mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.initializers import random_normal

batch_size = 128
num_classes = 10
epochs = 8

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

fns = ['linear', 'sigmoid', 'relu', 'tanh']
sds = [0.05,0.5,1]
val_ac_mat = []
val_l_mat = []
ac_mat = []
l_mat = []

for std_dev in sds:
    ac_sd = []
    l_sd = []
    val_ac_sd = []
    val_l_sd = []
    for i in range(3):
        ac_mod = []
        l_mod = []
        val_ac_mod = []
        val_l_mod = []
        for fn in fns:

            m = Sequential()
            m.add(Dense(128, activation = fn, input_shape = (784,), kernel_initializer = random_normal(stddev=std_dev)))

            if(i==1):
                m.add(Dense(64, activation = fn))
            if(i==2):
                m.add(Dense(64, activation = fn))
                m.add(Dense(64, activation = fn))

            m.add(Dense(num_classes, activation = 'softmax'))
            m.summary()
            m.compile(loss = 'categorical_crossentropy', optimizer = SGD(), metrics = ['accuracy'])
            record = m.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test,y_test))
            score = m.evaluate(x_test, y_test, verbose=0)
            m.save('sd: '+str(std_dev)+'fn: '+fn+'i: '+str(i+1)+".h5")
            ac_mod.append(record.history['acc'])
            l_mod.append(record.history['loss'])
            val_ac_mod.append(record.history['val_acc'])
            val_l_mod.append(record.history['val_loss'])

            plt.figure()
            plt.title("Accuracy vs epochs std_dev: "+str(std_dev)+' fn: '+fn+' i: '+str(i+1)+'.png')
            plt.plot(record.history['val_acc'],label="Validation-Accuracy")
            plt.plot(record.history['acc'],label = 'Training-Accuracy')
            plt.legend(loc='lower right')
            plt.savefig("Accuracy vs epochs std_dev: "+str(std_dev)+' fn: '+fn+' i: '+str(i+1)+'.png')
            plt.figure()
            plt.title("Loss vs epochs std_dev: "+str(std_dev)+' fn: '+fn+' i: '+str(i+1)+'.png')
            plt.plot(record.history['val_loss'],label = 'Validation-Loss')
            plt.plot(record.history['loss'], label = 'Training-Loss')
            plt.legend(loc='upper right')
            plt.savefig("Loss vs epochs std_dev: "+str(std_dev)+' fn: '+fn+' i: '+str(i+1)+'.png')
        
        ac_sd.append(ac_mod)
        l_sd.append(l_mod)
        val_ac_sd.append(val_ac_mod)
        val_l_sd.append(val_ac_mod)
    
    ac_mat.append(ac_sd)
    l_mat.append(ac_sd)
    val_ac_mat.append(val_ac_sd)
    val_l_mat.append(val_ac_sd)

np.reshape(ac_mat, newshape = (3,3,4,epochs))
np.reshape(l_mat, newshape = (3,3,4,epochs))
np.reshape(val_ac_mat, newshape = (3,3,4,epochs))
np.reshape(val_l_mat, newshape = (3,3,4,epochs))

ac_mat = np.array(ac_mat)
l_mat = np.array(l_mat)
val_ac_mat = np.array(val_ac_mat)
val_l_mat = np.array(val_l_mat)

dic = {}
dic['accuracy_matrix'] = ac_mat
dic['loss_matrix'] = l_mat
dic['validation_accuracy_matrix'] = val_ac_mat
dic['validation_loss_matrix'] = val_l_mat
scipy.savemat(mdict = dic, file_name = 'final_matrices.mat')