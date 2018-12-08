import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras import initializers

(X_train_a, y_train_a), (X_test, y_test) = mnist.load_data()


print("Machine Learning assignment1_code")

print("X_train's dimensions", X_train_a.shape)
print("y_train's dimensions", y_train_a.shape)
print("X_train's dimensions", X_test.shape)
print("y_train's dimensions", y_test.shape)

# building the input vector from the 28x28 pixels
X_train_a = X_train_a.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train_a = X_train_a.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train_a /= 255
X_test /= 255


idx = np.random.randint(60000, size=10000)
X_train = X_train_a[idx,:]
y_train = y_train_a[idx]
# print the final input shape ready for training
print("Here is a train matrix dimension", X_train.shape)
print("Here is a test matrix dimension", X_test.shape)


print(np.unique(y_train, return_counts=True))

n_classes = 10
print("here is a dimension before one-hot encoding were: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("here are the dimensions after one-hot encoding were : ", Y_train.shape)


acc_train = [[[0 for kk in xrange(4)] for jj in xrange(3)] for ii in xrange(3)]
loss_train = [[[0 for kkk in xrange(4)] for jjj in xrange(3)] for iii in xrange(3)]
acc_test = [[[0 for kk in xrange(4)] for jj in xrange(3)] for ii in xrange(3)]
loss_test = [[[0 for kkk in xrange(4)] for jjj in xrange(3)] for iii in xrange(3)]
# matrix_acc[0][0][3] = [1,2,3,4]
# print(matrix_acc)

stdv = 0.05
stdname = '005'
archname = '1'
funct = 'linear'
for i in range(3):
	for j in range(3):
		for k in range(4):
			archname='1'
			if i==0:
				stdv=0.05
				stdname='005'
			elif i==1:
				stdv=0.5
				stdname='050'
			else:
				stdv=1
				stdname='100'

			if k==0:
				funct = 'linear'
			elif k==1:
				funct = 'sigmoid'
			elif k==2:
				funct = 'tanh'
			else:
				funct = 'relu'

			model = Sequential()
			model.add(Dense(128, input_shape=(784,), kernel_initializer=initializers.random_normal(stddev=stdv)))
			model.add(Activation(funct))                            
			model.add(Dropout(0.2))

			if j>0:
				model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=stdv)))
				model.add(Activation(funct))
				model.add(Dropout(0.2))
				archname = '2'

			if j>1:
				model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=stdv)))
				model.add(Activation(funct))
				model.add(Dropout(0.2))
				archname = '3'

			model.add(Dense(10,  kernel_initializer=initializers.random_normal(stddev=stdv)))
			model.add(Activation('softmax'))

			model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

			history = model.fit(X_train, Y_train,
			          batch_size=128, epochs=15,
			          verbose=2,
			          validation_data=(X_test, Y_test))

			save_dir = "./results/"
			model_name = 'ML4DM_assignment_code'+'_arch_'+archname+'__std_'+stdname+'__act_'+funct+'.h5'
			model_path = os.path.join(save_dir, model_name)
			model.save(model_path)
			print('Let us Save a trained model at %s  for future use' % model_path)

			acc_train[i][j][k] = history.history['acc']
			loss_train[i][j][k] = history.history['loss']
			acc_test[i][j][k] = history.history['val_acc']
			loss_test[i][j][k] = history.history['val_loss']
			
			fig = plt.figure()
			plt.subplot(2,1,1)
			plt.plot(history.history['acc'])
			plt.plot(history.history['val_acc'])
			plt.title('accuracy level'+'_arrch_'+archname+'__standers_'+stdname+'__act_'+funct)
			plt.ylabel('accuracy on y axis')
			plt.xlabel('epoch on x axis')
			plt.legend(['train model', 'test model'], loc='lower right')

			plt.subplot(2,1,2)
			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.title('loss level'+'_arch_'+archname+'__standers_'+stdname+'__act_'+funct)
			plt.ylabel('loss on y axis')
			plt.xlabel('epoch on x axis')
			plt.legend(['train model', 'test model'], loc='upper right')

			plt.tight_layout()
			plt.show()
			fig

			mnist_model = load_model("./results/"+model_name)
			loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

			print("Total Loss of test", loss_and_metrics[0])
			print("Overall Test Accuracy", loss_and_metrics[1])

			mnist_model = load_model("./results/"+model_name)
			predicted_classes = mnist_model.predict_classes(X_test)

			correct_indices = np.nonzero(predicted_classes == y_test)[0]
			incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
			print()
			print()
			print()
			print(len(correct_indices)," hey it is classified correctly")
			print(len(incorrect_indices)," it is classified incorrectly")


sio.savemat('./acc_train.mat', mdict={'acc_train': acc_train})
sio.savemat('./loss_train.mat', mdict={'loss_train': loss_train})
sio.savemat('./acc_test.mat', mdict={'acc_test': acc_test})
sio.savemat('./loss_test.mat', mdict={'loss_test': loss_test})

