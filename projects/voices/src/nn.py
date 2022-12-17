import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras import layers, regularizers
import numpy as np
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

train = False

############## DATA ############################
dis = 50  # Distance between points
x_list = np.array(range(0, 100000, dis))
y_list = np.array(range(0, 10000, dis))
x_1, y_1 = np.meshgrid(x_list, y_list)   # a grid is created by standard straighten

x_train = np.array([x_1[:, i] for i in range(x_1.shape[1])]).reshape(-1, )
y_train = np.array([y_1[:, i] for i in range(y_1.shape[1])]).reshape(-1, )

y_train = np.concatenate((y_train.reshape(-1, 1), x_train.reshape(-1, 1)), axis=-1)
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

def model():
  inputs = Input(shape=[2, 1])
  x = LSTM(units=10, return_sequences=True, activation='relu')(inputs)
  x = LSTM(units=10, return_sequences=False, activation='relu')(x)
  x = tf.keras.layers.Dense(1,   activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(x)
  m = Model(inputs, x, name='resnet34')
  return m

model_full = model()
model_full.summary()

if train:
  model_full.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
  model_full.fit(y_train, X_train,
            epochs=100,
            shuffle=True,
            batch_size=32,
            validation_data = (y_test, X_test))

  model_full.save('model.h5')