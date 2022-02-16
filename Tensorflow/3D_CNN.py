import tensorflow as tf
import json, os
import time
import numpy as np
from sklearn import metrics
from data_tf import DataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
epoch = 100
# path
save_early_path = '/home/zxs/model/checkpoint_t2.hdf5'
save_hist_path = '/home/zxs/model/hist_t2.pckl'

# data
n_filters = [32,64,128]
pool_k = [2, 2]
conv_k = [5,5]

batch_size = 100
n_samplest = 8000
n_samplesv = 1000
n_samplesp = 1000
partition = {'train': ['Idlt-' + str(i) for i in range(n_samplest)],
             'valid': ['Idlv-' + str(i) for i in range(n_samplesv)],
             'pred': ['Idlp-' + str(i) for i in range(n_samplesp)]}

train_generator = DataGenerator(list_IDs=partition['train'], batch_size=batch_size)
valid_generator = DataGenerator(list_IDs=partition['valid'], batch_size=batch_size)

steps_per_epoch = n_samplest // batch_size
eval_steps_per_epoch = n_samplesv // batch_size

#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():

model_exists = os.path.exists("weights.f18.hdf5")
if (model_exists):
    print (1)
else:
    inputs = tf.keras.layers.Input((66, 66, 660, 1))
    x=tf.keras.layers.Conv3D(
                    filters = n_filters[0],
                    kernel_size=conv_k[0],
                    strides=2,
                    input_shape=(66,66,660,1),
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0) * 9**(-3/2)),
                    bias_initializer='zeros',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    bias_regularizer=None,
                    data_format='channels_last'
    )(inputs)
    x=tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(x)
    x=tf.keras.layers.Activation('relu')(x)

    x=tf.keras.layers.MaxPooling3D(pool_size=(pool_k[0], pool_k[0], pool_k[0]),data_format='channels_last')(x)
    x=tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format='channels_last')(x)
    x=tf.keras.layers.Conv3D(
                    filters = n_filters[1],
                    kernel_size=conv_k[0],
                    strides=1,
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0) * 9**(-3/2)),
                    bias_initializer='zeros',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    bias_regularizer=None,
                    data_format='channels_last'
    )(x)
    x=tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(x)
    x=tf.keras.layers.Activation('relu')(x)

    x=tf.keras.layers.MaxPooling3D(pool_size=(pool_k[0], pool_k[0], pool_k[0]),data_format='channels_last')(x)
    x=tf.keras.layers.SpatialDropout3D(0.4, data_format='channels_last')(x)

    x=tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format='channels_last')(x)
    x=tf.keras.layers.Conv3D(
                    filters = n_filters[2],
                    kernel_size=conv_k[1],
                    strides=1,
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0) * 9**(-3/2)),
                    bias_initializer='zeros',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    bias_regularizer=None,
                    data_format='channels_last'
    )(x)
    x=tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(x)
    x=tf.keras.layers.Activation('relu')(x)

    x=tf.keras.layers.MaxPooling3D(pool_size=(pool_k[0], pool_k[0], pool_k[0]),data_format='channels_last')(x)
    x=tf.keras.layers.SpatialDropout3D(0.4, data_format='channels_last')(x)

    x=tf.keras.layers.Flatten()(x)

    x=tf.keras.layers.Dense(units = 64,
                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                   bias_initializer='zeros',
                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                   bias_regularizer=None)(x)
    x=tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(x)
    x=tf.keras.layers.Activation('relu')(x)

    x=tf.keras.layers.Dropout(0.4)(x)
    x=tf.keras.layers.Dense(units = 16,
                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                   bias_initializer='zeros',
                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                   bias_regularizer=None)(x)
    x=tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(x)
    x=tf.keras.layers.Activation('relu')(x)

    x=tf.keras.layers.Dense(4)(x)
    x=tf.keras.layers.Activation('relu')(x)
    predictions1=tf.keras.layers.Dense(1,activation=None)(x)
    predictions2=tf.keras.layers.Dense(1,activation=None)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=[predictions1, predictions2])
    RMSprop = tf.keras.optimizers.RMSprop(lr=0.0085, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=RMSprop, loss='mean_absolute_error',metrics=['mean_absolute_error'],loss_weights=[0.8,1])

    ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    filepath=save_early_path
    ModelCheckpoint=tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=8,
                              verbose=1, mode='auto')
model.summary()

start_time = time.time()
hist = model.fit(train_generator,
               validation_data=valid_generator,
               steps_per_epoch=steps_per_epoch,
               validation_steps=eval_steps_per_epoch,
               epochs=epoch,
               callbacks = [ModelCheckpoint, ReduceLROnPlateau, EarlyStopping],
               #callbacks = [ModelCheckpoint, ReduceLROnPlateau],
               workers = 8,
               verbose=False)
end_time = time.time()
print('Done in {0:.6f} s'.format(end_time - start_time))
model.save("/home/zxs/model/model_tf0.h5")

#save or create concatenated history
import pickle
history_exists = os.path.exists('/home/zxs/model/hist_tn.pckl')
if (history_exists):
    f = open(save_hist_path, 'rb+')
    a = pickle.load(f)
    hist.history = hist.history
    for k2, v2 in hist.history.items():
        for k1, v1 in a.items():
            if k2 == k1:
                hist.history[k2] = v1+v2
    f.seek(0)
    f.truncate()
    pickle.dump(hist.history, f)
    f.close()
else:
    f = open(save_hist_path, 'wb')
    pickle.dump(hist.history, f)
    f.close()


# test results
test_generator = DataGenerator(list_IDs=partition['pred'], batch_size=1)
net = tf.keras.models.load_model(save_early_path)
true1, true2= [], []
pred1, pred2 = [], []

for i, data in enumerate(test_generator, 0):
    images, paras = data[0], data[1]
    outputs = net.predict(images)
    true1.append(paras[0])
    true2.append(paras[1])
    pred1.append(outputs[0][0])
    pred2.append(outputs[1][0])

true = np.hstack((np.array(true1),np.array(true2))).T
pred = np.hstack((np.array(pred1),np.array(pred2))).T

print (true.shape)
print (pred.shape)
#coefficient of determination
coefficient1 = metrics.r2_score(true[:,0],pred[:,0])
print ("coefficient of determination of Tvir(on test data): %0.5f" % coefficient1)
coefficient2 = metrics.r2_score(true[:,1],pred[:,1])
print ("coefficient of determination of Zeta(on test data): %0.5f" % coefficient2)
