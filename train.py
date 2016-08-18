from __future__ import print_function

import cv2
import numpy as np
from sklearn.cross_validation import KFold
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils
from keras import backend as K

from data import load_train_data, load_test_data

img_rows = 64 #160
img_cols = 80 #200
batch_size = 32
epochs = 50
folds = 5

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', init='he_normal')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def stan(X, Y=None, mean=None, std=None):
    X = X.astype('float32')
    if mean is None:
        mean = np.mean(X)  # mean for data centering
    if std is None:
        std = np.std(X)  # std for data normalization
    X-= mean
    X/= std
    if Y is not None:
        Y= Y.astype('float32')
        Y/= 255.  # scale masks to [0, 1]
        return X,Y,mean,std
    else:
        return X


def run_cross_validation(nfolds=5):
    random_state = 51
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    train_data, train_target = load_train_data()
    train_data, train_target, mean, std = stan(preprocess(train_data), preprocess(train_target))

    imgs_test, imgs_id_test = load_test_data()
    imgs_test = stan(preprocess(imgs_test), None, mean, std)

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(train_data), n_folds=nfolds,  shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_index, valid_index in kf:

        X_train, X_valid = train_data[train_index], train_data[valid_index]
        Y_train, Y_valid = train_target[train_index], train_target[valid_index]

        num_fold += 1

        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
            ModelCheckpoint('unet.' + str(num_fold) + '.hdf5', monitor='loss', save_best_only=True)
        ]

        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        model = get_unet()

        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        print('-'*30)
        print('Making predictions...')
        print('-'*30)

        ## Load best model
        model.load_weights('unet.' + str(num_fold) + '.hdf5')

        # Store valid predictions
        predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
        for i in range(len(valid_index)):
            yfull_train[valid_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(imgs_test, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)
        np.save('imgs_mask_test.fold' + str(num_fold) + '.npy', test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    np.save('imgs_mask_test.npy', test_res)
    #np.save('imgs_mask_train_cv.npy', yfull_train)

def run_main_model():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    train_data, train_target = load_train_data()
    train_data, train_target, mean, std = stan(preprocess(train_data), preprocess(train_target))

    imgs_test, imgs_id_test = load_test_data()
    imgs_test = stan(preprocess(imgs_test), None, mean, std)

    X_train, X_valid = train_data, train_data
    Y_train, Y_valid = train_target, train_target


    print('Main Model')
    print('Len train: ', len(X_train), len(Y_train))
    print('Len valid: ', len(X_valid), len(Y_valid))

    callbacks = [
        ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    ]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()

    # If existing model -> load it
    #model.load_weights('unet.hdf5')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
        shuffle=True, verbose=2, callbacks=callbacks)

    ## Load best model
    model.load_weights('unet.hdf5')

    test_prediction = model.predict(imgs_test, verbose=2)
    np.save('imgs_mask_test.npy', test_prediction)


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


if __name__ == '__main__':
    run_cross_validation(folds)
    #run_main_model()
