import os
import fnmatch
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, Lambda, ELU
from keras.optimizers import SGD


FISH_CATEGORIES = ['ALB',  'BET',  'DOL',  'LAG',  'NoF',  'OTHER',  'SHARK',  'YFT']

DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 144

def read_data(size=(DEFAULT_WIDTH, DEFAULT_HEIGHT)):
    data = {}
    folder_names = FISH_CATEGORIES
    print 'Reading files into memory...',
    cat_count = np.zeros(len(folder_names))
    for i in range(len(folder_names)):
        folder = folder_names[i]
        path = os.path.join('data', 'train', folder, '*.jpg')
        filenames = glob.glob(path)
        imgs = []
        cats = []
        for fn in filenames:
            img = cv2.imread(fn)
            img = cv2.resize(img, size)
            cat = np.zeros((len(folder_names)), dtype='float32')
            cat[i] = 1.0
            cat_count[i] += 1

            imgs.append(img)
            cats.append(cat)
            # print cat
            # print folder
            # cv2.imshow('test', img)
            # cv2.waitKey(0)

        data[folder] = (imgs, cats)
    print 'Done.'

    return data

def merge_data(data):
    X = []
    y = []
    for fish_type in FISH_CATEGORIES:
        imgs, cats = data[fish_type]
        for i in range(len(imgs)):
            X.append(imgs[i])
            y.append(cats[i])

    return np.asarray(X), np.asarray(y)

def get_model():
    row, col, ch = DEFAULT_HEIGHT, DEFAULT_WIDTH, 3
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(32, 5, 5, border_mode='same', dim_ordering='tf'))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, border_mode='same',  dim_ordering='tf'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(64, 3, 3, border_mode='same',  dim_ordering='tf'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='same',  dim_ordering='tf'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(128, 3, 3, border_mode='same',  dim_ordering='tf'))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, border_mode='same',  dim_ordering='tf'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(256, 3, 3, border_mode='same',  dim_ordering='tf'))
    model.add(ELU())
    model.add(Convolution2D(256, 3, 3, border_mode='same',  dim_ordering='tf'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    # model.add(Convolution2D(512, 3, 3, border_mode='same', dim_ordering='tf'))
    # model.add(ELU())
    # model.add(Convolution2D(512, 3, 3, border_mode='same', dim_ordering='tf'))
    # model.add(ELU())
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(8))
    model.add(Activation('sigmoid'))

    print model.summary()

    return model


def main():
    model = get_model()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    data = read_data()
    X_all, y_all = merge_data(data)


    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=23)


    model.fit(X_train, y_train, batch_size=32, nb_epoch=20,
                  validation_split=0.2, verbose=1, shuffle=True)

    preds = model.predict(X_test, verbose=1)
    print "Validation Log Loss: " + repr(log_loss(y_test, preds))

if __name__ == '__main__':
    main()
