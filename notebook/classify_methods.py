import os
import shutil

from PIL import Image as pil_image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import h5py
import numpy as np
import pandas
import progressbar
import requests

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils


def get_photo(url):
    """ download photo from cress api """
    fn = url.split('/')[-1]
    cycle = url.split('/')[-2]
    dirname = os.path.join('data/photo', cycle)
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, fn)
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as fp:
            shutil.copyfileobj(response.raw, fp)
        del response
    return filename


def load_image(fn, image_width, image_height):
    """ load image and convert to array """
    img = pil_image.open(fn)
    img = img.resize([image_width, image_height])
    img_a = np.asarray(img)
    return img_a


def img_cache(url, cycle, image_width, image_height):
    """ cache image array in hdf5 for faster loading """
    fn = url.split('/')[-1].split('.')[0] + '-' + str(image_width) + '.hdf5'
    dirname = os.path.join('data/cache', str(cycle))
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, fn)
    if not os.path.exists(filename):
        img_a = load_image(get_photo(url), image_width, image_height)
        x = img_a.astype('float32')/255
        with h5py.File(filename, "w") as fp:
            fp.create_dataset("x", data=x, compression="gzip", compression_opts=9)
        return x
    with h5py.File(filename,'r') as fp:
        return fp['x'][()]


def get_dataset(cycle, image_width, image_height, threshold, seed=42):
    # get csv of sensors and photos and correlate watermark to photos
    from bootstrap_cress import get_csv_files
    get_csv_files(cycle)

    df_photos = pandas.read_csv("data/photo_cycle_{}_enriched.csv".format(cycle))
    bar = progressbar.ProgressBar(max_value=len(df_photos))
    x = []
    y = []
    watermarks = []
    for idx, photo_ds in bar(df_photos.iterrows()):
        url = photo_ds['photo']
        if photo_ds['watermark'] > threshold:
            cls = 1
        else:
            cls = 0
        watermarks.append(photo_ds['watermark'])
        x.append(img_cache(url, cycle, image_width, image_height))
        y.append(cls)
    X = np.array(x)
    save_wet_dry_split(cycle, watermarks)
    return X, y


def render_dataset(cycle, image_width, image_height, threshold, seed=42):
    X, y = get_dataset(cycle, image_width, image_height, threshold)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)
    cls_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return X_train, X_test, Y_train, Y_test, cls_weight


def save_wet_dry_split(cycle, watermarks):
    from collections import defaultdict
    data = defaultdict(list)
    for threshold in [3000, 6000, 8000]:
        filename = os.path.join('data', 'dry-wet-split--{}.csv'.format(cycle))
        wet_count = len(list(filter(lambda x: x > threshold, watermarks)))
        dry_count = len(watermarks) - wet_count
        data['cycle'].append(cycle)
        data['count'].append(len(watermarks))
        data['threshold'].append(threshold)
        data['dry_count'].append(dry_count)
        data['wet_count'].append(wet_count)
    pandas.DataFrame.from_dict(data).to_csv(filename, sep='|')


def save_scores(fn, score_test, score_train):
    filename = os.path.join('data', fn + '--scores.csv')
    scores = {
        'loss_test': [score_test[0]],
        'acc_test': [score_test[1]],
        'loss_train': [score_train[0]],
        'acc_train': [score_train[1]]
    }
    pandas.DataFrame.from_dict(scores).to_csv(filename, sep='|')


def get_model(image_width, image_height, dense_param):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(image_height, image_width, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(dense_param, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
