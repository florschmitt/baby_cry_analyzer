import os
from keras.utils import to_categorical
from PIL import Image
import numpy as np

from main import baby_model


def load_spectrogram():
    data_path = '/content/drive/MyDrive/Baby_cry_data/Spectograms'
    classes = {'belly_pain': 0,
               'burping': 1,
               'discomfort': 2,
               'hungry': 3,
               'tired': 4}
    imgs = []
    labels = []
    for (cl, i) in classes.items():
        # images_path = ''
        images_path = [elt for elt in os.listdir(os.path.join(data_path, cl)) if elt.find('.png') > 0]
        # print('images_path:  ', images_path)
        for img in images_path[:100]:
            # path = ''
            path = os.path.join(data_path, cl, img)
            # print('second for loop', path)
            # print(os.path.exists(path))
            if os.path.exists(path):
                image = Image.open(path)
                # print(np.array(image))
                # return 'bye...'
                image = image.convert('RGB')
                image = image.resize((224, 224))
                imgs.append(np.array(image))
                labels.append(i)

    X = np.array(imgs)
    num_classes = len(set(labels))
    y = to_categorical(labels, num_classes)

    p = np.random.permutation(len(X))
    X, y = X[p], y[p]

    first_split = int(len(imgs) / 6.)
    second_split = first_split + int(len(imgs) * 0.2)
    X_test, X_val, X_train = X[:first_split], X[first_split:second_split], X[second_split:]
    y_test, y_val, y_train = y[:first_split], y[first_split:second_split], y[second_split:]

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes
