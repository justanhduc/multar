from scipy import  misc
import numpy as np
import neuralnet as nn
import csv
from random import shuffle
import os


def prep_image(im, size=256, resize=512):
    h, w, _ = im.shape
    if h < w:
        new_sh = (resize, int(w * resize / h))
    else:
        new_sh = (int(h * resize / w), resize)
    im = misc.imresize(im, new_sh, interp='bicubic')

    # Crop random
    im = nn.utils.crop_random(im, size)
    im = im.astype('float32')
    return np.transpose(im[None], (0, 3, 1, 2)) / 255.


def prep_image_test(im, size=512, resize=550):
    im = nn.utils.crop_center(im, [size, size], resize)
    im = im.astype('float32')
    return np.transpose(im[None], (0, 3, 1, 2)) / 255.


class DataManager(nn.DataManager):
    def __init__(self, placeholders, path, bs, n_epochs, shuffle=False, **kwargs):
        super(DataManager, self).__init__(None, placeholders, path=path, batch_size=bs, n_epochs=n_epochs, shuffle=shuffle, **kwargs)
        self.load_data()

    def load_data(self):
        source = os.listdir(self.path[0])
        source = [self.path[0] + '/' + f for f in source]
        if 'val' in self.path[0] or 'test' in self.path[0]:
            source = source[:self.kwargs.get('num_val_imgs', 100)]

        file = open(os.path.join(self.kwargs.get('style_path'), self.path[1]), 'r')
        contents = csv.reader(file, delimiter=',')
        style = [self.kwargs.get('style_path') + '/' + row[0] for row in contents]

        shuffle(source)
        shuffle(style)

        self.dataset = (source, style)
        self.data_size = min(len(source), len(style))

    def generator(self):
        source, style = list(self.dataset[0]), list(self.dataset[1])

        if self.shuffle:
            shuffle(source)
            shuffle(style)

        if len(source) > len(style):
            source = source[:len(style)]
        else:
            style = style[:len(source)]

        for idx in range(0, self.data_size, self.batch_size):
            source_batch = source[idx:idx + self.batch_size]
            style_batch = style[idx:idx + self.batch_size]
            imgs, stys = [], []
            exit = False
            for sou_f, sty_f in zip(source_batch, style_batch):
                try:
                    img, sty = misc.imread(sou_f), misc.imread(sty_f)
                except FileNotFoundError:
                    exit = True
                    break

                if len(img.shape) < 3 or len(sty.shape) < 3:
                    exit = True
                    break

                imgs.append(prep_image_test(img) if 'test' in self.path[0] else prep_image(img))
                stys.append(prep_image_test(sty, 256, 512) if 'test' in self.path[0] else prep_image(sty))

            if exit:
                continue

            imgs = np.concatenate(imgs)
            stys = np.concatenate(stys)
            yield imgs, stys


def get_weights(vgg_info):
    weights = [np.random.uniform(.5, 1.5, size=s).astype('float32') for s in vgg_info]
    return weights
