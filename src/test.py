import argparse
parser = argparse.ArgumentParser('Multimodal arbitrary style transfer')
parser.add_argument('input_path', type=str, help='path to a folder of input images')
parser.add_argument('style_path', type=str, help='path to a folder of style images')
parser.add_argument('weight_file', type=str, help='path to a trained weight file')
parser.add_argument('-n', '--n_styles', type=int, default=5, help='number of outputs per style')
parser.add_argument('--gpu', type=int, default=0, help='gpu nummber')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from theano import tensor as T
import neuralnet as nn
import numpy as np
from scipy import misc
import time

from net import Encoder, Decoder
from data_loader import prep_image_test, get_weights

input_size = (3, 256, 256)
vgg_param_file = 'vgg19_weights_normalized.h5'
vgg_info = [64, 64, 128, 128, 256, 256, 256, 256, 512]
num_styles = args.n_styles
style_img_folder = args.style_path
input_img_folder = args.input_path
dec_param_file = args.weight_file


def test_random():
    enc = Encoder((None,) + input_size, vgg_param_file)
    dec = Decoder(enc.output_shape, dec_param_file)

    X = T.tensor4('input')
    Y = T.tensor4('style')
    weights = [T.vector('weights') for i in range(len(vgg_info))]

    nn.set_training_off()
    X_styled = dec(enc((X, Y), weights))
    test = nn.function([X, Y] + weights, X_styled, name='test generator')

    style_folder = os.listdir(style_img_folder)
    input_folder = os.listdir(input_img_folder)

    time_list = []
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    for style_file in style_folder:
        sty_img = prep_image_test(misc.imread(style_img_folder + '/' + style_file))
        for input_file in input_folder:
            try:
                input_img = prep_image_test(misc.imread(input_img_folder + '/' + input_file))
            except ValueError:
                continue

            for i in range(num_styles):
                start = time.time()
                output = test(input_img, sty_img, *get_weights(vgg_info))
                time_list.append(time.time() - start)
                output = np.transpose(output[0], (1, 2, 0))
                misc.imsave(os.path.join('outputs', input_file[:-4] + '_' + style_file[:-4] + '_%d.jpg' % i), output)

    print('Took %f s/image' % np.mean(time_list))
    print('Testing finished!')


if __name__ == '__main__':
    test_random()
