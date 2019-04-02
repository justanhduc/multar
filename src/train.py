import argparse

parser = argparse.ArgumentParser('Multimodal arbitrary style transfer')
parser.add_argument('img_path', help='path to the folder containing MS COCO train/val folders', type=str)
parser.add_argument('style_path', help='path to the WikiArt dataset', type=str)
parser.add_argument('--bs', help='batch size', type=int, default=8)
parser.add_argument('--weight', help='weight between two loss terms', default=1., type=float)
parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
parser.add_argument('--lr_decay', help='learning rate decay', default=5e-5, type=float)
parser.add_argument('--n_epochs', help='number of epochs', type=int, default=20)
parser.add_argument('--gpu', help='gpu number', type=int, default=1)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import time
import theano

from data_loader import *
from net import *

srng = theano.sandbox.rng_mrg.MRG_RandomStreams(np.random.randint(1, int(time.time())))
vgg_param_file = 'vgg19_weights_normalized.h5'
dec_param_file = 'decoder.npz'
model_name = 'multar'
input_path = args.img_path
style_path = args.style_path
num_val_imgs = 80
input_size = (3, 256, 256)
bs = args.bs
weight = args.weight
lr = args.lr
lr_decay_rate = args.lr_decay
n_epochs = args.n_epochs
print_freq = 100
val_freq = 500
vgg_info = [64, 64, 128, 128, 256, 256, 256, 256, 512]


def train():
    enc = Encoder((None,) + input_size, vgg_param_file)
    dec = Decoder(enc.output_shape, dec_param_file)

    X = T.tensor4('input')
    Y = T.tensor4('style')
    iter = T.scalar('iteration', 'int64')
    weights = [T.vector('weights') for i in range(len(vgg_info))]
    X_ = nn.placeholder((bs,) + input_size, name='input_plhd')
    Y_ = nn.placeholder((bs,) + input_size, name='style_plhd')
    lr_ = nn.placeholder(value=lr, name='lr_plhd')

    data_train = DataManager((X_, Y_), (
        os.path.join(input_path, 'MS_COCO_train'), 'style_train.csv'), bs, n_epochs, True, style_path=style_path)
    data_test = DataManager((X_, Y_), (
        os.path.join(input_path, 'MS_COCO_val'), 'style_val.csv'), bs, 1, style_path=style_path)

    nn.set_training_on()
    latent = enc(T.concatenate((X, Y)), weights)
    X_styled = dec(latent)
    latent_cycle = enc[0](X_styled)

    content_loss = nn.norm_error(latent, latent_cycle)
    style_loss = enc.vgg19_loss(Y, X_styled, weights=weights)
    loss = content_loss + weight * style_loss
    updates, _, _ = nn.adam(loss * 1e6, dec.trainable, lr_)
    nn.anneal_learning_rate(lr_, iter, 'inverse', decay=lr_decay_rate)
    train = nn.function([iter] + weights, [content_loss, style_loss], updates=updates, givens={X: X_, Y: Y_},
                        name='train generator')

    nn.set_training_off()
    X_styled1 = dec(enc(T.concatenate((X, Y)), weights))
    X_styled2 = dec(enc(T.concatenate((X, Y)), weights))
    X_styled3 = dec(enc(T.concatenate((X, Y)), weights))
    test1 = nn.function(weights, X_styled1, givens={X: X_, Y: Y_}, name='test generator 1')
    test2 = nn.function(weights, X_styled2, givens={X: X_, Y: Y_}, name='test generator 2')
    test3 = nn.function(weights, X_styled3, givens={X: X_, Y: Y_}, name='test generator 3')

    mon = nn.Monitor(model_name=model_name, print_freq=print_freq)
    print('Training...')
    for it in data_train:
        with mon:
            weights_ = get_weights(vgg_info)
            c_loss_, s_loss_ = train(it, *weights_)
            if np.isnan(c_loss_ + s_loss_) or np.isinf(c_loss_ + s_loss_):
                raise ValueError('Training failed because loss went nan!')
            mon.plot('content loss', c_loss_)
            mon.plot('style loss', s_loss_)
            mon.plot('learning rate', lr_.get_value())

            if it % val_freq == 0:
                for i in data_test:
                    weights_ = get_weights(vgg_info)
                    img_styled1 = test1(*weights_)
                    weights_ = get_weights(vgg_info)
                    img_styled2 = test2(*weights_)
                    weights_ = get_weights(vgg_info)
                    img_styled3 = test3(*weights_)
                    mon.hist('output histogram 1 %d' % i, img_styled1)
                    mon.hist('output histogram 2 %d' % i, img_styled2)
                    mon.hist('output histogram 3 %d' % i, img_styled3)
                    mon.imwrite('stylized image 1 %d' % i, img_styled1, callback=unnormalize)
                    mon.imwrite('stylized image 2 %d' % i, img_styled2, callback=unnormalize)
                    mon.imwrite('stylized image 3 %d' % i, img_styled3, callback=unnormalize)
                    mon.imwrite('input %d' % i, X_.get_value(), callback=unnormalize)
                    mon.imwrite('style %d' % i, Y_.get_value(), callback=unnormalize)
                mon.dump(nn.utils.shared2numpy(dec.params), 'decoder.npz', keep=5)
    print('Training finished!')


if __name__ == '__main__':
    train()
