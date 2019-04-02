from theano import tensor as T
import numpy as np
import h5py
import pickle as pkl

import neuralnet as nn


def unnormalize(x):
    x = np.clip(x, 0., 1.)
    return x


def std(x, axis):
    return T.sqrt(T.var(x, axis=axis) + 1e-8)


def convert_kernel(kernel):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.
    Also works reciprocally, since the transformation is its own inverse.
    # Arguments
        kernel: Numpy array (3D, 4D or 5D).
    # Returns
        The converted kernel.
    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = np.asarray(kernel)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return np.copy(kernel[tuple(slices)])


def prep(x):
    conv = nn.Conv2DLayer((1, 3, 224, 224), 3, 1, no_bias=False, activation='linear', filter_flip=False, border_mode='valid')
    kern = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], 'float32')[:, :, None, None]
    conv.W.set_value(kern)
    conv.b.set_value(np.array([-103.939, -116.779, -123.68], 'float32'))
    return conv(x)


class VGG19(nn.Sequential):
    def __init__(self, input_shape, param_file, name='vgg19'):
        super(VGG19, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv1_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv1_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool1'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool2'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_2'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_3'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool3'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv4_1'))
        self.load_params(param_file)

    def get_output(self, input, weights=None):
        input = prep(input)
        if weights is None:
            return super(VGG19, self).get_output(input)
        else:
            out = input
            idx = 0
            for layer in self:
                out = layer(out)
                if 'maxpool' not in layer.layer_name:
                    mean = T.mean(out, 1, keepdims=True)
                    out = (out - mean) * weights[idx].dimshuffle('x', 0, 'x', 'x')
                    out += mean
                    idx += 1
            return out

    def get_output_for(self, input, index, weights=None):
        input = prep(input)
        if weights is None:
            out = self[:index](input)
            return out
        else:
            out = input
            idx = 0
            for layer in self[:index]:
                out = layer(out)
                if 'maxpool' not in layer.layer_name:
                    mean = T.mean(out, 1, keepdims=True)
                    out = (out - mean) * weights[idx].dimshuffle('x', 0, 'x', 'x')
                    out += mean
                    idx += 1
            return out

    def load_params(self, param_file=None):
        if param_file is not None:
            f = h5py.File(param_file, mode='r')
            trained = [layer[1].value for layer in list(f.items())]
            weight_value_tuples = []
            for p, tp in zip(self.params, trained):
                if len(tp.shape) == 4:
                    tp = np.transpose(convert_kernel(tp), (3, 2, 0, 1))
                weight_value_tuples.append((p, tp))
            nn.utils.batch_set_value(weight_value_tuples)
            print('Pretrained weights loaded successfully!')


def norm_error(x, y):
    return T.sum((x - y) ** 2) / T.cast(x.shape[0], 'float32')


class Encoder(nn.Sequential):
    def __init__(self, input_shape, param_file, name='Encoder'):
        super(Encoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(VGG19(self.output_shape, param_file, name=name+'/vgg19'))
        self.append(nn.AdaIN2DLayer(self.output_shape, layer_name=name+'/adain'))

    def get_output(self, input, weights):
        if isinstance(input, (tuple, list)):
            x, y = input
            x, y = self[self.layer_name+'/vgg19'](x), self[self.layer_name+'/vgg19'](y, weights)
        else:
            num_ins = input.shape[0] // 2
            x, y = input[:num_ins], input[num_ins:]
            x, y = self[self.layer_name+'/vgg19'](x), self[self.layer_name+'/vgg19'](y, weights)
        muy, sigma = T.mean(y, (2, 3)), std(y, (2, 3))
        out = self[self.layer_name+'/adain'](x, T.concatenate((sigma, muy), 1))
        return out

    def vgg19_loss(self, x, y, weights):
        conv1_x = self[self.layer_name+'/vgg19'].get_output_for(x, 1, weights)
        conv1_y = self[self.layer_name+'/vgg19'].get_output_for(y, 1)
        loss = norm_error(T.mean(conv1_x, (2, 3)), T.mean(conv1_y, (2, 3))) \
               + norm_error(std(conv1_x, (2, 3)), std(conv1_y, (2, 3)))

        conv4_x = self[self.layer_name+'/vgg19'].get_output_for(x, 4, weights)
        conv4_y = self[self.layer_name+'/vgg19'].get_output_for(y, 4)
        loss += norm_error(T.mean(conv4_x, (2, 3)), T.mean(conv4_y, (2, 3))) \
                + norm_error(std(conv4_x, (2, 3)), std(conv4_y, (2, 3)))

        conv7_x = self[self.layer_name+'/vgg19'].get_output_for(x, 7, weights)
        conv7_y = self[self.layer_name+'/vgg19'].get_output_for(y, 7)
        loss += norm_error(T.mean(conv7_x, (2, 3)), T.mean(conv7_y, (2, 3))) \
                + norm_error(std(conv7_x, (2, 3)), std(conv7_y, (2, 3)))

        conv12_x = self[self.layer_name+'/vgg19'].get_output_for(x, 12, weights)
        conv12_y = self[self.layer_name+'/vgg19'].get_output_for(y, 12)
        loss += norm_error(T.mean(conv12_x, (2, 3)), T.mean(conv12_y, (2, 3))) \
                + norm_error(std(conv12_x, (2, 3)), std(conv12_y, (2, 3)))
        return loss


class Decoder(nn.Sequential):
    def __init__(self, input_shape, param_file, name='Decoder'):
        super(Decoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv1_1', activation='relu'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name+'/up1'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_1', activation='relu'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_2', activation='relu'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_3', activation='relu'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv2_4', activation='relu'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name + '/up2'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv3_1', activation='relu'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv3_2', activation='relu'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name + '/up3'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           layer_name=name + '/conv4_1', activation='relu'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 3, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False,
                           activation='tanh', layer_name=name + '/output'))
        self.load_params(param_file)

    def get_output(self, input):
        out = super(Decoder, self).get_output(input)
        out = out / 2. + .5
        return out

    def load_params(self, param_file=None):
        if param_file is not None:
            with open(param_file, 'rb') as f:
                weights = pkl.load(f)
                f.close()
            nn.utils.numpy2shared(weights, self.params)
