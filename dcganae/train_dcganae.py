import sys
import os
import numpy as np
import time

import lasagne as nn
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer
from lasagne.layers import Upscale2DLayer, ReshapeLayer

sys.path.append("..")
import utils as u
import config as c
from batch_norm_layer import batch_norm

def build_nets(input_var, channels=1, do_batchnorm=True, z_dim=100):
    
    def ns(shape):
        ret=list(shape)
        ret[0]=[0]
        return tuple(ret)
    
    ret = {}
    bn = batch_norm if do_batchnorm else lambda x:x
    ret['ae_in'] = layer = InputLayer(shape=(None,channels,28,28), input_var=input_var)
    ret['ae_conv1'] = layer = bn(Conv2DLayer(layer, num_filters=64, filter_size=5))
    ret['ae_pool1'] = layer = MaxPool2DLayer(layer, pool_size=2)
    ret['ae_conv2'] = layer = bn(Conv2DLayer(layer, num_filters=128, filter_size=3))
    ret['ae_pool2'] = layer = MaxPool2DLayer(layer, pool_size=2)
    ret['ae_enc'] = layer = DenseLayer(layer, num_units=z_dim,
            nonlinearity=nn.nonlinearities.tanh)
    ret['ae_unenc'] = layer = bn(nn.layers.DenseLayer(layer,
        num_units = np.product(nn.layers.get_output_shape(ret['ae_pool2'])[1:])))
    ret['ae_resh'] = layer = ReshapeLayer(layer,
            shape=ns(nn.layers.get_output_shape(ret['ae_pool2'])))
    ret['ae_depool2'] = layer = Upscale2DLayer(layer, scale_factor=2)
    ret['ae_deconv2'] = layer = bn(Conv2DLayer(layer, num_filters=64, filter_size=3,
        pad='full'))
    ret['ae_depool1'] = layer = Upscale2DLayer(layer, scale_factor=2)
    ret['ae_out'] = Conv2DLayer(layer, num_filters=1, filter_size=5, pad='full',
            nonlinearity=nn.nonlinearities.sigmoid)
    
    ret['disc_in'] = layer = InputLayer(shape=(None,channels,28,28), input_var=input_var)
    ret['disc_conv1'] = layer = bn(Conv2DLayer(layer, num_filters=64, filter_size=5))
    ret['disc_pool1'] = layer = MaxPool2DLayer(layer, pool_size=2)
    ret['disc_conv2'] = layer = bn(Conv2DLayer(layer, num_filters=128, filter_size=3))
    ret['disc_pool2'] = layer = MaxPool2DLayer(layer, pool_size=2)
    ret['disc_hid'] = layer = bn(DenseLayer(layer, num_units=100))
    ret['disc_out'] = DenseLayer(layer, num_units=1, nonlinearity=nn.nonlinearities.sigmoid)
    
    return ret

def main(batch_size = 128, num_epochs = 30, learning_rate = 1e-4, d_ratio=4):

    print("Building model and compiling functions...")
    X = T.tensor4('inputs')
    Z = T.matrix('Z')
    ldict = build_nets(input_var = X)

    def build_loss(deterministic):
        # this currently has the problem that these 3 expressions come from 3 different
        # get_output calls, so they won't return the same mask if dropout or other
        # noise is used. Currently not using dropout so not a problem.
        ae = nn.layers.get_output(ldict['ae_out'], deterministic=deterministic)
        disc_real = nn.layers.get_output(ldict['disc_out'], deterministic=deterministic)
        disc_fake = nn.layers.get_output(ldict['disc_out'], { ldict['disc_in']:ae },
                deterministic=deterministic)

        d_cost_real=nn.objectives.binary_crossentropy(disc_real, T.ones_like(disc_real)).mean()
        d_cost_fake=nn.objectives.binary_crossentropy(disc_fake, T.zeros_like(disc_fake)).mean()
        g_cost=nn.objectives.binary_crossentropy(disc_fake, T.ones_like(disc_fake)).mean()
        d_cost = d_cost_real + d_cost_fake
        mse = nn.objectives.squared_error(ae, X).mean()
        return g_cost, d_cost, mse

    g_cost, d_cost, _ = build_loss(deterministic=False)
    g_cost_det, d_cost_det, mse = build_loss(deterministic=True)

    d_lr = theano.shared(nn.utils.floatX(learning_rate))
    d_params = nn.layers.get_all_params(ldict['disc_out'], trainable=True)
    d_updates = nn.updates.adam(d_cost, d_params, learning_rate=d_lr)
    g_lr = theano.shared(nn.utils.floatX(learning_rate))
    g_params = nn.layers.get_all_params(ldict['ae_out'], trainable=True)
    g_updates = nn.updates.adam(g_cost, g_params, learning_rate=g_lr)

    _train_g = theano.function([X], g_cost, updates=g_updates)
    _train_d = theano.function([X], d_cost, updates=d_updates)
    _test_g = theano.function([X], g_cost_det)
    _test_d = theano.function([X], d_cost_det)
    mse = theano.function([X], mse)

    print("Starting training...")
    data = u.DataH5PyStreamer(os.path.join(c.external_data, 'mnist.hdf5'),
            batch_size=batch_size)
    def train_fn(x, d_ratio=d_ratio):
        cost = 0
        for i in xrange(d_ratio):
            cost += _train_d(x)
        cost += _train_g(x)
        return cost
    transform_data = lambda x: u.raw_to_floatX(x[0], pixel_shift=0.)

    hist = u.train_with_hdf5(data, num_epochs=num_epochs, train_fn = train_fn, test_fn = mse,
                            tr_transform=transform_data, te_transform=transform_data)

if __name__ == '__main__':
    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    import argparse, inspect
    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name,arg_type in zip(ma.args[-len(ma.defaults):],[type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
