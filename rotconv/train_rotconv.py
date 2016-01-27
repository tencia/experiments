import os
import numpy as np
import sys

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import lasagne as nn

sys.path.append("..")
import utils as u
import config as c

srng = RandomStreams()
data = u.DataH5PyStreamer(os.path.join(c.external_data, 'cifar10.hdf5'), batch_size=128)

def transform_data(imb):
    # data augmentation: flip = -1 if we do flip over y-axis, 1 if not
    flip = -2*np.random.binomial(1, p=0.5) + 1
    return u.raw_to_floatX(imb[0], pixel_shift=0)[:,:,::flip], imb[1].flatten()

def rectify(X):
    return T.maximum(X, 0.)

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def conv_and_pool(input_expr, w, convs_mult, p_drop_conv):
    conv_w = w
    if convs_mult == 2:
        conv_w = T.concatenate([w, w[:,:,::-1,::-1]], axis=0)
    elif convs_mult == 4:
        conv_w = T.concatenate([w, w[:,:,::-1], w[:,:,:,::-1], w[:,:,::-1,::-1]], axis=0)
    e1 = rectify(conv2d(input_expr, conv_w))
    e2 = max_pool_2d(e1, (2, 2), ignore_border=False)
    return dropout(e2, p_drop_conv)

def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hid, convs_mult):
    l1 = conv_and_pool(X, w, convs_mult, p_drop_conv)
    l2 = conv_and_pool(l1, w2, convs_mult, p_drop_conv)
    l3 = conv_and_pool(l2, w3, convs_mult, p_drop_conv)
    l4 = rectify(conv2d(l3, w4))
    l4 = dropout(l4, p_drop_hid)
    l4 = T.flatten(l4, outdim=2)
    pyx = nn.nonlinearities.softmax(T.dot(l4, w_o))
    return pyx

def main(num_epochs=150, orig_filts=128, intermed_dim=64, batch_size=128,
        p_drop_conv=0.5, p_drop_hid=0.5, learning_rate=0.001,
        max_per_epoch=-1):
    X = T.ftensor4()
    Y = T.ivector()

    hist = {}
    accuracies = []
    nparams = []
    for convs_mult in [1,2,4]:
        num_filts= orig_filts / convs_mult
        norm_init = nn.init.Normal(std=0.01, mean=0.0)
        w = theano.shared(norm_init.sample((num_filts, 3, 3, 3)))
        w2 = theano.shared(norm_init.sample((num_filts, num_filts*convs_mult, 3, 3)))
        w3 = theano.shared(norm_init.sample((num_filts, num_filts*convs_mult, 3, 3)))
        w4 = theano.shared(norm_init.sample((intermed_dim, num_filts*convs_mult, 1, 1)))
        w_o = theano.shared(norm_init.sample((intermed_dim*3*3, 10)))

        py_x = model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hid, convs_mult=convs_mult)
        py_x_det = model(X, w, w2, w3, w4, w_o, 0., 0., convs_mult=convs_mult)
        y_x = T.argmax(py_x_det, axis=1)

        cost = nn.objectives.categorical_crossentropy(py_x, Y).mean()
        cost_det = nn.objectives.categorical_crossentropy(py_x_det, Y).mean()
        acc = nn.objectives.categorical_accuracy(py_x_det, Y).mean()
        params = [w, w2, w3, w4, w_o]
        updates = nn.updates.adam(cost, params, learning_rate=learning_rate)

        train_fn = theano.function([X, Y], outputs=cost, updates=updates,
                allow_input_downcast=True)
        test_fn = theano.function([X, Y], outputs=cost, allow_input_downcast=True)
        predict_fn = theano.function([X], outputs=y_x, allow_input_downcast=True)
        acc_fn = theano.function([X, Y], acc, allow_input_downcast=True)

        hist[convs_mult] = np.asarray(u.train_with_hdf5(data, num_epochs=num_epochs,
            train_fn=train_fn, test_fn=test_fn,
            max_per_epoch=max_per_epoch,
            tr_transform=transform_data, te_transform=transform_data))
        streamer = data.streamer(training=False)
        accur = 0
        batches = 0
        for tup in streamer.get_epoch_iterator():
            x,y = transform_data(tup)
            accur += acc_fn(x,y)
            batches += 1
        accuracies.append(accur/batches)
        nparams.append(sum([p.get_value().size for p in params]))

        np.savetxt('rotconv_hist_{}.csv'.format(convs_mult), hist[convs_mult], delimiter=',',
                fmt='%.5f')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    c1, = plt.plot(hist[1][:,1], label='orig')
    c2, = plt.plot(hist[2][:,1], label='2x')
    c4, = plt.plot(hist[4][:,1], label='4x')
    plt.ylabel('Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.legend(handles=[c1, c2, c4], loc=1)
    plt.savefig('validation_acc.jpg')
    np.savetxt('accuracies.csv', np.asarray(accuracies), delimiter=',', fmt='%.5f')
    np.savetxt('num_params.csv', np.asarray(nparams), delimiter=',', fmt='%d')

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
