import sys
import os
import numpy as np

import theano
import theano.tensor as T
import lasagne as nn

sys.path.append("..")
import utils as u
import config as c
from batch_norm_layer import batch_norm as bn
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.layers import GlobalPoolLayer, NonlinearityLayer, InverseLayer

def build_fcae(input_var, channels=1):
    ret = {}
    ret['input'] = layer = InputLayer(shape=(None, channels, None, None), input_var=input_var)
    ret['conv1'] = layer = bn(Conv2DLayer(layer, num_filters=128, filter_size=5, pad='full'))
    ret['pool1'] = layer =  MaxPool2DLayer(layer, pool_size=2)
    ret['conv2'] = layer = bn(Conv2DLayer(layer, num_filters=256, filter_size=3, pad='full'))
    ret['pool2'] = layer = MaxPool2DLayer(layer, pool_size=2)
    ret['conv3'] = layer = bn(Conv2DLayer(layer, num_filters=32, filter_size=3, pad='full'))
    ret['enc'] = layer = GlobalPoolLayer(layer)
    ret['ph1'] = layer = NonlinearityLayer(layer, nonlinearity=None)
    ret['ph2'] = layer = NonlinearityLayer(layer, nonlinearity=None)
    ret['unenc'] = layer = bn(InverseLayer(layer, ret['enc']))
    ret['deconv3'] = layer = bn(Conv2DLayer(layer, num_filters=256, filter_size=3))
    ret['depool2'] = layer = InverseLayer(layer, ret['pool2'])
    ret['deconv2'] = layer = bn(Conv2DLayer(layer, num_filters=128, filter_size=3))
    ret['depool1'] = layer = InverseLayer(layer, ret['pool1'])
    ret['output'] = layer = Conv2DLayer(layer, num_filters=1, filter_size=5,
                                     nonlinearity=nn.nonlinearities.sigmoid)
    return ret

def main(num_epochs = 20):

    print("Building model and compiling functions...")
    input_var = T.tensor4('inputs')
    fcae = build_fcae(input_var)

    output = nn.layers.get_output(fcae['output'])
    output_det = nn.layers.get_output(fcae['output'], deterministic=True)
    loss = nn.objectives.binary_crossentropy(output, input_var).mean()
    test_loss = nn.objectives.binary_crossentropy(output_det, input_var).mean()

    # ADAM updates
    params = nn.layers.get_all_params(fcae['output'], trainable=True)
    updates = nn.updates.adam(loss, params, learning_rate=1e-3)
    train_fn = theano.function([input_var], loss, updates=updates)
    val_fn = theano.function([input_var], test_loss)
    ae_fn = theano.function([input_var], nn.layers.get_output(fcae['output']))

    data = u.DataH5PyStreamer(os.path.join(c.external_data, 'mnist.hdf5'), batch_size=128)
    hist = u.train_with_hdf5(data, num_epochs=num_epochs,
            train_fn = train_fn, test_fn = val_fn,
            max_per_epoch=40,
            tr_transform = lambda x: u.raw_to_floatX(x[0], pixel_shift=0.),
            te_transform = lambda x: u.raw_to_floatX(x[0], pixel_shift=0.))

    u.save_params(fcae['output'], 'fcae_params_{}.npz'.format(np.asarray(hist)[-1,-1]))

    from PIL import Image
    from matplotlib import pyplot as plt

    streamer = data.streamer()
    imb = next(streamer.get_epoch_iterator())
    batch = u.raw_to_floatX(imb[0], pixel_shift=0.).transpose((0,1,3,2))

    orig_dim = 28
    im = Image.new("RGB", (orig_dim*20, orig_dim*20))
    for j in xrange(10):
        dim = orig_dim
        orig_im = Image.fromarray(u.get_picture_array(batch,
            np.random.randint(batch.shape[0]), shift=0.0))
        im.paste(orig_im.resize((2*orig_dim, 2*orig_dim), Image.ANTIALIAS),
                box=(0,j*orig_dim*2))
        new_im = {}
        for i in xrange(9):
            new_im = orig_im.resize((dim, dim), Image.ANTIALIAS)
            new_im = ae_fn(u.arr_from_img(new_im, shift=0.).reshape(1,-1,dim,dim))
            new_im = Image.fromarray(u.get_picture_array(new_im, 0, shift=0.))\
                    .resize((orig_dim*2, orig_dim*2), Image.ANTIALIAS)
            im.paste(new_im, box=((i+1)*orig_dim*2, j*orig_dim*2))
            dim = int(dim * 1.2)
    im.save('increasing_size_autoencoded.jpg')

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
