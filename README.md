### Fully convolutional autoencoder
Autoencoder that contains no fully connected layers. The encoding layer is calculated by global mean pooling over the last convolutional layer, which has as many filters as there are dimensions in the code.

### DCGAN autoencoder
Inspired by Alec Radford's DCGAN ( http://arxiv.org/abs/1511.06434 ). An attempt to train an autoencoder using generative adverserial training with the autoencoder acting as the generator network, and a separate discriminator network.

Based on code found here:
https://github.com/Newmu/dcgan_code

This approach trains poorly; the generator seems to find the degenerate solution of outputting a single solution designed to exploit the discriminator.

### Rotated convolutions
Compares performance obtained on the CIFAR-10 classification task using
- a typical CNN
- a CNN with half as many filters per layer, but each outputs its own activations and those of its weights if rotated 180 degrees
- a CNN with a quarter as many filters per layer, but each filter outputs both its own activations, and those of its weights if rotated 90, 180, and 270 degrees
