# StereoConvNet
Stereo fully convolutional neural network for depth map prediction from stereo images. The network is implemented using 
the nn library Lasagne, and training-time/over-fitting are reduced significantly with the help of the recently developed Batch normalization technique (http://arxiv.org/abs/1502.03167).

The network is fully convolutional, and takes a couple of grayscale stereoscopic images concatenated along the channel axis,
and ouputs  a single image representing the depth map. A series of convolutional and maxpooling layers followed by a series of upscalling and deconvolutional layers allow the network to
extract image disparity features at the smaller scale (object edges), and generate a smooth estimate of the depth map at the larger scale (full object). The main advantage of this technique over other methods from computer vision research (based on an explicit computation of image disparity) is its robustness, in particular the fact that it is able to produce smooth estimates of the depth map even on textureless region.

The traing/validation sets are created using the random virtual 3d scene generator (see https://github.com/LouisFoucard/DepthMap_dataset). The objective function used here is Eulerian distance. 
Thanks to batch normalization, training only takes 10 minutes (2000 images) on a K520 (AWS GPU), or 2 hours on a mac book pro.

Below are examples of a random 3d scene, its ground truth depth map, and the predictions computed with the fully convolutional stereo neural network:

![alt tag](https://github.com/LouisFoucard/StereoConvNet/blob/master/examples.png)


To check whether the network is truly learning stereo features, or merely learning to associate a closer distance to whatever object it sees on top of the background, let's run the same network (trained on stereo images) on 2 identical left images, instead of the stereo couple left/right images (similar to depthmap estimation from single images, which has been explored recently). In this case, there are no disparities between the 2 images, so this should throw the depthmap estimation off:

![alt tag](https://github.com/LouisFoucard/StereoConvNet/blob/master/check.png)

We see indeed that the depth map calculated from 2 left images is off (last column), and objects are simply not registered and disappear altogether. This seems to support the idea that the network is learning true stereo features based on image disparity.


The architecture is roughly as follows:

Input layer (batch_size x 2 x width x height)

conv1 (batch_size x 16 x width x height)

maxpool1 (batch_size x 16 x width/2 x height/2)

conv2 (batch_sizex 32 x width/2 x height/2)

maxpool2  (batch_size x 32 x width/4 x height/4)

conv3 (batch_size x 64 x width/4 x height/4)

maxool3 (batch_size x 64 x width/8 x height/8)

conv4 (batch_size x 128, x width/8 x height/8)

deconv1 (batch_size x 32, x width/8 x height/8)

upscale1 (batch_size x 32 x width/4 x height/4)

deconv2 (batch_size x 16, x width/4 x height/4)

upscale2 (batch_size x 16 x width/2 x height/2)

deconv3(batch_size x 8 x width/2 x height/2)

upscale3 (batch_size x 8 x width x height)

deconv4  (batch_size x 1 x width x height)

