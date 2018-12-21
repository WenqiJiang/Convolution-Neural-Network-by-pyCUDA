# Convolution-Neural-Network-by-pyCUDA
## Introduction
We build parallel algorithms of forward propagation for general convolutional neural networks and implemented ZF-Net for performance testing. Several methods include tiling, diminishing control divergence and carefully setting block size are used to optimize the algorithm speed. Finally, the most optimized algorithm is 107.35% and 1287 times faster than naive parallel method and serial method respectively.
The code is the convolution part of CNN. Fully-connected layer part was implemented by my teammate and I didn't upload them.
