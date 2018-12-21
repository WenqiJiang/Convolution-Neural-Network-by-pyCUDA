# Convolution-Neural-Network-by-pyCUDA
## Read our [REPORT](https://github.com/WenqiJiang/Convolution-Neural-Network-by-pyCUDA/blob/master/report.pdf) in the repository to see all implementation details 
## Introduction
We build parallel algorithms of forward propagation for general convolutional neural networks and implemented ZF-Net for performance testing. Several methods include tiling, diminishing control divergence and carefully setting block size are used to optimize the algorithm speed. Finally, the most optimized algorithm is 107.35% and 1287 times faster than naive parallel method and serial method respectively.

The code is the convolution part of CNN. Fully-connected layer part was implemented by my teammate and I didn't upload them.
## Environment
We use the pyCUDA environment of tesseract server provided by Columbia University.
## Components
(1) report.pdf: our report that contains all details  
(2) result.out: the result of our program, contains all profiling (time measuring) information  
(3) convolution.py: contains all kernel code of several methods  
(4) test_funcs.py: testing functions, verifying correctness and measuring time  
(5) utils.py: other functions, such as automatically setting block size  
(6) run.py: run this file as the main function **if you know the running command**  
(7) cnn.sh: run this file if you don't know the running command  
