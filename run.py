import numpy as np
import time

# plotting
import matplotlib as mpl

mpl.use('agg')
import matplotlib.pyplot as plt

# pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

# local functions
from convolution import *
from utils import *
from test_funcs import *
conv = convolution()

test_conv_5methods(conv, conv.par_conv_gl_coal_nopad_block1,
                    conv.par_conv_sh_coal_nopad_noiter_block1,
                    conv.par_conv_sh_coal_nopad_iter_block1,
                    conv.par_conv_sh_coal_nopad_iter_block2,
                    print_time=True, verify_answer=True,
                    whole_time1='par_conv_gl_coal_nopad_block1_whole_time',
                    kernel_time1='par_conv_gl_coal_nopad_block1_kernel_time',
                    whole_time2='par_conv_sh_coal_nopad_noiter_block1_whole_time',
                    kernel_time2='par_conv_sh_coal_nopad_noiter_block1_kernel_time',
                    whole_time3='par_conv_sh_coal_nopad_iter_block1_whole_time',
                    kernel_time3='par_conv_sh_coal_nopad_iter_block1_kernel_time',
                    whole_time4='par_conv_sh_coal_nopad_iter_block2_whole_time',
                    kernel_time4='par_conv_sh_coal_nopad_iter_block2_kernel_time')