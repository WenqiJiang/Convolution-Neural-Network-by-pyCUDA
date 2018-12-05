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


def assign_block_grid_block1(filter_num, new_height, new_width):
    """
    Tesla K40 restrictions:
    Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)

    Dimension correspondence:
    filter_num --> block_dim0
    height --> block_dim1
    width --> block_dim2

    Params:
    filter_num: number of filters, e.g. 384 in ZF-net layer4
    height, width: input size of a layer, e.g. an image
                   , not the filter height and width

    Return:
    block_size: a tuple (dim0, dim1, dim2)
    grid_size: a tuple (dim0, dim1, dim2)
    """

    ##### compute block size #####
    # filter num should < 1024
    assert filter_num <= 1024
    b_dim0 = int(filter_num)
    # maximum value of block_dim1 * block_dim1
    b_dim1by2 = int(np.floor(1024 / b_dim0))
    # b_dim2 is always < 64 using the following method
    b_dim1 = b_dim2 = int(np.sqrt(b_dim1by2))

    ##### compute grid size #####
    g_dim0 = int(1)
    g_dim1 = int(np.ceil(new_height / float(b_dim1)))
    g_dim2 = int(np.ceil(new_width / float(b_dim2)))

    assert b_dim1 >= 1 and b_dim2 >= 1
    assert g_dim0 >= 1 and g_dim1 >= 1 and g_dim2 >= 1

    block_size = (b_dim0, b_dim1, b_dim2)
    grid_size = (g_dim0, g_dim1, g_dim2)

    return block_size, grid_size

def assign_block_grid_block2(num_filters, filter_height, filter_width,
                             stride, new_height, new_width):
    """
    compute block and grid size for block setting 2

    filter shape(num_filters, filter_width, filter_height, channels)
    channels is not used here
    :param height: input shape
    :param width: input shape
    :param stride: convolution stride
    :param new_height: output shape
    :param new_width: output shape
    :return: block size, grid size
            both are tuples with 3 elements in them
    """
    ##### compute block size #####
    # filter num should < 1024
    assert num_filters <= 1024
    b_dim0 = 1
    # input floats covered in a block should
    # be less than 10,000
    # filter_height + stride * blockDim1 <= 100
    b_dim1 = int(np.floor((100 - filter_height) / stride))
    b_dim2 = int(np.floor((100 - filter_width) / stride))
    # b_dim0 * b_dim1 * b_dim2 should be less than 1024
    if b_dim1 > 32:
        b_dim1 = 32
    if b_dim2 > 32:
        b_dim2 = 32

    ##### compute grid size #####
    g_dim0 = int(num_filters)
    g_dim1 = int(np.ceil(new_height / float(b_dim1)))
    g_dim2 = int(np.ceil(new_width / float(b_dim2)))

    block_size = (b_dim0, b_dim1, b_dim2)
    grid_size = (g_dim0, g_dim1, g_dim2)

    assert b_dim1 >= 1 and b_dim2 >= 1
    assert g_dim0 >= 1 and g_dim1 >= 1 and g_dim2 >= 1

    return block_size, grid_size

def compute_memory_iter(input_shape, filter_w_shape,
                        filter_b_shape, block_size, stride):
    """use shared memory iteratively

    for Tesla K40
    Total amount of constant memory:               65,536 bytes / 16,384 floats
    Total amount of shared memory per block:       49,152 bytes / 12,288 floats

    load input to shared memory for input_iter times,
    split input to input_iter sections, each contains input_split_size channels

    load filter to shared memory for filter_iter times,
    split input to filter_iter sections, each with a size of
    filter_w_split_size and filter_b_split_size channels

    :param input_shape:  has a size of (height, width, channels)
    :param filter_w_shape: (num_filters, filter_height, filter_width, channels)
    :param filter_b_shape: (num_filters, )
    :param block_size: (block_dim0, block_dim1, block_dim2)
    :param stride: convolution stride
    :return: input_iter, input_split_size,
    filter_iter, filter_w_split_size, filter_b_split_size
    """
    ##### preprocessing #####

    _, __, input_dim2 = input_shape
    num_filters, filter_height, filter_width, channels = filter_w_shape
    num_filters_b = filter_b_shape
    _, block_dim1, block_dim2 = block_size
    assert num_filters == num_filters_b and input_dim2 == channels

    # shape of input per block
    input_per_block = (int(filter_height + stride * (block_dim1 - 1)),
                       int(filter_width + stride * (block_dim2 - 1)),
                       int(channels))

    # floats number of input per block
    input_per_block_num = input_per_block[0] * input_per_block[1] \
                          * input_per_block[2]

    # floats number of filter_w per thread
    filter_w_num = filter_height * \
                   filter_width * channels
    # floats number of filter_b
    filter_b_num = 1

    # input_per_block_num or filter should be less or equal to
    # half of shared memory, 8,192 floats
    half_share = 8192
    ##### compute input_iter, input_split_size #####
    if input_per_block_num <= half_share:
        input_iter = 1
        input_split_size = channels
    else:
        unit_size = input_per_block[0] * input_per_block[1]
        input_split_size = int(half_share / unit_size)
        input_iter = np.ceil(input_split_size * unit_size
                             / float(half_share))

    ##### filter_iter, filter_w_split_size, filter_b_split_size #####
    # give filter_b 1 float
    # give filter_w 8192-1 = 8191
    half_share_w = 8191
    half_share_b = 1
    if filter_w_num <= half_share_w and \
            filter_b_num <= half_share_b:
        filter_iter = 1
        filter_w_split_size = num_filters
        filter_b_split_size = num_filters
    else:
        # only use w to compute
        # because space for b is definitely enough
        unit_size = filter_height * filter_width * channels
        filter_w_split_size = int(half_share_w / unit_size)
        filter_b_split_size = int(half_share_w / unit_size)
        filter_iter = np.ceil(filter_w_split_size * unit_size
                             / float(half_share_w))

    return input_iter, input_split_size, \
    filter_iter, filter_w_split_size, filter_b_split_size

def relu(input):
    """
    :param input: a matrix
    :return: relu matrix
    """
    return np.maximum(input, 0)

