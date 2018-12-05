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

def find_dif(fal_arr, cor_arr, atol=2e-1, rtol=2e-2):
    """
    print out where the incorrect array is different with the correct one
    :param fal_arr:
    :param cor_arr:

    array_shape: (height, width, filter_num)
    :return: None
    """
    assert fal_arr.shape == cor_arr.shape

    for i in range(fal_arr.shape[0]):
        for j in range(fal_arr.shape[1]):
            for k in range(fal_arr.shape[2]):
                if not np.isclose(fal_arr[i][j][k], cor_arr[i][j][k], atol=atol, rtol=rtol):
                    print("idx: ({}, {}, {}) cor: {} fal: {}".format(
                        i, j, k, cor_arr[i][j][k], fal_arr[i][j][k]))

    print("length: {} Failed!".format(len))


def test_padding(instance):
    """ print whether the padding is true
    :param instance: a instance of class convolution
    :return: None
    """
    input_size = [224, 55, 13, 13, 13]
    ch_num = [3, 96, 256, 384, 384]
    pad = [1, 0, 1, 1, 1]
    for i in range(len(input_size)):

        x = np.random.randn(input_size[i], input_size[i], ch_num[i]).astype(np.float32)
        par_pad = instance.par_padding(x, pad[i])
        ser_pad = instance.ser_padding(x, pad[i])
        try:
            assert np.allclose(par_pad, ser_pad, atol=1e-3, rtol=1e-3)
            print('correct')
        except:
            print('ser_pad:{}{}{}par_pad:{}'.format('\n',ser_pad,'\n',par_pad))
            print('failed')
            break

def test_pooling(instance):
    """ print whether the pooling is true
    :param instance: a instance of class convolution
    :return: None
    """
    input_size = [112, 28, 15]
    ch_num = [96, 256, 256]
    pool_size = [3, 3, 3]
    stride = [2, 2, 2]
    for i in range(len(input_size)):

        x = relu(np.random.randn(input_size[i], input_size[i], ch_num[i]).astype(np.float32))
        par_pool = instance.par_max_pool(x, pool_size[i], stride[i])
        ser_pool = instance.ser_max_pool(x, pool_size[i], stride[i])
        try:
            assert np.allclose(par_pool, ser_pool, atol=1e-3, rtol=1e-3)
            print('correct')
        except:
            print('ser_pad:{}{}{}par_pad:{}'.format('\n',ser_pool,'\n',par_pool))
            print('failed')
            find_dif(par_pool ,ser_pool)
            break

def test_conv_nopad(instance, method1, method2, print_time=False,
                    verify_answer = False, atol=5e-1, rtol=2e-2,
                    whole_time1=None, kernel_time1=None,
                    whole_time2=None, kernel_time2=None):
    """
    put definitely correct method in method 1 !
    :param method1: e.g. parallel convolution method,
                    e.g. instance.par_conv_gl_coal_nopad_block1
    :param method2: e.g.serial convolution method,
                    e.g. instance.ser_conv
    :param print_time: whether print time
    :param xxx_time: e.g. instance.par_conv_sh_coal_nopad_noiter_block1_whole_time
    :return:
    """
    input_size = [226, 55, 15, 15, 15]
    f_num = [96, 256, 384, 384, 256]
    ch_num = [3, 96, 256, 384, 384]
    f_size = [7, 5, 3, 3, 3]
    stride = [2, 2, 1, 1, 1]
    for i in range(len(input_size)):

        x = np.random.randn(input_size[i], input_size[i], ch_num[i]).astype(np.float32)
        f = np.random.randn(f_num[i], f_size[i], f_size[i], ch_num[i]).astype(np.float32)
        b = np.random.randn(f_num[i]).astype(np.float32)

        print('x:{} f:{} b:{}'.format(x.shape, f.shape, b.shape))

        result1 = method1(x, f, b, pad=0, stride=stride[i]) #par_conv_gl_coal_nopad_block1
        result2 = method2(x, f, b, pad=0, stride=stride[i]) #ser_conv
        if verify_answer == True:
            try:
                assert np.allclose(result1, result2, atol=atol, rtol=rtol)
                print('correct')
            except:
                find_dif(result1, result2)
                print("x:{}, filter:{}".format(x.shape, f.shape))
                break

        if print_time == True:
            print('method1:{}{}:{}{}:{} '.format('\n',
                  str(whole_time1), getattr(instance, whole_time1),
                  str(kernel_time1), getattr(instance, kernel_time1)))
            print('method2:{}{}:{}{}:{} '.format('\n',
                  str(whole_time2), getattr(instance, whole_time2),
                  str(kernel_time2), getattr(instance, kernel_time2)))

def test_conv_4methods(instance, method1, method2, method3, method4,
                       print_time=False,
                       verify_answer = False, atol=5e-1, rtol=2e-2,
                       whole_time1=None, kernel_time1=None,
                       whole_time2=None, kernel_time2=None,
                       whole_time3=None, kernel_time3=None,
                       whole_time4=None, kernel_time4=None):
    """
    test all methods except serial one
    put definitely correct method in method 1 !
    :param method1: e.g. parallel convolution method,
                    e.g. instance.par_conv_gl_coal_nopad_block1
    :param method2: e.g.serial convolution method,
                    e.g. instance.ser_conv
    :param print_time: whether print time
    :param xxx_time: e.g. instance.par_conv_sh_coal_nopad_noiter_block1_whole_time
    :return:
    """
    input_size = [226, 55, 15, 15, 15]
    f_num = [96, 256, 384, 384, 256]
    ch_num = [3, 96, 256, 384, 384]
    f_size = [7, 5, 3, 3, 3]
    stride = [2, 2, 1, 1, 1]

    iter_time = 10

    temp_whole_time1 = 0
    temp_whole_time2 = 0
    temp_whole_time3 = 0
    temp_whole_time4 = 0
    temp_kernel_time1 = 0
    temp_kernel_time2 = 0
    temp_kernel_time3 = 0
    temp_kernel_time4 = 0

    for i in range(len(input_size)):

        x = np.random.randn(input_size[i], input_size[i], ch_num[i]).astype(np.float32)
        f = np.random.randn(f_num[i], f_size[i], f_size[i], ch_num[i]).astype(np.float32)
        b = np.random.randn(f_num[i]).astype(np.float32)

        print('testing computing time ...')
        print('x:{} f:{} b:{}'.format(x.shape, f.shape, b.shape))
        for j in range(iter_time):
            result1 = method1(x, f, b, pad=0, stride=stride[i]) #par_conv_gl_coal_nopad_block1
            result2 = method2(x, f, b, pad=0, stride=stride[i]) #ser_conv
            result3 = method3(x, f, b, pad=0, stride=stride[i])
            result4 = method4(x, f, b, pad=0, stride=stride[i])

            temp_whole_time1 += getattr(instance, whole_time1)
            temp_whole_time2 += getattr(instance, whole_time2)
            temp_whole_time3 += getattr(instance, whole_time3)
            temp_whole_time4 += getattr(instance, whole_time4)
            temp_kernel_time1 += getattr(instance, kernel_time1)
            temp_kernel_time2 += getattr(instance, kernel_time2)
            temp_kernel_time3 += getattr(instance, kernel_time3)
            temp_kernel_time4 += getattr(instance, kernel_time4)
            if verify_answer == True:
                try:
                    assert np.allclose(result1, result2, atol=atol, rtol=rtol)
                    assert np.allclose(result1, result3, atol=atol, rtol=rtol)
                    assert np.allclose(result1, result4, atol=atol, rtol=rtol)
                    print('correct')
                except:
                    find_dif(result1, result2)
                    find_dif(result1, result3)
                    find_dif(result1, result4)
                    print("x:{}, filter:{}".format(x.shape, f.shape))
                    break

        temp_whole_time1 /= iter_time
        temp_whole_time2 /= iter_time
        temp_whole_time3 /= iter_time
        temp_kernel_time1 /= iter_time
        temp_kernel_time2 /= iter_time
        temp_kernel_time3 /= iter_time
        temp_kernel_time4 /= iter_time

        if print_time == True:
            print('method1:{}{}:{}{}:{} '.format('\n',
                                                 str(whole_time1), temp_whole_time1,
                                                 str(kernel_time1), temp_kernel_time1))
            print('method2:{}{}:{}{}:{} '.format('\n',
                                                 str(whole_time2), temp_whole_time2,
                                                 str(kernel_time2), temp_kernel_time2))
            print('method3:{}{}:{}{}:{} '.format('\n',
                                                 str(whole_time3), temp_whole_time3,
                                                 str(kernel_time3), temp_kernel_time3))
            print('method4:{}{}:{}{}:{} '.format('\n',
                                                 str(whole_time4), temp_whole_time4,
                                                 str(kernel_time4), temp_kernel_time4))

def test_conv_5methods(instance, method1, method2, method3, method4,
                       print_time=False,
                       verify_answer = False, atol=5e-1, rtol=2e-2,
                       whole_time1=None, kernel_time1=None,
                       whole_time2=None, kernel_time2=None,
                       whole_time3=None, kernel_time3=None,
                       whole_time4=None, kernel_time4=None):
    """
    compare serial methods to parallel methods
    put definitely correct method in method 1 !
    :param method1: e.g. parallel convolution method,
                    e.g. instance.par_conv_gl_coal_nopad_block1
    :param method2: e.g.serial convolution method,
                    e.g. instance.ser_conv
    :param print_time: whether print time
    :param xxx_time: e.g. instance.par_conv_sh_coal_nopad_noiter_block1_whole_time
    :return:
    """
    input_size = [226, 55, 15, 15, 15]
    f_num = [96, 256, 384, 384, 256]
    ch_num = [3, 96, 256, 384, 384]
    f_size = [7, 5, 3, 3, 3]
    stride = [2, 2, 1, 1, 1]

    iter_time = 10

    temp_ser_time = 0
    temp_whole_time1 = 0
    temp_whole_time2 = 0
    temp_whole_time3 = 0
    temp_whole_time4 = 0
    temp_kernel_time1 = 0
    temp_kernel_time2 = 0
    temp_kernel_time3 = 0
    temp_kernel_time4 = 0

    total_whole_speedup1_to_ser = 0
    total_whole_speedup2_to_ser = 0
    total_whole_speedup3_to_ser = 0
    total_whole_speedup4_to_ser = 0
    total_kernel_speedup1_to_ser = 0
    total_kernel_speedup2_to_ser = 0
    total_kernel_speedup3_to_ser = 0
    total_kernel_speedup4_to_ser = 0

    total_whole_speedup2_to_naive = 0
    total_whole_speedup3_to_naive = 0
    total_whole_speedup4_to_naive = 0
    total_kernel_speedup2_to_naive = 0
    total_kernel_speedup3_to_naive = 0
    total_kernel_speedup4_to_naive = 0

    print('testing computing time ...')
    for i in range(len(input_size)):

        x = np.random.randn(input_size[i], input_size[i], ch_num[i]).astype(np.float32)
        f = np.random.randn(f_num[i], f_size[i], f_size[i], ch_num[i]).astype(np.float32)
        b = np.random.randn(f_num[i]).astype(np.float32)

        print('x:{} f:{} b:{}'.format(x.shape, f.shape, b.shape))

        result_ser = instance.ser_conv(x, f, b, pad=0, stride=stride[i])
        temp_ser_time = instance.ser_conv_time

        for j in range(iter_time):
            result1 = method1(x, f, b, pad=0, stride=stride[i]) #par_conv_gl_coal_nopad_block1
            result2 = method2(x, f, b, pad=0, stride=stride[i]) #ser_conv
            result3 = method3(x, f, b, pad=0, stride=stride[i])
            result4 = method4(x, f, b, pad=0, stride=stride[i])

            temp_whole_time1 += getattr(instance, whole_time1)
            temp_whole_time2 += getattr(instance, whole_time2)
            temp_whole_time3 += getattr(instance, whole_time3)
            temp_whole_time4 += getattr(instance, whole_time4)
            temp_kernel_time1 += getattr(instance, kernel_time1)
            temp_kernel_time2 += getattr(instance, kernel_time2)
            temp_kernel_time3 += getattr(instance, kernel_time3)
            temp_kernel_time4 += getattr(instance, kernel_time4)
            if verify_answer == True:
                try:
                    assert np.allclose(result1, result2, atol=atol, rtol=rtol)
                    assert np.allclose(result1, result3, atol=atol, rtol=rtol)
                    assert np.allclose(result1, result4, atol=atol, rtol=rtol)
                except:
                    find_dif(result1, result2)
                    find_dif(result1, result3)
                    find_dif(result1, result4)
                    print("x:{}, filter:{}".format(x.shape, f.shape))
                    break

        print('----------------- layer {} -----------------'.format(i+1))
        temp_whole_time1 /= iter_time
        temp_whole_time2 /= iter_time
        temp_whole_time3 /= iter_time
        temp_whole_time4 /= iter_time
        temp_kernel_time1 /= iter_time
        temp_kernel_time2 /= iter_time
        temp_kernel_time3 /= iter_time
        temp_kernel_time4 /= iter_time

        total_whole_speedup1_to_ser += temp_ser_time / temp_whole_time1
        total_whole_speedup2_to_ser += temp_ser_time / temp_whole_time2
        total_whole_speedup3_to_ser += temp_ser_time / temp_whole_time3
        total_whole_speedup4_to_ser += temp_ser_time / temp_whole_time4
        total_kernel_speedup1_to_ser += temp_ser_time / temp_kernel_time1
        total_kernel_speedup2_to_ser += temp_ser_time / temp_kernel_time2
        total_kernel_speedup3_to_ser += temp_ser_time / temp_kernel_time3
        total_kernel_speedup4_to_ser += temp_ser_time / temp_kernel_time4

        total_whole_speedup2_to_naive += temp_whole_time1 / temp_whole_time2
        total_whole_speedup3_to_naive += temp_whole_time1 / temp_whole_time3
        total_whole_speedup4_to_naive += temp_whole_time1 / temp_whole_time4
        total_kernel_speedup2_to_naive += temp_kernel_time1 / temp_kernel_time2
        total_kernel_speedup3_to_naive += temp_kernel_time1 / temp_kernel_time3
        total_kernel_speedup4_to_naive += temp_kernel_time1 / temp_kernel_time4

        if print_time == True:
            print('serial:{}ser_conv_time:{}'.format('\n', temp_ser_time))
            print('method1:{}{}:{}{}:{}'.format('\n',
                                                str(whole_time1), temp_whole_time1,
                                                str(kernel_time1), temp_kernel_time1))
            print('speed up to serial: kernel:{} whole:{}'.format(
                temp_ser_time / temp_kernel_time1,
                temp_ser_time / temp_whole_time1))
            print('method2:{}{}:{}{}:{}'.format('\n',
                                                str(whole_time2), temp_whole_time2,
                                                str(kernel_time2), temp_kernel_time2))
            print('speed up to serial: kernel:{} whole:{}'.format(
                temp_ser_time / temp_kernel_time2,
                temp_ser_time / temp_whole_time2))
            print('speed up to naive parallel: kernel:{} whole:{}'.format(
                temp_kernel_time1 / temp_kernel_time2,
                temp_whole_time1 / temp_whole_time2))
            print('method3:{}{}:{}{}:{} '.format('\n',
                                                 str(whole_time3), temp_whole_time3,
                                                 str(kernel_time3), temp_kernel_time3))
            print('speed up to serial: kernel:{} whole:{}'.format(
                temp_ser_time / temp_kernel_time3,
                temp_ser_time / temp_whole_time3))
            print('speed up to naive parallel: kernel:{} whole:{}'.format(
                temp_kernel_time1 / temp_kernel_time3,
                temp_whole_time1 / temp_whole_time3))
            print('method4:{}{}:{}{}:{} '.format('\n',
                                                 str(whole_time4), temp_whole_time4,
                                                 str(kernel_time4), temp_kernel_time4))
            print('speed up to serial: kernel:{} whole:{}'.format(
                temp_ser_time / temp_kernel_time4,
                temp_ser_time / temp_whole_time4))
            print('speed up to naive parallel: kernel:{} w3_hole:{}'.format(
                temp_kernel_time1 / temp_kernel_time4,
                temp_whole_time1 / temp_whole_time4))

    total_whole_speedup1_to_ser /= len(input_size)
    total_whole_speedup2_to_ser /= len(input_size)
    total_whole_speedup3_to_ser /= len(input_size)
    total_whole_speedup4_to_ser /= len(input_size)
    total_kernel_speedup1_to_ser /= len(input_size)
    total_kernel_speedup2_to_ser /= len(input_size)
    total_kernel_speedup3_to_ser /= len(input_size)
    total_kernel_speedup4_to_ser /= len(input_size)

    total_whole_speedup2_to_naive /= len(input_size)
    total_whole_speedup3_to_naive /= len(input_size)
    total_whole_speedup4_to_naive /= len(input_size)
    total_kernel_speedup2_to_naive /= len(input_size)
    total_kernel_speedup3_to_naive /= len(input_size)
    total_kernel_speedup4_to_naive /= len(input_size)

    if print_time == True:
        print('----------------- Overall Result -----------------')
        print('method1:')
        print('speed up to serial: kernel:{} whole:{}'.format(
            total_kernel_speedup1_to_ser,
            total_whole_speedup1_to_ser))
        print('method2:')
        print('speed up to serial: kernel:{} whole:{}'.format(
            total_kernel_speedup2_to_ser,
            total_whole_speedup2_to_ser))
        print('speed up to naive parallel: kernel:{} whole:{}'.format(
            total_kernel_speedup2_to_naive,
            total_whole_speedup2_to_naive))
        print('method3:')
        print('speed up to serial: kernel:{} whole:{}'.format(
            total_kernel_speedup3_to_ser,
            total_whole_speedup3_to_ser))
        print('speed up to naive parallel: kernel:{} whole:{}'.format(
            total_kernel_speedup3_to_naive,
            total_whole_speedup3_to_naive))
        print('method4:')
        print('speed up to serial: kernel:{} whole:{}'.format(
            total_kernel_speedup4_to_ser,
            total_whole_speedup4_to_ser))
        print('speed up to naive parallel: kernel:{} whole:{}'.format(
            total_kernel_speedup4_to_naive,
            total_whole_speedup4_to_naive))

def zf_net_conv(instance, par_conv_method, atol=2e-1, rtol=2e-2):
    """ test if zf-fp is correct
    :param instance: a instance of class convolution
    :param par_conv_method: parallel convolution method,
                    e.g. instance.par_conv_gl_coal_nopad
    :return: None
    """

    ##### Layer 1 #####
    in_layer1 = np.random.randn(224, 224, 3).astype(np.float32)
    f1 = np.random.randn(96, 7, 7, 3).astype(np.float32)
    b1 = np.random.randn(96).astype(np.float32)
    out_layer1 = test_conv_pool(instance, par_conv_method,
                   in_layer1, f1, b1, conv_stride=2, conv_pad=1,
                   pool_size=3, pool_stride=2, pool_pad=1
                                , atol=atol, rtol=rtol)

    ##### Layer 2 #####
    f2 = np.random.randn(256, 5, 5, 96).astype(np.float32)
    b2 = np.random.randn(256).astype(np.float32)
    out_layer2 = test_conv_pool(instance, par_conv_method,
                                out_layer1, f2, b2, conv_stride=2, conv_pad=0,
                                pool_size=3, pool_stride=2, pool_pad=1
                                , atol=atol, rtol=rtol)

    ##### Layer 3 #####
    f3 = np.random.randn(384, 3, 3, 256).astype(np.float32)
    b3 = np.random.randn(384).astype(np.float32)
    out_layer3 = test_conv(instance, par_conv_method,
              out_layer2, f3, b3, conv_stride=1, conv_pad=1,
                           atol=atol, rtol=rtol)

    ##### Layer 4 #####
    f4 = np.random.randn(384, 3, 3, 384).astype(np.float32)
    b4 = np.random.randn(384).astype(np.float32)
    out_layer4 = test_conv(instance, par_conv_method,
              out_layer3, f4, b4, conv_stride=1, conv_pad=1,
                           atol=atol, rtol=rtol)

    ##### Layer 5 #####
    f5 = np.random.randn(256, 3, 3, 384).astype(np.float32)
    b5 = np.random.randn(256).astype(np.float32)
    out_layer5 = test_conv_pool(instance, par_conv_method,
                                out_layer4, f5, b5, conv_stride=1, conv_pad=1,
                                pool_size=3, pool_stride=2, pool_pad=0,
                                atol=atol, rtol=rtol)

    print('The ZF-Net convolution part is correct!')

def test_conv_pool(instance, par_conv_method,
                   input_tensor, filter_w, filter_b, conv_stride, conv_pad,
                   pool_size, pool_stride, pool_pad, atol=2e-1, rtol=2e-2):
    """
    A layer containing convolution + max pooling,
    this process will verify the parallel results
    by computing serial results

    :param instance: a instance of class convolution
    :param par_conv_method: parallel convolution method,
                    e.g. instance.par_conv_gl_coal_nopad
    :param input:  has a size of (height, width, channels)
    :param filter_w: (num_filters, filter_height, filter_width, channels)
    :param filter_b: (num_filters, )
    :param conv_stride: convolution stride
    :param conv_pad: convolution pad width
    :param pool_size: max pool width
    :param pool_stride: max pool stride
    :param pool_pad: max pool pad width
    :return: Output (new_height, new_width, num_filters)
    """
    inter_result = input_tensor
    if conv_pad > 0:
        par_pad = instance.par_padding(input_tensor, conv_pad)
        ser_pad = instance.ser_padding(input_tensor, conv_pad)
        assert np.allclose(par_pad, ser_pad, atol=atol, rtol=rtol)
        inter_result = par_pad

    ser_conv = instance.ser_conv(inter_result, filter_w, filter_b, stride=conv_stride)
    par_conv = par_conv_method(inter_result, filter_w, filter_b, stride=conv_stride)
    try:
        assert np.allclose(par_conv, ser_conv, atol=atol, rtol=rtol)
    except:
        find_dif(par_conv, ser_conv)
        raise Exception('convolution is wrong')

    inter_result = par_conv
    if pool_pad > 0:
        par_pad = instance.par_padding(inter_result, pool_pad)
        ser_pad = instance.ser_padding(inter_result, pool_pad)
        assert np.allclose(par_pad, ser_pad, atol=atol, rtol=rtol)
        inter_result = par_pad

    par_pool = instance.par_max_pool(inter_result, pool_size, pool_stride)
    ser_pool = instance.ser_max_pool(inter_result, pool_size, pool_stride)
    assert np.allclose(par_pool, ser_pool, atol=atol, rtol=rtol)
    result = par_pool

    return result

def test_conv(instance, par_conv_method,
              input_tensor, filter_w, filter_b,
              conv_stride, conv_pad, atol=2e-1, rtol=2e-2):
    """
    A layer containing convolution, no pooling
    this process will verify the parallel results
    by computing serial results

    :param instance: a instance of class convolution
    :param par_conv_method: parallel convolution method,
                    e.g. instance.par_conv_gl_coal_nopad
    :param input:  has a size of (height, width, channels)
    :param filter_w: (num_filters, filter_height, filter_width, channels)
    :param filter_b: (num_filters, )
    :param conv_stride: convolution stride
    :param conv_pad: convolution pad width
    :return: Output (new_height, new_width, num_filters)
    """
    inter_result = input_tensor
    if conv_pad > 0:
        par_pad = instance.par_padding(inter_result, conv_pad)
        ser_pad = instance.ser_padding(inter_result, conv_pad)
        assert np.allclose(par_pad, ser_pad, atol=atol, rtol=rtol)
        inter_result = par_pad

    ser_conv = instance.ser_conv(inter_result, filter_w, filter_b, stride=conv_stride)
    par_conv = par_conv_method(inter_result, filter_w, filter_b, stride=conv_stride)
    try:
        assert np.allclose(par_conv, ser_conv, atol=atol, rtol=rtol)
    except:
        find_dif(par_conv, ser_conv)
        raise Exception('convolution is wrong')

    result = par_conv

    return result