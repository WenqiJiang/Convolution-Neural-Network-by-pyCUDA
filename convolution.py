
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

class convolution:

    def __init__(self):
        """ Constructor
        set basic parameters
        finish and build kernel
        """
        # Kernel code
        self.kernel = """
        #include <stdio.h>
        #include <math.h>
        /////////////               advanced input loading                ////////////
        /////////////                  block setting 2                    ////////////
        ///////////// shared + coalescing + no padding + filter iteration ////////////
        
        __global__ void conv_sh_coal_nopad_iter_block2(
            const int height, const int width, const int num_filters, 
            const int channels, const int filter_height, const int filter_width, 
            const int stride, const int new_height, const int new_width, 
            float* input, float* output, float* filter, float* bias)
        {
        /* iteratively load filter to shared memory
        /* input: height x width x channels
         * filter: num_filters x filter_height x filter_width x channels, 
         * stride, padding
         * bias: (num_filters, )
         * output: new_height x new_width x num_filters
         * Relu */
        /* load input to shared memory for input_iter times,
         * split input to input_iter sections, 
         * each contains input_split_size channels */
        
            
            __shared__ float input_ds[10000]; // (input_ds_height, input_ds_width)
            __shared__ float filter_w_ds[1024]; // max filter size 32 * 32
            __shared__ float filter_b_ds;
        
            /* layer --> filter, row --> height, col --> width
             * num_filters --> block_dim0
             * height --> block_dim1
             * width --> block_dim2 */
            const int col = blockIdx.z * blockDim.z + threadIdx.z;
            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            const int layer = blockIdx.x * blockDim.x + threadIdx.x;
        
            const int ty_row = threadIdx.y;
            const int tz_col = threadIdx.z;
        
            const int input_ds_width = filter_width + stride * (blockDim.z - 1);
        
            /* use these when loading input to input_ds */
            int start_row_input_ds = ty_row * stride;
            int start_col_input_ds = tz_col * stride;;
        
            /* use this when doing convolution */
            float temp = 0;
            
            /* following variables are used to load filter */
            int thread_per_block = blockDim.y * blockDim.z;
            int filter_per_channel = filter_height * filter_width;
            int filter_load_per_thread = ceil(float(filter_per_channel)
                                            / thread_per_block);
            
            /* following variables are used to load input */
            int block_start_row = blockIdx.y * blockDim.y * stride;
            int block_end_row = block_start_row + filter_height
                                + stride * (blockDim.y - 1) < height?
                                block_start_row + filter_height
                                + stride * (blockDim.y - 1) : height;
            int thread_num_row = blockDim.y; //new_height - blockIdx.y * blockDim.y < blockDim.y?
                            //new_height - blockIdx.y * blockDim.y : blockDim.y;

            int block_start_col = blockIdx.z * blockDim.z * stride;
            int block_end_col = block_start_col + filter_width
                            + stride * (blockDim.z - 1) < width?
                            block_start_col + filter_width
                            + stride * (blockDim.z - 1) : width;
            int thread_num_col = blockDim.z; //new_width - blockIdx.z * blockDim.z < blockDim.z?
                             //new_width - blockIdx.z * blockDim.z : blockDim.z;
            
            /* load thread_num_row rows of input per thread
             * load thread_num_col cols of input per thread */
            int load_height_per_thread = ceil(float(block_end_row - block_start_row)
                                        / thread_num_row);
            int load_width_per_thread = ceil(float(block_end_col - block_start_col)
                                    / thread_num_col);
           
            /* each iteration, load a channel of inputs
             * and a channel of filter,
             * then do convolution */
            for(int iter = 0; iter < channels; iter++)
            {
                /* load input to shared memory */
                if(layer < num_filters)// &&
                   //row < new_height && col < new_width)
                {
                    if(threadIdx.y * load_height_per_thread < block_end_row - block_start_row &&
                       threadIdx.z * load_width_per_thread < block_end_col - block_start_col)
                    {   
                        for(int i = block_start_row + threadIdx.y * load_height_per_thread;
                            i < block_start_row + (threadIdx.y + 1) * load_height_per_thread
                            && i < height; i++)
                        {
                            for(int j = block_start_col + threadIdx.z * load_width_per_thread;
                                j < block_start_col + (threadIdx.z + 1) * load_width_per_thread
                                && j < width; j++)
                            {
                                input_ds[(i - block_start_row) * input_ds_width + (j - block_start_col)]
                                = input[i * width * channels + j * channels + iter];
                            }
                        }
                    }
                } 
                __syncthreads();
                
                /* use many thread to load filter */
                if(layer < num_filters) // we need all threads to load
                                        // even its out of output range
                {
                    if((ty_row * blockDim.z + tz_col) 
                    * filter_load_per_thread  < filter_per_channel)
                    {
                        for(int i = (ty_row * blockDim.z + tz_col) 
                            * filter_load_per_thread;
                        i < (ty_row * blockDim.z + tz_col + 1) 
                            * filter_load_per_thread
                             && i < filter_per_channel; i++)
                        {
                            filter_w_ds[i] = filter[layer * filter_per_channel
                                            * channels + i * channels + iter];;
                        }
                    }
                }
                __syncthreads();
                
                /* do convolution */
                if(layer < num_filters &&
                   row < new_height && col < new_width)
                {
                    for(int i = 0; i < filter_height; i++)
                    {
                        for(int j = 0; j < filter_width; j++)
                        {
                            temp += input_ds[(start_row_input_ds + i) * input_ds_width + (start_col_input_ds + j)]
                                    * filter_w_ds[i * filter_width + j];
                        }
                    }
                } 
                __syncthreads();
            }
                        
            /* load bias to shared memory */
            if(layer < num_filters && row < new_height && col < new_width)
            {
                if(ty_row == 0 && tz_col == 0)
                {
                    filter_b_ds = bias[layer];
                }
            }
            __syncthreads();
            
            /* add bias + Relu */
            if(layer < num_filters && row < new_height && col < new_width)
            {  
                /* add bias */
                temp += filter_b_ds;
                /* ReLu */
                if(temp < 0)
                {
                    temp = 0;
                }
                /* write our resultes at output index */
                /* output[row, col, layer] */
                output[row * new_width * num_filters + col * num_filters + layer] = temp;
            }
            __syncthreads();
            
        }
        
        /////////////               advanced input loading                ////////////  
        ///////////// shared + coalescing + no padding + filter iteration ////////////
        __global__ void conv_sh_coal_nopad_iter_block1(
            const int height, const int width, const int num_filters, 
            const int channels, const int filter_height, const int filter_width, 
            const int stride, const int new_height, const int new_width, 
            float* input, float* output, float* filter, float* bias)
        {
        /* iteratively load filter to shared memory
        /* input: height x width x channels
         * filter: num_filters x filter_height x filter_width x channels, 
         * stride, padding
         * bias: (num_filters, )
         * output: new_height x new_width x num_filters
         * Relu */
        /* load input to shared memory for input_iter times,
         * split input to input_iter sections, 
         * each contains input_split_size channels */
            
            __shared__ float input_ds[8192];
            __shared__ float filter_w_ds[2048];
        
            /* layer --> filter, row --> height, col --> width
             * num_filters --> block_dim0
             * height --> block_dim1
             * width --> block_dim2 */
            const int col = blockIdx.z * blockDim.z + threadIdx.z;
            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            const int layer = blockIdx.x * blockDim.x + threadIdx.x;
        
            const int ty_row = threadIdx.y;
            const int tz_col = threadIdx.z;
        
            const int input_ds_width = filter_width + stride * (blockDim.z - 1);
        
            /* use these when loading input to input_ds */
            int start_row_input_ds = ty_row * stride;
            int start_col_input_ds = tz_col * stride;
            
            /* use these when doing convolution */
            float temp = 0;
            /* filter[layer, 0, 0, 0] */
            const int filter_start_idx = layer * filter_height * filter_width * channels;
            /* variables, will be changed latter */
            int input_ds_idx = 0;
            /* load many floats per layer per time 
             * e.g. section_num = 8
             * filter_w 0 ~ 7 for filter 1
             * 8 ~ 15 for filter 2, etc */
            const int section_num = blockDim.y * blockDim.z; 
            int conv_iter_num = int(ceil(filter_height * filter_width * channels 
                                / float(section_num))); 
            int count_load = 0;
            int count_conv = 0;
            int max_count = filter_height * filter_width * channels;
            int i = 0;
            int j = 0;
            int k = 0;
            
            /* starting input index of a block */  
            int block_start_row = blockIdx.y * blockDim.y * stride;
            int block_end_row = block_start_row + filter_height
                            + stride * (blockDim.y - 1) < height?
                            block_start_row + filter_height
                            + stride * (blockDim.y - 1) : height;
            int thread_num_row = blockDim.y;

            int block_start_col = blockIdx.z * blockDim.z * stride;
            int block_end_col = block_start_col + filter_width
                            + stride * (blockDim.z - 1) < width?
                            block_start_col + filter_width
                            + stride * (blockDim.z - 1) : width;
            int thread_num_col = blockDim.z;

            /* load thread_num_row rows per thread */
            /* load thread_num_col cols per thread */
            int load_height_per_thread = ceil(float(block_end_row - block_start_row)
                                        / thread_num_row);
            int load_width_per_thread = ceil(float(block_end_col - block_start_col)
                                    / thread_num_col);

            /* each iteration, load a channel of inputs
             * and a channel of filter,
             * then do convolution */
             
            /* load input to shared memory */
            if(layer < num_filters )
            {
                if(ty_row * load_height_per_thread < block_end_row - block_start_row &&
                   tz_col * load_width_per_thread < block_end_col - block_start_col
                   && layer == 0)
                {
                    for(int i = block_start_row + ty_row * load_height_per_thread;
                        i < block_start_row + (ty_row + 1) * load_height_per_thread
                        && i < block_end_row; i++)
                    {
                        for(int j = block_start_col + tz_col * load_width_per_thread;
                            j < block_start_col + (tz_col + 1) * load_width_per_thread
                            && j < block_end_col; j++)
                        {
                            for(int k = 0; k < channels; k++)
                            {
                                input_ds[(i - block_start_row) * input_ds_width * channels 
                                + (j - block_start_col) * channels + k]
                                = input[i * width * channels + j * channels + k];
                            }
                        }
                    }
                }
            } 
        
            __syncthreads();
            
            /* iteratively load filter */
            /* do convolution */
            for(int iter = 0; iter < conv_iter_num; iter++)
            {
                /* load section_num floats per filter */
                if(count_load + threadIdx.y * blockDim.z + threadIdx.z < max_count)
                {
                    filter_w_ds[(threadIdx.y * blockDim.z + threadIdx.z) 
                                * num_filters + layer] = 
                    filter[filter_start_idx + count_load + 
                                threadIdx.y * blockDim.z + threadIdx.z];
                    count_load += section_num;
                }
                __syncthreads();
                
                if(layer < num_filters && row < new_height && col < new_width)
                {
                    for(int m = 0; m < section_num && count_conv < max_count; m++)
                    {
                        i = count_conv / (filter_width * channels);
                        j = (count_conv - i * filter_width * channels) / channels;
                        k = count_conv - i * filter_width * channels - j * channels;
                        
                        input_ds_idx = (start_row_input_ds + i) * input_ds_width * channels
                                     + (start_col_input_ds + j) * channels + k;
                        temp += filter_w_ds[m * num_filters + layer] * input_ds[input_ds_idx];
                        count_conv++;
                    }
                }
                __syncthreads();
            }
            
            /* add bias and write output */
            
            if(layer < num_filters && row < new_height && col < new_width)
            {
                /* add bias */
                temp += bias[layer];
                /* ReLu */
                if(temp < 0)
                {
                    temp = 0;
                }
                /* write our resultes at output index */
                /* output[row, col, layer] */
                output[row * new_width * num_filters + col * num_filters + layer] = temp;
            }
            __syncthreads();
            
        }
        
        ///////////// shared + coalescing + no padding + no iteration ////////////
        __global__ void conv_sh_coal_nopad_noiter_block1(
            const int height, const int width, const int num_filters, 
            const int channels, const int filter_height, const int filter_width, 
            const int stride, const int new_height, const int new_width, 
            float* input, float* output, float* filter, float* bias)
        {
        /* input: height x width x channels
         * filter: num_filters x filter_height x filter_width x channels, 
         * stride, padding
         * bias: (num_filters, )
         * output: new_height x new_width x num_filters
         * Relu */
        /* load input to shared memory for input_iter times,
         * split input to input_iter sections, 
         * each contains input_split_size channels */
        
            
            __shared__ float input_ds[8192];
        
            /* layer --> filter, row --> height, col --> width
             * num_filters --> block_dim0
             * height --> block_dim1
             * width --> block_dim2 */
            const int col = blockIdx.z * blockDim.z + threadIdx.z;
            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            const int layer = blockIdx.x * blockDim.x + threadIdx.x;
        
            const int ty_row = threadIdx.y;
            const int tz_col = threadIdx.z;
        
            const int input_ds_width = filter_width + stride * (blockDim.z - 1);
        
            /* use these when loading input to input_ds */
            int start_row_input_ds, start_col_input_ds;
        
            /* use these when doing convolution */
            float temp = 0;
            /* filter[layer, 0, 0, 0] */
            const int filter_start_idx = layer * filter_height 
                                    * filter_width * channels;
            /* variables, will be changed latter */
            int input_ds_idx = 0;
            int filter_idx = 0;
            
            /* starting input index of a block */  
            int block_start_row = blockIdx.y * blockDim.y * stride;
            int block_end_row = block_start_row + filter_height
                            + stride * (blockDim.y - 1) < height?
                            block_start_row + filter_height
                            + stride * (blockDim.y - 1) : height;
            int thread_num_row = blockDim.y;

            int block_start_col = blockIdx.z * blockDim.z * stride;
            int block_end_col = block_start_col + filter_width
                            + stride * (blockDim.z - 1) < width?
                            block_start_col + filter_width
                            + stride * (blockDim.z - 1) : width;
            int thread_num_col = blockDim.z;

            /* load thread_num_row rows per thread */
            /* load thread_num_col cols per thread */
            int load_height_per_thread = ceil(float(block_end_row - block_start_row)
                                        / thread_num_row);
            int load_width_per_thread = ceil(float(block_end_col - block_start_col)
                                    / thread_num_col);

            /* each iteration, load a channel of inputs
             * and a channel of filter,
             * then do convolution */
            
            /* load input to shared memory */
            if(layer < num_filters )
            {
                if(ty_row * load_height_per_thread < block_end_row - block_start_row &&
                   tz_col * load_width_per_thread < block_end_col - block_start_col
                   && layer == 0)
                {
                    for(int i = block_start_row + ty_row * load_height_per_thread;
                        i < block_start_row + (ty_row + 1) * load_height_per_thread
                        && i < block_end_row; i++)
                    {
                        for(int j = block_start_col + tz_col * load_width_per_thread;
                            j < block_start_col + (tz_col + 1) * load_width_per_thread
                            && j < block_end_col; j++)
                        {
                            for(int k = 0; k < channels; k++)
                            {
                                input_ds[(i - block_start_row) * input_ds_width * channels 
                                + (j - block_start_col) * channels + k]
                                = input[i * width * channels + j * channels + k];
                            }
                        }
                    }
                }
            } 
        
            __syncthreads();
            
            /* for each sample in batch, for each filter in filters */
            /* for each points within new_height, new_width */
            /* do convolution */
            /* input[sample, range_h, range_w, :] * filter[flt,:,:,:] + b[flt] */
        
            if(layer < num_filters && row < new_height && col < new_width)
            {
                start_row_input_ds = ty_row * stride;
                start_col_input_ds = tz_col * stride;
                for(int i = 0; i < filter_height; i++)
                {
                    for(int j = 0; j < filter_width; j++)
                    {
                        for(int cn = 0; cn < channels; cn++)
                        {
                            /* input[layer, row * stride + i, col * stride + j, cn]
                             * filter[flt,:,:,:] + b[flt] */
                            input_ds_idx = (start_row_input_ds + i) * input_ds_width * channels
                                     + (start_col_input_ds + j) * channels + cn;
                            filter_idx = filter_start_idx 
                                         + i * filter_width * channels
                                         + j * channels + cn;
                            temp += input_ds[input_ds_idx] * filter[filter_idx];
                        }
                    }
                }
                /* add bias */
                temp += bias[layer];
                /* ReLu */
                if(temp < 0)
                {
                    temp = 0;
                }
                /* write our resultes at output index */
                /* output[row, col, layer] */
                output[row * new_width * num_filters + col * num_filters + layer] = temp;
            }
            __syncthreads();
        }
        
        ///////////// global + coalescing + no padding ////////////////
        
        __global__ void conv_gl_coal_nopad(
            const int height, const int width, const int num_filters, 
            const int channels, const int filter_height, const int filter_width, 
            const int stride, const int new_height, const int new_width, 
            float* input, float* output, float* filter, float* bias)
        {
        /* input: height x width x channels
         * filter: num_filters x filter_height x filter_width x channels, 
         * stride, padding
         * bias: (num_filters, )
         * output: new_height x new_width x num_filters
         * Relu */

            /* layer --> filter, row --> height, col --> width
             * num_filters --> block_dim0
             * height --> block_dim1
             * width --> block_dim2 */
            const int col = blockIdx.z * blockDim.z + threadIdx.z;
            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            const int layer = blockIdx.x * blockDim.x + threadIdx.x;

            /* define start index where convolution start from */
            /* input[row * stride, col * stride, 0]*/
            const int input_start_idx = row * stride * width  * channels 
                                        + col * stride * channels;
            /* filter[layer, 0, 0, 0] */
            const int filter_start_idx = layer * filter_height * filter_width * channels;
            /* variables, will be changed latter */
            int input_idx = 0;
            int filter_idx = 0;

            /* for each sample in batch, for each filter in filters */
            /* for each points within new_height, new_width */
            /* do convolution */
            /* input[sample, range_h, range_w, :] * filter[flt,:,:,:] + b[flt] */
            float temp = 0; // count for conv result

            if(layer < num_filters && row < new_height && col < new_width)
            {
                for(int i = 0; i < filter_height; i++)
                {
                    for(int j = 0; j < filter_width; j++)
                    {
                        for(int cn = 0; cn < channels; cn++)
                        {
                            /* input[layer, row * stride + i, col * stride + j, cn]
                             * filter[flt,:,:,:] + b[flt] */
                            input_idx = input_start_idx 
                                        + i * width * channels
                                        + j * channels + cn;
                            filter_idx = filter_start_idx 
                                         + i * filter_width * channels
                                         + j * channels + cn;
                            temp += input[input_idx] * filter[filter_idx];
                        }
                    }
                }
                /* add bias */
                temp += bias[layer];
                /* ReLu */
                if(temp < 0)
                {
                    temp = 0;
                }
                /* write our resultes at output index */
                /* output[row, col, layer] */
                output[row * new_width * num_filters + col * num_filters + layer] = temp;
            }
            __syncthreads();
        }
        
        //////////////////// padding function /////////////////////

        __global__ void padding(
            const int height, const int width, const int channels, 
            const int new_height, const int new_width, const int padding,
            float* input, float* output)
        {
        /* input: height x width x channels
         * padding width: padding
         * new height: height + 2 * padding
         * new width: width + 2 * padding
         * output: new_height x new_width x channels*/
        
            /* height --> block_dim0 --> row
             * width --> block_dim1 --> col
             * channels --> block_dim2 --> layer */
            const int layer = blockIdx.z * blockDim.z + threadIdx.z;
            const int col = blockIdx.y * blockDim.y + threadIdx.y;
            const int row = blockIdx.x * blockDim.x + threadIdx.x;
        
            /* index of threads, w.r.t. output */
            const int idx = row * new_width * channels + col * channels + layer;
            /* index of corresponding input 
             * assign value now may cause overflow
             *  but overflowed value will not be used */
            const int in_idx = (row - padding) * width * channels 
                                + (col - padding) * channels + layer;
        
            if(((col < padding && row < new_height) || 
               (col >= width + padding && col < new_width && row < new_height) ||
               (row < padding && col < new_width) ||
               (row >= height + padding && row < new_height && col < new_width))
               && layer < channels)
            /*((col < padding && row < padding) ||
               (col < padding && row >= height + padding && row < new_height) ||
               (col >= width + padding && col < new_width && row < padding) ||
               (col >= width + padding && col < new_width 
               && row >= height + padding && row < new_height))*/
            {
                /* output[row][col][layer] = 0 */
                output[idx] = 0;
            }
            else if(col >= padding && col < width + padding
                    && row >= padding && row < height + padding
                    && layer < channels)
            {
        
                output[idx] = input[in_idx]; 
            }
            __syncthreads();
        }
        
        //////////////////// max pooling function /////////////////////
        
        __global__ void max_pool(
            const int height, const int width, const int channels, 
            const int filter_height, const int filter_width, const int stride, 
            const int new_height, const int new_width, 
            float* input, float* output)
        {
        /* input: convolutional layer output, all >= 0
         * height x width x channels
         * filter: filter_height x filter_width, 
         * stride, padding
         * output: new_height x new_width x num_filters */
        
            /* height --> block_dim0 --> row
             * width --> block_dim1 --> col
             * channels --> block_dim2 --> layer */
            const int layer = blockIdx.z * blockDim.z + threadIdx.z;
            const int col = blockIdx.y * blockDim.y + threadIdx.y;
            const int row = blockIdx.x * blockDim.x + threadIdx.x;
            const int out_idx = row * new_width * channels
                                + col * channels + layer;
            int idx;
            float max_val = 0;
        
            /* if thread in range */
            if(row < new_height && col < new_width && layer < channels)
            {
                /* find the max value in specific range */
                for(int i = 0; i < filter_height; i++)
                {
                    for(int j = 0; j < filter_width; j++)
                    {
                        /* input[row * stride + i, col * stride +j, layer]
                         * > max_val?*/
                        idx = (row * stride + i) * width * channels
                              + (col * stride +j) * channels + layer;
                        if(input[idx] > max_val)
                        {
                            max_val = input[idx];
                        }
                    }
                }
        
                /* copy to output */
                output[out_idx] = max_val;
            }
            
            __syncthreads();
        }
        """
        self.mod = SourceModule(self.kernel)

    def par_padding(self, x, pad):
        """
        :param x: Input data. Should have size (height, width, channels).
        :param pad: padding width
        :return: x_pad: padded tensor
        """
        input_cpu = np.array(x, dtype=np.float32)
        height, width, channels = np.int32(input_cpu.shape[0]), \
                                  np.int32(input_cpu.shape[1]), \
                                  np.int32(input_cpu.shape[2])

        new_height = np.int32(height + 2 * pad)
        new_width = np.int32(width + 2 * pad)
        pad = np.int32(pad)
        print('pad:{}'.format(pad))
        output_cpu = np.ones((new_height, new_width, channels)
                              , dtype=np.float32)
        block_size = (int(8), int(8), int(8))
        grid_size = (int(np.ceil(new_height / 8.0)),
                     int(np.ceil(new_width / 8.0)),
                     int(np.ceil(channels / 8.0)))
        print('block_size:{}, grid_size:{}'.format(block_size, grid_size))
        padding = self.mod.get_function("padding")

        # define time record api
        kernel_start = cuda.Event()
        whole_start = cuda.Event()
        kernel_end = cuda.Event()
        whole_end = cuda.Event()

        # memory allocation / to device
        whole_start.record()
        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        output_gpu = cuda.mem_alloc(output_cpu.nbytes)
        cuda.memcpy_htod(input_gpu, input_cpu)
        cuda.memcpy_htod(output_gpu, output_cpu)

        kernel_start.record()
        # parameters:
        # height, width, channels, new_height, new_width, padding, input, output
        padding(height, width, channels,
                new_height, new_width, pad,
                input_gpu, output_gpu,
                block=block_size, grid=grid_size)
        # block=(block_size[0], block_size[1], block_size[2]),
        # grid=(grid_size[0], grid_size[1], grid_size[2]))
        kernel_end.record()
        cuda.memcpy_dtoh(output_cpu, output_gpu)
        whole_end.record()

        kernel_end.synchronize()
        whole_end.synchronize()

        self.padding_kernel_time = kernel_start.time_till(kernel_end) * 1e-3
        self.padding_whole_time = whole_start.time_till(whole_end) * 1e-3

        return output_cpu

    def ser_padding(self, x, pad):
        """
        :param x: input matrix (height, width, channels)
        :param pad: padding width
        :return: x_pad (height + 2*pad, width+2*pad, channels)
        """
        height, width, channels = x.shape
        x_pad = np.zeros((height + 2 * pad, width + 2 * pad, channels))
        for i in range(height):
            for j in range(width):
                for cn in range(channels):
                    x_pad[i + pad, j + pad, cn] = x[i, j, cn]

        return x_pad

    def ser_conv(self, x, f, b, pad=0, stride=1):
        """
        ########## global memory + coalescing + no padding ###########

        A Numpy implementation of 3-D image convolution:
        Inputs:
        :param x: Input data. Should have size (height, width, channels).
        :param f: Filter. Should have size (num_filters, filter_height, filter_width, channels).
        :param b: Bias term. Should have size (num_filters, ).
        :param pad: Integer. The number of zeroes to pad along the height and width axis.
        :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

        :return: A 3-D array. Should have size (new_height, new_width, num_filters).

        To calculate the output shape of your convolution, we need the following equations:
        new_height = ((height - filter_height + 2 * pad) // stride) + 1
        new_width = ((width - filter_width + 2 * pad) // stride) + 1
        """
        x = np.array(x).astype(np.float32)
        height, width, channels = x.shape
        num_filters, filter_height, filter_width, channels_f = f.shape
        assert channels == channels_f
        assert b.shape[0] == num_filters

        new_height = int(np.floor((height - filter_height + 2 * pad) / float(stride)) + 1)
        new_width = int(np.floor((width - filter_width + 2 * pad) / float(stride)) + 1)

        start = time.time()
        A = np.zeros((new_height, new_width, num_filters))
        x_pad = self.ser_padding(x, pad)
        for ft in range(num_filters):
            for i in range(new_height):
                for j in range(new_width):
                    # assert f[ft,:,:,:].shape == x_pad[bt,i*stride :i*stride +
                    # filter_height,j * stride: j*stride + filter_width,:].shape
                    A[i, j, ft] = b[ft] + np.sum(f[ft, :, :, :] *
                                                 x_pad[i * stride: i * stride +
                                                               filter_height, j * stride: j * stride + filter_width, :])

        self.ser_conv_time = time.time() - start

        return relu(A)

    def ser_max_pool(self, x, pool_size, stride):
        """
        A Numpy implementation of 2-D image max pooling.

        Inputs:
        :params x: Input data. Should have size (height, width, channels).
        :params pool_size: Integer. The size of a window in which you will perform max operations.
        :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
        :return :A 3-D array. Should have size (new_height, new_width, num_of_filters).
        """

        height, width, channels = x.shape

        new_height = int(np.floor((height - pool_size) / float(stride)) + 1)
        new_width = int(np.floor((width - pool_size) / float(stride)) + 1)

        A = np.min(x) * np.ones((new_height, new_width, channels))

        for row in range(new_height):
            for col in range(new_width):
                for cn in range(channels):
                    for i in range(pool_size):
                        for j in range(pool_size):
                            if (x[row * stride + i, col * stride + j, cn] > A[row, col, cn]):
                                A[row, col, cn] = x[row * stride + i, col * stride + j, cn]

        return A

    def par_max_pool(self, x, pool_size, stride):
        """
        Inputs:
        :params x: Input data. Should have size (height, width, channels).
        :params pool_size: Integer. The size of a window in which you will perform max operations.
        :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
        :return :A 3-D array. Should have size (new_height, new_width, num_of_filters).
        """
        input_cpu = np.array(x, dtype=np.float32)
        height, width, channels = np.int32(input_cpu.shape[0]), \
                                  np.int32(input_cpu.shape[1]), \
                                  np.int32(input_cpu.shape[2])
        stride = np.int32(stride)
        filter_height = filter_width = np.int32(pool_size)

        new_height = np.int32(np.floor((height - pool_size) / float(stride)) + 1)
        new_width = np.int32(np.floor((width - pool_size) / float(stride)) + 1)

        print('max_pool: in_size:({},{},{}) pool_size:{} stride:{}'.
              format(height, width, channels, pool_size, stride))
        output_cpu = np.ones((new_height, new_width, channels)
                             , dtype=np.float32)
        block_size = (int(8), int(8), int(8))
        grid_size = (int(np.ceil(new_height / 8.0)),
                     int(np.ceil(new_width / 8.0)),
                     int(np.ceil(channels / 8.0)))

        print('block_size:{}, grid_size:{}'.format(block_size, grid_size))
        max_pool = self.mod.get_function("max_pool")

        # define time record api
        kernel_start = cuda.Event()
        whole_start = cuda.Event()
        kernel_end = cuda.Event()
        whole_end = cuda.Event()

        # memory allocation / to device
        whole_start.record()
        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        output_gpu = cuda.mem_alloc(output_cpu.nbytes)
        cuda.memcpy_htod(input_gpu, input_cpu)
        cuda.memcpy_htod(output_gpu, output_cpu)

        kernel_start.record()
        # parameters:
        # const int height, const int width, const int channels,
        # const int filter_height, const int filter_width,
        # const int stride, const int new_height, const int new_width,
        # float* input, float* output
        max_pool(height, width, channels,
                 filter_height, filter_width,
                 stride, new_height, new_width,
                 input_gpu, output_gpu,
                block=block_size, grid=grid_size)

        kernel_end.record()
        cuda.memcpy_dtoh(output_cpu, output_gpu)
        whole_end.record()

        kernel_end.synchronize()
        whole_end.synchronize()

        self.max_pool_kernel_time = kernel_start.time_till(kernel_end) * 1e-3
        self.max_pool_whole_time = whole_start.time_till(whole_end) * 1e-3

        return output_cpu

    def par_conv_gl_coal_nopad_block1(self, x, f, b, pad=0, stride=1):
        """
        Inputs:
        :param x: Input data. Should have size (height, width, channels).
        :param f: Filter. Should have size (num_filters, filter_height, filter_width, channels).
        :param b: Bias term. Should have size (num_filters, ).
        :param pad: Integer. The number of zeroes to pad along the height and width axis.
        :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

        :return: A 3-D array. Should have size (new_height, new_width, num_filters).
        """

        input_cpu = np.array(x, dtype=np.float32)
        filter_cpu = np.array(f, dtype=np.float32)
        bias_cpu = np.array(b, dtype=np.float32)
        height, width, channels = np.int32(input_cpu.shape[0]), \
                                  np.int32(input_cpu.shape[1]), np.int32(input_cpu.shape[2])
        num_filters, filter_height, filter_width, channels_f = \
            np.int32(f.shape[0]), np.int32(f.shape[1]), \
            np.int32(f.shape[2]), np.int32(f.shape[3])
        stride = np.int32(stride)
        assert channels == channels_f
        assert b.shape[0] == num_filters

        new_height = np.int32(np.floor((height - filter_height + 2 * pad)
                                       / float(stride)) + 1)
        new_width = np.int32(np.floor((width - filter_width + 2 * pad)
                                      / float(stride)) + 1)

        output_cpu = np.zeros((new_height, new_width, num_filters)
                              , dtype=np.float32)
        block_size, grid_size = assign_block_grid_block1(num_filters, new_height, new_width)
        #print('block_size:{}, grid_size:{}'.format(block_size, grid_size))
        conv_gl_coal_nopad = self.mod.get_function("conv_gl_coal_nopad")

        # define time record api
        kernel_start = cuda.Event()
        whole_start = cuda.Event()
        kernel_end = cuda.Event()
        whole_end = cuda.Event()

        # memory allocation / to device
        whole_start.record()
        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        output_gpu = cuda.mem_alloc(output_cpu.nbytes)
        filter_gpu = cuda.mem_alloc(filter_cpu.nbytes)
        bias_gpu = cuda.mem_alloc(bias_cpu.nbytes)

        cuda.memcpy_htod(input_gpu, input_cpu)
        cuda.memcpy_htod(output_gpu, output_cpu)
        cuda.memcpy_htod(filter_gpu, filter_cpu)
        cuda.memcpy_htod(bias_gpu, bias_cpu)

        kernel_start.record()
        # parameters:
        # const int height, const int width, const int num_filters,
        # const int channels, const int filter_height, const int filter_width,
        # const int stride, const int new_height, const int new_width,
        # float* input, float* output, float* filter, float* bias
        conv_gl_coal_nopad(height, width, num_filters, channels,
                           filter_height, filter_width, stride,
                           new_height, new_width,
                           input_gpu, output_gpu, filter_gpu, bias_gpu,
                           block=block_size, grid=grid_size)
        # block=(block_size[0], block_size[1], block_size[2]),
        # grid=(grid_size[0], grid_size[1], grid_size[2]))
        kernel_end.record()
        cuda.memcpy_dtoh(output_cpu, output_gpu)
        whole_end.record()

        kernel_end.synchronize()
        whole_end.synchronize()

        self.par_conv_gl_coal_nopad_block1_kernel_time \
            = kernel_start.time_till(kernel_end) * 1e-3
        self.par_conv_gl_coal_nopad_block1_whole_time \
            = whole_start.time_till(whole_end) * 1e-3

        return output_cpu

    def par_conv_sh_coal_nopad_noiter_block1(self, x, f, b, pad=0, stride=1):
        """
        Inputs:
        :param x: Input data. Should have size (height, width, channels).
        :param f: Filter. Should have size (num_filters, filter_height, filter_width, channels).
        :param b: Bias term. Should have size (num_filters, ).
        :param pad: Integer. The number of zeroes to pad along the height and width axis.
        :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

        :return: A 3-D array. Should have size (new_height, new_width, num_filters).
        """

        input_cpu = np.array(x, dtype=np.float32)
        filter_cpu = np.array(f, dtype=np.float32)
        bias_cpu = np.array(b, dtype=np.float32)
        height, width, channels = np.int32(input_cpu.shape[0]), \
                                  np.int32(input_cpu.shape[1]), np.int32(input_cpu.shape[2])
        num_filters, filter_height, filter_width, channels_f = \
            np.int32(f.shape[0]), np.int32(f.shape[1]), \
            np.int32(f.shape[2]), np.int32(f.shape[3])
        stride = np.int32(stride)
        assert channels == channels_f
        assert b.shape[0] == num_filters

        new_height = np.int32(np.floor((height - filter_height + 2 * pad)
                                       / float(stride)) + 1)
        new_width = np.int32(np.floor((width - filter_width + 2 * pad)
                                      / float(stride)) + 1)

        output_cpu = np.zeros((new_height, new_width, num_filters)
                              , dtype=np.float32)
        block_size, grid_size = assign_block_grid_block1(num_filters, new_height, new_width)
        #print('block_size:{}, grid_size:{}'.format(block_size, grid_size))
        conv_sh_coal_nopad_noiter_block1 = self.mod.get_function("conv_sh_coal_nopad_noiter_block1")

        # define time record api
        kernel_start = cuda.Event()
        whole_start = cuda.Event()
        kernel_end = cuda.Event()
        whole_end = cuda.Event()

        # memory allocation / to device
        whole_start.record()
        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        output_gpu = cuda.mem_alloc(output_cpu.nbytes)
        filter_gpu = cuda.mem_alloc(filter_cpu.nbytes)
        bias_gpu = cuda.mem_alloc(bias_cpu.nbytes)

        cuda.memcpy_htod(input_gpu, input_cpu)
        cuda.memcpy_htod(output_gpu, output_cpu)
        cuda.memcpy_htod(filter_gpu, filter_cpu)
        cuda.memcpy_htod(bias_gpu, bias_cpu)

        kernel_start.record()
        # parameters:
        # const int height, const int width, const int num_filters,
        # const int channels, const int filter_height, const int filter_width,
        # const int stride, const int new_height, const int new_width,
        # float* input, float* output, float* filter, float* bias
        conv_sh_coal_nopad_noiter_block1(height, width, num_filters, channels,
                                         filter_height, filter_width, stride,
                                         new_height, new_width,
                                         input_gpu, output_gpu, filter_gpu, bias_gpu,
                                         block=block_size, grid=grid_size)
        # block=(block_size[0], block_size[1], block_size[2]),
        # grid=(grid_size[0], grid_size[1], grid_size[2]))
        kernel_end.record()
        cuda.memcpy_dtoh(output_cpu, output_gpu)
        whole_end.record()

        kernel_end.synchronize()
        whole_end.synchronize()

        self.par_conv_sh_coal_nopad_noiter_block1_kernel_time \
            = kernel_start.time_till(kernel_end) * 1e-3
        self.par_conv_sh_coal_nopad_noiter_block1_whole_time \
            = whole_start.time_till(whole_end) * 1e-3

        return output_cpu

    def par_conv_sh_coal_nopad_iter_block1(self, x, f, b, pad=0, stride=1):
        """
        Inputs:
        :param x: Input data. Should have size (height, width, channels).
        :param f: Filter. Should have size (num_filters, filter_height, filter_width, channels).
        :param b: Bias term. Should have size (num_filters, ).
        :param pad: Integer. The number of zeroes to pad along the height and width axis.
        :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

        :return: A 3-D array. Should have size (new_height, new_width, num_filters).
        """

        input_cpu = np.array(x, dtype=np.float32)
        filter_cpu = np.array(f, dtype=np.float32)
        bias_cpu = np.array(b, dtype=np.float32)
        height, width, channels = np.int32(input_cpu.shape[0]), \
                                  np.int32(input_cpu.shape[1]), np.int32(input_cpu.shape[2])
        num_filters, filter_height, filter_width, channels_f = \
            np.int32(f.shape[0]), np.int32(f.shape[1]), \
            np.int32(f.shape[2]), np.int32(f.shape[3])
        stride = np.int32(stride)
        assert channels == channels_f
        assert b.shape[0] == num_filters

        new_height = np.int32(np.floor((height - filter_height + 2 * pad)
                                       / float(stride)) + 1)
        new_width = np.int32(np.floor((width - filter_width + 2 * pad)
                                      / float(stride)) + 1)

        output_cpu = np.zeros((new_height, new_width, num_filters)
                              , dtype=np.float32)
        block_size, grid_size = assign_block_grid_block1(num_filters, new_height, new_width)
        #print('block_size:{}, grid_size:{}'.format(block_size, grid_size))
        conv_sh_coal_nopad_iter_block1 = self.mod.get_function("conv_sh_coal_nopad_iter_block1")

        # define time record api
        kernel_start = cuda.Event()
        whole_start = cuda.Event()
        kernel_end = cuda.Event()
        whole_end = cuda.Event()

        # memory allocation / to device
        whole_start.record()
        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        output_gpu = cuda.mem_alloc(output_cpu.nbytes)
        filter_gpu = cuda.mem_alloc(filter_cpu.nbytes)
        bias_gpu = cuda.mem_alloc(bias_cpu.nbytes)

        cuda.memcpy_htod(input_gpu, input_cpu)
        cuda.memcpy_htod(output_gpu, output_cpu)
        cuda.memcpy_htod(filter_gpu, filter_cpu)
        cuda.memcpy_htod(bias_gpu, bias_cpu)

        kernel_start.record()
        # parameters:
        # const int height, const int width, const int num_filters,
        # const int channels, const int filter_height, const int filter_width,
        # const int stride, const int new_height, const int new_width,
        # float* input, float* output, float* filter, float* bias
        conv_sh_coal_nopad_iter_block1(height, width, num_filters, channels,
                           filter_height, filter_width, stride,
                           new_height, new_width,
                           input_gpu, output_gpu, filter_gpu, bias_gpu,
                           block=block_size, grid=grid_size)
        # block=(block_size[0], block_size[1], block_size[2]),
        # grid=(grid_size[0], grid_size[1], grid_size[2]))
        kernel_end.record()
        cuda.memcpy_dtoh(output_cpu, output_gpu)
        whole_end.record()

        kernel_end.synchronize()
        whole_end.synchronize()

        self.par_conv_sh_coal_nopad_iter_block1_kernel_time \
            = kernel_start.time_till(kernel_end) * 1e-3
        self.par_conv_sh_coal_nopad_iter_block1_whole_time \
            = whole_start.time_till(whole_end) * 1e-3

        return output_cpu

    def par_conv_sh_coal_nopad_iter_block2(self, x, f, b, pad=0, stride=1):
        """
        Inputs:
        :param x: Input data. Should have size (height, width, channels).
        :param f: Filter. Should have size (num_filters, filter_height, filter_width, channels).
        :param b: Bias term. Should have size (num_filters, ).
        :param pad: Integer. The number of zeroes to pad along the height and width axis.
        :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

        :return: A 3-D array. Should have size (new_height, new_width, num_filters).
        """

        input_cpu = np.array(x, dtype=np.float32)
        filter_cpu = np.array(f, dtype=np.float32)
        bias_cpu = np.array(b, dtype=np.float32)
        height, width, channels = np.int32(input_cpu.shape[0]), \
                                  np.int32(input_cpu.shape[1]), np.int32(input_cpu.shape[2])
        num_filters, filter_height, filter_width, channels_f = \
            np.int32(f.shape[0]), np.int32(f.shape[1]), \
            np.int32(f.shape[2]), np.int32(f.shape[3])
        stride = np.int32(stride)
        assert channels == channels_f
        assert b.shape[0] == num_filters

        new_height = np.int32(np.floor((height - filter_height + 2 * pad)
                                       / float(stride)) + 1)
        new_width = np.int32(np.floor((width - filter_width + 2 * pad)
                                      / float(stride)) + 1)

        output_cpu = np.zeros((new_height, new_width, num_filters)
                              , dtype=np.float32)
        block_size, grid_size = assign_block_grid_block2(num_filters, filter_height, filter_width,
                                                        stride, new_height, new_width)
        #print('block_size:{}, grid_size:{}'.format(block_size, grid_size))
        conv_sh_coal_nopad_iter_block2 = self.mod.get_function("conv_sh_coal_nopad_iter_block2")

        # define time record api
        kernel_start = cuda.Event()
        whole_start = cuda.Event()
        kernel_end = cuda.Event()
        whole_end = cuda.Event()

        # memory allocation / to device
        whole_start.record()
        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        output_gpu = cuda.mem_alloc(output_cpu.nbytes)
        filter_gpu = cuda.mem_alloc(filter_cpu.nbytes)
        bias_gpu = cuda.mem_alloc(bias_cpu.nbytes)

        cuda.memcpy_htod(input_gpu, input_cpu)
        cuda.memcpy_htod(output_gpu, output_cpu)
        cuda.memcpy_htod(filter_gpu, filter_cpu)
        cuda.memcpy_htod(bias_gpu, bias_cpu)

        kernel_start.record()

        conv_sh_coal_nopad_iter_block2(height, width, num_filters, channels,
                           filter_height, filter_width, stride,
                           new_height, new_width,
                           input_gpu, output_gpu, filter_gpu, bias_gpu,
                           block=block_size, grid=grid_size)

        kernel_end.record()
        cuda.memcpy_dtoh(output_cpu, output_gpu)
        whole_end.record()

        kernel_end.synchronize()
        whole_end.synchronize()

        self.par_conv_sh_coal_nopad_iter_block2_kernel_time \
            = kernel_start.time_till(kernel_end) * 1e-3
        self.par_conv_sh_coal_nopad_iter_block2_whole_time \
            = whole_start.time_till(whole_end) * 1e-3

        return output_cpu
