import numpy as np
from numpy.lib.stride_tricks import as_strided

import math
import os

# fully connected layers
class fc:
    def __init__(self, in_channels, out_channels):
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.init_param()
    
    def init_param(self):
        # kernel = [out_channels x in_channels]
        self.kernel = np.random.uniform(
            low = -np.sqrt(6.0/(self.out_channels + self.in_channels)),
            high = np.sqrt(6.0/(self.in_channels + self.out_channels)),
            size = (self.out_channels, self.in_channels)
        )
        self.bias = np.zeros([self.out_channels])
        
    def forward(self, inputs):
        self.shape = inputs.shape
        # self.in [batch_num, in_channels]
        self.inputs = inputs.reshape(inputs.shape[0], -1).copy()
        assert self.inputs.shape[1] == self.kernel.shape[1]
        # Linear Transform: y = xw^T + b
        self.out_tensor = np.dot(self.inputs, self.kernel.T) + self.bias.T
        return self.out_tensor
        
    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        kernel_diff = np.dot(out_diff_tensor.T, self.inputs).squeeze()
        bias_diff = np.sum(out_diff_tensor, axis=0).reshape(self.bias.shape)
        self.in_diff_tensor = np.dot(out_diff_tensor, self.kernel).reshape(self.shape)
        self.kernel -= lr * kernel_diff
        self.bias -= lr * bias_diff

        return self.in_diff_tensor
        
    def save(self, path, num):
        if os.path.exists(path) == False:
            os.mkdir(path)
        
        np.save(os.path.join(path, "fc_lr_{}_weight.npy").format(num), self.kernel)
        np.save(os.path.join(path, "fc_lr_{}.bias.npy").format(num), self.bias)

        return num + 1
        
    def load(self, path, num):
        assert os.path.exists(path)
        print(os.path.join(path, "fc_lr_{}_weight.npy").format(num))
        self.kernel = np.load(os.path.join(path, "fc_lr_{}_weight.npy").format(num))
        self.bias = np.load(os.path.join(path, "fc_lr_{}_bias.npy").format(num))
        
        return num + 1
        

class mfm:  
    def __init__(self,in_channels, out_channels, kernel_h=3, kernel_w=3, stride=1, padding=1, type1=1, pad_h=0, pad_w=0):
        self.out_channels = out_channels
        self.type1 = type1
        if type1 == 1:
            self.filter = conv_layer(in_channels, 2*out_channels, kernel_h, kernel_w, padding, stride, True, pad_h=pad_h, pad_w=pad_w)
        else:
            self.filter = fc(in_channels, 2*out_channels)
            
    def forward(self, inputs):
        self.inputs = inputs
        self.out_tensor = self.filter.forward(inputs)
        out_tensor = self.split(self.out_tensor)
        return np.maximum(out_tensor[0], out_tensor[1])

    def split(self, inputs, dim=1):
        """
            split the input into 2 EQUAL size tensor
        """
        idx = inputs.shape[dim]
        half_idx = idx // 2
        idx1 = list(range(0, half_idx))
        idx2 = list(range(half_idx, idx))
        return np.take(inputs, idx1, dim), np.take(inputs, idx2, dim)
    
    def backward(self, out_diff_tensor, lr):
        out_tensor1, out_tensor2 = self.split(self.out_tensor)
        # get the element wise maximum indices
        # and only backward on these elements
        mask1 = out_tensor1 > out_tensor2
        out_diff_tensor1 = mask1 * out_diff_tensor
        out_diff_tensor2 = (1 - mask1) * out_diff_tensor
        out_diff_tensor = np.concatenate([out_diff_tensor1, out_diff_tensor2], axis=1)
        if self.type1 == 1:
            in_diff_tensor = self.filter.backward(out_diff_tensor, lr)
        else:
            in_diff_tensor = self.filter.backward(out_diff_tensor, lr)

        self.in_diff_tensor = in_diff_tensor
        return in_diff_tensor

    def save(self, path, conv_num):
        if not os.path.exists(path):
            os.mkdir(path)

        self.filter.save(path, conv_num)
        return conv_num + 1
    
    def load(self, path, conv_num):
        if not os.path.exists(path):
            os.mkdir(path)

        self.filter.load(path, conv_num)
        return conv_num + 1

# a block in the lightcnn
class group:
    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, stride=1, padding=0):
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_h, kernel_w, stride, padding)
        
    def forward(self, inputs):
        x = self.conv_a.forward(inputs)
        x = self.conv.forward(x)
        return x

    def backward(self, out_diff_tensor, lr):
        x = self.conv.backward(out_diff_tensor, lr)
        x = self.conv_a.backward(x, lr)

        self.in_diff_tensor = x
        return x

    def save(self, path, conv_num):
        if os.path.exists(path) == False:
            os.mkdir(path)

        conv_num = self.conv_a.save(path, conv_num)
        conv_num = self.conv.save(path, conv_num)

        return conv_num
    
    def load(self, path, conv_num):

        conv_num = self.conv_a.load(path, conv_num)
        conv_num = self.conv.load(path, conv_num)

        return conv_num 

    def train(self):
        pass
    
    def eval(self):
        pass

    
class conv_layer:

    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, padding=False, stride = 1, shift=True, pad_h=0, pad_w=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.padding = padding
        self.stride = stride
        self.shift = shift
        self.pad_h = pad_h
        self.pad_w = pad_w

        self.init_param()

    def init_param(self):
        # use xwaiver to initialize weights
        # use guassian to initialize bias
        n = self.kernel_h * self.kernel_w * self.out_channels
        self.kernel = np.random.normal(0, math.sqrt(2./n), size=(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w))
        self.bias = np.zeros([self.out_channels]) if self.shift else None

    def pad(self, inputs, pad_h, pad_w):
        # pad the input tensor
        # add pad_h and pad_w at the FOUR direction 
        batch_num = inputs.shape[0]
        in_channels = inputs.shape[1]
        in_h = inputs.shape[2]
        in_w = inputs.shape[3]
        padded = np.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
        padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w] = inputs
        return padded
    
    def convolution(self, inputs, kernel, stride=1):
        """
            input: [batch_num x in_channels x in_h x in_w]
            kernels: [out_channels x in_channels x kernel_h x kernel_w]
            extended_in: [in_channels*kernel_h*kernel_w x batch_num*out_h*out_w]
            output: [batch_num x out_channels x out_h x out_w]
        """
        
        # kernels: [out_channels x in_channels x kernel_h x kernel_w]
        kernel = kernel.transpose(2, 3, 1, 0)
        inputs_copy = inputs.transpose(0, 2, 3, 1)
        Hout = inputs_copy.shape[1] - kernel.shape[0] + 1
        Wout = inputs_copy.shape[2] - kernel.shape[1] + 1

        # strided the blocks 
        inputs_copy = as_strided(inputs_copy, (inputs_copy.shape[0], Hout, Wout, kernel.shape[0], kernel.shape[1], inputs_copy.shape[3]), inputs_copy.strides[:3] + inputs_copy.strides[1:])

        # directly multiply it with filters
        out = np.tensordot(inputs_copy, kernel, axes=3)
        return out.transpose(0, 3, 1, 2)
    
    
    def forward(self, inputs):
        if self.padding:
            inputs = self.pad(inputs, int((self.kernel_h-1)/2), int((self.kernel_w-1)/2))
        
        self.inputs = inputs.copy()
        
        self.out_tensor = self.convolution(inputs, self.kernel, self.stride)

        if self.shift:
            self.out_tensor += self.bias.reshape(1,self.out_channels,1,1)

        return self.out_tensor
    
    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        # In the backward process, we should get the right weights for
        # out_diff tensor to flow back
        if self.shift:
            bias_diff = np.sum(out_diff_tensor, axis = (0,2,3)).reshape(self.bias.shape)
            self.bias -= lr * bias_diff
        
        batch_num = out_diff_tensor.shape[0]
        out_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]

        # 1. The diff of kernel can be view as an multiplication between extended input tensor
        # and the out_diff tensor, which thus is an convolution between input tensor and the out_diff tensor
        extend_out = np.zeros([batch_num, out_channels, out_h, out_w, self.stride * self.stride])
        extend_out[:, :, :, :, 0] = out_diff_tensor
        extend_out = extend_out.reshape(batch_num, out_channels, out_h, out_w, self.stride, self.stride)
        extend_out = extend_out.transpose(0,1,2,4,3,5).reshape(batch_num, out_channels, out_h*self.stride, out_w*self.stride)

        kernel_diff = self.convolution(self.inputs.transpose(1,0,2,3), extend_out.transpose(1,0,2,3))
        kernel_diff = kernel_diff.transpose(1,0,2,3)

        # 2. The in_diff should be the convolution between extended out_diff tensor 
        # and kernel. We first reshape kernel and multiply them directly
        padded = self.pad(extend_out, self.kernel_h-1, self.kernel_w-1)
        kernel_trans = self.kernel.reshape(self.out_channels, self.in_channels, self.kernel_h*self.kernel_w)
        kernel_trans = kernel_trans[:,:,::-1].reshape(self.kernel.shape)
        
        self.in_diff_tensor = self.convolution(padded, kernel_trans.transpose(1,0,2,3))
        assert self.in_diff_tensor.shape == self.inputs.shape

        if self.padding:
            pad_h = int((self.kernel_h-1)/2)
            pad_w = int((self.kernel_w-1)/2)
            if pad_h == 0 and pad_w != 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, :, pad_w:-pad_w]
            elif pad_h !=0 and pad_w == 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, :]
            elif pad_h != 0 and pad_w != 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]

        self.kernel -= lr * kernel_diff

        return self.in_diff_tensor

    def save(self, path, conv_num):
        if os.path.exists(path) == False:
            os.mkdir(path)

        np.save(os.path.join(path, "conv{}_weight.npy".format(conv_num)), self.kernel)
        if self.shift:
            np.save(os.path.join(path, "conv{}_bias.npy".format(conv_num)), self.bias)
        
        return conv_num + 1

    def load(self, path, conv_num):
        assert os.path.exists(path)
        print("load conv{}_weight.npy".format(conv_num))
        self.kernel = np.load(os.path.join(path, "conv{}_weight.npy".format(conv_num)))
        if self.shift:
            self.bias = np.load(os.path.join(path, "conv{}_bias.npy").format(conv_num))
        
        return conv_num + 1



class max_pooling:
    
    def __init__(self, kernel_h, kernel_w, stride, padding=False, pad=0):
        assert stride > 1
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.padding = padding
        self.stride = stride

    @staticmethod
    def pad(inputs, pad_h=1, pad_w=1):
        batch_num = inputs.shape[0]
        in_channels = inputs.shape[1]
        in_h = inputs.shape[2]
        in_w = inputs.shape[3]
        padded = np.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
        padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w] = inputs
        return padded

    def forward(self, inputs):
        if self.padding:
            inputs = max_pooling.pad(inputs)
        self.shape = inputs.shape

        batch_num = inputs.shape[0]
        in_channels = inputs.shape[1]
        in_h = inputs.shape[2]
        in_w = inputs.shape[3]
        out_h = int((in_h - self.kernel_h) / self.stride) + 1
        out_w = int((in_w - self.kernel_w) / self.stride) + 1

        # iterate over each block to get the maximum elements
        out_tensor = np.zeros([batch_num, in_channels, out_h, out_w])
        self.maxindex = np.zeros([batch_num, in_channels, out_h, out_w], dtype = np.int32)
        for i in range(out_h):
            for j in range(out_w):
                part = inputs[:, :, i*self.stride:i*self.stride+self.kernel_h, j*self.stride:j*self.stride+self.kernel_w].reshape(batch_num, in_channels, -1)
                out_tensor[:, :, i, j] = np.max(part, axis = -1)
                self.maxindex[:, :, i, j] = np.argmax(part, axis = -1)
        self.out_tensor = out_tensor
        return self.out_tensor

    def backward(self, out_diff_tensor, lr=0):
        assert out_diff_tensor.shape == self.out_tensor.shape
        batch_num = out_diff_tensor.shape[0]
        in_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        in_h = self.shape[2]
        in_w = self.shape[3]

        out_diff_tensor = out_diff_tensor.reshape(batch_num*in_channels, out_h, out_w)
        self.maxindex = self.maxindex.reshape(batch_num*in_channels, out_h, out_w)
        
        if in_h < self.stride*out_h:
            self.in_diff_tensor = np.zeros([batch_num*in_channels, self.stride*out_h, self.stride*out_w])
        else:
            self.in_diff_tensor = np.zeros([batch_num*in_channels, in_h, in_w])

        h_index = (self.maxindex/self.kernel_h).astype(np.int32)
        w_index = self.maxindex - h_index * self.kernel_h

        for i in range(out_h):
            for j in range(out_w):
                self.in_diff_tensor[range(batch_num * in_channels), i*self.stride+h_index[:,i,j], j*self.stride+w_index[:,i,j]] += out_diff_tensor[:,i,j]
        if in_h < self.stride*out_h:
            self.in_diff_tensor = self.in_diff_tensor[:, :in_h, :in_w]
        self.in_diff_tensor = self.in_diff_tensor.reshape(batch_num, in_channels, in_h, in_w)

        if self.padding:
            pad_h = int((self.kernel_h - 1) / 2)
            pad_w = int((self.kernel_w - 1) / 2)
            self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]

        return self.in_diff_tensor






