import cv2
import random
import numpy as np
import os

class dataloader:

    def __init__(self, filename, batch_size, image_w, image_h):
        with open(filename) as file:
            self.datalist = file.readlines()
        random.shuffle(self.datalist)
        self.batch_size = batch_size
        self.len = len(self.datalist)
        self.index = 0
        self.image_w = image_w
        self.image_h = image_h

    def reset(self):
        # get batch from scratch
        self.index = 0
        random.shuffle(self.datalist)

    def get_trans_img(self, path):
        """
            get grayscale image in [image_w x image_h]
        """
        img = cv2.imread(path, 0)
        shape1, shape2 = img.shape
        if shape1 != self.image_w or shape2 != self.image_h:
            img = cv2.resize(img, (self.image_w, self.image_h))
        img = img.reshape(1, img.shape[0], img.shape[1])
        
        # normalize iamges
        img = img / 255
        return img

    def get_next_batch(self):
        """
            return a batch of images and their labels
        """
        if self.index + self.batch_size >= self.len:
            self.reset()
            
        images = np.zeros([self.batch_size, 1, self.image_w, self.image_h],dtype=np.float32)
        labels = np.zeros([self.batch_size],dtype=np.int32)
        
        # iterate over batchsize to get each image
        for i in range(self.batch_size):
            path, label = self.datalist[i + self.index].split(' ')
            images[i] = self.get_trans_img(path)
            labels[i] = int(label)
        self.index += self.batch_size
        return images, labels

    
    