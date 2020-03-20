import cv2
import numpy as np
from model import *
from scipy.spatial.distance import cosine

def read_pairs(path):
    files = []
    with open(path) as f:
        files = f.readlines()
        files = [afile[:-1].split(' ') for afile in files]
        files = [[afile[0], afile[1], afile[2]=='1'] for afile in files]

    return files

def get_test_pair_images(filename, image_h, image_w):
    """
        read in file list
    """
    with open(filename) as file1:
        filelist = file1.readlines()
    img_tensor1 = np.zeros([len(filelist), 1, image_h, image_w])
    img_tensor2 = np.zeros([len(filelist), 1, image_h, image_w])
    label_tensor = np.zeros([len(filelist)])
    for i in range(len(filelist)):
        path1, path2, label = filelist[i].split(" ")
        img1 = cv2.imread(path1, 0)
        img2 = cv2.imread(path2, 0)
        shape1, shape2 = img1.shape
        if shape1 != image_w or shape2 != image_h:
            img1 = cv2.resize(img1, (image_w, image_h))
            img2 = cv2.resize(img2, (image_w, image_h))
        img1 = img1.reshape(1, img1.shape[0], img1.shape[1])
        img2 = img2.reshape(1, img2.shape[0], img2.shape[1])

        img1 = img1[:,:,::-1].astype(np.float32)
        img2 = img2[:,:,::-1].astype(np.float32)
        img_tensor1[i] = img1
        img_tensor2[i] = img2
        label_tensor[i] = int(label)
    return img_tensor1, img_tensor2, label_tensor

def extract_features(img_path):
    """
        read in images in [1 x 1 x 128 x 128]
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img0 = np.zeros((1, 1, 128, 128))
    img = np.reshape(img, (1, 128, 128))
    img0[0] = img
    return img0

def test(model, img_list, image_h=128, image_w=128, write=True):
    model.eval()
    read_list = read_pairs(img_list)
    tp, tn, fp, fn = 0, 0, 0, 0
    acc = 0
    with open('predict.txt', "w+") as f:
        for idx, (path1, path2, label) in enumerate(read_list):
            
            img1 = extract_features(path1)
            img2 = extract_features(path2)
            y = model.lfw_inference(img1, img2)
            if y == label and label:
                tp += 1
                acc += 1
            elif y != label and not label:
                fp += 1
            elif y == label and not label:
                tn += 1
                acc += 1
            else:
                fn += 1
            if write:
                f.write(str(y) + '\n')

            if not (idx+1) % 100:
                print('TP: {} TN: {} FP: {} FN: {} ACC: {}'.format(tp, tn, fp, fn, float(acc) / (idx + 1)))


    print('TP: {} TN: {} FP: {} FN: {} ACC: {}'.format(tp, tn, fp, fn, acc))
    return acc

if __name__ == "__main__":
    model_path = 'model'
    test_path = 'train-aligned-label.txt'
    model = lightcnn29(79077, 0.2)
    model.load(model_path)
    acc = test(model, test_path)
    print(acc)