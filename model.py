from layers import *
from scipy.spatial.distance import cosine

class lightcnn:
    def __init__(self, num_classes, threshold, in_channels=1):
        self.num_classes = num_classes
        self.threshold = threshold
        self.layer1 = mfm(in_channels, 48, 9, 9, 1, 1)
        self.layer2 = max_pooling(2, 2, 2)
        self.layer3 = mfm(48, 96, 5, 5, 1, 1)
        self.layer4 = max_pooling(2, 2, 2)
        self.layer5 = mfm(96, 128, 5, 5, 1, 1)
        self.layer6 = max_pooling(2, 2, 2)
        self.layer7 = mfm(128, 192, 4, 4, 1, 1)
        self.layer8 = max_pooling(2, 2, 2)
        self.layer10 = mfm(9408, 256, type1=0)
        self.fc = fc(256, num_classes)

    def forward(self, inputs):
        x = self.layer2.forward(self.layer1.forward((inputs)))
        x = self.layer4.forward(self.layer3.forward((x)))
        x = self.layer6.forward(self.layer5.forward((x)))
        x = self.layer7.forward((x))
        x = self.layer8.forward(x)
        x = self.layer10.forward(x)
        self.extract = x
        x = self.fc.forward(x)
        
        return x

    def backward(self, out_diff_tensor, lr):
        x = self.fc.backward(out_diff_tensor, lr)
        x = self.layer10.backward(x, lr)
        x = self.layer8.backward(x, lr)
        x = self.layer7.backward(x, lr)
        x = self.layer6.backward(x, lr)
        x = self.layer5.backward(x, lr)
        x = self.layer4.backward(x, lr)
        x = self.layer3.backward(x, lr)
        x = self.layer2.backward(x, lr)
        x = self.layer1.backward(x, lr)
        self.in_diff_tensor = x
        return x

    def lfw_inference(self, inputs1, inputs2):
        out_tensor1 = self.forward(inputs1).reshape(inputs1.shape[0], -1)
        out_tensor2 = self.forward(inputs2).reshape(inputs2.shape[0], -1)
        score = cosine(out_tensor1, out_tensor2)
        return score > self.threshold

    def save(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)
            
        conv_num = self.layer1.save(path, 0)
        conv_num = self.layer3.save(path, conv_num)
        conv_num = self.layer5.save(path, conv_num)
        conv_num = self.layer7.save(path, conv_num)
        conv_num = self.layer10.save(path, conv_num)

        self.fc.save(path, conv_num)
    
    def load(self, path):

        conv_num = self.layer1.load(path, 0)
        conv_num = self.layer3.load(path, conv_num)
        conv_num = self.layer5.load(path, conv_num)
        conv_num = self.layer7.load(path, conv_num)
        conv_num = self.layer10.load(path, conv_num)
        
        self.fc.load(path, conv_num)

    def train(self):
        pass
    
    def eval(self):
        pass


class lightcnn29:
    def __init__(self, num_classes, threshold, in_channels=1):
        self.num_classes = num_classes
        self.threshold = threshold
        self.layer1 = mfm(in_channels, 48, 5, 5, padding=True, stride=1, pad_h=2, pad_w=2)
        self.layer2 = max_pooling(2, 2, 2)
        self.layer3 = group(48, 96, 3, 3, 1, 1)
        self.layer4 = max_pooling(2, 2, 2)
        self.layer5 = group(96, 192, 3, 3, 1, 1)
        self.layer6 = max_pooling(2, 2, 2)
        self.layer7 = group(192, 128, 3, 3, 1, 1)
        self.layer8 = group(128, 128, 3, 3, 1, 1)
        self.layer9 = max_pooling(2, 2, 2)
        self.layer10 = mfm(8*8*128, 256, type1=0)
        self.fc = fc(256, num_classes)

    def forward(self, inputs):
        x = self.layer2.forward(self.layer1.forward((inputs)))
        x = self.layer4.forward(self.layer3.forward((x)))
        x = self.layer5.forward((x))
        x = self.layer6.forward(x)
        x = self.layer7.forward((x))
        x = self.layer8.forward(x)
        x = self.layer9.forward(x)
        x = self.layer10.forward(x)
        self.extract_ = x[0]
        x = self.fc.forward(x)
        
        return x

    def extract(self):
        return self.extract_

    def backward(self, out_diff_tensor, lr):
        x = self.fc.backward(out_diff_tensor, lr)
        x = self.layer10.backward(x, lr)
        x = self.layer9.backward(x, lr)
        x = self.layer8.backward(x, lr)
        x = self.layer7.backward(x, lr)
        x = self.layer6.backward(x, lr)
        x = self.layer5.backward(x, lr)
        x = self.layer4.backward(x, lr)
        x = self.layer3.backward(x, lr)
        x = self.layer2.backward(x, lr)
        x = self.layer1.backward(x, lr)
        self.in_diff_tensor = x
        return x

    def lfw_inference(self, inputs1, inputs2):
        self.forward(inputs1)
        out_tensor1 = self.extract()
        self.forward(inputs2)
        out_tensor2 = self.extract()
        score = 1 - cosine(out_tensor1, out_tensor2)
        return score > self.threshold

    def save(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)
            
        conv_num = self.layer1.save(path, 0)
        conv_num = self.layer3.save(path, conv_num)
        conv_num = self.layer5.save(path, conv_num)
        conv_num = self.layer7.save(path, conv_num)
        conv_num = self.layer8.save(path, conv_num)
        conv_num = self.layer10.save(path, conv_num)

        self.fc.save(path, conv_num)
    
    def load(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)
            
        conv_num = self.layer1.load(path, 0)
        conv_num = self.layer3.load(path, conv_num)
        conv_num = self.layer5.load(path, conv_num)
        conv_num = self.layer7.load(path, conv_num)
        conv_num = self.layer8.load(path, conv_num)
        conv_num = self.layer10.load(path, conv_num)

        self.fc.load(path, conv_num)

    def train(self):
        pass
    
    def eval(self):
        pass

