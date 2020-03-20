from data import *
from model import *
from test import *
from layers import *
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class trainer:
    def __init__(self, model, dataset, num_classes, init_lr):
        self.dataset = dataset
        self.net = model
        self.lr = init_lr
        self.cls_num = num_classes

    def set_lr(self, lr):
        self.lr = lr

    def iterate(self):
        """
            compute loss and perform back propagation
        """
        images, labels = self.dataset.get_next_batch()

        out_tensor = self.net.forward(images)

        loss = self.loss_forward(out_tensor, labels)
        out_diff_tensor = self.loss_backward(out_tensor, labels)
        
        self.net.backward(out_diff_tensor, self.lr)
        
        return loss
    
    
    def loss_forward(self, embs, y):
        """
            compute Cross Entropy Error Function
            Note that y is NOT one-hot encoding
            embs is a [batch_size, num_classes] tensor
        """
        N = embs.shape[0]
        exps = np.exp(embs)
        sfm = exps / np.sum(exps, axis=1, keepdims=True)
        logprobs = - np.log(sfm[[list(range(self.dataset.batch_size))], y-1]).mean()
        
        return logprobs
    
    
    def init_params(self):
        self.r = 1.0
        
        
    def loss_backward(self, embs, y):
        N = embs.shape[0]
        y = np.eye(self.cls_num)[(y - 1)]
        embs = (embs - embs.max(axis=1, keepdims=True))
        exps = np.exp(embs)
        sfm = exps / np.sum(exps, axis=1, keepdims=True)
        
        return (sfm - y) / N
    
    def l2_norm_loss_forward(self, embs, y):
        """
            Ring Loss computation
            L = ring loss + softmax based loss
        """
        embs = (embs - embs.min(axis=1, keepdims=True)) / (embs.max(axis=1, keepdims=True) - embs.min(axis=1, keepdims=True))
        exps = np.exp(embs)
        sfm = exps / np.sum(exps, axis=1, keepdims=True)
        logprobs = - np.log(sfm[[list(range(16))], y-1]).mean()

        
        embs_norm = np.norm(embs, axis=1, ord=2, keepdims=True)
        r = np.repeat(self.r, self.dataset.batch_size).reshape(embs_norm.shape)
        # Since it is the 'auto' mode in the paper
        # the 'lambda' term is m
        ring_loss = 2 * ((embs_norm - r) ** 2).mean() * self.m 

        return logprobs + ring_loss
    
    def ring_loss_backward(self, embs, y):
        # gradient of softmax
        y = np.eye(self.cls_num)[(y - 1)]
        embs = (embs - embs.min(axis=1, keepdims=True)) / (embs.max(axis=1, keepdims=True) - embs.min(axis=1, keepdims=True))
        exps = np.exp(embs)
        sfm = exps / np.sum(exps, axis=1, keepdims=True)
        soft_gradient = (sfm - y) / self.dataset.batch_size
        
        
        norm_embs = np.norm(embs, axis=1, ord=2, keepdims=True)
        embs_norm = embs / norm_embs
        r = np.repeat(self.r, self.dataset.batch_size).reshape(norm_embs.shape)
        ring_gradient = 2 * embs_norm * (norm_embs - r) / self.dataset.batch_size
        
        # gradient of R
        r_gradient = - ((norm_embs - r) ** 2).mean() * self.m
        self.r -= self.lr * r_gradient

        return soft_gradient + ring_gradient

def train_lfw():
    
    batch_size = 128
    image_h = 128
    image_w = 128
    num_classes = 5750
    init_lr = 0.001
    threshold = 0.5
    save_per_n = 50
    n_epochs = 1000
    data_path = "train-aligned.txt"
    
    print(datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = dataloader(data_path, batch_size, image_h, image_w)
    print('Data load done!')

    # initialize the model and trainer
    model = lightcnn29(num_classes, threshold)
    train = trainer(model, dataset, num_classes, init_lr)


    model.train()

    for i in tqdm(range(n_epochs)):
        aloss = train.iterate()
        out_str = 'epoch: {} loss: {:.4f}'.format(i, aloss)
        print(out_str)
        
        if i % save_per_n == 0:
            # model.eval()
            # acc = test(model, "test-match.txt", image_h, image_w)
            # print('Acc: {:.4f}'.format(acc))
            # accs.append(acc)
            model.save("model")
            # model.train()

        # reschedule learning rate each 100 iteration
        if i % 100 and i != 0:
            init_lr = init_lr / 5
            train.set_lr(init_lr)


    
if __name__ == "__main__":
    train_lfw()