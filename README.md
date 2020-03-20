# Face Verification System
This is the course project for CS210709 Machine Learning at University of Science and Technology of China.

## Requirements
- Implement a face verification system to verify whether two facial images belong to the same person. 
- Do **NOT** use any machine learning packages, e.g., scikit-learn, TensorFlow, or PyTorch. You are supposed to implement your face verification system from scratch. For example, if you want to use SVM, you are expected to implement the model and the training algorithm; if you want to use CNN, you are expected to implement both the neural network and the backpropagation.
- The offered dataset is a subset [LFW](http://vis-www.cs.umass.edu/lfw/).

## Model 

We have implemented LightCNN based on numpy with its pretrained model.

## Usage

To train the model, you first download the datasets [here](https://github.com/AlfredXiangWu/LightCNN) and configure the following parameters and path(default parameters are preferred) in `train.py`
```python
    batch_size = 128
    image_h = 128
    image_w = 128
    num_classes = 5750
    init_lr = 0.001
    threshold = 0.2
    save_per_n = 50
    n_epochs = 1000
    data_path = "train-aligned.txt"
```

where the format of `train-aligned.txt` should be 
```text
Anthony_Pico/Anthony_Pico_0001.bmp 0
```

And run the following command
```python
python train.py
```

To test the model, you need to configure the following paths in `test.py`
```python
model_path = 'model'
test_path = 'train-aligned-label.txt'
```
And run the following command
```python
python test.py
```