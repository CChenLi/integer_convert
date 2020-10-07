# Integer_convert

This project recognizes Chinese characters in a image file and convert them into Arabic numerals

### Prerequisites

[Pytorch](https://pytorch.org/)  
[OpenCV](https://opencv.org/)  
[torchvision](https://pytorch.org/docs/stable/torchvision/index.html)  

### Training Dataset
[Chinese MNIST](https://www.kaggle.com/gpreda/chinese-mnist/discussion/173101)

### Files
- `src/`
  - `model.py` Defination of `class IntRec`, the model trained to convert chinese character to Arabic numerals.
  - `utils.py` 
    - Implementation of `DataLoader`, which load image files from dataset into tensor of shape   
    `[batch, channel, height, width]` along with groundtruth label
    - `HelperFunc` a class of helper function to extract single characters from each image
  - `train.py` traning loops for the model, aumomatically use [cuda](https://developer.nvidia.com/cuda-downloads) if availiable. Tracking training and validation accuracy in `train_acc.pickle` and `valid_acc.pickle`
- `data/` containing training data images and a csv file `chinese_mnist.csv`, which contains paths to images and labels.
- `pictures/` the program will read image files from this folder and produce prediction

### Example
```

```

## Author

* **Chen Li** - *Initial work* - 

