# Integer_convert

- This project recognizes Chinese characters in a image file and convert them into Arabic numerals  
Check [demo.ipynb](https://github.com/CChenLi/integer_convert/blob/main/demo.ipynb) for the whole process 
- Future work: to handle rotated input, use [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering) for number seperation. Define the similarity matrix with gaussian kernal and run the K-mean on the eigenspace.


### Prerequisites

[Pytorch](https://pytorch.org/)  
[OpenCV](https://opencv.org/)  
[torchvision](https://pytorch.org/docs/stable/torchvision/index.html)  

### Example

 | label            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|:-----------------:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Chinese Character | 零 | 一| 二 | 三 | 四 | 五 | 六 | 七 | 八 | 九 | 十 | 百 | 千 | 万 | 亿 |
| numeral           | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 100 | 1000 | 10000 | 10-e8 |

- Original photo
<img width="302" alt="Screen Shot 2020-10-17 at 12 35 32 AM" src="https://user-images.githubusercontent.com/63531857/96331349-67639500-1011-11eb-9973-3d97b9bcff67.png">

- After process
<img width="40%" alt="Screen Shot 2020-10-17 at 12 35 56 AM" src="https://user-images.githubusercontent.com/63531857/96331379-8f52f880-1011-11eb-9372-2dca1236f291.png">
<img width="50%" alt="Screen Shot 2020-10-17 at 12 36 06 AM" src="https://user-images.githubusercontent.com/63531857/96331378-8eba6200-1011-11eb-81b8-f115e44b7bf9.png">
<img width="40%" alt="Screen Shot 2020-10-17 at 12 36 15 AM" src="https://user-images.githubusercontent.com/63531857/96331377-8e21cb80-1011-11eb-8791-2639067cea12.png">

### Training Dataset
[Chinese MNIST](https://www.kaggle.com/gpreda/chinese-mnist/discussion/173101)


### Files
- `demo.ipynb` a example to show how the program works.
- `src/`
  - `model.py` Defination of `class IntRec`, the model trained to convert chinese character to Arabic numerals.
  - `utils.py` 
    - Implementation of `DataLoader`, which load image files from dataset into tensor of shape   
    `[batch, channel, height, width]` along with groundtruth label
    - `HelperFunc` a class of helper function to extract single characters from each image
  - `train.py` traning loops for the model, aumomatically use [cuda](https://developer.nvidia.com/cuda-downloads) if availiable. Tracking training and validation accuracy in `train_acc.pickle` and `valid_acc.pickle`
- `data/` containing training data images and a csv file `chinese_mnist.csv`, which contains paths to images and labels.
- `pictures/` the program will read image files from this folder and produce prediction


## Author

* **Chen Li** - *Initial work* - 

