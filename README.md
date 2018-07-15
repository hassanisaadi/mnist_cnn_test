# mnist_cnn_test
2 Class MNIST Classification Using CNN

# Goal
Using CNN to classify the MNIST dataset for two classes: the input digit image >= 3 or < 3. Thus it's a two class classification problem.

# Input Images
![input_image](./img/input_image.png)

## Downsampling
The input samples first downsampled to 14x14 (from 28x28) using tf.image.resize function.
![input_downsampled_image](./img/input_ds_image.png)

## Blurring
Then, a blurring filter applied to all the input samples. The blurring filter is (1/9) * np.ones(3,3)
![input_blurred_image](./img/input_blurr_image.png)

# Architecture
A big picture of architecture for three models (1, 2, 3 CNN layers) is in img directory.

# Number of CNN Layers Experiment
In this experiment, I changed the number of CNN layers from 1 to 3.
Every CNN layer is followed by a ReLU activation function and Batch Normalization operation. 
The model continues with a fully connected layer, a dropout layer, and a signle fully connected layer with 2 units to predict the two classes.

Test and Train accuracy for different number of filter (N) and different number of layers (left: 1, middle: 2, right: 3):
### Test Accuracy
![test_all](img/test_all.png)

### Train Accuracy
![train_all](img/train_all.png)

Legend for different number of filter on each CNN layer (Train/Test):

![legend](img/legend.png)

# Conclusion
All the models have a reasonable train accuracy. However, train accuracy for N=8 is smalled than others. 
Since the stride in CNN layers is 2 and the input image size is small (14x14), adding more CNN layers means we have a tiny image in the output of the model before the fully connected layer(7x7 for 1, 4x4 for 2, and 2x2 for 3 layers). Also, because of down-sampled and blurred input images, the output of CNN part of the model tends to a constant tensor (all black or white) and there are no useful feature which the network can classify samples. That's why a 3 layer model does not a good accuracy. 

Furthermore, According to the results, for two layer model we have a better performance for all N=8, 16, 32 compare to one layer model.
For N=64 the performance for both one and two layer models decrease which can be overfitting because the model becomes complex.
Therefore, I choose the two layer model. However, for N=16 performance of one layer model is a little bit better than the two layer model.

In summary the two layer model works better for N=8, 16, 32 and one layer model works better for only N=16.

# Performance
* Maximum accuracy after 1000 steps for 2 layer model (N = 32): 
** Test: 0.8221
** Train: 0.9600


# Parameters
* Learning rate: 0.001
* Batch size: 100
* Number of Train examples: 60,000
* Number of Test examples: 10,000
* Keep_prob for dropout layer: 0.9
* Kernel size for conv2d layer: (3,3)
* strides: (2,2)


## Prerequisites
* Python 3.5
* Tensorflow 1.4.1

