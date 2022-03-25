# Self-Drivig-Car-Image-Semantic-Segmentation


## Introduction

Here is an example of creating an AI model for on-board computer vision systems for autonomous vehicles. The system being designed is composed of the following parts:

1. Acquisition of images in real time.
2. Image processing.
3. Image segmentation.
4. Decision system.

In this repository we focus on 3.'Image segmentation' part. The goal will be to design a first image segmentation model that should be easily integrated into the fully embedded IoT system.

## Dataset

The model is creted and tested on [Cityscapes Dataset](https://www.cityscapes-dataset.com/dataset-overview/)

## Model Logic

Semantic segmentation is a computer vision task of grouping similar parts of the image that belong to the same class.
Here it is not a question of predicting a value per image, but of predicting each pixel and its class. This therefore requires the following changes:
  1. The neural network model: The neural network model must be adapted in order to be able to output a matrix equal to that of the input (W x H x Class). To do this, it is necessary to adopt encoder-decoder structures because conventional networks use pooling in order to reduce the size of the inputs and to be able to interpret the details. Here, once arrived at the point of compression, it is necessary to decompress the data on the second part of the network in order to find the original size. We will therefore use the U-type models (named by their form of compression-decompression). We need, next to the image to also have the masks (or the notation of the points of interest in classes)
  2. Metrics: Classic metrics no longer work. It would then be necessary to use the appropriate metrics for this type of problem, such as:
  - IoU (Intersection over Union) or Jaccard Index (the most used). It is the intersection area (between prediction and mask) divided by the union area = I / U.
            For the multiclass classification, the average must be taken over all the classes 
  - Dice Coefficient (F1 Score): 2 * area of the intersection divided by the total number of pixels in the two images. = 2 * I / (2 * Pixels) 
  - Focal Tversky etc.
  3. Loss Functions:  Loss functions tested are those commonly used in this type of problem:
  - Dice Loss is used to calculate the overlap between the predicted class and the ground truth class. Our goal is to minimize 1- overlap between the class of predicted truth and ground truth.
  - Since cross entropy evaluates class predictions for each pixel vector individually and then averages over all pixels, we are essentially asserting equal learning for every pixel in the image. This can be a problem if your different classes have an unbalanced representation in the image, as the formation can be dominated by the most prevalent class
  - We preferred Dice Loss because the images have a lot of predominant classes (road, buildings)

  For more on metrics and loss check for exemple [Lars' Blog here](https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html)
  
## Image transformation
  
  It would then be interesting to be able to simulate real behavior by performing transformations on the data. This consists of applying variations to our images, both qualitative and spatial: for example a horizontal translation and noisy other images. This can also be beneficial when training, as our dataset is small and it can allow us to reduce overfitting. The idea here is to simulate for example rainy days when camera does not catch perfect images or some camera dirt etc. 

To apply these transformations, we decided to use the [Albumentations library](https://pypi.org/project/albumentations/), which also allows us to maintain consistency between the images and the segmentation masks.

## Unet Model
  
  As said earlier, here we have to choose the family of models of the encoder/decoder type. Here, the architecture consists of two paths:
  1. Encoder path — The left side of the “U” shaped architecture is called the encoder path/contraction. This path consists of a convolutional layer stack and a maximum pooling layer. This is used to capture the context of the image.
  2. Decoder path — The right side of the “U” shaped architecture is called the decoder/extension path. The path consists of transposed convolutional layers. This is used to extend the precise localization (spatialization).

It is an end-to-end Fully Convolutional Network (FCN). It can accept any image size because it has no fully connected dense layers.

  ### Baseline

For our base case we used a mini Unet network with 3 convolution blocks in the encoder and 2 deconvolution blocks, followed by an output block for the deconvolution and output phase. Unsurprisingly, the model being shallow, cannot distinguish the details well and even confuses the large masses with each other. It gets a Dice score of around 0.68 on the validation game. Below are examples of predictions from this model.

  ### Transfer Learning

In order to obtain better results, we tested the transfer learning on a pretrained EfficientNet type model. This is a [Google model](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html), trained on 1.5 million images. We used the first (about 50%) layers of Efficient net as an encoder to extract a number of common features. The image therefore passes through the encoder, decreasing in size with each layer, but extracting more and more features. The encoder being frozen at first, we only trained the decoder on our data set. Once the decoder was trained we freed up 10% encoder layers in order to fine-tune the entire model. This approach gave the best results with a Dice score of more than 0.9 on the validation game.

## Results

Here is an example of a prediction coming out of this model. We clearly see that the first pedestrians and the first objects (traffic light) begin to appear. Below is the confusion matrix emerging from this model:
 

We can therefore see what we have already seen in the photo, it is that the classes less well represented are the small classes, humans and objects. Objects are often confused with construction or nature while humans are confused with construction and vehicles.

For the other classes, the rate of true positives is quite high.


## Flaask Rest API
The model is deployed as endpoint on Azure, via simple Flask webpage. 

