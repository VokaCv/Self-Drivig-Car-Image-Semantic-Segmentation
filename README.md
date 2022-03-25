# Self-Drivig-Car-Image-Semantic-Segmentation


## Introduction

Here is an example of creating an AI model for on-board computer vision systems for autonomous vehicles. The system being designed is composed of the following parts:

1. Acquisition of images in real time.
2. Image processing.
3. Image segmentation.
4. Decision system.

In this repository we focus on 3.'Image segmentation' part. The goal will be to design a first image segmentation model that should be easily integrated into the fully embedded IoT system.

## Dataset
the model is creted and tested on [Cityscapes Dataset](#https://www.cityscapes-dataset.com/dataset-overview/)

## Dependencies

The notebooks were developed using the Microsoft Azure Machine Learning Studio web interface.

All dependencies are listed in `*.yml` files.

All scripts are available in the `/scripts` folder.

## Installation of conda and packages

To use notebooks, import this project into Microsoft Azure Machine Learning Studio.

- Launch a calculation instance.
- Open a terminal and go to the project directory:
- Create a conda virtual environment and install the dependencies:
```
./conda_create_env.sh
``` 
