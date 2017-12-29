# Zalando-Neural-Network

## Structure of project:
- MnistDataset:
  - genMnistDataset.py
  - mnist_test.csv
  - mnist_train.csv
- NeuralNet.py
- performance_history.csv
- README.md

## Description:
#### MnistDataset/genMnistDataset.py:
After running this file, from Zalando github repository will be download of four archives. After it they will be unpacking, on the next step they will be converted into CSV files type. Files what we are can to use for training out neural network.
#### MnistDataset/mnist_test.csv:
File for testing out neural network after training. Available only after running “genMnistDataset.py” script.
#### MnistDataset/mnist_train.csv:
File for training our neural network. Available only after running “genMnistDataset.py” script.

#### NeuralNet.py:
Inside of this file we are have a code with class initialization and few functions which make some actions, I mean: training, recording results of training, testing, writing results to a file and so on.
#### performance_history.csv:
Here we are record your results of training with different parameters.
#### README.md:
  All is simple, it’s Readme file :)

## System requirements:
- **Python3;**		I was used python 3.5
- **Python libraries:**	matplotlib, scipy, numpy
- **Linux distributive;** 	I was used Ubuntu 16.04 LTS
- **Little interest in this topic and patience**

