import os
from convertToCSV import *

try:
    os.system("wget github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz")
    os.system("wget github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz")
    os.system("wget github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz")
    os.system("wget github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz")

    os.system("gunzip t10k-images-idx3-ubyte.gz")
    os.system("gunzip t10k-labels-idx1-ubyte.gz")
    os.system("gunzip train-images-idx3-ubyte.gz")
    os.system("gunzip train-labels-idx1-ubyte.gz")

    convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train.csv", 60000)
    convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test.csv", 10000)

    os.system("rm train-images-idx3-ubyte")
    os.system("rm train-labels-idx1-ubyte")
    os.system("rm t10k-images-idx3-ubyte")
    os.system("rm t10k-labels-idx1-ubyte")

    os.system("rm -r __pycache__")
except:
    print("Happened somethink really terrible, or you just use Windows!")
finally:
    print("Script executed succesfull!")
