import numpy as np

import random
from sklearn.datasets import load_boston
from DataReader_1_2 import *
from HyperParameters_1_0 import *
from NeuralNet_1_1 import *

if __name__ == '__main__':
    # data
    reader = DataReader_1_2(load_boston().data,load_boston().target)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    hp = HyperParameters_1_0(13, 1, eta=0.01, max_epoch=2000, batch_size=40, eps = 1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader,checkpoint=0.1)
    # inference
    x = np.array([0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900,  1, 296.0, 15.30, 396.90,  4.98]).reshape(1,13)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    Z_real = z * reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print("Z=", Z_real,'\n')
