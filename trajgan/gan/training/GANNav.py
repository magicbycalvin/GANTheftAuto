# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:33:22 2021

@author: ckielasjensen
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets, transforms


if __name__ == '__main__':
    pass