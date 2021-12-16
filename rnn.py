
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset.iloc[:,1:2].values


