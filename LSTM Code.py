# multivariate lstm libraries
import numpy as np
import pandas as pd
import xlrd
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
# extraction of data for dissertation
data = pd.read_excel(r'D:\Desktop\Coursework for Masters\7070MAA\bug_ml_updated.xlsx')
#extraction of average temp, humidity and bug count
pheno = pd.DataFrame(data, columns= ['avg_scaled','avg_humid_scaled','c_cnt'])
print(pheno)

