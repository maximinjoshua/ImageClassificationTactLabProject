import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r"D:\tacti and tact\csvfiles\imagedata.csv")
X=dataset.iloc[0:,:-1]
y=dataset.iloc[0:,-1]

corrmat=dataset.corr()

corr_with_output=abs(corrmat['Labels'])

req=corr_with_output[corr_with_output>0.33]

print(req)
