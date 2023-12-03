
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('new.csv', sep=";", decimal=".")

sns.set() 
data.plot(kind="scatter", x="BestValue", y="WorseValue")