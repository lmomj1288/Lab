import numpy as np
import pandas as pd 

data = np.load("C:/Users/lmomj/Desktop/연구실/인수인계/UIU-net datasets/test_set.npz")

for arr_name in data.files:
    arr = data[arr_name]
    print(f"\n배열 이름: {arr_name}")
    print(f"Shape: {arr.shape}")
    