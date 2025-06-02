import multiprocessing as mp
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def create_dataset(example, df):
    df_update = df.loc[df[1].isin((df[1].value_counts()[(example[1] >= df[1].value_counts()) & (df[1].value_counts() >= example[0])]).index)]
    return df_update[0].values, df_update[1].values