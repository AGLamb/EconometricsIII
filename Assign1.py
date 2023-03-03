import matplotlib.pyplot as plt
from numpy.linealg import inv
from numpy.linalg import eig
import pandas as pd
import numpy as np


def get_data(file_path):
    return pd.read_csv(file_path)