import matplotlib.pyplot as plt
from numpy.linealg import inv
from numpy.linalg import eig
import pandas as pd
import numpy as np


def get_data(file_path: str)-> pd.DataFrame:
    return pd.read_csv(file_path)


def ProblemA(input_df: pd.DataFrame)-> None:
    """"""
    input_df['date'] = pd.to_datetime(input_df['date'])

    plt.plot(input_df.date, input_df.gdp, linestyle = 'solid')
    plt.show()
    plt.plot(input_df.date, input_df.ir, linestyle = 'solid')
    plt.show()
    plt.plot(input_df.date, input_df.cpi, linestyle = 'solid')
    plt.show()
    return


def ProblemB():
    return


def ProblemC():
    return


def ProblemD():
    return


def ProblemE():
    return


def main()-> None:
    path = "Data/data_assignment1_2023.csv"
    df = get_data(path)
    
    return None


if __name__ == "__main__":
    main()