from numpy.linalg import inv, eig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def get_data(file_path: str) -> pd.DataFrame:
    output_df = pd.read_csv(file_path)
    output_df['date'] = pd.to_datetime(output_df['date'])
    output_df.set_index(keys='date', inplace=True)
    output_df.drop(columns=['Unnamed: 0'], inplace=True)
    return output_df


def ProblemA(input_df: pd.DataFrame) -> None:
    plt.plot(input_df.index, input_df.gdp, linestyle='solid')
    plt.show()
    plt.plot(input_df.index, input_df.ir, linestyle='solid')
    plt.show()
    plt.plot(input_df.index, input_df.cpi, linestyle='solid')
    plt.show()
    return None


def ProblemB(input_df: pd.DataFrame):
    exog_df = input_df
    exog_df.drop(columns='cpi', inplace=True)
    model = VAR_Model(exog_df)
    model.regress(lags=3)
    return


class VAR_Model:
    def __init__(self, input_df: pd.DataFrame):
        self.coef = None
        self.exog = input_df

    def regress(self, lags: int):
        exo = self.exog
        exo = exo.to_numpy()

        for i in range(1, lags):
            lag = np.roll(exo, shift=i, axis=0)
            lag[i, :] = 0
            exo = np.concatenate((exo, lag), axis=1)

        cons = np.ones((len(exo), 1))
        exo = np.concatenate((cons, exo), axis=1)

        col_num = self.exog.shape[1]
        beta_left = np.linalg.inv(np.matmul(exo[:, col_num:].T,
                                            exo[:, col_num:]))
        beta_right = np.matmul(exo[:, col_num:].T, exo[:, :col_num])
        self.coef = np.matmul(beta_left, beta_right)
        print(f'The following are the coefficients of the regression \n {self.coef}')
        return


def ProblemC():
    return


def ProblemD():
    return


def ProblemE():
    return


def main() -> None:
    path = "../Data/data_assignment1_2023.csv"
    df = get_data(path)
    ProblemA(df)
    ProblemB(df)
    return None


if __name__ == "__main__":
    main()
