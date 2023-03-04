from statsmodels.tsa.api import VAR
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
    exog_df = input_df.copy()
    exog_df.drop(columns='cpi', inplace=True)
    model = VAR_Model(exog_df, lags=3)
    model.regress()
    return


def select_order(models: list):
    output = dict()
    for model in models:
        output["VAR(" + str(model.order) + ")"] = [model.aic, model.sic, model.hqic]
    return output


class VAR_Model:
    def __init__(self, input_df: pd.DataFrame, lags: int):
        self.coef = None
        self.pred = None
        self.exog = input_df
        self.aic = None
        self.sic = None
        self.hqic = None
        self.order = lags

    def drop_obs(self, no_to_drop: int):
        inter = self.exog.iloc[no_to_drop - 1:, :]
        self.exog = inter
        return

    def regress(self):
        exo = self.exog.copy()
        exo = exo.to_numpy()

        exo = np.concatenate((exo, np.ones((len(exo), 1))), axis=1)

        for i in range(1, self.order + 1):
            lag = np.roll(exo[:, :len(self.exog.columns)], shift=i, axis=0)
            lag[:i, :] = 0
            exo = np.concatenate((exo, lag), axis=1)
        exo = exo[self.order:, :]
        col_num = self.exog.shape[1]
        self.coef = np.linalg.inv(exo[:, col_num:].T @ exo[:, col_num:]) @ exo[:, col_num:].T @ exo[:, :col_num]
        print(f'The following are the coefficients of the regression: \nC =  {self.coef[0, :]}')

        row = 1
        for i in range(1, self.order + 1):
            print(f'B_{i} = {self.coef[row:row + len(self.exog.columns), :]}')
            row += len(self.exog.columns)

        print("\n")
        self.pred = (exo[:, col_num:] @ self.coef)
        self.AIC()
        self.SIC()
        self.HQIC()
        return

    def AIC(self):
        if self.pred is None:
            raise Exception(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = p ** 2 * self. order + p
            resid = self.exog.iloc[self.order:, :] - self.pred
            sigma2 = (1 / n) * (resid.T @ resid)
            self.aic = np.log(np.linalg.det(sigma2)) + (2 * k) / n
        return

    def SIC(self):
        if self.pred is None:
            raise Exception(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = p ** 2 * self.order + p
            resid = self.exog.iloc[self.order:, :] - self.pred
            sigma2 = (1 / n) * (resid.T @ resid)
            self.sic = np.log(np.linalg.det(sigma2)) + (k * np.log(n)) / n
        return

    def HQIC(self):
        if self.pred is None:
            raise Exception(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = p ** 2 * self.order + p
            resid = self.exog.iloc[self.order:, :] - self.pred
            sigma2 = (1 / n) * (resid.T @ resid)
            self.hqic = np.log(np.linalg.det(sigma2)) + (2 * k * np.log(np.log(n))) / n
        return

    # def impulse_response(self):
    #     return


def ProblemC(input_df: pd.DataFrame):
    exog_df = input_df.copy()
    exog_df.drop(columns='cpi', inplace=True)

    model_1 = VAR_Model(exog_df, lags=1)
    model_1.drop_obs(no_to_drop=2)
    model_1.regress()

    model_2 = VAR_Model(exog_df, lags=2)
    model_2.drop_obs(no_to_drop=1)
    model_2.regress()

    model_3 = VAR_Model(exog_df, lags=3)
    model_3.regress()

    order_res = select_order(models=[model_1, model_2, model_3])
    print(order_res)
    return


def ProblemD():
    return


def ProblemE():
    return


def main() -> None:
    path = "./Data/data_assignment1_2023.csv"
    df = get_data(path)
    # ProblemA(df)
    ProblemB(df)
    ProblemC(df)
    return None


if __name__ == "__main__":
    main()
