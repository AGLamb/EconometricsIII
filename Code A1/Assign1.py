from statsmodels.tsa.api import VAR
from numpy.linalg import inv, eig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class VAR_Model:
    def __init__(self, input_df: pd.DataFrame, lags: int):
        self.coef = None
        self.pred = None
        self.exog = input_df
        self.aic = None
        self.sic = None
        self.hqic = None
        self.order = lags
        self.resid = None
        self.sigma2_ML = None
        self.sigma2_LS = None
        self.exo_lagged = None

    def drop_obs(self, no_to_drop: int) -> None:
        inter = self.exog.iloc[no_to_drop - 1:, :]
        self.exog = inter
        return None

    def regress(self) -> None:
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
        self.exo_lagged = exo[:, col_num:]
        print(f'The following are the coefficients of the regression: \nC =  {self.coef[0, :]}')

        row = 1
        for i in range(1, self.order + 1):
            print(f'B_{i} = {self.coef[row:row + len(self.exog.columns), :]}')
            row += len(self.exog.columns)

        print("\n")
        self.pred = (exo[:, col_num:] @ self.coef)
        self.resid = self.exog.iloc[self.order:, :] - self.pred
        self.Sigma2()
        self.AIC()
        self.SIC()
        self.HQIC()
        return None

    def Sigma2(self) -> None:
        n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
        k = p ** 2 * self.order + p
        self.sigma2_ML = (1 / n) * (self.resid.T @ self.resid)
        self.sigma2_LS = (1 / (n - (k * p) - 1)) * (self.resid.T @ self.resid)
        return None

    def AIC(self) -> None:
        if self.pred is None:
            raise ValueError(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = p ** 2 * self.order + p
            self.aic = np.log(np.linalg.det(self.sigma2_ML)) + (2 * k) / n
        return None

    def SIC(self) -> None:
        if self.pred is None:
            raise ValueError(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = p ** 2 * self.order + p
            self.sic = np.log(np.linalg.det(self.sigma2_ML)) + (k * np.log(n)) / n
        return None

    def HQIC(self) -> None:
        if self.pred is None:
            raise ValueError(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = p ** 2 * self.order + p
            self.hqic = np.log(np.linalg.det(self.sigma2_ML)) + (2 * k * np.log(np.log(n))) / n
        return None

    # def impulse_response(self):
    #     return

    def granger_causality(self, alpha: float, caused_var: list, causing_var: list, constants: bool) -> None:

        if not (0 < alpha < 1):
            raise ValueError("Alpha value has to be between 0 and 1")

        n = len(causing_var) * len(caused_var) * self.order

        if constants:
            m = (len(self.exog.columns) * self.order + 1) * len(self.exog.columns)
            C = np.zeros((n, m))
            restriction_matrix = np.zeros((len(self.exog.columns), len(self.exog.columns) * self.order + 1))

            i = 0
            for var1 in causing_var:
                for var2 in caused_var:
                    for j in range(1, self.order + 1):
                        restriction_matrix[var2, var1 * j + 1] = 1
                        C[i, :] = vec_operator(restriction_matrix)
                        restriction_matrix[:, :] = 0
                        i += 1
        else:
            restriction_matrix = np.zeros((len(self.exog.columns), len(self.exog.columns) * self.order))
            m = (len(self.exog.columns) ** 2) * self.order
            C = np.zeros((n, m))

            i = 0
            for var1 in causing_var:
                for var2 in caused_var:
                    for j in range(self.order):
                        restriction_matrix[var2, var1 * j] = 1
                        C[i, :] = vec_operator(restriction_matrix)
                        restriction_matrix[:, :] = 0
                        i += 1

        Cb = np.dot(C, vec_operator(self.coef))
        kron_prod = np.kron(np.linalg.inv(self.resid.T @ self.resid), self.sigma2_LS)
        middle = np.linalg.inv(C @ kron_prod @ C.T)
        lam = (Cb.T @ middle @ Cb)
        statistic = lam / restrictions
        df = (restrictions, k * self.df_resid)
        dist = stats.f(*df)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - alpha)
        return


def vec_operator(input_matrix: np.ndarray) -> np.ndarray:
    output_vector = np.zeros((input_matrix.shape[0] * input_matrix.shape[1]))
    slicer = 0
    for i in range(input_matrix.shape[1]):
        end_slice = (input_matrix.shape[0] * (i + 1))
        output_vector[slicer:end_slice] = input_matrix[:, i]
        slicer += input_matrix.shape[0]
    return output_vector


def get_data(file_path: str) -> pd.DataFrame:
    output_df = pd.read_csv(file_path)
    output_df['date'] = pd.to_datetime(output_df['date'])
    output_df.set_index(keys='date', inplace=True)
    output_df.drop(columns=['Unnamed: 0'], inplace=True)
    return output_df


def display_order(order_results: dict) -> None:
    for key1 in order_results:
        print(f'{key1}: ')
        for key2 in order_results[key1]:
            print(f'{key2} = {order_results[key1][key2]:.2f}')
    return None


def select_order(models: list) -> None:
    output = dict()
    for model in models:
        output["VAR(" + str(model.order) + ")"] = {'AIC': model.aic, 'SIC': model.sic, 'HQIC': model.hqic}
    display_order(output)
    print(f'The best model according to AIC is: {get_min_dict_name(output, "AIC")}')
    print(f'The best model according to SIC is: {get_min_dict_name(output, "SIC")}')
    print(f'The best model according to HQIC is: {get_min_dict_name(output, "HQIC")}')
    return None


def get_min_dict_name(d, key):
    min_name = None
    min_val = float('inf')
    for name, inner_dict in d.items():
        if key in inner_dict:
            if inner_dict[key] < min_val:
                min_val = inner_dict[key]
                min_name = name
    return min_name


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


def ProblemC(input_df: pd.DataFrame) -> None:
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

    select_order(models=[model_1, model_2, model_3])
    return None


def ProblemD(input_df: pd.DataFrame) -> None:
    return None


def ProblemE(input_df: pd.DataFrame) -> VAR_Model:
    exog_df = input_df.copy()

    output_model = VAR_Model(exog_df, lags=2)
    output_model.regress()
    vec_matrix(output_model.coef)
    return output_model


def ProblemF(input_model: VAR_Model) -> None:
    input_model.granger_causality(caused_var=[0, 1], causing_var=[2], alpha=0.05, constants=True)
    return None


def main() -> None:
    path = "./Data/data_assignment1_2023.csv"
    df = get_data(path)
    # ProblemA(df)
    # ProblemB(df)
    # ProblemC(df)
    # ProblemD()
    Model_E = ProblemE(df)
    ProblemF(Model_E)
    return None


if __name__ == "__main__":
    main()
