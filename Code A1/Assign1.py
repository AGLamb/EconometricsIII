from statsmodels.tsa.api import VAR
from numpy.linalg import inv, eig
import matplotlib.pyplot as plt
from itertools import accumulate
import scipy.stats as stats
from pprint import pprint
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
        self.exog = self.exog.iloc[no_to_drop:, :]
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
        self.exo_lagged = exo[:, col_num:]

        coef_left = np.linalg.inv(self.exo_lagged.T @ self.exo_lagged)
        coef_right = self.exo_lagged.T @ exo[:, :col_num]
        self.coef = coef_left @ coef_right

        print(f'The following are the coefficients of the regression: \nC =  {self.coef[0, :]}')

        row = 1
        for i in range(1, self.order + 1):
            print(f'B_{i} = {self.coef[row:row + len(self.exog.columns), :].T}')
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
        self.sigma2_LS = (n / (n - k * p - 1)) * self.sigma2_ML
        return None

    def AIC(self) -> None:
        if self.pred is None:
            raise ValueError(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = (p ** 2) * self.order + p
            self.aic = np.log(np.linalg.det(self.sigma2_ML)) + (2 * k) / n
        return None

    def SIC(self) -> None:
        if self.pred is None:
            raise ValueError(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = (p ** 2) * self.order + p
            self.sic = np.log(np.linalg.det(self.sigma2_ML)) + (k * np.log(n)) / n
        return None

    def HQIC(self) -> None:
        if self.pred is None:
            raise ValueError(f'The model has not been estimated')
        else:
            n, p = self.exog.to_numpy().shape[0] - self.order, self.exog.to_numpy().shape[1]
            k = (p ** 2) * self.order + p
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
                    for j in range(self.order):
                        restriction_matrix[var2, var1 + j * len(self.exog.columns) + 1] = 1
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
                        restriction_matrix[var2, var1 + j * len(self.exog.columns)] = 1
                        C[i, :] = vec_operator(restriction_matrix)
                        restriction_matrix[:, :] = 0
                        i += 1
        print(C)
        Cb = np.dot(C, vec_operator(self.coef.T))

        kron_prod = np.kron(np.linalg.inv(self.exo_lagged.T @ self.exo_lagged), self.sigma2_LS)
        middle = np.linalg.inv(C @ kron_prod @ C.T)
        lam = (Cb.T @ middle @ Cb) / n
        print(f'The statistic is: {lam:.2f}')

        dist = stats.f(n, len(self.exog) - self.order * len(self.exog.columns) - len(self.exog.columns))
        pvalue = dist.sf(lam)
        crit_value = dist.ppf(1 - alpha)

        print(f'The critical values is:{crit_value:.2f}\nThe p-value is:{pvalue:.2f}')
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
    exog_df.drop(columns='ir', inplace=True)
    model = VAR_Model(exog_df, lags=3)
    model.regress()
    return


def ProblemC(input_df: pd.DataFrame) -> None:
    exog_df = input_df.copy()
    exog_df.drop(columns='ir', inplace=True)

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
    exog_df = input_df.copy()
    exog_df.drop(columns=['ir'], inplace=True)

    print(exog_df)

    model_3 = VAR_Model(exog_df, lags=3)
    model_3.regress()

    B_hat = model_3.coef.T
    # print(B_hat)

    ## calculate and plot 4 impulse response functions with 10 periods
    horizon = 11

    # Initialize arrays
    IR_array = np.zeros((horizon + 1, 2, 2))
    IR_array[0, :, :] = np.identity(2)
    # print(IR_array)

    # 2x2 array for each horizon
    A_hat_array = np.zeros((2, horizon * 2))
    A_hat_array[:, :6] = B_hat[:, 1:]

    # print(A_hat_array)

    # Calculate impulse responses, the range is correct cause 0 indexing
    for i in range(1, horizon + 1):
        # print(i)
        IR_here = np.zeros((2, 2))
        previous_j = 0
        for j in range(1, i + 1):
            IR_here += IR_array[i - j, :, :] @ A_hat_array[:, previous_j:(2 * j)]
            # print(i,j)
            # print('ir',IR_array[(i-1)-j, :, :])
            # print('A_hat', A_hat_array[:, previous_j:(2*j)])
            previous_j = 2 * j
        IR_array[i, :, :] = IR_here

    print(IR_array)

    # create lists with the impulse responses (this isnt general cause I already spent to much time on generality)

    # list for ir -> ir
    cpi_cpi = []
    cpi_gpd = []
    gpd_cpi = []
    gdp_gdp = []

    for i in range(horizon):
        cpi_cpi.append(IR_array[i, 1, 1])
        cpi_gpd.append(IR_array[i, 0, 1])
        gpd_cpi.append(IR_array[i, 1, 0])
        gdp_gdp.append(IR_array[i, 1, 1])

    cpi_cpi_acc = list(accumulate(cpi_cpi))
    cpi_gpd_acc = list(accumulate(cpi_gpd))
    gpd_cpi_acc = list(accumulate(gpd_cpi))
    gdp_gdp_acc = list(accumulate(gdp_gdp))

    # Define impulse response data
    impulse_responses = [cpi_cpi, cpi_gpd, gpd_cpi, gdp_gdp]
    acc_imulse_responses = [cpi_cpi_acc, cpi_gpd_acc, gpd_cpi_acc, gdp_gdp_acc]

    title_names = ['CPI -> CPI', 'CPI -> GDP', 'GDP -> CPI', 'GDP -> GDP']

    # Create figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=2)

    # Loop over impulse responses and plot in separate subplots
    for i, ir in enumerate(impulse_responses):
        # Determine row and column indices for current subplot
        row = i // 2
        col = i % 2

        # Plot impulse response and set x-axis limits and labels
        axs[row, col].plot(ir)
        axs[row, col].set_xlim([0, 10])
        axs[row, col].set_xticks(range(1, 11))
        axs[row, col].set_xticklabels(range(1, 11))
        axs[row, col].set_title(title_names[i])

    # Create figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=2)

    # Loop over impulse responses and plot in separate subplots
    for i, ir in enumerate(acc_imulse_responses):
        # Determine row and column indices for current subplot
        row = i // 2
        col = i % 2

        # Plot impulse response and set x-axis limits and labels
        axs[row, col].plot(ir)
        axs[row, col].set_xlim([0, 10])
        axs[row, col].set_xticks(range(1, 11))
        axs[row, col].set_xticklabels(range(1, 11))
        axs[row, col].set_title(title_names[i])

    # Adjust layout and show plot
    plt.show()

    return None


def ProblemE(input_df: pd.DataFrame) -> VAR_Model:
    exog_df = input_df.copy()

    output_model = VAR_Model(exog_df, lags=2)
    output_model.regress()
    return output_model


def ProblemF(input_model: VAR_Model) -> None:
    input_model.granger_causality(caused_var=[0, 2], causing_var=[1], alpha=0.05, constants=True)
    return None


def main() -> None:
    path = "./Data/data_assignment1_2023.csv"
    df = get_data(path)
    # ProblemA(df)
    # ProblemB(df)
    # ProblemC(df)
    # ProblemD(df)
    Model_E = ProblemE(df)
    ProblemF(Model_E)

    return None


if __name__ == "__main__":
    main()