import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from multiprocessing import Pool, Process

from Strategy import *


def Granger_Causality_Pred(num):
    # loading dataset
    df = pd.read_csv("Y:\\Dropbox\\Dropbox (MIT)\\Robinhood Trading\\Stock Data\\broader_stock.csv")
    df = df.set_index(pd.to_datetime(df['Date']))
    df.drop(['Date'], axis=1, inplace=True)
    pct_df = df.pct_change().shift(1).iloc[2:]

    # set up global variables
    leader_tick_dict = {}
    leader_t_val = {}
    perf_res = {}
    w_mktre = (1 + pct_df['SPY_Close']).resample('W').prod() - 1

    _ = 0

    # identify leaders
    for tick in pct_df.columns[::3][(num * 100):((num + 1) * 100)]:
        # picking leaders for each stocks
        target_arr = pct_df[tick].dropna()
        w_target = (1 + target_arr).resample('W').prod() - 1
        Y = w_target.shift(-1)
        leader_set = []
        leader_tval = []

        for leader in pct_df.columns[::3]:
            if leader != tick:
                leader_arr = pct_df[leader].dropna()
                w_leader = (1 + leader_arr).resample('W').prod() - 1

                tempreg_dta = pd.concat([Y, w_target, w_mktre, w_leader], axis=1).dropna()
                tempreg_dta.columns = ['Y', 'Y-1', 'Mkt', 'Lead']

                if tempreg_dta.shape[0] >= 36 * 4:
                    ols = sm.OLS(tempreg_dta['Y'].iloc[-36 * 4:],
                                 sm.add_constant(tempreg_dta[['Y-1', 'Mkt', 'Lead']].iloc[-36 * 4:]))
                    res = ols.fit(cov_type='HC0')
                    leader_sig = res.pvalues[3]

                elif tempreg_dta.shape[0] >= 12 * 4:
                    ols = sm.OLS(tempreg_dta['Y'].iloc[-12 * 4:],
                                 sm.add_constant(tempreg_dta[['Y-1', 'Mkt', 'Lead']].iloc[-12 * 4:]))
                    res = ols.fit(cov_type='HC0')
                    leader_sig = res.pvalues[3]

                else:
                    leader_sig = 1

                if leader_sig <= 1e-3:
                    leader_set.append(leader)
                    leader_tval.append(abs(res.tvalues[3]))

        leader_tick_dict[tick] = leader_set
        leader_t_val[tick] = leader_tval

        # evaluate performance
        leader = leader_tick_dict[tick]
        t_val = leader_t_val[tick]
        if len(leader) > 1:
            # simple average
            avg_signal = ((1 + pct_df[leader_tick_dict[tick]]).resample('W').prod() - 1).mean(axis=1)
            # only evaluate at short term period
            val_avg = pd.concat([w_target.shift(1).iloc[-12 * 4:], avg_signal.iloc[-12 * 4:]], axis=1).dropna().values
            # metrics
            mu_avg = mean_squared_error(val_avg[:, 0], val_avg[:, 1]) * 100
            acc_avg = accuracy_score((val_avg[:, 0] > 0).astype(int), (val_avg[:, 1] > 0).astype(int))
            perf_res[tick] = [mu_avg, acc_avg]

        _ += 1
        print("{}/100".format(_))

    perf_pls = {}
    N = len(leader_tick_dict.keys())
    count = 0

    for tick in leader_tick_dict.keys():
        leader = leader_tick_dict[tick]
        if len(leader) > 1:
            leader_arr = df[leader_tick_dict[tick]]
            target_arr = pct_df[tick].dropna()
            w_target = (1 + target_arr).resample('W').prod() - 1
            pls_set = []

            for col in leader_arr.columns:
                ind_arr = []
                for t in range(leader_arr.shape[0] - 300, leader_arr.shape[0]):
                    macd = MACD(leader_arr[col], 5, t)
                    booling = BoolingerBands(leader_arr[col], 5, t)
                    volcof = Vol_Coefficient(leader_arr[col], 5, t)
                    anvol = AnnVol(leader_arr[col], 5, t)
                    phl = Price_High_Low(leader_arr[col], 5, t)
                    prev = PriceReverse(leader_arr[col], 5, t)

                    ind_arr.append([macd, booling, volcof, anvol, phl, prev])

                w_X = pd.DataFrame(data=ind_arr, index=leader_arr.index[-300:]).resample('W').mean()
                temp_dta = pd.concat([w_target, w_X], axis=1).dropna().values[-12 * 4:, ]
                pls = PLSRegression(n_components=1)
                pls_x = pls.fit_transform(X=temp_dta[:, 1:], y=temp_dta[:, :1])[0]
                pls_set.append(pls_x)
            pls_X = np.column_stack(pls_set)

            signal = np.mean(pls_X, axis=1)
            actual = w_target.iloc[-49:-1]

            mu_pls = mean_squared_error(actual, signal) * 100
            acc_pls = accuracy_score((actual > 0).astype(int), (signal > 0).astype(int))
            perf_pls[tick] = [mu_pls, acc_pls]

        count += 1
        print("{}/{}".format(count, N))

    avg_res = pd.DataFrame(perf_res).T
    pls_res = pd.DataFrame(perf_pls).T
    ttl_res = pd.concat([pls_res, avg_res], axis=1)
    ttl_res.columns = ['MSE_PLS', 'ACC_PLS', 'MSE_AVG', 'ACC_AVG']
    ttl_res.to_csv('Granger_Causality_Res%s.csv' % num)


Pros = []
if __name__ == "__main__":
    print('Threads Started')
    for i in range(2, 4):
        p = Process(target=Granger_Causality_Pred, args=(i,))
        Pros.append(p)
        p.start()

    for t in Pros:
        t.join()

    print("Done")
