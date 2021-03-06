import math
import numpy as np


def PriceReverse(df, cycle, time):
    """
    Compute 1M Price Reversal
    Order: Ascending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: PM_{i,t} = (Close_{i,t} - Close_{i, t-1}) / Close_{i, t-1}
    """
    try:
        previous_price = df.iloc[time - cycle]
        i = 1
        while previous_price == 0:
            previous_price = df.iloc[time-cycle-i]
            i += 1
        return (df.iloc[time] - previous_price) / previous_price
    except KeyError:
        pass


def PriceMomentum(df, cycle, time):
    """
    Compute 1M Price Reversal
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: PM_{i,t} = (Close_{i,t} - Close_{i, t-1}) / Close_{i, t-1}
    """
    try:
        previous_price = df.iloc[time - cycle]
        i = 1
        while previous_price == 0:
            previous_price = df.iloc[time-cycle-i]
            i += 1
        return -(df.iloc[time] - previous_price) / previous_price
    except KeyError:
        pass


def MomentumReturn(df, cycle, time):
    try:
        x_0 = df.iloc[time - 2*cycle]
        x_t = df.iloc[time - cycle]
        x_T = df.iloc[time]

        i = 1
        while x_0 == 0:
            x_0 = df.iloc[time-2*cycle-i]
            i += 1

        ttl_ret = (x_T - x_0) / x_0
        half_ret = (x_t - x_0) / x_0
        return ttl_ret + half_ret
    except KeyError:
        pass


def MeanCutOff(df, cycle, time):
    try:
        mu = df.iloc[time - 2*cycle:time].mean()
        upr = (df.iloc[time - 2*cycle:time] > mu).sum()
        lwr = (df.iloc[time - 2*cycle:time] <= mu).sum()
        return upr - lwr
    except KeyError:
        pass


def Price_High_Low(df, cycle, time):
    """
    Compute High-minus-low:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: HL_{i,t} = (High_{i,t} - Close_{i,t}) / (Close_{i,t} - Low_{i,t})
    """
    try:
        arr = df.iloc[time-cycle:time].values
        Current = df.iloc[time]
        High = max(arr)
        Low = min(arr)
        if Current == Low:
            return -(High - Current) / 1e-8
        return -(High - Current) / (Current - Low)
    except KeyError:
        pass


def RelativeStrengh(df, cycle, time):
    """
    Compute relative strength index
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: RSI_{i,t} = 100 - (100 / (1 + avg. gain / avg. loss))
    """
    try:
        arr = df.iloc[time - cycle:time].values
        pct_arr = np.diff(arr) / arr[:len(arr)-1]
        gain = pct_arr[pct_arr > 0].mean()
        loss = pct_arr[pct_arr < 0].mean()
        return 100 - (100 / (1 + gain / loss))
    except KeyError:
        pass

def Vol_Coefficient(df, cycle, time):
    """
    Compute Coefficient of Variation:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: CV_{i,t} = Std(Close_i, cycle) / Ave(Close_i, cycle)
    """
    try:
        arr = df.iloc[time - cycle:time].values
        arr_pct = np.diff(arr) / arr[:len(arr)-1]
        std = np.std(arr_pct)
        avg = np.mean(arr_pct)
        return -std / (avg+1e-6)
    except KeyError:
        pass


def AnnVol(df, cycle, time):
    """
    Compute Annual Volatility:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: AnnVol = sqrt(252) * sqrt(1/21 * sum(r_{i,t-j}^2))
    where r_{i,s} = log(Close_{i,t} / Close_{i,t-1})
    """
    try:
        r_2 = int(0)
        for i in range(1, cycle):
            log = np.log(df.iloc[time - i] / df.iloc[time - i - 1])
            r_2 += log ** 2
        result = np.sqrt(252 / cycle * r_2)
        return -result
    except KeyError:
        pass


def MovingAverage(df, cycle, time):
    """
    Compute Moving Average:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: (MA_10 - Price) + (MA_20 - Price) * 2 + (MA_50 - Price) * 5
    """
    if time - 50 < 0 and not math.isnan(df.iloc[time - cycle]):
        return 0
    elif time - 50 >= 0 and not math.isnan(df.iloc[time - 50]):
        try:
            arr = df.iloc[time - 50:time].values
            cumsum = np.cumsum(np.insert(arr, 0, 0))
            ma10 = (cumsum[cycle:] - cumsum[:-cycle]) / cycle
            ma20 = (cumsum[cycle*2:] - cumsum[:-cycle*2]) / 2 / cycle
            ma50 = (cumsum[50:] - cumsum[:-50]) / 50
            res = ma10[-1] + ma20[-1] + ma50 - df.iloc[time]*3
            return res[0]
        except KeyError:
            pass


def MACD(df, cycle, time):
    """
    Compute Moving Average Convergence Divergence:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: cycle-Period EMA - cycle*2-Period EMA
    where EMA = Price_t * k + EMA_t-1 * (1-k)
    k = 2 / (N+1)
    """
    try:
        data = df.iloc[time - cycle:time]
        EMA_SR = data.ewm(span=cycle).mean()
        EMA_LR = data.ewm(span=cycle*2).mean()
        res = list(EMA_SR)[-1] - list(EMA_LR)[-1]
        return res
    except KeyError:
        pass


def BoolingerBands(df, cycle, time):
    """
    Compute Boolinger Bands:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: Ave(cycle) +- 2 * Std(cycle)
    """
    if time - 2 * cycle <= 0 and not math.isnan(df.iloc[time - cycle]):
        return 0
    try:
        arr_lr = np.nan_to_num(df.iloc[time - 2*cycle+1:time].values)
        arr_sr = np.nan_to_num(df.iloc[time - cycle:time].values)
        # moving average for long-run
        cumsum = np.cumsum(np.insert(arr_lr, 0, 0))
        ma_cycle = (cumsum[cycle:] - cumsum[:-cycle]) / cycle
        delta = np.std(arr_sr)
        up_bound = ma_cycle + delta
        lw_bound = ma_cycle - delta
        midpoint = len(arr_sr) // 2
        res = sum(arr_sr[:midpoint] > up_bound[:midpoint]) - sum(arr_sr[midpoint:] < lw_bound[midpoint:])
        # calculate pct_change
        arr_pct = np.diff(arr_sr) / arr_sr[:len(arr_sr)-1]
        return res * np.std(arr_pct)
    except ValueError:
        pass


trading_strategies = [MovingAverage, PriceReverse, PriceMomentum, MomentumReturn, MeanCutOff, Price_High_Low, Vol_Coefficient, AnnVol, MACD, BoolingerBands]


def MinVariance(data, ranking, time, cycle):
    """
    MinVariance minimizes variance (needs short positions)
    Argument ranking: list of stocks from PitchStock
            return weighting for each stock (in percentage)
    """
    covar = np.zeros(shape=(len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = data[ranking[i]].iloc[time-cycle:time].fillna(method='Backfill')
    inv_cov_matrix = np.linalg.pinv(np.cov(covar))
    ita = np.ones(inv_cov_matrix.shape[0])
    weight = (inv_cov_matrix @ ita) / (ita @ inv_cov_matrix @ ita)
    return weight


def EqualWeight(data, ranking, time, cycle):
    """
    EqualWeight assign weight by 1/N
    return weighting for each stock (in percentage)
    """
    N = len(ranking)
    weight = np.ones(shape=N) / N
    return weight



def RiskParity(data, ranking, time, cycle):
    """
    RiskParity inversely invest for stock according to their volatility
    disregards covariance is the major drawback
    return weighting for each stock (in percentage)
    """
    covar = np.zeros(shape=(len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = data[ranking[i]].iloc[time - cycle:time].fillna(method='Backfill')
    vol = np.array(covar.std(axis=1))
    vol = 1 / (vol + 1e-8)
    weight = vol / vol.sum()
    return weight


rebalancing_strategies = [MinVariance, EqualWeight, RiskParity]
