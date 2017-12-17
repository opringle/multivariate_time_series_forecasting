from pandas_datareader import data
import datetime
import pandas as pd
import numpy as np

#get a list of all tickers
tickers = ['GLD', 'AAPL']

#choose the dates to fetch data for
begdate = datetime.datetime(2014, 11, 11)
enddate = datetime.datetime(2016, 11, 11)

#loop through them, appending data to a single dataframe
label_cols = []
feature_cols = []
for i, ticker in enumerate(tickers):
    label_cols.append(ticker + "_label")
    feature_cols.append(ticker + "_price")

    #read in the data
    df = data.DataReader(ticker, 'yahoo', begdate, enddate)

    #get all price information
    df1 = df[["Open"]].rename(columns={"Open": "price"})
    df2 = df[["Close"]].rename(columns={"Close": "price"})

    #add time to the price values that correspond to close
    df1.index = df1.index + pd.DateOffset(hours=0)
    df2.index = df2.index + pd.DateOffset(hours=12)

    #row bind
    df = pd.concat([df1, df2], axis = 0)

    #sort by date index
    df = df.sort_index().rename(columns={"price": ticker + "_price"})

    #annotate data
    df[ticker + "_label"] = np.where(df[ticker + "_price"] < df[ticker + "_price"].shift(-1), "buy", "sell")

    #column bind to existing data
    if i == 0:
        output = df
    else:
        output = pd.concat([output, df], axis = 1)

#index labels
output = output.replace(to_replace={"buy": 1, "sell": 0})

#save the resulting data as a numpy arrays for mxnet
x = output.as_matrix(columns=feature_cols)
y = output.as_matrix(columns=label_cols)
np.save("../data/x.npy", x)
np.save("../data/y.npy",y)


print(output, x, y)

