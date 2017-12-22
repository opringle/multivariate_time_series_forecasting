#read in the txt file
import pandas as pd
import numpy as np

#read in the data
df = pd.read_csv("../data/household_power_consumption.txt", sep=";")

print(df.shape)

df = df.iloc[:6000, :]

def get_ts_label(array):
    """outputs an array of labels"""

    #shift input array backwards one value backwards
    shifted_array = np.roll(array, -1)

    #remove last value from input and shifted array since we do not know the next value
    shifted_array = shifted_array[:-1]
    array = array[:-1]

    #boolean array denoting if next value was higher than last in the input array
    label_array = array < shifted_array

    return label_array


def get_label_df(feature_df):

    #fill NA's with previous value
    feature_df = feature_df.fillna(method='ffill')

    cols = list(feature_df)

    for i, col in enumerate(cols):

        label_array = get_ts_label(feature_df[col].values)
        label_array = np.array([int(x) for x in label_array.tolist()])

        if i == 0:
            label_matrix = label_array
        else:
            label_matrix = np.column_stack((label_matrix, label_array))

    label_cols = [col + "_label" for col in cols]

    label_df = pd.DataFrame(columns = label_cols, data = label_matrix)

    return label_df

#extract feature values
feature_df = df.iloc[:, 3:].astype(float)

#extract label values
label_df = get_label_df(feature_df)

#exclude last feature values since we dont have label here
feature_df = feature_df.iloc[:-1, :]

#convert to numpy matrix
x = feature_df.as_matrix()
y = label_df.as_matrix()

#save files
np.save("../data/x_electric.npy", x)
np.save("../data/y_electric.npy", y)

print(x)
print(y)
