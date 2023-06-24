import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(random_state=None):
    exp_data = pd.read_pickle('./experimental data.pickle')

    data_train, data_test = train_test_split(exp_data, test_size=0.2, shuffle=True, random_state=random_state)
    data_val, data_test = train_test_split(data_test, test_size=0.5, shuffle=True, random_state=random_state)

    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    return data_train, data_val, data_test
