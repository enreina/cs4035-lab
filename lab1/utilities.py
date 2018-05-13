import seaborn as sns
import numpy as np
import pandas as pd

def fraud_heatmap(data, column_name_a, column_name_b, frac=True):
    fraud = data.loc[data['is_fraud'] == 1]
    non_fraud = data.loc[data['is_fraud'] == 0]
    # column_name_a = 'currencycode'
    # column_name_b = 'accountcode'

    x_axis = list(data[column_name_a][data.is_fraud == 1].unique())
    y_axis = list(data[column_name_b][data.is_fraud == 1].unique())
    values = []

    length_x = len(x_axis)
    length_y = len(y_axis)
    x_axis = np.repeat(x_axis, length_y)
    y_axis = y_axis*length_x


    for combination in zip(x_axis, y_axis):
        # number of fraud
        num_fraud = len(fraud.loc[(fraud[column_name_a] == combination[0]) & (fraud[column_name_b] == combination[1])])
        num_total = len(data.loc[(data[column_name_a] == combination[0]) & (data[column_name_b] == combination[1])])
        #num_total = len(data)
        
        if frac:
            if num_total == 0:
                values.append(0)
            else:
                values.append(num_fraud / float(num_total))
        else: 
            values.append(num_fraud)

    df = pd.DataFrame({column_name_a: x_axis, column_name_b: y_axis, 'value': values })

    # plot it
    df_wide=df.pivot_table( index=column_name_a, columns=column_name_b, values='value' )
    sns.set(rc={"figure.figsize": (10, 10)})
    sns.heatmap( df_wide, cmap="Blues", annot=True)