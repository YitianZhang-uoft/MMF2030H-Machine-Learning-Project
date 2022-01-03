import pandas as pd
import numpy as np
import math


if __name__ == '__main__':
    iv_tables = []
    for i in range(1, 118):
        filename = '/Users/zackwang/Desktop/iv_tables/DF' + str(i) + '.feather'
        DF = pd.read_feather(filename)
        iv_tables.append(DF)

    data = pd.read_csv('/Users/zackwang/Desktop/Logistic Regression/data.csv')
    woe_data = data[['SK_ID_CURR', 'TARGET']]

    for table in iv_tables:
        column_name = table.columns[0]
        woe_column_name = 'WOE_' + column_name

        feature_data = data[column_name].tolist()
        feature_woe = [0] * len(feature_data)

        if table[column_name][1][0] == '[':
            for i in range(len(table[column_name])):
                if table[column_name][i] == 'NA':
                    for j in range(len(feature_data)):
                        if math.isnan(feature_data[j]):
                            feature_woe[j] = table['WOE'][i]
                else:
                    data_range = table[column_name][i]
                    index_1 = data_range.find(',')
                    index_2 = data_range.find(']')
                    lower_b = float(data_range[1:index_1])
                    upper_b = float(data_range[index_1+1:index_2])
                    for j in range(len(feature_data)):
                        if lower_b <= feature_data[j] <= upper_b:
                            feature_woe[j] = table['WOE'][i]
        else:
            for i in range(len(table[column_name])):
                if table[column_name][i] == '':
                    for j in range(len(feature_data)):
                        if feature_data[j] != feature_data[j]:
                            feature_woe[j] = table['WOE'][i]
                else:
                    for j in range(len(feature_data)):
                        if feature_data[j] == table[column_name][i]:
                            feature_woe[j] = table['WOE'][i]

        woe_data[woe_column_name] = feature_woe

    woe_data.to_csv('data_woe.csv', index=False)

