import pandas as pd

if __name__ == '__main__':
    # application_train
    application_train = pd.read_csv('application_train.csv')
    # Drop columns that cannot be explained
    application_train = application_train.drop(columns=['FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_DOCUMENT_2',
                                                        'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                                                        'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
                                                        'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
                                                        'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                                                        'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                                                        'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                                                        'FLAG_DOCUMENT_21'])

    # bureau_balance
    bureau_balance = pd.read_csv('bureau_balance.csv')
    # Drop irrelevant columns
    bureau_balance = bureau_balance.drop(columns=['MONTHS_BALANCE'])
    # Data transformation
    bureau_balance['STATUS_0'] = (bureau_balance['STATUS'] == '0').astype(int)
    bureau_balance['STATUS_TOTAL'] = 1
    bureau_balance = bureau_balance.drop(columns=['STATUS'])
    bureau_balance = bureau_balance.groupby(['SK_ID_BUREAU']).sum()
    bureau_balance['STATUS_0'] = bureau_balance['STATUS_0'] / bureau_balance['STATUS_TOTAL']
    bureau_balance = bureau_balance.drop(columns=['STATUS_TOTAL'])

    # bureau
    bureau = pd.read_csv('bureau.csv')
    # Drop irrelevant columns
    bureau = bureau.drop(columns=['CREDIT_CURRENCY', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE',
                                  'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE',
                                  'AMT_ANNUITY'])

    # Merge bureau_balance to bureau
    bureau_1 = pd.merge(bureau, bureau_balance, how='left', on=['SK_ID_BUREAU'])

    # Drop columns with more than 90% NaN
    bureau_2 = bureau_1.dropna(thresh=int(0.1*len(bureau)), axis=1)

    # Drop bureau ID
    bureau_2 = bureau_2.drop(columns=['SK_ID_BUREAU'])

    # Fill NaN
    bureau_3 = bureau_2.copy()
    bureau_3['CREDIT_ACTIVE'] = bureau_3['CREDIT_ACTIVE'].fillna('Active')
    bureau_3['AMT_CREDIT_MAX_OVERDUE'] = bureau_3['AMT_CREDIT_MAX_OVERDUE'].fillna(0)
    bureau_3['CNT_CREDIT_PROLONG'] = bureau_3['CNT_CREDIT_PROLONG'].fillna(0)
    bureau_3['AMT_CREDIT_SUM'] = bureau_3['AMT_CREDIT_SUM'].fillna(0)
    bureau_3['AMT_CREDIT_SUM_DEBT'] = bureau_3['AMT_CREDIT_SUM_DEBT'].fillna(0)
    bureau_3['AMT_CREDIT_SUM_LIMIT'] = bureau_3['AMT_CREDIT_SUM_LIMIT'].fillna(0)
    bureau_3['AMT_CREDIT_SUM_OVERDUE'] = bureau_3['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

    # Data transformation
    bureau_4 = bureau_3.copy()
    bureau_4['TOTAL'] = 1
    bureau_4['CREDIT_ACTIVE'] = (bureau_4['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau_4['CREDIT_TYPE_CREDIT_CARD'] = (bureau_4['CREDIT_TYPE'] == 'Credit card').astype(int)
    bureau_4['CREDIT_TYPE_CONSUMER_CREDIT'] = (bureau_4['CREDIT_TYPE'] == 'Consumer credit').astype(int)
    bureau_4 = bureau_4.drop(columns=['CREDIT_TYPE'])

    # Sum
    bureau_sum = bureau_4[['SK_ID_CURR', 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                           'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE_CREDIT_CARD',
                           'CREDIT_TYPE_CONSUMER_CREDIT', 'TOTAL']]
    bureau_sum = bureau_sum.groupby(['SK_ID_CURR']).sum()
    bureau_sum['CREDIT_ACTIVE'] = bureau_sum['CREDIT_ACTIVE'] / bureau_sum['TOTAL']
    bureau_sum['CREDIT_TYPE_CREDIT_CARD'] = bureau_sum['CREDIT_TYPE_CREDIT_CARD'] / bureau_sum['TOTAL']
    bureau_sum['CREDIT_TYPE_CONSUMER_CREDIT'] = bureau_sum['CREDIT_TYPE_CONSUMER_CREDIT'] / bureau_sum['TOTAL']
    bureau_sum = bureau_sum.drop(columns=['TOTAL'])

    # Mean
    bureau_mean = bureau_4[['SK_ID_CURR', 'AMT_CREDIT_MAX_OVERDUE']]
    bureau_mean = bureau_mean.groupby(['SK_ID_CURR']).mean()

    # Max
    bureau_max = bureau_4[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']]
    bureau_max = bureau_max.groupby(['SK_ID_CURR']).max()

    # NaN mean
    # Fill NaN
    bureau_nanmean = bureau_4[['SK_ID_CURR', 'STATUS_0']]
    bureau_nanmean = bureau_nanmean.groupby(['SK_ID_CURR']).agg({'STATUS_0': lambda x: x.mean(skipna=True)})

    # Merge to application_train
    application_train = pd.merge(application_train, bureau_sum, how='left', on='SK_ID_CURR')
    application_train = pd.merge(application_train, bureau_mean, how='left', on='SK_ID_CURR')
    application_train = pd.merge(application_train, bureau_max, how='left', on='SK_ID_CURR')
    application_train = pd.merge(application_train, bureau_nanmean, how='left', on='SK_ID_CURR')
    # bureau done

    # installments_payments
    installments_payments = pd.read_csv('installments_payments.csv')
    installments_payments['AMT_PAYMENT_GREATER_EQUAL_INSTALMENT'] = (installments_payments['AMT_PAYMENT'] >=
                                                                     installments_payments['AMT_INSTALMENT']).astype(int)
    installments_payments_max = installments_payments[['SK_ID_CURR', 'AMT_PAYMENT_GREATER_EQUAL_INSTALMENT']]
    installments_payments_max = installments_payments_max.groupby(['SK_ID_CURR']).max()
    application_train = pd.merge(application_train, installments_payments_max, how='left', on='SK_ID_CURR')
    # print(installments_payments_max.isna().sum())

    # credit_card_balance
    credit_card_balance = pd.read_csv('credit_card_balance.csv')
    credit_card_balance_mean = credit_card_balance[['SK_ID_CURR', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT',
                                                    'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
                                                    'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE']]
    credit_card_balance_mean = credit_card_balance_mean.fillna(0)
    credit_card_balance_mean = credit_card_balance_mean.groupby(['SK_ID_CURR']).mean()
    application_train = pd.merge(application_train, credit_card_balance_mean, how='left', on='SK_ID_CURR')

    # POS_CASH_balance
    POS_CASH_balance = pd.read_csv('POS_CASH_balance.csv')
    POS_CASH_balance = POS_CASH_balance[['SK_ID_CURR', 'CNT_INSTALMENT_FUTURE', 'SK_DPD']]
    POS_CASH_balance = POS_CASH_balance.fillna(0)
    POS_CASH_balance['SK_DPD'] = (POS_CASH_balance['SK_DPD'] != 0).astype(int)
    POS_CASH_balance_max = POS_CASH_balance[['SK_ID_CURR', 'SK_DPD']]
    POS_CASH_balance_max = POS_CASH_balance_max.groupby(['SK_ID_CURR']).max()
    application_train = pd.merge(application_train, POS_CASH_balance_max, how='left', on='SK_ID_CURR')

    POS_CASH_balance_mean = POS_CASH_balance[['SK_ID_CURR', 'CNT_INSTALMENT_FUTURE']]
    POS_CASH_balance_mean = POS_CASH_balance_mean.groupby(['SK_ID_CURR']).mean()
    application_train = pd.merge(application_train, POS_CASH_balance_mean, how='left', on='SK_ID_CURR')

    # Drop columns with more than 90% NaN
    data = application_train.dropna(thresh=int(0.1 * len(application_train)), axis=1)

    # Output result
    # data.to_csv('data.csv', index=False)

    # Sampling
    training = 0.7
    training_df = data.sample(n=int(training*len(data)), replace=False, random_state=1)
    testing_df = data[~data['SK_ID_CURR'].isin(training_df['SK_ID_CURR'])]

    # training_df.to_csv('training.csv', index=False)
    # testing_df.to_csv('testing.csv', index=False)

    # Edit training set
    data = pd.read_csv('training_adjusted.csv')

    data_0 = data[data['TARGET'] == 0]
    data_1 = data[data['TARGET'] == 1]

    new_data_0 = data_0.sample(n=17375, replace=False, random_state=1)
    new_data = pd.concat([data_1, new_data_0])
    new_data = new_data.sort_values(by=['SK_ID_CURR'])
    new_data = new_data.drop(columns=['Unnamed: 0'])
    # new_data.to_csv('training_new.csv', index=False)


