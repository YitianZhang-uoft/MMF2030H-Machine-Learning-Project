import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance


pd.set_option('display.max_columns', None)

training =  pd.read_csv('training.csv')

testing = pd.read_csv('testing.csv')

files = [training, testing]

for i in range(len(files)):

	#data = pd.read_csv('data.csv')
	data = files[i]

	data = data[data.CODE_GENDER != 'XNA']

	data = data[data.NAME_FAMILY_STATUS != 'Unknown']

	for col in data:
		if data[col].nunique() <= 75:	
			data[col] = data[col].fillna(data[col].mode().iloc[0])
		else:
			data[col] = data[col].fillna(0)#data[col].median()


	data = pd.get_dummies(data, prefix=['NAME_CONTRACT_TYPE'], columns = ['NAME_CONTRACT_TYPE'])

	data = pd.get_dummies(data, prefix=['CODE_GENDER'], columns = ['CODE_GENDER'])

	data = pd.get_dummies(data, prefix=['FLAG_OWN_CAR'], columns = ['FLAG_OWN_CAR'])

	data = pd.get_dummies(data, prefix=['FLAG_OWN_REALTY'], columns = ['FLAG_OWN_REALTY'])

	data = pd.get_dummies(data, prefix=['NAME_TYPE_SUITE'], columns = ['NAME_TYPE_SUITE'])

	data = pd.get_dummies(data, prefix=['NAME_INCOME_TYPE'], columns = ['NAME_INCOME_TYPE'])

	data = pd.get_dummies(data, prefix=['NAME_EDUCATION_TYPE'], columns = ['NAME_EDUCATION_TYPE'])

	data = pd.get_dummies(data, prefix=['NAME_FAMILY_STATUS'], columns = ['NAME_FAMILY_STATUS'])

	data = pd.get_dummies(data, prefix=['NAME_HOUSING_TYPE'], columns = ['NAME_HOUSING_TYPE'])

	data = pd.get_dummies(data, prefix=['OCCUPATION_TYPE'], columns = ['OCCUPATION_TYPE'])

	data = pd.get_dummies(data, prefix=['WEEKDAY_APPR_PROCESS_START'], columns = ['WEEKDAY_APPR_PROCESS_START'])

	data = pd.get_dummies(data, prefix=['ORGANIZATION_TYPE'], columns = ['ORGANIZATION_TYPE'])

	data = pd.get_dummies(data, prefix=['FONDKAPREMONT_MODE'], columns = ['FONDKAPREMONT_MODE'])

	data = pd.get_dummies(data, prefix=['HOUSETYPE_MODE'], columns = ['HOUSETYPE_MODE'])

	data = pd.get_dummies(data, prefix=['WALLSMATERIAL_MODE'], columns = ['WALLSMATERIAL_MODE'])

	data = pd.get_dummies(data, prefix=['EMERGENCYSTATE_MODE'], columns = ['EMERGENCYSTATE_MODE'])




	data.to_csv('{}_adjusted.csv'.format(i))

	print('saved {} to csv'.format(i))


	#x = data.drop(columns=['SK_ID_CURR','TARGET'])

	#y = data['TARGET']

	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
