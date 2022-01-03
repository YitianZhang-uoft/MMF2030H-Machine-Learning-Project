import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, plot_roc_curve, f1_score, precision_score, recall_score, auc, roc_curve, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
import sys

#XNA and XAP mean not available
#Days worked at last job can not be positive especially positive 100 years (365243 is numerical NA value)
#Compare number of members in family to how many children, remove inconsistencies
#Look into unreasonable car ages
#Two columns have data about home phones

#CREDIT_TYPE_CREDIT_CARD, CREDIT_TYPE_CONSUMER_CREDIT, STATUS_0, AMT_PAYMENT_GREATER_EQUAL_INSTALMENT

#Make a correlation matrix for features and Information Value

data = pd.read_csv('training_new.csv')


#pd.set_option('display.max_columns', None)




#x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['SK_ID_CURR','TARGET']), data['TARGET'], test_size=0.30, random_state=0)


x_train = data.drop(columns=['SK_ID_CURR','TARGET'])

y_train = data['TARGET']

data_test = pd.read_csv('testing_adjusted.csv')

x_test = data_test.drop(columns=['Unnamed: 0','SK_ID_CURR','TARGET'])

y_test = data_test['TARGET']

correlation_matrix = x_train.join(y_train).corr()


corr_median =  np.median(np.absolute(correlation_matrix['TARGET']))

print('correlation matrix median',corr_median)


i = 0
for ind in correlation_matrix.index:
	if abs(correlation_matrix['TARGET'][ind]) < 10 **(-2):		
		#print(i)
		#print(ind, correlation_matrix['TARGET'][ind])
		
		#x_train = x_train.drop(columns=[ind])
		try:
			#x_test = x_test.drop(columns=[ind])
			x_train = x_train.drop(columns=[x_train.columns[i]])
			x_test = x_test.drop(columns=[x_test.columns[i]])
		except:
			print('Unable to drop {} from the test set'.format(ind))
	i += 1

#pd.set_option('display.max_columns', None)
#print(x_train)

#data.to_csv('data_adjusted.csv')

#sns.heatmap(correlation_matrix, cmap=plt.cm.Reds, xticklabels = 1, yticklabels = 1, annot=True)
#plt.show()

def get_models_trees():
	models = dict()
	trees = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
	for n in trees:
		models[str(n)] = XGBClassifier(n_estimators=n, eval_metric = 'logloss',use_label_encoder=False)
	return models


def get_models_depth():
	models = dict()
	for i in range(1,11):
		models[str(i)] = XGBClassifier(max_depth=i, n_estimators=40, eval_metric = 'logloss',use_label_encoder=False)
	return models

def get_models_rate():
	models = dict()
	rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
	for r in rates:
		key = '%.4f' % r
		models[key] = XGBClassifier(eta=r, max_depth=3, n_estimators=40, eval_metric = 'logloss',use_label_encoder=False)
	return models


def get_models_samples():
	models = dict()
	for i in np.arange(0.1, 1.1, 0.1):
		key = '%.1f' % i
		models[key] = XGBClassifier(subsample=i, eta=0.1, max_depth=3, n_estimators=40, eval_metric = 'logloss',use_label_encoder=False)
	return models

def get_models_features():
	models = dict()
	for i in np.arange(0.1, 1.1, 0.1):
		key = '%.1f' % i
		models[key] = XGBClassifier(colsample_bytree=i, subsample=0.9, eta=0.1, max_depth=3, n_estimators=40, eval_metric = 'logloss',use_label_encoder=False)
	return models

def get_models_alpha():
	models = dict()
	for i in np.arange(0.1, 1.1, 0.1):
		key = '%.1f' % i
		models[key] = XGBClassifier(reg_alpha=i, colsample_bytree=0.3, subsample=0.9, eta=0.1, max_depth=3, n_estimators=40, eval_metric = 'logloss',use_label_encoder=False)
	return models

def get_models_lambda():
	models = dict()
	for i in np.arange(0.5, 2.5, 0.25):
		key = '%.2f' % i
		models[key] = XGBClassifier(reg_lambda =i, reg_alpha=0.3, colsample_bytree=0.3, subsample=0.9, eta=0.1, max_depth=3, n_estimators=40, eval_metric = 'logloss',use_label_encoder=False)
	return models


def get_models():
	models = XGBClassifier(reg_lambda =1, reg_alpha=0.3, colsample_bytree=0.3, subsample=0.9, eta=0.1, max_depth=3, n_estimators=180, eval_metric = 'logloss',use_label_encoder=False)
	#models = XGBClassifier(eval_metric = 'logloss',use_label_encoder=False)
	return models

# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 
'''
# get the models to evaluate
models = get_models_trees()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.8f (%.8f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()
'''

model_1 = get_models()

print('Cross-Validation Accuracy %.8f' % (np.mean(evaluate_model(model_1))))

model_1.fit(x_train, y_train)

y_pred_train = model_1.predict(x_train)
y_pred_test = model_1.predict(x_test)

print('Preiction AUC Train: %.8f' % roc_auc_score(y_train, y_pred_train))
print('Preiction AUC Test: %.8f' % roc_auc_score(y_test, y_pred_test))


plot_confusion_matrix(model_1, x_test, y_test, normalize = 'all')
plt.show()

print('Confusion Matrix', confusion_matrix(y_test, y_pred_test))

print('Accuracy', accuracy_score(y_test, y_pred_test))

print('Recall', recall_score(y_test, y_pred_test))

print('Precision', precision_score(y_test, y_pred_test))

print('F-Score', f1_score(y_test, y_pred_test))

print('Total Misclassification rate', np.mean(y_pred_test != y_test))

plot_roc_curve(model_1, x_test, y_test)
plt.show()

plot_roc_curve(model_1, x_train, y_train)
plt.show()


x = pd.concat([x_train, x_test])

y = pd.concat([y_train, y_test])

model_2 = get_models()

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 1)

tprs = []
aucs = []

i=1

for train,test in k_fold.split(x,y):
    prob = model_2.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])[:,1]
    fpr, tpr, t = roc_curve(y.iloc[test], prob)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC AUC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1

print('Average AUC', np.mean(aucs))
    
plt.legend(loc="lower right", fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('ROC Curve and AUC for Tuned Gradient Boosted Trees Model', fontsize = 24)
plt.show()

plot_importance(model_2)
plt.show()

'''
#model = XGBClassifier(learning_rate = 0.3, n_estimators=40, min_samples_split = len(x_train.index)/100, min_samples_leaf = 50, max_depth = 8, max_features = 'sqrt', subsample = 0.8, eval_metric = 'auc',use_label_encoder=False, seed=0)


#model = XGBClassifier(subsample = 0.9, reg_lambda = 1, reg_alpha = 0.1, n_estimators = 60, min_child_weight = 20, max_depth = 6, learning_rate = 0.1, gamma = 0.5, eval_metric = 'logloss',use_label_encoder=False, seed=0)



model = XGBClassifier(eval_metric = 'logloss',use_label_encoder=False, seed=0)


parameters = {      
              'learning_rate': [0.1,0.2,0.3,0.4,0.5],
              'n_estimators': [30,40,50,60,70],
              'max_depth': [6,7,8,9,10],
              'min_samples_leaf' : [25,50,75,100]
              }


search = RandomizedSearchCV(model,parameters, cv=5,n_iter=100,scoring='roc_auc',verbose=2,n_jobs=-1)
search.fit(x_train, y_train,verbose = 2)


print('Best Parameters : ', search.best_params_)

y_pred_train = search.predict(x_train)
y_pred_test = search.predict(x_test)



print('Preiction ROC Train: %.8f' % roc_auc_score(y_train, y_pred_train))
print('Preiction ROC Test: %.8f' % roc_auc_score(y_test, y_pred_test))



print(y_pred_test)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

print('>%s %.8f (%.8f)' % ('Training Accuracy', np.mean(scores), np.std(scores)))

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('Preiction Accuracy: %.8f' % accuracy_score(y_test, y_pred))
print('ROC AUC Score: %.8f' % roc_auc_score(y_test, y_pred))

#np.set_printoptions(threshold=sys.maxsize)

print(y_pred)

plot_importance(model)
plt.show()
'''

#Validation Default parameters, removed low correlations 0.91907662 (0.00035002)
#Prediction Default parameters, removed low correlations 0.92499059



#Best Parameters :  {'subsample': 0.9, 'reg_lambda': 1, 'reg_alpha': 0.1, 'n_estimators': 60, 'min_child_weight': 2.0, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0.5}

#Best Parameters :  {'subsample': 0.7, 'reg_lambda': 1, 'reg_alpha': 1, 'n_estimators': 60, 'min_child_weight': 2.0, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 1}

#Best Parameters : {'n_estimators': 70, 'min_samples_leaf': 75, 'max_depth': 6, 'learning_rate': 0.2}