# For more interactive code, see fraud_detection.ipynb (a Jupyter notebook) 

from __future__ import division
import pandas as pd
import numpy as np
from currency_converter import CurrencyConverter
from utilities import fraud_heatmap, run_cross_validation, derive_transaction_average_amount, derive_transaction_count
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score, auc, precision_recall_curve, accuracy_score, f1_score, confusion_matrix, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
import pickle
from sklearn import tree
import graphviz

# Preprocessing
print "Preprocessing data..."
data = pd.read_csv('data/data_for_student_case.csv')

## Discard 'refused' transactions
data_preproc = data.loc[data['simple_journal'] != 'Refused'].copy()
data_preproc.head()

## Transform date and extract date-related features
data_preproc['creationdate'] = pd.to_datetime(data_preproc['creationdate'])
## save month, date, weekday, hour
data_preproc['creation_month'] = data_preproc.creationdate.dt.month
data_preproc['creation_month'] = data_preproc['creation_month'].astype('category')
data_preproc['creation_weekday'] = data_preproc.creationdate.dt.weekday
data_preproc['creation_weekday'] = data_preproc['creation_weekday'].astype('category')
data_preproc['creation_day'] = data_preproc.creationdate.dt.day
data_preproc['creation_day'] = data_preproc['creation_day'].astype('category')
data_preproc['creation_hour'] = data_preproc.creationdate.dt.hour
data_preproc['creation_hour'] = data_preproc['creation_hour'].astype('category')

## map cvcresponsecode with value 3-6 to 3
data_preproc['cvcresponsecode'] = map(lambda x: 3 if int(x) >= 3 else int(x), data_preproc['cvcresponsecode'])

## map to either categorical or object (for identifier) datatype
data_preproc['bin'] = data_preproc['bin'].astype(int)
data_preproc['bin'] = data_preproc['bin'].astype(str)
data_preproc['txid'] = data_preproc['txid'].astype(str) # txid is an identifier
for category_column in ["issuercountrycode","txvariantcode", "bin", "currencycode","shopperinteraction","simple_journal","cardverificationcodesupplied","cvcresponsecode","accountcode", "shoppercountrycode"]:  
    data_preproc[category_column] = data_preproc[category_column].astype("category")

## convert amount to euro
currency_converter = CurrencyConverter()
data_preproc['amount_euro'] = map(lambda x,y: currency_converter.convert(x,y, 'EUR'), data_preproc['amount'],data_preproc['currencycode'])

## create column for storing label 'is_fraud' (1 = fraud, 0 = legitimate)
## mapped from the simple_journal value (chargeback -> 1, settled -> 0)
data_preproc['is_fraud'] = data_preproc.apply(lambda x: 1 if x['simple_journal'] == "Chargeback" else 0, axis=1)
data_preproc = data_preproc.drop(['simple_journal'],axis=1)

## drop booking date as we may not use it for training
data_preproc = data_preproc.drop(['bookingdate'],axis=1)

print data_preproc.head()

## we remove attributes which are identifier (txid, card_id, mail_id, ip_id)
print "Encoding data using one-hot encoding..."
data_encoded = data_preproc.copy()
for x in data_preproc.columns:
    if data_preproc[x].dtypes == np.dtype('O'):
        data_encoded = data_encoded.drop([x], axis=1)

## removing creationdate as we already extracted hour, weekday, month        
data_encoded = data_encoded.drop(['creationdate'], axis=1)

## separating labels from features
labels = data_encoded['is_fraud']
data_encoded = data_encoded.drop(['is_fraud'], axis=1)

## dropping bin to speed up imbalance task
data_encoded_without_bin = data_encoded.drop(['bin'], axis=1)

## apply one hot encoding
data_encoded = pd.get_dummies(data_encoded, dummy_na=True)
data_encoded_without_bin = pd.get_dummies(data_encoded_without_bin, dummy_na=True)
## store the feature_names for later to be used in decision tree
feature_names = data_encoded.columns
## convert to numpy array
labels = np.array(labels)
data_encoded = np.array(data_encoded)


# Visualization
print "1. Visualization Task"
## Distribution Histogram
fraud = data_preproc.loc[data_preproc['is_fraud'] == 1]
non_fraud = data_preproc.loc[data_preproc['is_fraud'] == 0]
plt.figure("Amount Distribution (Fraud")
fraud_plot = sns.distplot(fraud['amount_euro'], label="fraud")
fraud_plot.set_xscale('log')
non_fraud_plot = sns.distplot(non_fraud['amount_euro'], label="legitimate")
non_fraud_plot.set_xscale('log')
plt.legend(fontsize="xx-large")
plt.show()

## Fraud transaction trend
temp_fraud = fraud.set_index('creationdate').resample('1D').count()['txid']
x_fraud = temp_fraud.index
y_fraud = temp_fraud.values
plt.figure()
plt.plot(x_fraud, y_fraud, 'g', label="fraud");
plt.legend()
plt.show()

## Legitimate transaction trend
temp_non_fraud = non_fraud.set_index('creationdate').resample('1D').count()['txid']
x_non_fraud = temp_non_fraud.index
y_non_fraud = temp_non_fraud.values
plt.figure()
plt.plot(x_non_fraud, y_non_fraud,  'b', label="non-fraud");
plt.legend()
plt.show()

## Heatmap
### select categorical columns
categorical_columns = ['txvariantcode', 'accountcode', 'issuercountrycode', 'currencycode']
for combination in list(itertools.combinations(categorical_columns,2)):
    fraud_heatmap(data_preproc, combination[0], combination[1], frac=True)
    plt.show()

# Imbalance Task
print "2. Imbalance Task1"
## create train test split
print "Splitting train and test dataset"
X_train, X_test, y_train, y_test = train_test_split(data_encoded_without_bin, labels, test_size=0.40, random_state=42)
## apply SMOTE to the training set
print "Applying SMOTE"
intended_ratio = 0.3
original_num_non_fraud = len([x for x in y_train if x == 0])
sm = SMOTE(random_state=12, ratio={1:int(intended_ratio * original_num_non_fraud)})
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

## create classifiers
knn = KNeighborsClassifier(n_neighbors=3)
lr = LogisticRegression(C=500, penalty='l1', random_state=12)
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=10, random_state=12)

## create dictionary to store values
clf_dict = {"knn": {"clf": knn}, "lr": {"clf": lr}, "rf": {"clf": rf}}

print "Benchmark between classifiers.."
for k,v in clf_dict.iteritems():
    ## train using original dataset
    print k.upper()
    print "Without SMOTE"
    clf = clone(v['clf'])
    clf.fit(X_train, y_train)
    pred_proba = clf.predict_proba(X_test)[:,1]
    pred = clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    print "Precision: ", precision_score(y_test, pred)
    print "Recall: ", recall_score(y_test, pred)
    print "F1 Score: ", f1_score(y_test, pred)
    print "AUC: ", auc_score
    precision, recall, _ = precision_recall_curve(y_test, pred_proba)
    v['non-smote'] = {}
    v['non-smote']['fpr'] = fpr
    v['non-smote']['tpr'] = tpr
    v['non-smote']['thresholds'] = thresholds
    v['non-smote']['auc'] = auc_score
    v['non-smote']['recall'] = recall
    v['non-smote']['precision'] = precision
    ## train using SMOTE
    print "With SMOTE"
    clf = clone(v['clf'])
    clf.fit(X_train_res, y_train_res)
    pred_proba = clf.predict_proba(X_test)[:,1]
    pred = clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    print "Precision: ", precision_score(y_test, pred)
    print "Recall: ", recall_score(y_test, pred)
    print "F1 Score: ", f1_score(y_test, pred)
    print "AUC: ", auc_score
    precision, recall, _ = precision_recall_curve(y_test, pred_proba)
    v['smote'] = {}
    v['smote']['fpr'] = fpr
    v['smote']['tpr'] = tpr
    v['smote']['thresholds'] = thresholds
    v['smote']['auc'] = auc_score
    v['smote']['recall'] = recall
    v['smote']['precision'] = precision
    print ""

## save the result to a pickle file    
# if it's already there uncoment the following two lines
# with open('imbalance_task_result.pickle', 'rb') as handle:
#    clf_dict = pickle.load(handle)
with open('imbalance_task_result.pickle', 'wb') as handle:
    pickle.dump(clf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot the ROCs
print "Plotting the ROCs.."
for k in clf_dict.keys():
    plt.title('Receiver Operating Characteristic: ' + k.upper())

    plt.plot(clf_dict[k]['smote']['fpr'], clf_dict[k]['smote']['tpr'], 'b',label='SMOTE: AUC = %0.2f'% clf_dict[k]['smote']['auc'])
    plt.plot(clf_dict[k]['non-smote']['fpr'], clf_dict[k]['non-smote']['tpr'], 'g',label='Original: AUC = %0.2f'% clf_dict[k]['non-smote']['auc'])

    plt.legend(loc='lower right', fontsize="xx-large")
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0,1.0])
    plt.ylim([0,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Classification Task
print "3. Classifier Task"

print "-- Black Box: Random Forest-- "
## Black Box: Random Forest
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=10, random_state=12)
run_cross_validation(clf, data_encoded, labels, threshold=0.6, verbose=True)

print "-- White Box: Decision Tree-- "
## White Box: Decision Tree
clf = tree.DecisionTreeClassifier(max_depth=10, random_state=12)
run_cross_validation(clf, data_encoded, labels, threshold=0.6, verbose=True)

## save the tree
print "Exporting decision tree figure"
modelDT = tree.DecisionTreeClassifier(max_depth=5, random_state=12)
modelDT.fit(data_encoded, labels)

dot_data = tree.export_graphviz(modelDT, out_file=None, 
                         feature_names=feature_names,  
                         class_names=['Legitimate', 'Fraud'],  
                         filled=True, rounded=True,  
                         special_characters=True, max_depth=5)  
graph = graphviz.Source(dot_data)  
graph.render("figures/decision_tree_fraud")


# Bonus Task
## store data into new variable
print "4. Bonus Task"
data_derived = data_preproc.copy()
print "Deriving attributes..."
## derive the attributes
derive_transaction_average_amount(data_derived) # deriving prev_transaction_
derive_transaction_count(data_derived, based_on=["card_id"], column_name="prev_transaction_count")
derive_transaction_count(data_derived, based_on=["card_id", "shoppercountrycode"], column_name="prev_transaction_count_shoppercountrycode")
derive_transaction_count(data_derived, based_on=["card_id", "currencycode"], column_name="prev_transaction_count_currencycode")
derive_transaction_count(data_derived, based_on=["card_id", "mail_id"], column_name="prev_transaction_count_mail_id")
## encode the data with derived attributes
print "Encoding data with derived attributes..." 
data_derived_encoded = data_derived.copy()
for x in data_derived_encoded.columns:
    if data_derived_encoded[x].dtypes == np.dtype('O'):
        data_derived_encoded = data_derived_encoded.drop([x], axis=1)
       
data_derived_encoded = data_derived_encoded.drop(['creationdate'], axis=1)

labels = data_derived_encoded['is_fraud']
data_derived_encoded = data_derived_encoded.drop(['is_fraud'], axis=1)

data_derived_encoded = pd.get_dummies(data_derived_encoded, dummy_na=True)
data_derived_encoded.head()

feature_names = data_derived_encoded.columns

labels = np.array(labels)
data_derived_encoded = np.array(data_derived_encoded)

print "Running cross-validation on encoded derived atributes..."
## run cross validation on RandomForest
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=10, random_state=12)
run_cross_validation(clf, data_derived_encoded, labels, threshold=0.6, verbose=True)

print "FINISHED!"