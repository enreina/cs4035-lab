import seaborn as sns
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score, auc, precision_recall_curve, accuracy_score, f1_score, confusion_matrix, average_precision_score

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

def run_cross_validation(clf, data, labels, threshold=0.5, undersampling=True, ratio=0.3, num_fold=10, verbose=False):
    all_precision = []
    all_recall = []
    all_f1 = []
    all_auc = []
    all_average_precision = []
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    
    skf = StratifiedKFold(n_splits=num_fold, random_state=13)

    for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
       

        # apply undersampling
        rus = RandomUnderSampler(ratio={0:int(((1.0 - ratio) / ratio) * len([x for x in y_train if x == 1]))}) 
        #smote = SMOTE(ratio={1: int((3/7.0) * len([x for x in y_train if x == 0]))})
        if undersampling:
            X_train_res, y_train_res = rus.fit_sample(X_train, y_train)
            clf.fit(X_train_res, y_train_res)
        else:
            clf.fit(X_train, y_train)
        
        pred = clf.predict(X_test)
        pred_proba = clf.predict_proba(X_test)[:,1]
        pred = [1 if y >= threshold else 0 for y in pred_proba]

        fpr, tpr, thresholds = roc_curve(y_test, pred_proba, pos_label=1)
        auc_score = auc(fpr, tpr)
        average_precision = average_precision_score(y_test, pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        if verbose:
            print "Processing fold ", fold," out of ", num_fold,"..."
            print "Precision: ", precision_score(y_test, pred)
            print "Recall: ", recall_score(y_test, pred)
            print "TP: ", tp
            print "FP: ", fp
            print "TN: ", tn
            print "FN: ", fn
            print "F1 Score: ", f1_score(y_test, pred)
            print "AUC: ", auc_score
            print "AUPRC: ", average_precision

        all_precision.append(precision_score(y_test, pred))
        all_recall.append(recall_score(y_test, pred))
        all_f1.append(f1_score(y_test, pred))
        all_auc.append(auc_score)
        all_average_precision.append(average_precision)
        total_tp = total_tp + tp
        total_fp = total_fp + fp
        total_tn = total_tn + tn
        total_fn = total_fn + fn
       
    print "----SUMMARY-----"
    print "TP: ", total_tp
    print "FP: ", total_fp
    print "TN: ", total_tn
    print "FN: ", total_fn
    print "Precision: ", np.mean(all_precision)
    print "Recall: ", np.mean(all_recall)
    print "F1: ", np.mean(all_f1)
    print "AUC: ", np.mean(all_auc)
    print "AUPRC: ", np.mean(all_average_precision)

def derive_transaction_average_amount(data):
    data.sort_values(by=["card_id","creationdate"], inplace=True)

    prev_card_id = None
    prev_amounts_euro = []
    prev_amounts = []

    ii = 0
    for index, row in data.iterrows():
        # print ii, " out of ", len(data)
        # if current card id equals to previous card, accumulate the amount and calculate the mean
        if row['card_id'] == prev_card_id:
            # update mean
            data.at[index, 'prev_amount_mean'] = np.mean(prev_amounts)
            data.at[index, 'prev_amount_euro_mean'] = np.mean(prev_amounts_euro)
            
            if row['is_fraud'] == 0:
                prev_amounts.append(row['amount'])
                prev_amounts_euro.append(row['amount_euro'])
        # if the card_id is a new one, reset amount list
        else:
            prev_amounts = []
            prev_amounts_euro = []
            
            data.at[index, 'prev_amount_mean'] = 0
            data.at[index, 'prev_amount_euro_mean'] = 0
        
        
        prev_card_id = row['card_id']
        ii = ii + 1

    data.fillna({"prev_amount_mean": 0, "prev_amount_euro_mean": 0}, axis=0, inplace=True)
    data.to_csv(open("fraud_data_derived.csv", "w"))

def derive_transaction_count(data, based_on=["card_id"], column_name="prev_transaction_count"):
    data.sort_values(by=based_on + ["creationdate"], inplace=True)
    
    count = 0
    prev_values = {}
    for attr in based_on:
        prev_values[attr] = None
    ii = 0
    for index, row in data.iterrows():
        # if current attribute group by equals to previous transaction, reset the count
        should_reset = False
        for key in prev_values.keys():
            should_reset = should_reset or row[key] != prev_values[key]
            
        if should_reset:
            count = 0

        # update transaction derived attribute
        data.at[index, column_name] = count
        # update counter
        if row['is_fraud'] == 0:
            count = count + 1

        for key in prev_values.keys():
            prev_values[key] = row[key]
            
        ii = ii + 1
    
    data.fillna({column_name: 0}, axis = 0, inplace=True)
    data[column_name] = data[column_name].astype(int)
    data.to_csv(open("fraud_data_derived.csv", "w"))