# For more interactive code, see fraud_detection.ipynb (a Jupyter notebook) 

from __future__ import division
import pandas as pd
from currency_converter import CurrencyConverter
from utilities import fraud_heatmap
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

# Preprocessing
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

print data_preproc

## we remove attributes which are identifier (txid, card_id, mail_id, ip_id)
data_encoded = data_preproc.copy()
for x in data_preproc.columns:
    if data_preproc[x].dtypes == np.dtype('O'):
        data_encoded = temp.drop([x], axis=1)

## removing creationdate as we already extracted hour, weekday, month        
data_encoded = data_encoded.drop(['creationdate'], axis=1)
## separating labels from features
labels = data_encoded['is_fraud']
data_encoded = data_encoded.drop(['is_fraud'], axis=1)
## apply one hot encoding
data_encoded = pd.get_dummies(data_encoded, dummy_na=True)
## store the feature_names for later to be used in decision tree
feature_names = data_encoded.columns
## convert to numpy array
labels = np.array(labels)
data_encoded = np.array(data_encoded)


# Visualization
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
# select categorical columns
categorical_columns = ['txvariantcode', 'accountcode', 'issuercountrycode', 'currencycode']
for combination in list(itertools.combinations(categorical_columns,2)):
    fraud_heatmap(data_preproc, combination[0], combination[1], frac=True)
    plt.show()

# Imbalance Task


# Classification Task

## Black Box: Random Forest

## White Box: Decision Tree


# Bonus Task
