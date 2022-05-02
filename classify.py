import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

test = pd.read_csv("test.csv.zip", parse_dates=['Dates'], index_col='Id')
train = pd.read_csv("train.csv.zip", parse_dates=['Dates'])

test.head()
train.head()
train.info()
test.info()


# Function for printing null_values and related info
def description(data):
    no_rows = data.shape[0]
    types = data.dtypes
    col_null = data.columns[data.isna().any()].to_list()
    counts = data.apply(lambda x: x.count())
    uniques = data.apply(lambda x: x.unique())
    nulls = data.apply(lambda x: x.isnull().sum())
    distincts = data.apply(lambda x: x.unique().shape[0])
    nan_percent = (data.isnull().sum() / no_rows) * 100
    cols = {'dtypes': types, 'counts': counts, 'distincts': distincts, 'nulls': nulls,
            'missing_percent': nan_percent, 'uniques': uniques}
    table = pd.DataFrame(data=cols)
    return table


# Checking Null Values In Train
details_tr = description(train)
details_tr.reset_index(level=[0], inplace=True)
details_tr.sort_values(by='missing_percent', ascending=False)

# Checking Null Values In Test
details_test = description(test)
details_test.reset_index(level=[0], inplace=True)
details_test.sort_values(by='missing_percent', ascending=False)

train.duplicated().sum()

# Checking the outliers
figure, axs = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=train[["X"]], ax=axs[0])
sns.boxplot(data=train[["Y"]], ax=axs[1])

train.drop_duplicates(inplace=True)
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='mean')

for district in train['PdDistrict'].unique():
    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train.loc[train['PdDistrict'] == district, ['X', 'Y']])
    test.loc[test['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
        test.loc[test['PdDistrict'] == district, ['X', 'Y']])

#figure, axs = plt.subplots(1, 2, figsize=(15, 5))
#sns.boxplot(data=train[["X"]], ax=axs[0])
#sns.boxplot(data=train[["Y"]], ax=axs[1])

#train = train[train["Y"] < 80]
#sns.displot(train[["X"]], kde=True)
#plt.show()

data = train.groupby('Category').count()
data = data['Dates'].sort_values(ascending=False)

plt.figure(figsize=(20, 12))
ax = sns.barplot(data.values, data.index, palette=cm.ScalarMappable(cmap='magma').to_rgba(data.values))

plt.title('Count by Category', fontdict={'fontsize': 24})
plt.xlabel('Count')
plt.grid()

train['DayOfWeek'] = train['Dates'].dt.weekday
train['Month'] = train['Dates'].dt.month
train['Year'] = train['Dates'].dt.year
train['Hour'] = train['Dates'].dt.hour

year = train.groupby('Year').count().iloc[:, 0]
month = train.groupby('Month').count().iloc[:, 0]
hour = train.groupby('Hour').count().iloc[:, 0]
dayofweek = train.groupby('DayOfWeek').count().iloc[:, 0]

figure, axs = plt.subplots(2, 2, figsize=(15, 10))

sns.barplot(x=year.index, y=year, ax=axs[0][0], palette=cm.ScalarMappable(cmap='Reds').to_rgba(data.values))
sns.barplot(x=month.index, y=month, ax=axs[0][1], palette=cm.ScalarMappable(cmap='viridis').to_rgba(data.values))
sns.barplot(x=hour.index, y=hour, ax=axs[1][0], palette=cm.ScalarMappable(cmap='Blues').to_rgba(data.values))
sns.barplot(x=dayofweek.index, y=dayofweek, ax=axs[1][1], palette=cm.ScalarMappable(cmap='cool').to_rgba(data.values))
plt.show()

#figure, axs = plt.subplots(figsize=(10, 5))
#sns.countplot(x=train["PdDistrict"])
#plt.show()

df_cr = pd.DataFrame(train['Category'].value_counts())
df_cr.tail()
plt.figure(figsize=(16, 10))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax1.set_title('Top 10', size=16)
sns.barplot(x=df_cr.head(10).index, y='Category', data=df_cr.head(10))
ax1.set_xticklabels(ax1.xaxis.get_ticklabels(), rotation=90)
ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.set_title('Bottom 10', size=16)
sns.barplot(x=df_cr.tail(10).index, y='Category', data=df_cr.tail(10))
ax2.set_xticklabels(ax2.xaxis.get_ticklabels(), rotation=90)
plt.show()

#top10cc = pd.Series(df_cr.head(10).index)
#top10 = train[train['Category'].isin(top10cc)]
#tmp = pd.DataFrame(top10.groupby(['PdDistrict', 'Category']).size(), columns=['count'])
#tmp.reset_index(inplace=True)
#tmp = tmp.pivot(index='PdDistrict', columns='Category', values='count')
#fig, axes = plt.subplots(1, 1, figsize=(15, 15))
#tmp.plot(ax=axes, kind='bar', stacked=True)


def feature_engineering(data):
    data['Date'] = pd.to_datetime(data['Dates'].dt.date)
    data['n_days'] = (data['Date'] - data['Date'].min()).apply(lambda x: x.days)
    data['Day'] = data['Dates'].dt.day
    data['DayOfWeek'] = data['Dates'].dt.weekday
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Block'] = data['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)
    data["X-Y"] = data["X"] - data["Y"]
    data["XY"] = data["X"] + data["Y"]
    data.drop(columns=['Dates', 'Date', 'Address'], inplace=True)
    return data


train = feature_engineering(train)
test = feature_engineering(test)
train.drop(columns=['Descript', 'Resolution'], inplace=True)

le1 = LabelEncoder()
train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])
test['PdDistrict'] = le1.transform(test['PdDistrict'])

le2 = LabelEncoder()
X = train.drop(columns=['Category'])
y = le2.fit_transform(train['Category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Fitting data in Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Predicting the results
dtree_prec = dtree.predict(X_test)

print(classification_report(y_test, dtree_prec))
print("Train Accuracy: ", accuracy_score(y_train, dtree.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, dtree_prec))

# Fitting In RandomForest Ensemble
rfc = RandomForestClassifier(n_estimators=40, min_samples_split=100)
rfc.fit(X_train, y_train)

# Predicting The Final Results
rfc_pred = rfc.predict(X_test)

print("Train Accuracy: ", accuracy_score(y_train, rfc.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, rfc_pred))

print(classification_report(y_test, rfc_pred))

#cm = confusion_matrix(y_test, dtree_prec)
#fig, ax = plt.subplots(figsize=(10, 7))
#sns.heatmap(cm, annot=False, ax=ax)
#ax.set_xlabel('Predicted labels')
#ax.set_ylabel('True labels')
#ax.set_title('Confusion Matrix')

n_features = X.shape[1]
plt.barh(range(n_features), rfc.feature_importances_)
plt.yticks(np.arange(n_features), train.columns[1:])
plt.show()

keys = le2.classes_
values = le2.transform(le2.classes_)
print(keys)

dictionary = dict(zip(keys, values))
print(dictionary)

crime_pred_dtree = dtree.predict_proba(test)

result_dtree = pd.DataFrame(crime_pred_dtree, columns=keys)
result_dtree.head()

crime_pred_rfc = rfc.predict_proba(test)

result_rfc = pd.DataFrame(crime_pred_rfc, columns=keys)
result_rfc.head()

result_rfc.to_csv(path_or_buf="random_forest.csv", index=True, index_label='Id')
