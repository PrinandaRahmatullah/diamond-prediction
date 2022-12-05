# %% [markdown]
# ##

# %% [markdown]
# # Exploratory Data Analysis

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# %%
train_df = pd.read_csv("data/train.csv")
test_df = (pd.read_csv("data/test.csv", index_col=0)).reset_index(drop=True)

# %%
train_df.head()

# %%
test_df.head()

# %%
train_df.info()

# %%
train_df.describe()

# %%
train_df.columns

# %%
test_df.columns

# %%
label_column = [column for column in list(
    train_df.columns) if column not in list(test_df.columns)]
label_column

# %% [markdown]
# ### Clear Null Data

# %%
train_df.isna().sum()

# %%
train_df.isnull().sum()

# %%
# px_height && sc_w have zero that supposed to be value
train_df = (train_df[(train_df.px_height != 0) & (
    train_df.sc_w != 0)]).reset_index(drop=True)
train_df

# %%
test_df = (test_df[(test_df.px_height != 0) & (
    test_df.sc_w != 0)]).reset_index(drop=True)
test_df

# %% [markdown]
# ### Count Each Price Range

# %%


def change_name(data):
    if data.price_range == 0:
        return "Cheap"
    elif data.price_range == 1:
        return "Medium"
    elif data.price_range == 2:
        return "Expensive"
    elif data.price_range == 3:
        return "Pricey"


# %%
# Assign new name to classify
new_train_df = train_df.copy()
new_train_df["price_range"] = train_df.apply(change_name, axis=1)
new_train_df

# %%
new_train_df.price_range.value_counts().plot.pie(
    y='price_range', autopct='%1.1f%%', startangle=0)

# %% [markdown]
# ### Correlation Each Columns

# %%
# calculate the correlation matrix
train_df_corr = train_df.corr()

mask = np.triu(np.ones_like(train_df_corr.corr(), dtype=bool))


plt.figure(figsize=(16, 10))
# plot the heatmap
sns.heatmap(
    train_df_corr,
    mask=mask,
    xticklabels=train_df_corr.columns,
    yticklabels=train_df_corr.columns,
    fmt=".1g",
    annot=True)

# %% [markdown]
# ### Results

# %% [markdown]
# 1. Ram highly affects the price range with correlation 0.9
# 2. Battery power, pixel width, and pixel height low affects the price range with correlation between 0.1 and 0.2
# 3. Primary camera megapixel (pc) has medium affects to Front camera mega pixel (fc) with correlation 0.6
# 4. Three G (three_g) has medium affects to four G (four_g) with correlation 0.6
# 5. Pixel width (px_width) has medium affects to pixel height (px_height) with correlation 0.5
# 6. Scree width (px_width) has medium affects to screen height (px_height) with correlation 0.5

# %% [markdown]
# # Model Development

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Load Data

# %%
X = train_df.drop("price_range", axis=1)
y = train_df["price_range"]
X_test = test_df

# %% [markdown]
# ### Stardize Data
#

# %%

# %% [markdown]
# #### MinMax Scaler

# %%
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X)

min_max_scaled_X_df = pd.DataFrame(
    min_max_scaler.transform(X), columns=X.columns)
min_max_scaled_X_df

# %%
min_max_scaled_X_test_df = pd.DataFrame(
    min_max_scaler.transform(X_test), columns=X_test.columns)
min_max_scaled_X_test_df

# %% [markdown]
# #### Standard Scaler

# %%
standard_scaler = StandardScaler()
standard_scaler.fit(X)

standard_scaled_X_df = pd.DataFrame(
    standard_scaler.transform(X), columns=X.columns)
standard_scaled_X_df

# %%
standard_scaled_X_test_df = pd.DataFrame(
    standard_scaler.fit_transform(X_test), columns=X_test.columns)
standard_scaled_X_test_df

# %% [markdown]
# ### Split Data

# %%

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=51)

# %%
len(X_train), len(X_val), len(X_test)

# %%
len(y_train), len(y_val)

# %% [markdown]
# ## Model

# %%
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['val_mse', 'val_acc'],
                      columns=['KNN', 'RandomForest', 'Boosting'])
models

# %% [markdown]
# ### KNN

# %%

# %%
# Find best k of KNN
acc = []
error_rate = []

for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)

    y_val_pred = [round(x) for x in knn.predict(X_val)]

    error_rate.append(mean_squared_error(y_pred=y_val_pred, y_true=y_val))
    acc.append(accuracy_score(y_val_pred, y_val))

# %%
# plot error
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-", min(error_rate), "at K =",
      error_rate.index(min(error_rate))+1)

# %%
# plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), acc, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc))+1)

# %%
models.loc['val_mse', 'KNN'] = error_rate[20]
models.loc['val_acc', 'KNN'] = acc[20]

# %% [markdown]
# ### RandomForest

# %%
# Impor library yang dibutuhkan

# %%
acc = []
error_rate = []

# FInd best estimator
for estimator in range(25, 1000, 25):
    RF = RandomForestRegressor(
        n_estimators=estimator, max_depth=8, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)

    y_val_pred = [round(x) for x in RF.predict(X_val)]

    error_rate.append(mean_squared_error(y_pred=y_val_pred, y_true=y_val))
    acc.append(accuracy_score(y_val_pred, y_val))

# %%
# plot error
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. N Estimator * 25')
plt.xlabel('N Estimator * 25')
plt.ylabel('Error Rate')
print("Minimum error:-", min(error_rate), "at N Estimator = ",
      error_rate.index(min(error_rate))+1, "x 25")

# %%
# plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), acc, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('accuracy vs. N Estimator * 25')
plt.xlabel('N Estimator * 25')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(acc),
      "at N Estimator =", acc.index(max(acc))+1, "x 25")

# %%
models.loc['val_mse', 'RandomForest'] = error_rate[3]
models.loc['val_acc', 'RandomForest'] = acc[3]

# %% [markdown]
# ### Boosting

# %%

# %%
# Base weak classifier

dtclf = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=1)
dtclf.fit(X_train, y_train)

dtclf_train_sc = accuracy_score(y_train, dtclf.predict(X_train))
dtclf_val_sc = accuracy_score(y_val, dtclf.predict(X_val))
print('Decision tree train/val accuracies %.3f/%.3f' %
      (dtclf_train_sc, dtclf_val_sc))

# %%
hyperparameter_space = {'n_estimators': list(range(1, 50)),
                        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]}


gs = GridSearchCV(AdaBoostClassifier(base_estimator=dtclf,
                                     algorithm='SAMME.R',
                                     random_state=1),
                  param_grid=hyperparameter_space,
                  scoring="accuracy", n_jobs=-1, cv=5)

gs.fit(X_train, y_train)
print("Optimal hyperparameter combination:", gs.best_params_)

y_val_pred = gs.predict(X_val)
print(mean_squared_error(y_pred=y_val_pred, y_true=y_val),
      accuracy_score(y_val_pred, y_val))

# %%
boosting = AdaBoostClassifier(
    base_estimator=dtclf, learning_rate=0.05, n_estimators=7, random_state=55)
boosting.fit(X_train, y_train)

y_val_pred = boosting.predict(X_val)
print(mean_squared_error(y_pred=y_val_pred, y_true=y_val),
      accuracy_score(y_val_pred, y_val))

# %%
models.loc['val_mse', 'Boosting'] = mean_squared_error(
    y_pred=y_val_pred, y_true=y_val)
models.loc['val_acc', 'Boosting'] = accuracy_score(y_val_pred, y_val)

# %% [markdown]
# ## Evaluasi Model

# %%
models

# %% [markdown]
# Results :
# 1. Best KNN
# 2. Worst Boosting

# %% [markdown]
# ## Prediksi Data Test

# %%
knn = KNeighborsRegressor(n_neighbors=21)
knn.fit(X_train, y_train)

y_val_pred = [round(x) for x in knn.predict(X_val)]
print("Accuracy Data Validation : ", round(
    accuracy_score(y_val_pred, y_val), 2))

# %%
y_test_pred = [round(x) for x in knn.predict(X_test)]

# %%
test_df["price_range"] = y_test_pred
test_df
