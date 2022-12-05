# %% [markdown]
# ##

# %% [markdown]
# # Exploratory Data Analysis

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
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
# ### Clear Null Data dan Zero Value

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
# 6. Screen width (px_width) has medium affects to screen height (px_height) with correlation 0.5

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


# %%
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['val_f1_score'],
                      columns=['KNN', 'RandomForest', 'Boosting'])
models

# %% [markdown]
# ### KNN

# %%

# %%
# Find best k of KNN
f_score = []

for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)

    y_val_pred = [round(x) for x in knn.predict(X_val)]

    f_score.append(f1_score(y_pred=y_val_pred,
                   y_true=y_val, average="weighted"))

# %%
# plot error
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), f_score, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('F1-score vs. K Value')
plt.xlabel('K')
plt.ylabel('F1-score')
print("Maximum F1-Score:", max(f_score),
      "at K =", f_score.index(max(f_score))+1)

# %%
knn = KNeighborsRegressor(n_neighbors=21)
knn.fit(X_train, y_train)

y_val_pred = [round(x) for x in knn.predict(X_val)]
confusion_matrix(y_val_pred, y_val)

# %%
models.loc['val_f1_score', 'KNN'] = max(f_score)

# %% [markdown]
# ### RandomForest

# %%
# Impor library yang dibutuhkan

# %%
f_score = []

# FInd best estimator
for estimator in range(5, 200, 5):
    RF = RandomForestRegressor(
        n_estimators=estimator, max_depth=8, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)

    y_val_pred = [round(x) for x in RF.predict(X_val)]

    f_score.append(f1_score(y_pred=y_val_pred,
                   y_true=y_val, average="weighted"))

# %%
# plot error
plt.figure(figsize=(10, 6))
plt.plot(range(5, 200, 5), f_score, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('F1-Score vs. N Estimator * 5')
plt.xlabel('N Estimator * 5')
plt.ylabel('F1-Score')
print("Maximum F1-score:", max(f_score), "at N Estimator = ",
      (f_score.index(max(f_score))+1)*5,)

# %%
RF = RandomForestRegressor(n_estimators=(f_score.index(
    max(f_score))+1)*5, max_depth=8, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

y_val_pred = [round(x) for x in RF.predict(X_val)]
confusion_matrix(y_val_pred, y_val)

# %%
models.loc['val_f1_score', 'RandomForest'] = max(f_score)

# %% [markdown]
# ### Boosting

# %%

# %%
# Base weak classifier

dtclf = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=1)
dtclf.fit(X_train, y_train)

dtclf_train_sc = f1_score(y_train, dtclf.predict(X_train), average="weighted")
dtclf_val_sc = f1_score(y_val, dtclf.predict(X_val), average="weighted")
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
print(f1_score(y_val_pred, y_val, average="weighted"))

# %%
boosting = AdaBoostClassifier(
    base_estimator=dtclf, learning_rate=0.05, n_estimators=7, random_state=55)
boosting.fit(X_train, y_train)

y_val_pred = boosting.predict(X_val)
print(f1_score(y_val_pred, y_val, average="weighted"))

# %%
models.loc['val_f1_score', 'Boosting'] = f1_score(
    y_val_pred, y_val, average="weighted")

# %% [markdown]
# ## Evaluasi Model

# %% [markdown]
# https://scikit-learn.org/stable/modules/model_evaluation.html

# %%
models

# %% [markdown]
# Results :
# 1. Best KNN
# 2. Worst Boosting

# %% [markdown]
# ## Prediksi Data Test

# %%
knn_test_pred = [round(x) for x in knn.predict(X_test)]
rf_test_pred = [round(x) for x in RF.predict(X_test)]
boosting_test_pred = [round(x) for x in boosting.predict(X_test)]

# %%
test_results_df = pd.DataFrame(
    {"KNN": knn_test_pred, "RandomForest": rf_test_pred, "Boosting": boosting_test_pred})
test_results_df
