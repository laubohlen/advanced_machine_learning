### TASK 1: PREDICT THE AGE OF A BRAIN FROM MRI FEATURES

This task is primarily concerned with regression. However, we have perturbed the original MRI features in several ways.

You will have to perform the following preprocessing steps:

- outlier detection
- feature selection
- imputation of missing values

You are required to document each of the three steps in the description that you will submit with your project. Besides the data processing steps, you have to provide a succint description of the regression model you used.


```python
import math
import time
import random
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

### Load dataset


```python
X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv')
X_test = pd.read_csv('./data/X_test.csv')

X_train['y'] = y_train['y']
mri = X_train
mri = mri.drop(columns='id')
```


```python
X_train = mri.loc[:, mri.columns != 'y']
y_train = mri['y']
X_test = X_test.drop(columns='id')
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
```

    (1212, 832)
    (1212,)
    (776, 832)


### Split dataset and remove outliers


```python
# shuffle image list
obs = mri.index.tolist()
random.seed(17)
random.shuffle(obs)

# set percentages of training set size
train_p = 99 # val_p = 100-train_p

# split into training and testing list
train_l = obs[0:math.floor(len(obs)/100*train_p)]
val_l = obs[len(train_l):]

training = mri.loc[train_l]
validation = mri.loc[val_l]

# detecting outliers with IQR (interquartile range)
Q1 = training.loc[:, training.columns != 'y'].quantile(0.25)
Q3 = training.loc[:, training.columns != 'y'].quantile(0.75)
IQR = Q3 - Q1

training_no = training[~((training < (Q1 - 100 * IQR)) |(training > (Q3 + 100 * IQR))).any(axis=1)]

X_train = training_no.loc[:, training.columns != 'y']
y_train = training_no['y']
X_val = validation.drop(columns='y')
y_val = validation['y']
```


```python
print('Outliers removed: {}'.format(training.shape[0] - training_no.shape[0]))
print('Remaining samples: {}'.format(X_train.shape[0]))
print('Validation samples: {}'.format(X_val.shape[0]))
```

    Outliers removed: 0
    Remaining samples: 1199
    Validation samples: 13


### Impute missing data


```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
```


```python
# simple impute data
def simple_impute(X_train, X_val, method):
    col_names = X_train.columns.tolist()
    
    imputer = SimpleImputer(missing_values=np.nan, strategy=method)
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    
    return X_train_imp, X_val_imp, imputer

# knn impute data
def knn_impute(X_train, X_val, k):
    col_names = X_train.columns.tolist()
    
    imputer = KNNImputer(n_neighbors=k)
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    
    return X_train_imp, X_val_imp, imputer

# iterative imputer
def iter_impute(X_train, X_val):
    col_names = X_train.columns.tolist()
    
    start_time = time.time()
    imputer = IterativeImputer(max_iter=10, verbose=1, n_nearest_features=20)
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    print('{} seconds'.format(round(time.time() - start_time)))
    
    return X_train_imp, X_val_imp, imputer

def check_for_nan(np_array):
    np_array_sum = np.sum(np_array)
    array_has_nan = np.isnan(np_array_sum)
    print(array_has_nan)
    
def check_for_finite(np_array):
    output = np.all(np.isfinite(np_array))
    print(output)
```


```python
X_train_imp, X_val_imp, imputer = simple_impute(X_train, X_test, 'median')
```


```python
check_for_nan(X_train_imp)
check_for_nan(X_val_imp)
```

    False
    False


### Feature selection


```python
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesRegressor
```


```python
# feature selection. 
# method = f_regression for correlation; mutual_info_regression for mutual information
def select_features(X_train, y_train, X_test, method, n_features): # 
    # configure to select all features
    fs = SelectKBest(score_func=method, k=n_features)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
```


```python
def recursive_feature_elimination(X_train, y_train, X_val, estim):
    rfecv = RFECV(estimator=estim, step=1, cv=KFold(2), n_jobs=-1)
    rfecv = rfecv.fit(X_train, y_train)
    X_train_selected = rfecv.transform(X_train)
    X_val_selected = rfecv.transform(X_val)
    print("Optimal number of features : %d" % rfecv.n_features_)
    return(X_train_selected, X_val_selected)
```


```python
X_train_cor, X_val_cor, cor = select_features(X_train_imp, y_train, X_val_imp, f_regression, 100)
```

    /Users/lau/opt/anaconda3/envs/face/lib/python3.6/site-packages/sklearn/feature_selection/_univariate_selection.py:302: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /Users/lau/opt/anaconda3/envs/face/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1932: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= _a)



```python
print(X_train_cor.shape)
print(X_val_cor.shape)
```

    (1212, 100)
    (776, 100)



```python
start_time = time.time()
X_train_sel, X_val_sel = recursive_feature_elimination(X_train_cor, y_train, X_val_cor,
                                                      ExtraTreesRegressor(n_estimators=1470, n_jobs=-1))
print('{} seconds'.format(round(time.time() - start_time)))
```

    Optimal number of features : 59
    1070 seconds



```python
print(X_train_sel.shape)
print(X_val_sel.shape)
```

    (1212, 59)
    (776, 59)


### Transformation


```python
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
```


```python
# mean=0, var=1
def standardize(X_train, X_val):
    transformer = preprocessing.StandardScaler()
    X_train_trans = transformer.fit_transform(X_train)
    X_val_trans = transformer.transform(X_val)
    return X_train_trans, X_val_trans, transformer

# scale to [min, max]
def minmax(X_train, X_val, range_tuple): # e.g. (-1, 1)
    transformer = preprocessing.MinMaxScaler(feature_range=range_tuple)
    X_train_trans = transformer.fit_transform(X_train)
    X_val_trans = transformer.transform(X_val)
    return X_train_trans, X_val_trans, transformer

# powertransform
def powertransform(X_train, X_val):
    transformer = PowerTransformer()
    X_train_trans = transformer.fit_transform(X_train)
    X_val_trans = transformer.transform(X_val)
    return X_train_trans, X_val_trans, transformer
```


```python
# transform data
X_train_stand, X_test_stand, standard_scaler = standardize(X_train_sel, X_val_sel)
X_train_minmax, X_test_minmax, minmax_scaler = minmax(X_train_sel, X_val_sel, (-1, 1))
X_train_minmax01, X_test_minmax01, minmax_scaler01 = minmax(X_train_sel, X_val_sel, (0, 1))
```

### Randomized Cross Validated Grid Search


```python
n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators' : n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
```


```python
est = ExtraTreesRegressor()
rf_random = RandomizedSearchCV(estimator = est, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=1, n_jobs = -1)
rf_random.fit(X_train_sel, y_train)
rf_random.best_params_
```

### Fine Tuning with Cross Validated Grid Search


```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
```


```python
parameter_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive']}

start_time = time.time()

regf_grid = GridSearchCV(MLPRegressor(max_iter=10000, activation='tanh', solver='sgd'), parameter_grid, n_jobs=-1)
regf_grid.fit(X_train_minmax01, y_train)

print('{} seconds'.format(round(time.time() - start_time)))
print()
print("Best parameters set found on development set:")
print(regf_grid.best_params_)
print()
print("Grid scores on development set:")
means = regf_grid.cv_results_['mean_test_score']
stds = regf_grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, regf_grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
```

### Cross Validation


```python
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
```


```python
regf = ExtraTreesRegressor(n_estimators=1740, max_depth=53)
regf_cv = cross_val_score(regf, X_train_minmax, y_train, cv=5, n_jobs=-1)
```


```python
print("r2: %0.4f (+/- %0.2f)" % (regf_cv.mean(), regf_cv.std() * 2))
```

### Prediction


```python
# extra trees regression
extra_tree = ExtraTreesRegressor(random_state=0, n_estimators=1740, max_depth=60, n_jobs=-1)
extra_tree.fit(X_train_minmax, y_train)
extra_pred = extra_tree.predict(X_test_minmax)

print(r2_score(y_val, extra_pred))
```


```python
ID = np.array(range(len(X_test_minmax)))
df = pd.DataFrame({'id': ID,
                    'y': extra_pred})
df.to_csv('/Users/lau/Desktop/prediction.csv', index=False)
```

### Result

public $R^{2}$ score: **0.656816298899** <br>
private $R^{2}$ score: **0.604249316334**


```python

```
