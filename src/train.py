import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from scipy.stats import shapiro


url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

def percentage_nulls(df):
    """
    This function returns a dictionary with the column and
    the porcentage of missing values
    """
    N_rows = df.shape[0]
    vars_ = {}
    for var in df.columns:
        vars_[var]=(df[var].isnull().sum() / N_rows)
    return vars_

percentage_nulls(df)
df=df.drop_duplicates().reset_index(drop = True)
df.columns[np.array(df.dtypes == "object")]

def binary(data):
  bin_reg = bin_reg = r"^[01](?:\.0)?\.?$"
  if str(data) == 'nan':
    return np.nan
  else:
    return bool(re.findall(bin_reg, str(data)))

def is_binary(df_, col):
  """
  to consider this as a pure binary var take into account that
  the others not reach a limit of range..
  #no puedehaber un porcentaje de varriable superior con más de un  digito!!!!!!
  """
  df = df_.copy()
  percent =  df[col].apply(binary).sum() / df[col].count()
  if percent > 0.5:
    return True
  else:
    return False

normal  = []
nonormal = []
binaries = []
cates = []

def tipo_var(df_):
  df = df_.copy()
  for col in df.columns:
    if df[col].dtypes.name=='int64' or df[col].dtypes.name == 'float64':
      if is_binary(df, col):
        binaries.append(col)
      else:
        if shapiro(df[col]).pvalue > 0.05:
          normal.append(col)
        else:
          nonormal.append(col)
    else:
      cates.append(col)
  return normal, nonormal, binaries,  cates

# Regex para seleccionar columnas numéricas
numerical_columns = [col for col in df.columns if re.match(r'^[A-Za-z]+$', col) and col != 'Outcome'] #quito manualmente Target
df[numerical_columns].describe()

#Separar el dataset en train y test
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 2. Dividir en train/test antes de escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Escalar solo usando la media y std de X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform SOLO (sin fit)

#algoritmo para arboles de decision

def grid_dt(X_train, y_train):
    model = DecisionTreeClassifier(random_state=666)

    class_weight = [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    max_depth = [None, 3, 5, 10] #cuidado si uso solo none es una tupla de un solo valor, no una lista o un valor único. agrego 3, 5, 10
    min_samples_leaf = [5, 10, 20, 50, 100]
    criterion  = ["gini", "entropy"]

    grid = dict(
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,
        cv=cv,
        scoring='f1',
        error_score=0,
        verbose=1
    )

    grid_result = grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

#entreno modelo arbol de decision
best_dt = grid_dt(X_train_scaled, y_train)
y_pred_dt = best_dt.predict(X_test_scaled)

print(classification_report(y_test, y_pred_dt))

with open("modelo.pkl", "wb") as f:
    pickle.dump(best_dt, f)

print("Modelo entrenado y guardado exitosamente.")
