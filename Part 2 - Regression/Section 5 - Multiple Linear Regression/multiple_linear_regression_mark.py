import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

data = pd.read_csv("50_Startups.csv")
data = data.iloc[:, 0:5]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_Lbl_Enc = LabelEncoder()
X[:, -1] = X_Lbl_Enc.fit_transform(X[:, -1])

X_OneH_Enc = OneHotEncoder(categorical_features=[-1])
X = X_OneH_Enc.fit_transform(X).toarray()

X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

X = np.append(np.ones(shape=(50,1), dtype=int), axis=1, values=X)
X_OLS = pd.DataFrame(X)

SL = 0.05
#Backward elimination
while True:
    regressor_OLS = sm.OLS(endog=y, exog=X_OLS).fit()
    summary = regressor_OLS.summary().tables[1].as_html()
    df = pd.read_html(summary, header=None, skiprows=1)[0]
    df.index = X_OLS.columns
    P_index = df.iloc[:, 4].argmax()
    P = df.iloc[:, 4].max()
    print(P)
    if P > SL:
        X_OLS.drop(columns=P_index, axis=1, inplace=True)
    else:
        break