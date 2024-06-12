# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle
import nbformat

# %%
data = pd.read_csv("CarPrice_Assignment.csv")
df = pd.DataFrame(data)

# %%
df.head()

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.duplicated().sum()

# %%
df.describe()

# %%
df.describe(include='object')

# %%
df['CarName'].unique()

# %%
df.insert(1,'Company',df['CarName'].str.split(' ').str[0])

# %%
df.drop(columns=['CarName'],inplace=True)

# %%
df.head()

# %%
df['Company'].unique()

# %%
company_name_mapping = {
    'maxda': 'mazda',
    'Nissan': 'nissan',
    'porcshce': 'porsche',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen'
}

df['Company'].replace(company_name_mapping, inplace=True)

# %%
df['Company'].unique()

# %%
df.drop(columns=['car_ID'],inplace=True)

# %%
df2 = df.drop(columns=['symboling','wheelbase','boreratio','stroke','compressionratio','peakrpm','Company'])

# %%
df2 = pd.get_dummies(columns=["fueltype","enginelocation","aspiration","doornumber","carbody","drivewheel","enginetype",
                                "cylindernumber","fuelsystem"],data=df2)

# %%
df2.info()

# %%
scaler = StandardScaler()

# %%
numerical_columns = df2.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_columns.remove('price')

# %%
numerical_columns

# %%
df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])

# %%
X = df2.drop(columns=['price'])
y = df2['price'].values.reshape(-1,1)

# %%
X.shape

# %%
y.shape

# %%
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
pickle.dump(scaler,open('scaling.pkl','wb'))

# %%
X_train

# %%
y_train = y_train.ravel()
y_test = y_test.ravel()

regression_models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100),
    SVR(),
    KNeighborsRegressor()
]

training_scores = []
testing_scores = []

def model_prediction(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    a = r2_score(y_train, x_train_pred) * 100
    b = r2_score(y_test, x_test_pred) * 100
    training_scores.append(a)
    testing_scores.append(b)
    
    print(f"R2 Score of {type(model).__name__} model on Training Data: {a:.2f}")
    print(f"R2 Score of {type(model).__name__} model on Testing Data: {b:.2f}")

for model in regression_models:
    print(f"Evaluating {type(model).__name__} model : ")
    model_prediction(model, X_train, y_train, X_test, y_test)
    print("\n")

best_model_index = testing_scores.index(max(testing_scores))
best_model = regression_models[best_model_index]
best_test_score = testing_scores[best_model_index]

print(f"The best model is {type(best_model).__name__} with an R2 score of {best_test_score:.2f} on the testing data.")

# %%
RandomForestRegressor = regression_models[2]

# %%
pickle.dump(RandomForestRegressor,open('RFRegressorModel.pkl','wb'))

# %%
pickled_model = pickle.load(open('RFRegressorModel.pkl','rb'))

# %%
row_to_predict = X_test.iloc[10]

# %%
row_to_predict

# %%
headers=[]
for i in df2.columns:
    if i.split('_')[0] not in headers:
        headers.append(i.split('_')[0])
headers

# %%
custom_input={
'carlength':[150],
 'carwidth':[64],
 'carheight':[52.6],
 'curbweight':[1837],
 'enginesize':[79],
 'horsepower':[60],
 'citympg':[38],
 'highwaympg':[42],
 'fueltype':['gas'],
 'enginelocation':['front'],
 'aspiration':['std'],
 'doornumber':['two'],
 'carbody':['convertible'],
 'drivewheel':['fwd'],
 'enginetype':['ohc'],
 'cylindernumber':['four'],
 'fuelsystem':['1bbl']
}
custom_input=pd.DataFrame(custom_input)
custom_input



# %%
custom_input = pd.get_dummies(columns=["fueltype","enginelocation","aspiration","doornumber","carbody","drivewheel","enginetype",
                                "cylindernumber","fuelsystem"],data=custom_input)


# %%
for i in df2.columns:
    if i not in custom_input.columns:
        custom_input[i]=False

# %%
custom_input[numerical_columns]=scaler.transform(custom_input[numerical_columns])
custom_input

# %%
prediction = pickled_model.predict([row_to_predict])
prediction


