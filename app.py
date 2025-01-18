from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

car = pd.read_csv("D:/projects/car prediction/cleardata.csv")
X = car.drop('Price', axis=1)
y = car['Price']

ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe.fit(X[['name', 'company', 'fuel_type']])
encoded_cat = ohe.transform(X[['name', 'company', 'fuel_type']])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=ohe.get_feature_names_out(['name', 'company', 'fuel_type']))

X = X.drop(['name', 'company', 'fuel_type'], axis=1)
X = pd.concat([X.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

app = Flask(__name__)

@app.route('/')
def home():
    data = pd.read_csv('cleardata.csv')
    companies = sorted(data['company'].unique())
    models = sorted(data['name'].unique())
    return render_template('index.html', companies=companies, models=models, r2=r2, mse=mse, rmse=rmse)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = pd.read_csv('cleardata.csv')
        companies = sorted(data['company'].unique())
        models = sorted(data['name'].unique())

        kms_driven = float(request.form['kms_driven'])
        year = int(request.form['year'])
        fuel_type = request.form['fuel_type']
        name = request.form['name']
        company = request.form['company']

        user_features = pd.DataFrame(columns=X.columns)
        user_features.loc[0] = 0
        user_features['kms_driven'] = kms_driven
        user_features['year'] = year

        if f'name_{name}' in X.columns:
            user_features[f'name_{name}'] = 1
        if f'company_{company}' in X.columns:
            user_features[f'company_{company}'] = 1
        if f'fuel_type_{fuel_type}' in X.columns:
            user_features[f'fuel_type_{fuel_type}'] = 1

        prediction = model.predict(user_features)
        return render_template(
            'index.html',
            prediction=f"Predicted Price: {prediction[0]:,.2f}",
            companies=companies,
            models=models,
            r2=r2,
            mse=mse,
            rmse=rmse
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}",
            companies=companies,
            models=models,
            r2=r2,
            mse=mse,
            rmse=rmse
        )

if __name__ == "__main__":
    app.run(debug=True)
