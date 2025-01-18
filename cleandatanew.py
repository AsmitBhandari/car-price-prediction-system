import pandas as pd

file = pd.read_csv("D:/projects/car prediction/quikr_car.csv")

file = file.dropna()

file['Price'] = file['Price'].str.replace(',', '', regex=True)
file = file[file['Price'].str.isnumeric()]
file['Price'] = file['Price'].astype(int)

file['kms_driven'] = file['kms_driven'].str.replace(' kms', '', regex=True).str.replace(',', '', regex=True)
file = file[file['kms_driven'].str.isnumeric()]
file['kms_driven'] = file['kms_driven'].astype(int)

file['year'] = pd.to_numeric(file['year'], errors='coerce')
file = file[file['year'].notna()]
file['year'] = file['year'].astype(int)

file['name'] = file['name'].str.split().str.slice(0, 3).str.join(' ')

mean_price = file['Price'].mean()
std_price = file['Price'].std()

threshold = 3
lower_bound = mean_price - (threshold * std_price)
upper_bound = mean_price + (threshold * std_price)

file = file[(file['Price'] >= lower_bound) & (file['Price'] <= upper_bound)]

fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
file['fuel_type'] = file['fuel_type'].map(fuel_mapping)
file = file.dropna(subset=['fuel_type'])
file['fuel_type']=file['fuel_type'].astype(int)

file = file.reset_index(drop=True)

file.to_csv("D:/projects/car prediction/cleardata.csv", index=False)
