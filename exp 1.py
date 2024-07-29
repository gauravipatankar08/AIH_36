import os
import zipfile
import pandas as pd
import numpy as np

# Download the dataset using Kaggle API
os.system("kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset")

# Unzip the dataset
with zipfile.ZipFile("novel-corona-virus-2019-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("covid19_data")

# Load the dataset
df = pd.read_csv('covid19_data/covid_19_data.csv')

# Perform 25 Pandas operations

# 1. Display the first 5 rows
print(df.head())

# 2. Display the last 5 rows
print(df.tail())

# 3. Display basic information about the dataset
print(df.info())

# 4. Display summary statistics
print(df.describe())

# 5. Check for missing values
print(df.isnull().sum())

# 6. Drop rows with missing values
df_clean = df.dropna()

# 7. Fill missing values with a specific value
df_filled = df.fillna(0)

# 8. Rename columns
df_renamed = df.rename(columns={'Country/Region': 'Country', 'ObservationDate': 'Date'})

# 9. Convert column to datetime
df_renamed['Date'] = pd.to_datetime(df_renamed['Date'])

# 10. Create a new column for active cases
df_renamed['Active'] = df_renamed['Confirmed'] - df_renamed['Deaths'] - df_renamed['Recovered']

# 11. Filter data by country
us_data = df_renamed[df_renamed['Country'] == 'US']

# 12. Group data by country and sum cases (excluding non-numeric columns)
country_data = df_renamed.groupby('Country')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()

# 13. Group data by date and sum cases (excluding non-numeric columns)
date_data = df_renamed.groupby('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()

# 14. Sort data by confirmed cases
sorted_data = df_renamed.sort_values(by='Confirmed', ascending=False)

# 15. Drop columns
df_dropped = df_renamed.drop(columns=['SNo', 'Province/State'])

# 16. Reset index
df_reset = df_dropped.reset_index(drop=True)

# 17. Set a new index
df_indexed = df_reset.set_index('Date')

# 18. Select specific columns
selected_columns = df_indexed[['Country', 'Confirmed', 'Deaths', 'Recovered', 'Active']]

# 19. Apply a function to a column, handling invalid values for log
df_indexed['Confirmed_log'] = df_indexed['Confirmed'].apply(lambda x: np.log(x + 1) if x > 0 else 0)


# 20. Merge two dataframes (resetting index to avoid merge issues)
df_indexed_reset = df_indexed.reset_index()
country_data_reset = country_data.reset_index()
merged_df = pd.merge(df_indexed_reset, country_data_reset, on='Country', suffixes=('_daily', '_total'))

# 21. Pivot table
pivot_df = df_renamed.pivot_table(values='Confirmed', index='Date', columns='Country', aggfunc='sum')

# 22. Melt dataframe
melted_df = pd.melt(df_renamed, id_vars=['Date', 'Country'], value_vars=['Confirmed', 'Deaths', 'Recovered', 'Active'])

# 23. Resample data
resampled_df = df_indexed.resample('ME').sum()

# 24. Rolling mean
rolling_mean = df_indexed['Confirmed'].rolling(window=7).mean()

# 25. Save cleaned dataframe to a new CSV file
df_indexed.to_csv('cleaned_covid19_data.csv')

print("25 Pandas operations completed!")
