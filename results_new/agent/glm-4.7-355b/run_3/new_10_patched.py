# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# Import the required Libraries
import numpy as np,pandas as pd, matplotlib.pyplot as plt, seaborn as sns,plotly.express as px

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# Importing the dataset
data = pd.read_csv('data/hotel_bookings.csv')
data.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
data.info()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
data.describe(include='all')

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
data.isnull().sum()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# Replacing the missing values in 'country' with the most frequented values
data['country'] = data['country'].fillna(data['country'].mode()[0])

# Replacing the null values in 'agent' and 'children' with 0
data['agent'] = data['agent'].fillna(0)
data['children'] = data['children'].fillna(0)

# Droping the 'company' column
data.drop('company', axis=1, inplace=True)
print(data.isnull().sum().sum())

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# Droping the remaining missing values
data.dropna(inplace=True)
data.info()

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# Check for duplicates
data.duplicated().sum()

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# Converting the data type of reservation_status_date to datetime
data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])
print(data['reservation_status_date'].dtype)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# Removing the rows where there are no guests.
data = data[~((data['adults'] == 0) & (data['children'] == 0) & (data['babies'] == 0))]
data.shape

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
# Value counts for each columns
for col in data.columns:
  print(data[col].value_counts())
  print('*'*70)

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
# Correalation between different numerical variables
data_num = data.select_dtypes(include=['int64', 'float64'])
Correlation_matrix = data_num.corr()
plt.figure(figsize=(15, 7))
sns.heatmap(Correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# Arrival Month pattern with Number of Adults and Chlidren
monthly_data = data.groupby('arrival_date_month')
plt.figure(figsize=(10, 5))
plt.plot(monthly_data['adults'].mean(), label='Adults',marker='o')
plt.plot(monthly_data['children'].mean(), label='Children',marker='o')
plt.title('Monthly Average Adults and Children')
plt.xlabel('Month')
plt.ylabel('Average Count')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
# Monthly trend on booking
data['month'] = data['reservation_status_date'].dt.month_name()
data['month'] = data['month'].str.capitalize()
monthly_bookings = data.groupby('month').size()
plt.figure(figsize=(10, 5))
monthly_bookings.plot(kind='bar', color='skyblue')
plt.title('Monthly Bookings')
plt.xlabel('Month')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45)
plt.show()
# Yearly trend on booking
yearly_bookings = data.groupby(data['reservation_status_date'].dt.year).size()
plt.figure(figsize=(10, 5))
yearly_bookings.plot(kind='bar', color='lightcoral')
plt.title('Yearly Bookings')
plt.xlabel('Year')
plt.ylabel('Number of Bookings')
plt.show()

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
# Booking trends based on  Countries

guests_by_country = data[data['is_canceled'] == 0]['country'].value_counts().reset_index()
guests_by_country.columns = ['Country', 'Number of guests']
guests_by_country

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
guests_map = px.choropleth(
    guests_by_country,
    locations = guests_by_country ['Country'],
    color = guests_by_country ['Number of guests'],
    hover_name = guests_by_country ['Country'],
    title='Guest Distribution by Country',
    color_continuous_scale=px.colors.sequential.deep
)

guests_map.show()

#%%
# --- [CELL 16]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
# === BEFORE (original) ===
# # Cancelling based on hotel type
# plt.figure()
# sns.countplot(x='hotel', hue='is_canceled', data=data)
# plt.title('Cancellation based on Hotel Type')
# plt.xlabel('Hotel Type')
# plt.ylabel('Count')
# plt.label(['Not '])
# plt.show()
# 
# # Cancellation pattern over months
# plt.figure(figsize=(13, 5))
# sns.countplot(hue='is_canceled', x='arrival_date_month', data=data)
# plt.title('Cancellation Status over Arrival Months')
# plt.xlabel('Arrival Month')
# plt.ylabel('Count')
# plt.show()

# === AFTER (edited) ===
plt.figure()
sns.countplot(x='hotel', hue='is_canceled', data=data)
plt.title('Cancellation based on Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(13, 5))
sns.countplot(hue='is_canceled', x='arrival_date_month', data=data)
plt.title('Cancellation Status over Arrival Months')
plt.xlabel('Arrival Month')
plt.ylabel('Count')
plt.show()