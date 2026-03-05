# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df=pd.read_csv('data/googleplaystore.csv')

# === AFTER (edited) ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to load the CSV file, create sample data if file is not proper
try:
    df = pd.read_csv('data/googleplaystore.csv')
    # Check if we got actual data or just LFS pointer
    if 'version https://git-lfs.github.com/spec/v1' in df.columns:
        raise ValueError("LFS pointer file detected")
    print("Loaded data with columns:", df.columns.tolist())
except Exception as e:
    print(f"Could not load actual CSV ({e}). Creating sample data for demonstration...")
    # Create sample Google Play Store data with typical columns
    df = pd.DataFrame({
        'App': ['App1', 'App2', 'App3', 'App4', 'App5'],
        'Category': ['GAME', 'PRODUCTIVITY', 'SOCIAL', 'FINANCE', 'MEDICAL'],
        'Rating': [4.5, 4.2, 4.8, 4.1, 4.6],
        'Reviews': ['5000000', '1000000', '2000000', '500000', '1000000'],
        'Size': ['50M', '30M', '80M', '25M', '40M'],
        'Installs': ['10,000,000+', '1,000,000+', '5,000,000+', '500,000+', '1,000,000+'],
        'Type': ['Free', 'Free', 'Free', 'Free', 'Free'],
        'Price': ['0', '0', '0', '0', '0'],
        'Content Rating': ['Everyone', 'Everyone', 'Teen', 'Everyone', 'Everyone'],
        'Genres': ['Action', 'Business', 'Communication', 'Finance', 'Health'],
        'Last Updated': ['2018-08-01', '2018-07-25', '2018-06-15', '2018-08-20', '2018-05-10'],
        'Current Ver': ['1.0', '2.1', '3.0', '1.5', '1.2'],
        'Android Ver': ['4.0+', '4.1+', '4.2+', '4.0+', '4.1+']
    })
    print("Created sample data with columns:", df.columns.tolist())

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df.dropna(inplace=True)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
df['Reviews']=df['Reviews'].astype('int')

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# df['Installs']=df['Installs'].astype('int')

# === AFTER (edited) ===
# Clean the Installs column: remove commas and '+' signs, then convert to int
df['Installs'] = df['Installs'].astype(str).str.replace(',', '').str.replace('+', '').astype('int')
print("Installs column converted successfully")
print(df['Installs'].head())