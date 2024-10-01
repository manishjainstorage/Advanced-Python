!pip install --upgrade pandas 

#conda update pandas

import pandas as pd
print(pd.__version__)

!pip show pandas
__________________________________________________________________________________
import pandas as pd

# https://github.com/manishjainstorage/Advanced-Python/blob/main/StockMarketData-13krows.xlsx
# Define the correct raw URL of the .xlsx file
url = "https://github.com/manishjainstorage/Advanced-Python/raw/main/StockMarketData-13krows.xlsx"

# Read the .xlsx file into a DataFrame, specifying the engine
df = pd.read_excel(url, engine='openpyxl')

# Display the first few rows of the DataFrame
df.head()
__________________________________________________________________________________

import pandas as pd
import polars as pl
import time

# Load large dataset using Pandas
start_time = time.time()
df_pandas = pd.read_excel("https://github.com/manishjainstorage/Advanced-Python/raw/main/StockMarketData-13krows.xlsx")
print(f'Pandas load time: {time.time() - start_time} seconds')

# Load large dataset using Polars
start_time = time.time()
df_polars = pl.read_excel("https://github.com/manishjainstorage/Advanced-Python/raw/main/StockMarketData-13krows.xlsx")
print(f'Polars load time: {time.time() - start_time} seconds')
___________________________________________________________________________________

def a(n):
  i = 1
  while i<=n:
    yield i
    i+=1

gen = a(5)

print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
________________________________________________________________________

import pandas as pd
import polars as pl
import numpy as np
import time

# Define the size of the dataset
N = 10**7  # 10 million rows

# Generate a large dataset
data = {
    'a': np.random.rand(N),
    'b': np.random.rand(N)
}

# Function to perform a computation in Pandas
def pandas_computation(data):
    df = pd.DataFrame(data)
    # Perform a computation (e.g., summing the product of two columns)
    result = (df['a'] * df['b']).sum()
    return result

# Function to perform a computation in Polars
def polars_computation(data):
    df = pl.DataFrame(data)
    # Perform a computation (e.g., summing the product of two columns)
    result = (df['a'] * df['b']).sum()
    return result

# Measure execution time for Pandas
start_time = time.time()
pandas_result = pandas_computation(data)
pandas_time = time.time() - start_time
print(f"Pandas result: {pandas_result}, Time taken: {pandas_time:.2f} seconds")

# Measure execution time for Polars
start_time = time.time()
polars_result = polars_computation(data)
polars_time = time.time() - start_time
print(f"Polars result: {polars_result}, Time taken: {polars_time:.2f} seconds")

________________________________________________________________________________________










