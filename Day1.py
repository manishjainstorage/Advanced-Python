import pandas as pd
import numpy as np
df = pd.read_csv("/content/Ajio.csv" , encoding = "unicode_escape")
df.head()

df.info()
df.describe()
df.describe(include=[object])

######################################################################3
import sweetviz as sv
import pandas as pd

# Load your dataset
df = pd.read_csv("/content/Ajio.csv" , encoding = "Unicode_escape")

# Generate the report
report = sv.analyze(df)

# Display the report
report.show_html("Sweetviz_Report.html")
########################################################################

# Analyzing features against a target variable
report = sv.analyze(df, target_feat='Sales')
report.show_html("Target_Analysis_Report.html")

#####################################################################

df = df.convert_dtypes()
df.info()

chunksize = 10
for chunk in pd.read_csv("/content/Ajio.csv" , encoding = "unicode_escape" , chunksize=chunksize):
  print(chunk)

######################################################################

df.info(memory_usage='deep')
df.to_excel('compressed_file.xlsx.gz', compression='gzip')

import gc
gc.collect()
################################################################

# drop columns where more then 50% of the values are missing

df_cleaned = df.dropna(thresh=0.5*len(df), axis=1)

# Visualize the % of missing values in the columns

missing_percentage = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
print(missing_percentage)

# missingvalues from heatmap

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned.isnull(), cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

#using missingno

import missingno as msno

msno.matrix(df_cleaned)
plt.show()

#############################################################33

# prompt: create a sample time series dataset with some missing valuesand perform forword and backward  filling for missing data

import pandas as pd
import numpy as np

# Create a sample time series dataset with some missing values
dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
values = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
df = pd.DataFrame({'Date': dates, 'Value': values})
df = df.set_index('Date')

print("Original DataFrame:")
print(df)

# Forward fill missing values
df_forward_filled = df.fillna(method='ffill')
print("\nDataFrame after Forward Fill:")
print(df_forward_filled)

# Backward fill missing values
df_backward_filled = df.fillna(method='bfill')
print("\nDataFrame after Backward Fill:")
print(df_backward_filled)

#####################################################3

import numpy as np
import modin.pandas as pd

#create a large dataframe
df = pd.DataFrame(np.random.randn(10000000, 10), columns=[f'col_{i}' for i in range(10)])

#perform some operations
result = df.mean()
print(result)

___________________________________________________________________________

#Using timing functions

import numpy as np
import pandas as pd
import modin.pandas as mpd
import time

#create a large dataframe

nrows = 10_000_000
data = np.random.rand(nrows,5)

#Timing with Pandas

start_time = time.time()
df_pandas = pd.DataFrame(data)
mean_pandas = df_pandas.mean()
end_time = time.time()
pandas_time = end_time - start_time
print(f"Time taken with Pandas: {pandas_time} seconds")

#Timing with Modin

start_time = time.time()
df_modin = mpd.DataFrame(data)
mean_modin = df_modin.mean()
end_time = time.time()
modin_time = end_time - start_time
print(f"Time taken with Modin: {modin_time} seconds")
#################################################################3

#Using timing functions

import numpy as np
import pandas as pd
import modin.pandas as mpd
import time

#create a large dataframe

nrows = 10_000_000
data = np.random.rand(nrows,5)

#Timing with Pandas

start_time = time.time()
df_pandas = pd.DataFrame(data)
mean_pandas = df_pandas.mean()
end_time = time.time()
pandas_time = end_time - start_time
print(f"Time taken with Pandas: {pandas_time} seconds")

#Timing with Modin

import os
os.environ["MODIN_ENGINE"] = "dask"
os.environ["MODIN_NUM_PARTITIONS"] = "3"

start_time = time.time()
df_modin = mpd.DataFrame(data)
mean_modin = df_modin.mean()
end_time = time.time()
modin_time = end_time - start_time
print(f"Time taken with Modin: {modin_time} seconds")
print("Number of partitions used :",os.environ.get("MODIN_NUM_PARTITIONS"))

#############################################################3

#using joblib

from joblib import Parallel, delayed

def square(x):
  return x**2

result = Parallel(n_jobs=5)(delayed(square)(i) for i in range(10))

print(result)

#########################################################3
# using multiprocessing

import multiprocessing

def square(x):
  return x**2

if __name__ == '__main__':
  numbers = [1,2,3,4,5]
  #create a pool of processors
  with multiprocessing.Pool(processes = 5) as pool:
    result = pool.map(square , numbers)
  print(result)
############################################################

df.memory_usage(deep=True)
