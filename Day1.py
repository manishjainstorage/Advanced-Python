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


