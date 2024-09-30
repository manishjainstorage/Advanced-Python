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





