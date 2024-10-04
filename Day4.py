
import plotly.express as px

fig = px.line(flights, x="month", y="passengers", color="year", title="Number of Passengers Over Time")
fig.update_layout(
    yaxis_title="Number of Passengers"
)
fig.show()


###########################################################

import streamlit as st
import pandas as pd 
import numpy as np 
st.title("My APP")
userinput = st.text_input("Enter a name","streamlit user")
st.write(f"Hello,{userinput}") 
data = np.random.randn(100,3) 
df = pd.DataFrame(data,columns=['A','B','C'])
st.line_chart(df)
num = st.slider('select a no',0,100,25) 
st.write(f"You selected: {num}")
____________________________________________________________

streamlit run app.py
__________________________________________________________________________

import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Function to update the text label when the slider value changes
def update_label(val):
    label_num.config(text=f"You selected: {slider.get()}")

# Function to update the greeting when user input changes
def update_greeting():
    label_greeting.config(text=f"Hello, {entry_name.get()}")

# Generate random data
data = np.random.randn(100, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])

# Create the main window
root = tk.Tk()
root.title("My APP")

# Add title label
title_label = tk.Label(root, text="My APP", font=("Helvetica", 16))
title_label.pack(pady=10)

# Text input for name
name_frame = tk.Frame(root)
name_frame.pack(pady=10)
label_name = tk.Label(name_frame, text="Enter a name: ")
label_name.pack(side=tk.LEFT)
entry_name = tk.Entry(name_frame)
entry_name.insert(0, "Tkinter user")  # Default text
entry_name.pack(side=tk.LEFT)
button_update_name = tk.Button(root, text="Submit", command=update_greeting)
button_update_name.pack(pady=5)

# Label for greeting
label_greeting = tk.Label(root, text="Hello, Tkinter user")
label_greeting.pack(pady=10)

# Create a slider
slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=update_label)
slider.set(25)  # Default value
slider.pack(pady=10)

# Label for the slider selection
label_num = tk.Label(root, text=f"You selected: {slider.get()}")
label_num.pack(pady=10)

# Plotting a line chart
fig, ax = plt.subplots()
ax.plot(df.index, df['A'], label="A")
ax.plot(df.index, df['B'], label="B")
ax.plot(df.index, df['C'], label="C")
ax.set_title("Line Chart")
ax.legend()

# Embedding Matplotlib graph in Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
____________________________________________________________________

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Generate date range
dates = pd.date_range(start='2024-09-01', end='2024-09-30', freq='D')

# Create demo data: simulating stock prices
np.random.seed(42)  # For reproducibility
prices = np.random.normal(loc=100, scale=5, size=len(dates)).cumsum()

# Create DataFrame
data = pd.DataFrame(data={'Date': dates, 'Stock Price': prices})

# Set Date as the index
data.set_index('Date', inplace=True)

print(data.head())


# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Stock Price'], marker='o', linestyle='-')
plt.title('Simulated Daily Stock Prices for September 2024')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid()
plt.show()


# Resample the data to weekly frequency and calculate the mean
weekly_data = data.resample('W').mean()

print(weekly_data)
_______________________________________________________________________________

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Generate date range
dates = pd.date_range(start='2024-09-01', end='2024-09-30', freq='D')

# Create demo data: simulating stock prices
np.random.seed(42)  # For reproducibility
prices = np.random.normal(loc=100, scale=5, size=len(dates)).cumsum()

# Create DataFrame
data = pd.DataFrame(data={'Date': dates, 'Stock Price': prices})

# Set Date as the index
data.set_index('Date', inplace=True)
print(data.head())



# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Stock Price'], marker='o', linestyle='-')
plt.title('Simulated Daily Stock Prices for September 2024')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid()
plt.show()

# Resample the data to weekly frequency and calculate the mean
weekly_data = data.resample('W').mean()
print(weekly_data)



import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
decomposition = seasonal_decompose(data['Stock Price'], model='additive')

# Create subplots
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])

# Add observed
fig.add_trace(go.Scatter(x=data.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)

# Add trend
fig.add_trace(go.Scatter(x=data.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)

# Add seasonal
fig.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)

# Add residual
fig.add_trace(go.Scatter(x=data.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)

# Update layout
fig.update_layout(height=800, width=1000, title_text="Seasonal Decomposition of Time Series", showlegend=False)
fig.update_xaxes(title_text='Date', row=4, col=1)

# Show plot
fig.show()
__________________________________________________________

"""Moving Average
Calculate and visualize a moving average to smooth out the fluctuations."""

# Calculate moving average
data['Moving Average'] = data['Stock Price'].rolling(window=7).mean()

# Plot original and moving average
plt.figure(figsize=(12, 6))
plt.plot(data['Stock Price'], label='Original', alpha=0.5)
plt.plot(data['Moving Average'], label='7-Day Moving Average', color='orange')
plt.title('Stock Prices with Moving Average')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()

____________________________________________________________

!pip install plotly

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Generate date range
dates = pd.date_range(start='2024-09-01', end='2024-09-30', freq='D')

# Create demo data: simulating stock prices
np.random.seed(42)  # For reproducibility
prices = np.random.normal(loc=100, scale=5, size=len(dates)).cumsum()

# Create DataFrame
data = pd.DataFrame(data={'Date': dates, 'Stock Price': prices})

# Set Date as the index
data.set_index('Date', inplace=True)


# Plotting the time series data with Plotly
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=data.index,
    y=data['Stock Price'],
    mode='lines+markers',
    name='Stock Price',
    marker=dict(color='blue')
))

fig1.update_layout(
    title='Simulated Daily Stock Prices for September 2024',
    xaxis_title='Date',
    yaxis_title='Stock Price',
    template='plotly_white'
)

fig1.show()


# Resample the data to weekly frequency and calculate the mean
weekly_data = data.resample('W').mean()

# Plotting the weekly average stock prices
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=weekly_data.index,
    y=weekly_data['Stock Price'],
    mode='lines+markers',
    name='Weekly Average Stock Price',
    marker=dict(color='orange')
))

fig2.update_layout(
    title='Weekly Average Stock Prices for September 2024',
    xaxis_title='Date',
    yaxis_title='Weekly Average Stock Price',
    template='plotly_white'
)

fig2.show()


# Plotting the observed stock prices for decomposition
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=data.index,
    y=data['Stock Price'],
    mode='lines+markers',
    name='Observed',
    marker=dict(color='green')
))

fig3.update_layout(
    title='Observed Stock Prices',
    xaxis_title='Date',
    yaxis_title='Stock Price',
    template='plotly_white'
)

fig3.show()


# Calculate moving average
data['Moving Average'] = data['Stock Price'].rolling(window=7).mean()

# Plotting original and moving average
fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=data.index,
    y=data['Stock Price'],
    mode='lines+markers',
    name='Original',
    line=dict(color='blue', width=2),
    marker=dict(size=5)
))

fig4.add_trace(go.Scatter(
    x=data.index,
    y=data['Moving Average'],
    mode='lines',
    name='7-Day Moving Average',
    line=dict(color='orange', width=2)
))

fig4.update_layout(
    title='Stock Prices with 7-Day Moving Average',
    xaxis_title='Date',
    yaxis_title='Stock Price',
    template='plotly_white'
)

fig4.show()

_____________________________________________

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Step 1: Create Sample Data
dates = pd.date_range(start='2024-01-01', periods=20)
np.random.seed(42)
prices = np.random.randint(100, 200, size=len(dates))

data = pd.DataFrame(data={'Date': dates, 'Price': prices})
data.set_index('Date', inplace=True)

# Step 2: Calculate Moving Averages
data['SMA_3'] = data['Price'].rolling(window=3).mean()  # 3-day SMA
data['SMA_5'] = data['Price'].rolling(window=5).mean()  # 5-day SMA
weights = np.arange(1, 4)  # For a 3-day window
data['WMA_3'] = data['Price'].rolling(window=3).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

# Step 3: Plot with Plotly
fig = go.Figure()

# Add traces for original prices and moving averages
fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines+markers', name='Original Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_3'], mode='lines+markers', name='3-Day SMA', line=dict(dash='dash', color='orange')))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], mode='lines+markers', name='5-Day SMA', line=dict(dash='dash', color='green')))
fig.add_trace(go.Scatter(x=data.index, y=data['WMA_3'], mode='lines+markers', name='3-Day WMA', line=dict(dash='dot', color='red')))

# Update layout
fig.update_layout(
    title='Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Legend',
    template='plotly_white'
)

# Show the plot
fig.show()

__________________________________________________________

pip install pyarrow

import pyarrow as pa

# Create an Arrow array
data = pa.array([1, 2, 3, 4, 5])
print(data)

# Create an Arrow table
table = pa.table({'column1': data, 'column2': pa.array(['A', 'B', 'C', 'D', 'E'])})
print(table)

import pyarrow.parquet as pq

# Create a sample Arrow table
data = {
    'column1': pa.array([1, 2, 3]),
    'column2': pa.array(['A', 'B', 'C'])
}
table = pa.table(data)

# Write the table to a Parquet file
pq.write_table(table, 'example.parquet')

# Read the table back from the Parquet file
read_table = pq.read_table('example.parquet')
print(read_table)

import pandas as pd

# Create a sample Pandas DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['X', 'Y', 'Z']
})

# Convert Pandas DataFrame to Arrow Table
arrow_table = pa.Table.from_pandas(df)
print(arrow_table)

# Convert Arrow Table back to Pandas DataFrame
df_from_arrow = arrow_table.to_pandas()
print(df_from_arrow)

# Create an Arrow array
data = pa.array([1, 2, 3, 4, 5])

# Convert to NumPy array without copying data
numpy_array = data.to_numpy(zero_copy_only=True)
print(numpy_array)

_______________________________________________________________________




