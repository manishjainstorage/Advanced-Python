
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
