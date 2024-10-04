
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
