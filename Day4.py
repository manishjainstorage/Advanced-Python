
import plotly.express as px

fig = px.line(flights, x="month", y="passengers", color="year", title="Number of Passengers Over Time")
fig.update_layout(
    yaxis_title="Number of Passengers"
)
fig.show()
