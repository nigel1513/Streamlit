import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
from ipyvizzu import Chart, Data, Config, Style


# 다른 데이터로 다시 예시 파일 만들기.
st.title('🎈 ipyvizzu')

def create_chart():

    # initialize chart
    chart = Chart(width="640px", height="360px", display="manual")

    # add data
    data = Data()
    data_frame = pd.read_csv("https://github.com/vizzuhq/ipyvizzu/raw/main/docs/examples/stories/titanic/titanic.csv")
    data.add_data_frame(data_frame)

    chart.animate(data)

    # add config
    chart.animate(Config({"x": "Count", "y": "Sex", "label": "Count","title":"Passengers of the Titanic"}))
    chart.animate(Config({"x": ["Count","Survived"], "label": ["Count","Survived"], "color": "Survived"}))
    chart.animate(Config({"x": "Count", "y": ["Sex","Survived"]}))

    # add style
    chart.animate(Style({"title": {"fontSize": 35}}))
    st.write(data_frame)
    return chart._repr_html_()


CHART = create_chart()
html(CHART, width=650, height=370)
