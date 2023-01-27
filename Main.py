## -- 라이브러리 불러오기 -- ##
import pandas as pd
import streamlit as st
from st_pages import Page
from st_pages import Section
from st_pages import show_pages
from st_pages import add_page_title
import numpy as np

#-- 햄버거 메뉴 안보이게 하기 / footer visibility:hidden; 처리하면 밑에 저작권 표시 지울 수 있음--
hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
}

footer
{ visibility:visible;   
}

</style>
"""


# -- 메인페이지 시작 -- 

st.set_page_config(
page_title='Data Analysis',
page_icon="🚀", initial_sidebar_state="collapsed")


show_pages([
    Page("Main.py", "Home", "🏠"),
    #Section(name="Data Analysis", icon="📘"),
    page("pages/Collection.py", "Collection", "📚")
    Page("pages/EDA.py", "EDA", "🔎"),
    Page("pages/Preprocessing.py", "Preprocessing", "📝"),      
    Page("pages/Visualization.py", "Visualization", "📊"),
    #Section(name="Auto-ML", icon="🤖"),
    Page("pages/pycaret.py", "Pycaret", "🥕"),
    #Section(name="Project", icon="🧪"),
    Page("pages/football.py", "Soccer-Dashboard", "⚽"),
    Page("pages/ipyvizzu.py", "Ipyvizzu", "🎈"),  
    #Section(name="Contact", icon= ":mailbox:"),
    Page("pages/Contact.py", "Contact", "📞"),
 ])


st.title('WELCOME Data Analysis page🥳 ')
st.text("Data Analysis page") 

st.markdown(hide_menu, unsafe_allow_html=True)

