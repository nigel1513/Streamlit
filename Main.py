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
page_title='Data Sientist?',
page_icon="🚀", initial_sidebar_state="collapsed"
)

#add_page_title()

#show_pages([
#    Page("Streamlit_app.py", "Home", "🏰"),
#    Section(name="EDA & Preprocessing", icon="🩺"),
#    Page("pages/EDA/EDA.py", "EDA", "📋"),
#    Page("pages/EDA/PRE-PROCESSING.py", "Preprocessing", "📝"),      
#    Section(name="Visualization", icon="📊"),
#   Page("pages/VISUALIZATION/VISUALIZATION.py", "Visualization", "📈"),
#    Page("pages/VISUALIZATION/ipyvizzu.py", "Ipyvizzu", "📉"),    
#    Section(name="Auto-ML", icon="🤖"),
#    Page("pages/AUTO-ML/Auto-ML.py", "Pycaret", "🥕"),
#    Page("pages/AUTO-ML/ChatGPT.py", "Chat-Bot", "👨‍💻"),
#    Section(name="Project", icon="🧪"),
#    Page("pages/PROJECT/Football Player Dashboard.py", "Football-Dashboard", "⚽"),
#    Section(name="Contact", icon= ":mailbox:"),
#   Page("pages/Contact/CONTACT.py", "Contact", "📞"),
#    Page("pages/Contact/ABOUTME.py", "About me", "🙂"),
# ])


st.title('WELCOME Data Analysis page🥳 ')
st.text("Data Analysis page") 

st.markdown(hide_menu, unsafe_allow_html=True)

