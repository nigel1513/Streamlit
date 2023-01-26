# pip install streamlit-authenticator
import yaml
from yaml.loader import SafeLoader
import streamlit as st  
import streamlit_authenticator as stauth  


st.set_page_config(
page_title='login_page',
layout='centered',
page_icon="🔑")

    
with open("./config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
    
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"]
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.title('Start Data Analysis 🥳')
    st.header('hi')
    st.subheader("asdf") 
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
