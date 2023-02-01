import streamlit as st
import json
from pathlib import Path
from PIL import Image
import requests
from streamlit_lottie import st_lottie

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottiefile("pages/contact/lottiefiles/coding.json")
lottie_hello = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ldulgcir.json")

st_lottie(
   lottie_coding,
   speed=1,
   reverse=False,
   loop=True,
   quality="high",
   height=600,
   width=None,
   key=None,
)

st.header("\n")
st.header(":mailbox: CONTACT ME")
st.subheader("")

contact_form = """
<form action="https://formsubmit.co/bigdatanigel1513@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder ="이름" required>
     <input type="email" name="email" placeholder ="이메일" required>
     <input type="text" name="phon_number" placeholder ="연락처" required>
     <textarea name="message" placeholder="문의사항"></textarea>
     <button type="submit">Send</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
local_css("pages/contact/contact_style/style.css")
