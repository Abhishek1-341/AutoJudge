import streamlit as st
from inference import predict_problem

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.write("Predict programming problem difficulty using ML")

title = st.text_input("Problem Title")
desc = st.text_area("Problem Description", height=200)
inp = st.text_area("Input Description", height=150)
out = st.text_area("Output Description", height=150)

if st.button("Predict"):
    if desc.strip() == "":
        st.warning("Please enter the problem description.")
    else:
        with st.spinner("Analyzing..."):
            difficulty, score = predict_problem(title, desc, inp, out)

        st.success(f"Difficulty: **{difficulty}**")
        st.info(f"Difficulty Score: **{score}**")
