import math
import numpy as np
import pandas as pd
import streamlit as st
import pickle as pickle
import json as json

file=open("./training_model.pickle", "rb")
file=pickle.load(file)
model=file["model"]
d=file["dataset"]
lc_country=file["lc_country"]
lc_edu=file["lc_edu"]
colums=d.columns
st.area_chart((pd.DataFrame(columns=colums,data=d)))
st.header("Software Developer Salary Prediction")

form=st.form("Fill out and Predict Salary of a Developer")
country=form.selectbox("Select your Country",options=['United States of America',
 'United Kingdom of Great Britain and Northern Ireland' ,'Finland',
 'Australia', 'Netherlands' ,'Germany', 'Sweden' ,'France' ,'Spain', 'Brazil',
 'Portugal' ,'Italy' ,'Canada' ,'Switzerland' ,'India' ,'Austria', 'Norway',
 'Russian Federation' ,'Poland' ,'Belgium', 'Denmark' ,'Israel' ,'Ukraine',
 'Czech Republic' ,'Romania' ,'New Zealand'])
edu_level=form.selectbox("Select Education Level",options=['Bachelor’s degree', 'Primary School', 'Master’s degree', 'Phd', 'Other' ])
years=form.slider("Select years of Experience",min_value=0,max_value=40)
submit=form.form_submit_button("Predict")


if submit:
 arr=np.array([[country,edu_level,years]])
 arr[:,0]=lc_country.transform(arr[:,0])
 arr[:,1]=lc_edu.transform(arr[:,1])
 ans=model.predict(arr)
 st.subheader(f"Salary is ${math.ceil(ans[0])}")