import pandas as pd
import pickle as pickle
import json as json

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

d=pd.read_csv("./public_survey_results.csv")
d=d[["Country","EdLevel","YearsCodePro","ConvertedCompYearly"]]
d=d.dropna()
d=d.rename({"ConvertedCompYearly":"Salary"},axis=1)

def country_categories(categories,cutoff):
    categories_map={}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categories_map[categories.index[i]]=categories.index[i]
        else:
            categories_map[categories.index[i]]="Other"
    return categories_map

country_map=country_categories(d["Country"].value_counts(), 400)
d["Country"]=d["Country"].map(country_map)

d=d[d["Salary"]<=230000]
d=d[d["Salary"]>5000]
d=d[d["Country"]!="Other"]
d["Country"].unique()

def cvtDegree(dgree):
    if "Bachelor’s degree" in dgree:
        return "Bachelor’s degree"
    elif "Master’s degree" in dgree:
        return "Master’s degree"
    elif "Professional degree" in dgree:
        return "Phd"
    elif "Primary/elementary school" in dgree:
        return "Primary School"
    else:
        return "Other"

def cvtExperience(dgree):
    if "Less than 1 year" == dgree:
        return float("0.5")
    elif 'More than 50 years' == dgree:
        return float(50.0)
    else:
        return float(dgree)

d["EdLevel"]=d["EdLevel"].apply(cvtDegree)
d["YearsCodePro"]=d["YearsCodePro"].apply(cvtExperience)
d["Country"].unique()
d["EdLevel"].unique()

lc_country=LabelEncoder()
lc_edu=LabelEncoder()
d["Country"]=lc_country.fit_transform(d["Country"])
d["EdLevel"]=lc_edu.fit_transform(d["EdLevel"])
output=d["Salary"]
d=d.drop("Salary",axis=1)

# model training
print(f'Model training... \n')

lrg=LinearRegression()
lrg.fit(d,output.values)
output_train=lrg.predict(d)
error=np.sqrt(mean_squared_error(output,output_train))
print(f'Linear Regression Error: ${error} \n')

dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(d,output.values)
output_train=dtr.predict(d)
error=np.sqrt(mean_squared_error(output,output_train))
print(f'Decision Tree Error: ${error} \n')

rfr=RandomForestRegressor(random_state=0)
rfr.fit(d,output.values)
output_train=rfr.predict(d)
error=np.sqrt(mean_squared_error(output,output_train))
print(f'Random Forest Error: ${error} \n')

parameters={"max_depth":[None,2,4,6,8,10,12]}
dtr=DecisionTreeRegressor(random_state=0)
gdv=GridSearchCV(dtr,parameters,scoring="neg_mean_squared_error")
gdv.fit(d,output.values)
gdv_o=gdv.best_estimator_
gdv_o.fit(d,output.values)
output_train=gdv_o.predict(d)
error=np.sqrt(mean_squared_error(output,output_train))
print(f'Model is using Decision Tree Regression (${gdv_o.get_depth}) \n')
print(f'Current Model Error: ${error}')

arr=np.array([['India',"Bachelor’s degree",1]])
print(lc_country.transform(arr[:,0]))
arr[:,0]=lc_country.transform(arr[:,0])
arr[:,1]=lc_edu.transform(arr[:,1])
output_train=gdv_o.predict(arr)

file=open("./training_model.pickle","ab")
dic={"model":gdv_o,"lc_country":lc_country,"lc_edu":lc_edu,"dataset":d,"output":output}
pickle.dump(dic,file)
file.close()