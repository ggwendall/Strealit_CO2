from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.web.bootstrap as stb
import os
from joblib import load

def main():
    data_model = pd.read_csv("data_model.csv")
    minmin,minmax,maxmin,maxmax=data_model['masse_ordma_min'].min(),data_model['masse_ordma_min'].max(),data_model['masse_ordma_max'].min(),data_model['masse_ordma_max'].max()
    model=load('Model.pkl')
    encoder=load('Encoder.pkl')
    scaler=load('Scaler.pkl')
    dataframe=pd.read_csv('carrosseries.csv')
    st.title('Determine the CO2 of the car of your dreams ! Doctors hate me !')
    with st.form("my_form"):
        carrosserie = st.selectbox('Choisissez votre carosserie', dataframe['Carrosserie'].unique())
        masse_min=st.slider("Masse Minimale Admissible", min_value=minmin, max_value=minmax, value=minmin)
        masse_max=st.slider("Masse Maximale Autorisée", min_value=maxmin, max_value=maxmax, value=maxmin)
        submitted=st.form_submit_button("Prédire le CO2")
        if submitted:
            carrosserie=encoder.transform([carrosserie])
            X = np.array([carrosserie[0],masse_min,masse_max])
            X = X.reshape(1, -1)
            X = scaler.transform(X)
            y_predict=model.predict(X).round(2)
            st.title("CO2 : "+str(y_predict[0]))
            st.write(model.score(X, y_predict))

if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        streamlit_script_path = os.path.join(os.getcwd(), "Main.py")
        stb.run(streamlit_script_path,'',[],flag_options=[])
        main()