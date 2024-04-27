import streamlit as st
import joblib
import numpy as np
import os
# from sklearn.preprocessing import StandardScaler

# print("Current Working Directory:", os.getcwd())

# Load the machine learning model
model = joblib.load('rf_model.pickle')

def main():
    st.title('Machine Learning Model Deployment')

    # Add user input components for 4 features
    Creditscore = st.slider('CreditScore', min_value=0.0, max_value=860.0, value=0.5)
    Age = st.slider('Age', min_value=0.0, max_value=95.0, value=1.0)
    Balance = st.slider('Balance', min_value=0.0, max_value=250000.0, value=0.5)
    EstimatedSalary = st.slider('EstimatedSalary', min_value=0.0, max_value=200000.0, value=0.5)
    
    # Add a dropdown menu for categorical prediction
    Geography = ['Spain', 'France', 'Germany']
    selected_category1 = st.selectbox('Select Geography', Geography)
    
    Gender = ['Female', 'Male']
    selected_category2 = st.selectbox('Select Gender', Gender)
    
    Tenure = ['0','1','2','3','4','5','6','7','8','9','10']
    selected_category3 = st.selectbox('Select Tenure', Tenure)

    NumOfProducts = ['1','2','3','4']
    selected_category4 = st.selectbox('Select NumOfProducts', NumOfProducts)

    HasCrCard = ['0','1']
    selected_category5 = st.selectbox('Select HasCrCard', HasCrCard)

    IsActiveMember = ['0','1']
    selected_category6 = st.selectbox('Select IsActiveMember', IsActiveMember)

    if st.button('Make Prediction'):
        features = [Creditscore,Age,Balance,EstimatedSalary,selected_category1,selected_category2,selected_category3,selected_category4,selected_category5
                    ,selected_category6]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# def revert_scaling(features):
#     # Revert scaling for numerical features
#     scaled_features = np.array(features[:4]).reshape(1, -1)
#     original_features = scaler.inverse_transform(scaled_features)
#     features[:4] = original_features.flatten()
#     return features

if __name__ == '__main__':
    main()
