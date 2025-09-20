import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#from sklearn.neighbors import KNeighborsClassifier   


st.set_page_config(page_title="Credit Card Fraud Message Detector", layout="centered")

st.title("💳 Credit Card Fraud Message Detector")

df = pd.read_csv("data.csv")

model = LogisticRegression(max_iter=1000, class_weight="balanced")


#model = KNeighborsClassifier(n_neighbors=5)
df['label_num'] = df['label'].map({'fraud': 0, 'safe': 1})        
x = df['message']
y = df['label_num']

X_train, X_test, y_train, y_act = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
joblib.dump((vectorizer, model), "fraud_model.pkl")



model.fit(X_train_vec, y_train)


#_____main_____

user_input = st.text_input(label="Enter the message to check for fraud:")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        user_input_vec = vectorizer.transform([user_input])
        pre = model.predict(user_input_vec)[0]
        st.write(pre)   
        prediction = model.predict(user_input_vec)[0]
        prediction.round(2)
        proba = model.predict_proba(user_input_vec)[0]
        fraud_score = proba[0]  # probability of fraud
        safe_score = proba[1]   # probability of safe

        st.write(f"Fraud Probability: {fraud_score:.2f}")
        st.write(f"Safe Probability: {safe_score:.2f}")
        vectorizer, model = joblib.load("fraud_model.pkl")


        if prediction == 1:
            st.error("✅ This is a SAFE message.")
            
        else:
            st.success("⚠️ This is a FRAUD message.")
