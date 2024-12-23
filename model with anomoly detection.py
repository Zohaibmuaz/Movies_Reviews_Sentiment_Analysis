import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report

data = pd.read_csv("movie_reviews.csv")

# Data cleaning function
def clean_text(text):
    text = re.sub(r'\b\d+\b', '', text) # remove numbers
    text = re.sub(r'\W', ' ', text)     # remove special characters
    text = re.sub(r'\s+', ' ', text)    # remove extra spaces
    text = text.lower().strip()         # convert to lowercase and strip whitespace
    return text

data['cleaned_review'] = data['review'].apply(clean_text)

print(data[['review','cleaned_review']].head())

x = data['cleaned_review']
y =  data['sentiment']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer()
x_train_norm = vectorizer.fit_transform(x_train)
x_test_norm = vectorizer.transform(x_test)

model = LogisticRegression()
model.fit(x_train_norm,y_train)
y_pred = model.predict(x_test_norm)

print(f"Accuracy = {accuracy_score(y_test,y_pred)}")
print(f"Classification = {classification_report(y_test,y_pred)}")

results = pd.DataFrame({"Actual Sentiments" : y_test,"Predicted Sentiments" : y_pred})
print(results.head())

anomolies = []
anomolies_count = 0
for i in range(len(y_test)):
    if (y_test.iloc[i] != y_pred[i]):
        anomolies.append((i,y_test.iloc[i],y_pred[i]))
        anomolies_count+=1

print(f"total anomolies are : {anomolies_count}")
print(f"anomolies are : {anomolies}")

print(len(x))