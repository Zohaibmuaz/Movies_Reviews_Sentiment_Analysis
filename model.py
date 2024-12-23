import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv("movie_reviews.csv")

# Data cleaning function
def clean_text(text):
    text = re.sub(r'\b\d+\b', '', text) # remove numbers
    text = re.sub(r'\W', ' ', text)     # remove special characters
    text = re.sub(r'\s+', ' ', text)    # remove extra spaces
    text = text.lower().strip()         # convert to lowercase and strip whitespace
    return text

# Apply the cleaning function to the 'review' column
data['cleaned_review'] = data['review'].apply(clean_text)

# Display the first few cleaned reviews
data[['cleaned_review', 'sentiment']].head()
print(data)

x = data['cleaned_review']
y = data['sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Fit and transform the training data, transform the test data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Print classification report and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

