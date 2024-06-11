import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import nltk
import re
from nltk.corpus import stopwords
stopword = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

import nltk
nltk.download('stopwords')

# Load data
data = pd.read_csv("data.csv")
print(data.head())

# Map class labels to text labels
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]

# Clean text function
def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w\d\w', '', text)
    
    # Fix the line below to use split() without an argument
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    
    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)
    
    return text

# Apply cleaning function to tweets
data["tweet"] = data["tweet"].apply(clean)

# Prepare data for model
x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluate model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Example prediction
i = "mother"
i = cv.transform([i]).toarray()
print(model.predict(i))

# Save model and vectorizer as pickle file
import pickle

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
    
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("Model and vectorizer saved as pickle files.")
