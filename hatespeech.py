import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Load the saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# Initialize stopwords and stemmer
stopword = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

# Function to clean the input text
def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w\d\w', '', text)
    
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    
    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)
    
    return text

# Streamlit UI
st.title("Hate Speech Detection")

user_input = st.text_area("Enter a tweet to check if it's hate speech:")

if st.button("Predict"):
    cleaned_input = clean(user_input)
    transformed_input = cv.transform([cleaned_input]).toarray()
    prediction = model.predict(transformed_input)
    
    if prediction == "Hate Speech":
        st.error("This tweet contains hate speech.")
    elif prediction == "Offensive Speech":
        st.warning("This tweet contains offensive speech.")
    else:
        st.success("This tweet does not contain hate or offensive speech.")
