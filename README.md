Hate Speech Detection App


This Hate Speech Detection App is a simple tool built with Streamlit that allows users to input a tweet and check if it contains hate speech, offensive speech, or neither. The app utilizes a trained Decision Tree Classifier model and a CountVectorizer to make predictions.

How to Use
Clone the Repository: Clone this repository to your local machine using the following command:

git clone <repository-url>
Install Dependencies: Navigate to the project directory and install the required dependencies using pip:

pip install -r requirements.txt
Run the App: Start the Streamlit app by running the following command:

streamlit run app.py
Interact with the App: Once the app is running, you can access it through your web browser. Enter a tweet in the provided text area and click the "Predict" button to see the prediction.

Requirements
Python 3.x
Streamlit
scikit-learn
pandas
numpy
nltk
Files
hatey.py: Contains the Streamlit application code.
model.pkl: Pickled Decision Tree Classifier model.
vectorizer.pkl: Pickled CountVectorizer.

How it Works
The app loads the pre-trained model and vectorizer from the pickle files (model.pkl and vectorizer.pkl). Users can input a tweet, which is then cleaned and transformed using the same preprocessing steps used during model training. The transformed text is passed to the model for prediction, and the result is displayed on the app interface.
