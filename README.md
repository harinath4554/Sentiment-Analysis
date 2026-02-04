# Sentiment Analysis Web Application

This project is a Flask-based web application for performing sentiment analysis on user-provided text reviews.  
The application predicts whether a given review is **Positive** or **Negative** along with a confidence score.

The trained machine learning model is deployed on **AWS EC2** and accessed through a web interface.

---

## Features
- Accepts user input text (reviews)
- Performs text preprocessing using NLTK
- Uses a trained TF-IDF + Logistic Regression model
- Returns sentiment prediction with confidence percentage
- Deployed on AWS EC2 using Flask

---

## Project Structure


<pre>
Sentiment-Analysis/
├── app/
│   └── flask_app.py
├── src/
│   ├── preprocess.py
│   └── train_model.py
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── data/
│   └── data.csv
├── requirements.txt
└── README.md
</pre>



---

## Technologies Used
- Python
- Flask
- Scikit-learn
- NLTK
- Pandas & NumPy
- AWS EC2

---

## How to Run Locally

1. Clone the repository:

git clone <your-github-repo-url>
cd Sentiment-Analysis

2. Install dependencies:

pip install -r requirements.txt

3. Run the Flask application:

python app/flask_app.py

4. Open your browser and go to:

http://127.0.0.1:5000

AWS Deployment

The application is deployed on an AWS EC2 instance.

Deployment Link:

http://13.60.186.53:5000

Note: The link works only while the EC2 instance and Flask server are running.

