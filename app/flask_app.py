from flask import Flask, request, jsonify, render_template_string
import joblib
import sys

sys.path.append("src")
from preprocess import clean_text

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_sentiment_with_confidence(text):
    clean_review = clean_text(text)
    vector = vectorizer.transform([clean_review])

    # Get prediction probabilities
    proba = model.predict_proba(vector)[0]

    # Probability of positive class (label = 1)
    positive_prob = proba[1]

    # Threshold tuned for text sentiment
    sentiment = "Positive" if positive_prob >= 0.4 else "Negative"

    return sentiment, round(positive_prob * 100, 2)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        review = request.form["review"]
        result, confidence = predict_sentiment_with_confidence(review)

    return render_template_string("""
        <html>
        <head>
            <title>Sentiment Analysis Website</title>
        </head>
        <body>
            <h2>Sentiment Analysis Website</h2>

            <form method="post">
                <textarea name="review" rows="5" cols="60"
                placeholder="Enter your review here"></textarea><br><br>
                <button type="submit">Analyze Sentiment</button>
            </form>

            {% if result %}
                <h3>Sentiment: {{ result }}</h3>
                <p>Positive confidence: {{ confidence }}%</p>
            {% endif %}
        </body>
        </html>
    """, result=result, confidence=confidence)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")

    sentiment, confidence = predict_sentiment_with_confidence(review)

    return jsonify({
        "sentiment": sentiment,
        "positive_confidence_percent": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)
