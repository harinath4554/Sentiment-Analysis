import sys
sys.path.append("src")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from preprocess import clean_text


def main():
    df = pd.read_csv("data/data.csv")

    # Use review text
    df["review"] = df["Review_Text"].astype(str)

    # Create sentiment labels
    # 4,5 -> Positive (1)
    # 1,2,3 -> Negative (0)
    df["sentiment"] = df["Reviewer_Rating"].apply(
        lambda x: 1 if x >= 4 else 0
    )

    # Remove very short reviews
    df = df[df["review"].str.len() > 10]

    # Balance dataset (equal positives & negatives)
    positive = df[df["sentiment"] == 1]
    negative = df[df["sentiment"] == 0]

    min_size = min(len(positive), len(negative))
    positive = positive.sample(min_size, random_state=42)
    negative = negative.sample(min_size, random_state=42)

    df = pd.concat([positive, negative])

    print("Balanced sentiment distribution:")
    print(df["sentiment"].value_counts())

    # Clean text
    df["clean_review"] = df["review"].apply(clean_text)

    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 🔑 IMPORTANT CHANGE: NO class_weight
    model = LogisticRegression(max_iter=2000)

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    f1 = f1_score(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("Model retrained successfully!")


if __name__ == "__main__":
    main()
