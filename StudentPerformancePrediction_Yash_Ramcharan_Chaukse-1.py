"""
Industrial Internship Project
Student Performance Prediction & Early Risk Detection Using Machine Learning

Student : Yash Ramcharan Chaukse
Domain  : Data Science and Machine Learning
Project : Student Performance Prediction & Early Risk Detection

The program creates a reproducible synthetic dataset, prepares the data,
trains Logistic Regression and Random Forest classifiers, evaluates them,
and provides an example prediction.

The dataset is synthetic and is used only for an educational prototype.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


RANDOM_SEED = 42


def create_dataset(n_records=1200):
    """Create a reproducible synthetic student dataset."""
    rng = np.random.default_rng(RANDOM_SEED)

    attendance = rng.uniform(35, 100, n_records)
    study_hours = rng.uniform(0.5, 9, n_records)
    assignment_score = rng.uniform(20, 100, n_records)
    internal_score = rng.uniform(15, 100, n_records)
    previous_gpa = rng.uniform(3.5, 10, n_records)
    sleep_hours = rng.uniform(4, 10, n_records)
    quiz_score = rng.uniform(15, 100, n_records)

    # Synthetic weighted score + noise.
    score = (
        0.25 * attendance
        + 2.5 * study_hours
        + 0.15 * assignment_score
        + 0.18 * internal_score
        + 4.0 * previous_gpa
        + 1.5 * sleep_hours
        + 0.12 * quiz_score
        + rng.normal(0, 10, n_records)
    )

    # Median threshold creates a balanced binary demonstration target.
    passed = (score >= np.median(score)).astype(int)

    return pd.DataFrame({
        "attendance": attendance,
        "study_hours": study_hours,
        "assignment_score": assignment_score,
        "internal_score": internal_score,
        "previous_gpa": previous_gpa,
        "sleep_hours": sleep_hours,
        "quiz_score": quiz_score,
        "passed": passed,
    })


def evaluate_model(name, model, x_test, y_test):
    """Print standard binary-classification metrics."""
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    return predictions


def main():
    # 1. Generate data.
    df = create_dataset()

    print("Dataset shape:", df.shape)
    print("\nFirst five records:")
    print(df.head())

    # 2. Basic data-quality check.
    print("\nMissing values:")
    print(df.isnull().sum())

    # 3. Separate features and target.
    features = [
        "attendance",
        "study_hours",
        "assignment_score",
        "internal_score",
        "previous_gpa",
        "sleep_hours",
        "quiz_score",
    ]

    X = df[features]
    y = df["passed"]

    # 4. Stratified 80:20 train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # 5. Baseline model.
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    logistic_predictions = evaluate_model(
        "Logistic Regression", logistic_model, X_test, y_test
    )

    # 6. Main ensemble model.
    random_forest_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        random_state=RANDOM_SEED,
    )
    random_forest_model.fit(X_train, y_train)
    rf_predictions = evaluate_model(
        "Random Forest", random_forest_model, X_test, y_test
    )

    # 7. Confusion matrix for the main model.
    print("\nRandom Forest confusion matrix:")
    print(confusion_matrix(y_test, rf_predictions))

    # 8. Example inference for a new student.
    new_student = pd.DataFrame([{
        "attendance": 82,
        "study_hours": 5.5,
        "assignment_score": 78,
        "internal_score": 75,
        "previous_gpa": 8.0,
        "sleep_hours": 7,
        "quiz_score": 80,
    }])

    predicted_class = random_forest_model.predict(new_student)[0]
    probability = random_forest_model.predict_proba(new_student)[0].max()

    label = "ON TRACK" if predicted_class == 1 else "AT RISK"

    print("\nExample student prediction:")
    print("Prediction :", label)
    print(f"Confidence : {probability:.2%}")

    print("\nNote:")
    print("This project uses synthetic data for educational demonstration.")
    print("Predictions must not be treated as final academic decisions.")


if __name__ == "__main__":
    main()
