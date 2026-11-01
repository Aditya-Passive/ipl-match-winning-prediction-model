"""
=============================================================================
IPL Match Winner Prediction System - Model Training Script
=============================================================================
This script performs:
1. Historical IPL match data loading from dataset/matches.csv
2. Data preprocessing & cleaning
3. Feature engineering (Runs Left, Balls Left, Wickets Left, CRR, RRR)
4. Train/Test split
5. Pipeline construction using OneHotEncoder & LogisticRegression
6. Model training & comprehensive evaluation (Accuracy, Precision, Recall, ROC-AUC)
7. Serialization of the trained model into 'model.pkl' using pickle
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def load_and_preprocess_data(dataset_path='dataset/matches.csv'):
    """
    Loads historical match dataset and performs preprocessing & feature engineering.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please verify the path.")

    print(f"[1/5] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    initial_rows = len(df)
    print(f"      Initial record count: {initial_rows:,}")

    # Remove invalid / edge records
    df = df.dropna(subset=['batting_team', 'bowling_team', 'city', 'target', 'current_score', 'overs', 'wickets_fallen', 'result'])
    df = df[df['batting_team'] != df['bowling_team']]
    df = df[df['target'] > 0]
    df = df[(df['overs'] > 0) & (df['overs'] <= 20)]
    df = df[(df['wickets_fallen'] >= 0) & (df['wickets_fallen'] <= 10)]

    print(f"[2/5] Performing Feature Engineering...")
    # Calculate fundamental chase dynamics
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_left'] = 120 - (df['overs'] * 6)
    df['wickets_remaining'] = 10 - df['wickets_fallen']

    # Filter out completed matches or invalid states
    df = df[(df['runs_left'] > 0) & (df['balls_left'] > 0)]

    # Compute Run Rates
    # CRR = (Current Score * 6) / (Balls Bowled)
    df['crr'] = np.round((df['current_score'] * 6) / (120 - df['balls_left']), 2)
    # RRR = (Runs Left * 6) / Balls Left
    df['rrr'] = np.round((df['runs_left'] * 6) / df['balls_left'], 2)

    # Ensure no infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['crr', 'rrr'])

    print(f"      Preprocessed record count: {len(df):,} ({initial_rows - len(df):,} filtered)")
    return df

def train_and_evaluate(df):
    """
    Builds the ML pipeline, trains the Logistic Regression model,
    and displays evaluation metrics.
    """
    # Select predictive features
    feature_cols = [
        'batting_team',
        'bowling_team',
        'city',
        'runs_left',
        'balls_left',
        'wickets_remaining',
        'target',
        'crr',
        'rrr'
    ]

    X = df[feature_cols]
    y = df['result'].astype(int)

    # Extract distinct teams and cities
    teams = sorted(list(set(df['batting_team'].unique()) | set(df['bowling_team'].unique())))
    cities = sorted(list(df['city'].unique()))

    print(f"[3/5] Dataset Overview:")
    print(f"      Total unique teams : {len(teams)}")
    print(f"      Total unique venues: {len(cities)}")
    print(f"      Class balance (1=Batting Win, 0=Defending Win):")
    print(f"        - Chasing Team Won : {(y == 1).sum():,} ({(y == 1).mean() * 100:.2f}%)")
    print(f"        - Defending Team Won: {(y == 0).sum():,} ({(y == 0).mean() * 100:.2f}%)")

    # Split dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training samples: {len(X_train):,}")
    print(f"      Testing samples : {len(X_test):,}")

    # Step 4: Construct Pipeline with Preprocessing & Logistic Regression
    print(f"[4/5] Constructing Pipeline & Training Logistic Regression Model...")
    categorical_features = ['batting_team', 'bowling_team', 'city']

    # OneHotEncoding categorical variables with handle_unknown='ignore' for robust inference
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Logistic Regression with liblinear solver for reliable probability calibration
    model = LogisticRegression(
        solver='liblinear',
        penalty='l2',
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('step1', preprocessor),
        ('step2', model)
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate model performance
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("           MODEL PERFORMANCE EVALUATION METRICS            ")
    print("=" * 60)
    print(f"  Model Architecture : Pipeline [OneHotEncoder + LogisticRegression]")
    print(f"  Test Accuracy      : {accuracy * 100:.2f}%")
    print(f"  Precision Score    : {precision * 100:.2f}%")
    print(f"  Recall Score       : {recall * 100:.2f}%")
    print(f"  F1 Score           : {f1 * 100:.2f}%")
    print(f"  ROC-AUC Score      : {roc_auc * 100:.2f}%")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Defending Team Won', 'Chasing Team Won']))
    print("=" * 60)

    # Package model metadata
    model_payload = {
        'model': pipeline,
        'teams': teams,
        'cities': cities,
        'features': feature_cols,
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'roc_auc': round(roc_auc * 100, 2)
    }

    return model_payload

def save_model(model_payload, output_path='model.pkl'):
    """
    Serializes the trained pipeline and metadata into a pickle file.
    """
    print(f"[5/5] Saving model to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(model_payload, f)
    print(f"[OK] Model successfully trained and saved to '{output_path}'!")

def main():
    print("=" * 60)
    print("       IPL MATCH WINNER PREDICTION - MODEL TRAINING        ")
    print("=" * 60)
    df = load_and_preprocess_data()
    payload = train_and_evaluate(df)
    save_model(payload)
    print("\nTraining completed successfully! You can now start 'app.py'.\n")

if __name__ == '__main__':
    main()
