# 🏏 Interactive IPL Match Winner Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black.svg?logo=flask)](https://palletsprojects.com/p/flask/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg?logo=bootstrap)](https://getbootstrap.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-coral.svg?logo=chart.js)](https://www.chartjs.org/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-85.87%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-93.22%25-success.svg)]()

An interactive, full-stack machine learning web application that calculates **calibrated winning probabilities** for Indian Premier League (IPL) matches during the second innings (the run chase). The project utilizes a calibrated **Logistic Regression** pipeline, feature-engineered cricket dynamics (Required Run Rate, Current Run Rate, Wickets in hand), and historical match data across 16+ seasons to deliver instantaneous probability forecasts with an interactive broadcast-grade web dashboard.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Introduction](#-introduction)
3. [Problem Statement](#-problem-statement)
4. [Methodology & Machine Learning Formulation](#-methodology--machine-learning-formulation)
5. [System Architecture](#-system-architecture)
6. [Dataset Description](#-dataset-description)
7. [Experimental Results & Evaluation](#-experimental-results--evaluation)
8. [Tech Stack](#-tech-stack)
9. [Folder Structure](#-folder-structure)
10. [Setup & Installation Instructions](#-setup--installation-instructions)
11. [API Reference](#-api-reference)
12. [Future Scope](#-future-scope)
13. [Conclusion](#-conclusion)

---

## 📖 Project Overview

In Twenty20 cricket, particularly the Indian Premier League (IPL), match momentum shifts dramatically on an over-by-over basis. A chase requiring 45 runs off 30 balls can transform into an improbable contest if two wickets fall, or into an easy chase with two consecutive sixes. 

The **Interactive IPL Match Winner Prediction System** provides sports enthusiasts, data analysts, commentators, and coaches with a data-driven probabilistic model. By entering the current match state (Batting team, Bowling team, Venue, Target, Current Score, Overs bowled, and Wickets lost), the application computes:
- Exact winning percentage for both teams (summing to 100%).
- Real-time diagnostic metrics: Runs Needed, Balls Left, Current Run Rate (CRR), Required Run Rate (RRR).
- Interactive Chart.js Donut Visualization & CRR vs. RRR pressure charts.
- Natural-language match situation insights.

---

## 🎯 Introduction

The Indian Premier League (IPL), established in 2008 by the BCCI, is the world's most lucrative and competitive T20 cricket tournament. Each fixture consists of two innings of maximum 20 overs (120 legal balls) per team.

While predicting pre-match winners relies largely on team composition and historical head-to-head records, in-play prediction during the run chase (second innings) is far more demanding. The target is fixed, balls decrease monotonically, and every lost wicket reduces batting depth. Machine Learning offers an objective mechanism to quantify this contest by learning complex probability boundaries from thousands of past match situations.

---

## ❗ Problem Statement

Existing conventional metrics, such as naive run-rate comparisons or historical win-loss ratios, fail to capture the multi-dimensional pressure of a modern T20 chase:
1. **Non-Linear Resource Depletion:** A required run rate of 10.0 RPO is manageable with 8 wickets in hand, but disastrous with only 2 wickets remaining.
2. **Probability Calibration:** Tree-based models (like uncalibrated Decision Trees or Random Forests) frequently produce erratic step jumps in predicted probabilities when small inputs change.
3. **Accessibility:** Most analytical models remain locked inside Jupyter Notebooks without accessible, responsive graphical interfaces for casual users.

This project addresses these gaps by creating a statistically calibrated, transparent, and aesthetically refined web application.

---

## 🔬 Methodology & Machine Learning Formulation

### 1. Match Calculations & Feature Formulation
During an in-progress IPL run chase, the system tracks 6 key dynamic match features:
- **Runs Left to Win:** `Target Score - Current Score`
- **Balls Remaining:** `120 - Total Balls Bowled`
- **Wickets in Hand:** `10 - Wickets Fallen`
- **Current Run Rate (CRR):** `(Current Score × 6) / Balls Bowled`
- **Required Run Rate (RRR):** `(Runs Left × 6) / Balls Left`
- **Match Context:** `Batting Team`, `Bowling Team`, and `Host City` (venue dew factor).

### 2. Logistic Regression & Probability Calibration
Logistic Regression models the probability of chasing team victory using the standard Sigmoid curve:

$$\text{Probability} = \frac{1}{1 + e^{-z}}$$

Where $z$ is the weighted sum of all 9 match features learned from 16+ seasons of IPL historical data.

**Why Logistic Regression?**
Unlike uncalibrated tree-based models, Logistic Regression produces monotonic, smooth, well-calibrated probabilities. As balls decrease or wickets fall, probability shifts smoothly without erratic random jumps, matching real-world cricket game theory.

---

## 🏛 System Architecture

```
+-------------------------------------------------------------------------+
|                              USER CLIENT                                |
|        (Web Browser: Desktop / Tablet / Smartphone Interface)           |
+-------------------------------------------------------------------------+
                                     │
           HTTP GET / POST           │          JSON REST API
         (Standard Web Form)         │        (Async AJAX Call)
                                     ▼
+-------------------------------------------------------------------------+
|                           FLASK WEB SERVER                              |
|                               (app.py)                                  |
|  - Request Validation & Boundary Checking                               |
|  - Live Cricket Metrics Computation (CRR, RRR, Balls Left)              |
|  - Match Intelligence & Natural Language Commentary Generator           |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                     SCIKIT-LEARN PIPELINE (model.pkl)                   |
|                                                                         |
|  ┌───────────────────────────────────────────────────────────────────┐  |
|  │  ColumnTransformer:                                               │  |
|  │  - OneHotEncoder: [Batting Team, Bowling Team, City]              │  |
|  │  - Passthrough / Scaler: [Runs Left, Balls Left, Wkts, CRR, RRR]  │  |
|  └───────────────────────────────────────────────────────────────────┘  |
|                                    │                                    |
|                                    ▼                                    |
|  ┌───────────────────────────────────────────────────────────────────┐  |
|  │  LogisticRegression Classifier (solver='liblinear')               │  |
|  │  predict_proba() -> [P(Defending Win), P(Chasing Win)]            │  |
|  └───────────────────────────────────────────────────────────────────┘  |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                           DYNAMIC FRONTEND                              |
|  - Bootstrap 5 Responsive Grid & Glassmorphism UI                       |
|  - Chart.js Interactive Win Probability Donut & Pressure Bar Charts     |
|  - Animated Dual Progress Bars with Franchise Color Themes              |
|  - Real-time Instant Preview Calculation Ribbon                         |
+-------------------------------------------------------------------------+
```

---

## 📊 Dataset Description

The dataset resides in `dataset/matches.csv` and captures over-by-over progression across 1,200 matches with **22,181 second-innings situation rows**.

| Column Name | Data Type | Description |
|---|---|---|
| `match_id` | Integer | Unique match identifier |
| `season` | Integer | IPL Season year (2008–2024) |
| `city` | String | Host city / stadium location |
| `batting_team` | String | Franchise chasing the target |
| `bowling_team` | String | Franchise defending the target |
| `target` | Integer | Target score required to win (1st innings score + 1) |
| `current_score`| Integer | Runs accumulated by batting team at this snapshot |
| `overs` | Float | Overs completed (e.g. 14.2) |
| `wickets_fallen`| Integer | Wickets lost so far (0 to 9) |
| `runs_left` | Integer | Runs remaining to reach target |
| `balls_left` | Integer | Legal deliveries remaining out of 120 |
| `wickets_remaining` | Integer | Available wickets in hand (10 - wickets_fallen) |
| `crr` | Float | Current Run Rate ($S \times 6 / B_{\text{bowled}}$) |
| `rrr` | Float | Required Run Rate ($R_{\text{left}} \times 6 / B_{\text{left}}$) |
| `result` | Integer | **Target Label:** 1 if chasing team won, 0 if defending team won |

---

## 📈 Experimental Results & Evaluation

The dataset was split using an 80:20 stratified train-test split (17,744 training instances, 4,437 holdout test instances).

```
============================================================
           MODEL PERFORMANCE EVALUATION METRICS            
============================================================
  Model Architecture : Pipeline [OneHotEncoder + LogisticRegression]
  Test Accuracy      : 85.87%
  Precision Score    : 81.39%
  Recall Score       : 78.90%
  F1 Score           : 80.13%
  ROC-AUC Score      : 93.22%
------------------------------------------------------------
Classification Report:
                    precision    recall  f1-score   support

Defending Team Won       0.88      0.90      0.89      2835
  Chasing Team Won       0.81      0.79      0.80      1602

          accuracy                           0.86      4437
         macro avg       0.85      0.84      0.85      4437
      weighted avg       0.86      0.86      0.86      4437
============================================================
```

### Analysis of Model Coefficients
- **Required Run Rate (RRR):** Strongly negative coefficient. As RRR escalates, the probability of the batting team declines sharply.
- **Wickets Remaining:** Strongly positive coefficient. More wickets allow batting teams to absorb overs of tight bowling and launch counter-attacks in death overs.
- **Balls Left:** Positive coefficient. Extra deliveries afford tactical flexibility.
- **Venue Influence:** Chasing-friendly grounds like Wankhede (Mumbai) and Chinnaswamy (Bengaluru) show positive bias for the batting team due to dew and short boundaries.

---

## 💻 Tech Stack

- **Backend:**
  - Python 3.9+ / 3.13
  - Flask (Web framework & REST API)
  - Scikit-Learn (Machine learning pipeline & preprocessing)
  - Pandas & NumPy (Data manipulation & numerical computing)
  - Pickle (Model serialization)
- **Frontend:**
  - HTML5 & CSS3 (Glassmorphism styling, dark stadium theme)
  - Bootstrap 5.3 (Mobile-first responsive grid system)
  - JavaScript ES6+ (AJAX requests, DOM updates, real-time calculators)
  - Chart.js 4.4 (Dynamic interactive charts)
  - SVG (High-definition vector logos for all 10 IPL franchises)

---

## 📂 Folder Structure

```
IPL_Prediction/
│
├── app.py                      # Flask backend, routing, API endpoints & model inference
├── requirements.txt            # Python dependencies
├── model.pkl                   # Trained Scikit-Learn pipeline & metadata
├── train_model.py              # ML preprocessing, training & evaluation script
├── dataset/
│   └── matches.csv             # Historical IPL 2nd innings match dataset
├── templates/
│   ├── index.html              # Home page with project overview, architecture & docs
│   └── prediction.html         # Live prediction interface & interactive dashboard
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark stadium & glassmorphism stylesheet
│   ├── js/
│   │   └── script.js           # Live calculator, AJAX API caller & Chart.js renderer
│   └── images/                 # All 10 IPL team logos & tournament SVGs
│       ├── csk.svg
│       ├── mi.svg
│       ├── rcb.svg
│       ├── kkr.svg
│       ├── srh.svg
│       ├── dc.svg
│       ├── rr.svg
│       ├── pbks.svg
│       ├── gt.svg
│       ├── lsg.svg
│       ├── ipl_trophy.svg
│       └── cricket_ball.svg
└── README.md                   # Complete documentation and setup manual
```

---

## 🚀 Setup & Installation Instructions

Follow these step-by-step instructions to run the project on your local machine:

### Prerequisites
- Python 3.9 or higher (Python 3.10, 3.11, 3.12, 3.13 supported)
- `pip` package manager

### Step 1: Clone or Navigate to Project Directory
```bash
git clone https://github.com/your-username/ipl-match-winning-prediction-model.git
cd ipl-match-winning-prediction-model
```

### Step 2: (Optional) Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model
Train the Logistic Regression pipeline and generate `model.pkl`:
```bash
python train_model.py
```
*(The script will output data statistics, classification reports, and save `model.pkl` with 85.87% test accuracy.)*

### Step 5: Launch the Flask Web Application
```bash
python app.py
```

### Step 6: Open in Your Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
- Visit `http://127.0.0.1:5000/` for the **Project Home & Overview**.
- Visit `http://127.0.0.1:5000/predict` for the **Interactive Match Winner Predictor**.

---

## 📡 API Reference

The application exposes a clean REST API for programmatic predictions:

### Endpoint: `POST /api/predict`
Calculates winning probabilities given a second-innings match situation.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Payload
```json
{
  "batting_team": "Chennai Super Kings",
  "bowling_team": "Mumbai Indians",
  "city": "Hyderabad",
  "target": 150,
  "current_score": 133,
  "overs": 18.2,
  "wickets_fallen": 5
}
```

#### Response Payload (`200 OK`)
```json
{
  "success": true,
  "probabilities": {
    "batting_team": "Chennai Super Kings",
    "bowling_team": "Mumbai Indians",
    "batting_prob": 47.6,
    "bowling_prob": 52.4
  },
  "match_info": {
    "runs_left": 17,
    "balls_left": 10,
    "wickets_remaining": 5,
    "crr": 7.25,
    "rrr": 10.2
  },
  "analysis": {
    "status": "Nail-Biter / Even Contest (50-50)",
    "badge_class": "warning",
    "commentary": "This match is on a knife's edge! With 17 needed off 10 balls and 5 wickets remaining, momentum could swing on a single over.",
    "pressure_level": "Moderate"
  }
}
```

---

## 🔮 Future Scope

1. **Live Cricinfo / Cricbuzz API Integration:** Stream real-time ball-by-ball updates directly without manual score entry.
2. **Player Matchup Matrix:** Incorporate batter vs. bowler head-to-head metrics, spin vs. pace strike rates, and death-bowling economy rates.
3. **Weather & Pitch Analytics:** Factor in temperature, humidity, boundary dimensions, and stadium-specific dew accumulation curves.
4. **First Innings Projection Engine:** Predict first-innings totals based on powerplay wickets and acceleration phases.

---

## 🏁 Conclusion

The **IPL Match Winner Prediction System** successfully combines sports domain knowledge, feature engineering, and regularized logistic classification to deliver real-time, explainable winning probabilities for IPL run chases. Achieving **85.87% accuracy** and an **ROC-AUC of 93.22%**, it illustrates how machine learning can transform raw sports data into actionable, visual intelligence for analysts and cricket fans alike.

---

## 👨‍💻 Author & Attribution
- **Project Title:** IPL Match Winner Prediction System
- **Technologies:** Python, Flask, Scikit-learn, Bootstrap 5, Chart.js, HTML5/CSS3/JavaScript
- **Coursework / Level:** College Final Year / Machine Learning Capstone Level
- **License:** Open Source for Educational and Research Purposes

