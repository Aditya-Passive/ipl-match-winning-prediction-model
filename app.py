"""
=============================================================================
IPL Match Winner Prediction System - Flask Web Application
=============================================================================
This application serves:
1. Home Page ('/') with project documentation, architecture, & model insights
2. Match Prediction Page ('/predict') with interactive match inputs
3. RESTful Prediction API ('/api/predict') for AJAX live calculation & Chart.js
4. Sample scenarios & team metadata endpoints
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Initialize Flask application
app = Flask(__name__)

# Model and metadata global containers
MODEL_PAYLOAD = None
MODEL = None
TEAMS = []
CITIES = []
ACCURACY = 85.87
PRECISION = 81.39
RECALL = 78.90
ROC_AUC = 93.22

# Metadata for IPL Teams (Colors, Short Codes, Stadiums, Logos)
TEAM_METADATA = {
    'Chennai Super Kings': {
        'code': 'CSK',
        'primary_color': '#FDB913',
        'secondary_color': '#005CA8',
        'text_color': '#000000',
        'logo': 'csk.svg',
        'stadium': 'MA Chidambaram Stadium, Chepauk',
        'titles': 5
    },
    'Mumbai Indians': {
        'code': 'MI',
        'primary_color': '#004BA0',
        'secondary_color': '#D4AF37',
        'text_color': '#FFFFFF',
        'logo': 'mi.svg',
        'stadium': 'Wankhede Stadium',
        'titles': 5
    },
    'Kolkata Knight Riders': {
        'code': 'KKR',
        'primary_color': '#3A225D',
        'secondary_color': '#F3CE52',
        'text_color': '#FFFFFF',
        'logo': 'kkr.svg',
        'stadium': 'Eden Gardens',
        'titles': 3
    },
    'Sunrisers Hyderabad': {
        'code': 'SRH',
        'primary_color': '#F26522',
        'secondary_color': '#111111',
        'text_color': '#FFFFFF',
        'logo': 'srh.svg',
        'stadium': 'Rajiv Gandhi Intl Cricket Stadium',
        'titles': 1
    },
    'Rajasthan Royals': {
        'code': 'RR',
        'primary_color': '#EA1A85',
        'secondary_color': '#004BA0',
        'text_color': '#FFFFFF',
        'logo': 'rr.svg',
        'stadium': 'Sawai Mansingh Stadium',
        'titles': 1
    },
    'Gujarat Titans': {
        'code': 'GT',
        'primary_color': '#1C2A39',
        'secondary_color': '#B89547',
        'text_color': '#FFFFFF',
        'logo': 'gt.svg',
        'stadium': 'Narendra Modi Stadium',
        'titles': 1
    },
    'Royal Challengers Bengaluru': {
        'code': 'RCB',
        'primary_color': '#EC1C24',
        'secondary_color': '#111111',
        'text_color': '#FFFFFF',
        'logo': 'rcb.svg',
        'stadium': 'M. Chinnaswamy Stadium',
        'titles': 2
    },
    'Delhi Capitals': {
        'code': 'DC',
        'primary_color': '#004C97',
        'secondary_color': '#EF3340',
        'text_color': '#FFFFFF',
        'logo': 'dc.svg',
        'stadium': 'Arun Jaitley Stadium',
        'titles': 0
    },
    'Punjab Kings': {
        'code': 'PBKS',
        'primary_color': '#DD1F2D',
        'secondary_color': '#D4AF37',
        'text_color': '#FFFFFF',
        'logo': 'pbks.svg',
        'stadium': 'PCA Stadium, Mohali',
        'titles': 0
    },
    'Lucknow Super Giants': {
        'code': 'LSG',
        'primary_color': '#00B4D8',
        'secondary_color': '#F77F00',
        'text_color': '#FFFFFF',
        'logo': 'lsg.svg',
        'stadium': 'BRSABV Ekana Cricket Stadium',
        'titles': 0
    }
}

# City stadium mapping for UI enrichment
CITY_VENUES = {
    'Mumbai': 'Wankhede Stadium / Brabourne',
    'Bengaluru': 'M. Chinnaswamy Stadium',
    'Kolkata': 'Eden Gardens',
    'Chennai': 'MA Chidambaram Stadium',
    'Delhi': 'Arun Jaitley Stadium',
    'Hyderabad': 'Rajiv Gandhi International Stadium',
    'Jaipur': 'Sawai Mansingh Stadium',
    'Ahmedabad': 'Narendra Modi Stadium',
    'Chandigarh': 'PCA IS Bindra Stadium, Mohali',
    'Lucknow': 'BRSABV Ekana Cricket Stadium'
}

# Curated real-life style IPL thriller match scenarios for 1-click testing
SAMPLE_SCENARIOS = [
    {
        'title': '2019 Final Thriller: CSK vs MI',
        'description': 'Target 150. CSK needs 17 runs off 10 balls with 5 wickets in hand at Hyderabad.',
        'city': 'Hyderabad',
        'batting_team': 'Chennai Super Kings',
        'bowling_team': 'Mumbai Indians',
        'target': 150,
        'current_score': 133,
        'overs': 18.2,
        'wickets_fallen': 5
    },
    {
        'title': 'High-Scoring Chase: RCB vs KKR',
        'description': 'Target 206. RCB needs 41 runs off 24 balls at Chinnaswamy with 7 wickets remaining.',
        'city': 'Bengaluru',
        'batting_team': 'Royal Challengers Bengaluru',
        'bowling_team': 'Kolkata Knight Riders',
        'target': 206,
        'current_score': 165,
        'overs': 16.0,
        'wickets_fallen': 3
    },
    {
        'title': 'Death Overs Pressure: GT vs RR',
        'description': 'Target 178. GT needs 36 runs off 18 balls at Ahmedabad with 4 wickets remaining.',
        'city': 'Ahmedabad',
        'batting_team': 'Gujarat Titans',
        'bowling_team': 'Rajasthan Royals',
        'target': 178,
        'current_score': 142,
        'overs': 17.0,
        'wickets_fallen': 6
    },
    {
        'title': 'Nail Biter Finish: PBKS vs SRH',
        'description': 'Target 183. PBKS needs 13 runs off 6 balls with 3 wickets remaining at Mohali.',
        'city': 'Chandigarh',
        'batting_team': 'Punjab Kings',
        'bowling_team': 'Sunrisers Hyderabad',
        'target': 183,
        'current_score': 170,
        'overs': 19.0,
        'wickets_fallen': 7
    }
]

def load_prediction_model():
    """
    Loads serialized model pipeline and metadata from 'model.pkl'.
    """
    global MODEL_PAYLOAD, MODEL, TEAMS, CITIES, ACCURACY, PRECISION, RECALL, ROC_AUC
    model_path = 'model.pkl'

    if not os.path.exists(model_path):
        print(f"[!] Warning: '{model_path}' not found. Training model now...")
        from train_model import main as train_main
        train_main()

    with open(model_path, 'rb') as f:
        MODEL_PAYLOAD = pickle.load(f)

    if isinstance(MODEL_PAYLOAD, dict):
        MODEL = MODEL_PAYLOAD.get('model')
        TEAMS = MODEL_PAYLOAD.get('teams', sorted(list(TEAM_METADATA.keys())))
        CITIES = MODEL_PAYLOAD.get('cities', sorted(list(CITY_VENUES.keys())))
        ACCURACY = MODEL_PAYLOAD.get('accuracy', 85.87)
        PRECISION = MODEL_PAYLOAD.get('precision', 81.39)
        RECALL = MODEL_PAYLOAD.get('recall', 78.90)
        ROC_AUC = MODEL_PAYLOAD.get('roc_auc', 93.22)
    else:
        MODEL = MODEL_PAYLOAD
        TEAMS = sorted(list(TEAM_METADATA.keys()))
        CITIES = sorted(list(CITY_VENUES.keys()))

    print(f"[OK] Model loaded successfully with {ACCURACY}% Accuracy.")

# Load model upon application startup
load_prediction_model()

def validate_and_compute_features(data):
    """
    Validates user match inputs and computes derived cricket metrics:
    - Runs Left = Target - Current Score
    - Balls Left = 120 - Balls Bowled
    - Wickets Left = 10 - Wickets Fallen
    - CRR = (Current Score * 6) / Balls Bowled
    - RRR = (Runs Left * 6) / Balls Left
    """
    batting_team = data.get('batting_team', '').strip()
    bowling_team = data.get('bowling_team', '').strip()
    city = data.get('city', '').strip()

    # Team validation
    if not batting_team or not bowling_team:
        return None, "Please select both Batting and Bowling teams."
    if batting_team == bowling_team:
        return None, "Batting and Bowling teams cannot be the same. Please choose two distinct teams."
    if batting_team not in TEAMS or bowling_team not in TEAMS:
        return None, "Selected team is not recognized in the IPL database."
    if not city or city not in CITIES:
        return None, "Please select a valid host city."

    # Numerical inputs validation
    try:
        target = int(data.get('target', 0))
        current_score = int(data.get('current_score', 0))
        overs = float(data.get('overs', 0.0))
        wickets_fallen = int(data.get('wickets_fallen', 0))
    except (ValueError, TypeError):
        return None, "Target, current score, overs, and wickets must be valid numbers."

    if target <= 0:
        return None, "Target score must be a positive integer greater than 0."
    if current_score < 0:
        return None, "Current score cannot be negative."
    if current_score >= target:
        return None, f"Current score ({current_score}) already equals or exceeds the target ({target}). Batting team has already won!"

    # Overs validation (e.g., 15.3 means 15 overs and 3 balls)
    completed_overs = int(overs)
    balls_in_current_over = round((overs - completed_overs) * 10)

    if balls_in_current_over > 5:
        return None, f"Invalid over notation: '{overs}'. Over decimals must be between .0 and .5 (representing 0 to 5 balls bowled)."
    if overs < 0.1 or overs > 19.5:
        return None, "Overs completed must be between 0.1 and 19.5 overs (during active 2nd innings play)."

    balls_bowled = (completed_overs * 6) + balls_in_current_over
    if balls_bowled <= 0 or balls_bowled >= 120:
        return None, "Match balls bowled must be between 1 and 119 balls."

    balls_left = 120 - balls_bowled
    runs_left = target - current_score

    # Wickets validation
    if wickets_fallen < 0 or wickets_fallen > 9:
        return None, "Wickets fallen must be between 0 and 9 (at 10 wickets, innings is complete and defending team wins)."

    wickets_remaining = 10 - wickets_fallen

    # Compute Run Rates
    crr = round((current_score * 6) / balls_bowled, 2)
    rrr = round((runs_left * 6) / balls_left, 2) if balls_left > 0 else 0.0

    computed_dict = {
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'city': city,
        'target': target,
        'current_score': current_score,
        'overs': overs,
        'balls_bowled': balls_bowled,
        'wickets_fallen': wickets_fallen,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_remaining': wickets_remaining,
        'crr': crr,
        'rrr': rrr
    }

    return computed_dict, None

def generate_match_analysis(batting_team, bowling_team, batting_prob, bowling_prob, crr, rrr, runs_left, balls_left, wickets_remaining):
    """
    Generates intelligent qualitative commentary and pressure factor insights.
    """
    if batting_prob >= 75:
        status = "Strong Chasing Position"
        badge_class = "success"
        commentary = (
            f"{batting_team} is in commanding control of this chase. Needing {runs_left} runs from {balls_left} balls "
            f"with {wickets_remaining} wickets in hand and a required rate of {rrr} RPO, the batting lineup has plenty "
            f"of firepower and depth to close out the victory."
        )
    elif batting_prob >= 55:
        status = "Batting Team Slightly Ahead"
        badge_class = "primary"
        commentary = (
            f"{batting_team} currently holds a slight tactical edge. The current run rate ({crr}) remains healthy, "
            f"but keeping wickets intact against {bowling_team}'s bowling attack will be decisive as the death overs approach."
        )
    elif batting_prob >= 45:
        status = "Nail-Biter / Even Contest (50-50)"
        badge_class = "warning"
        commentary = (
            f"This match is on a knife's edge! With {runs_left} needed off {balls_left} balls and {wickets_remaining} wickets remaining, "
            f"momentum could swing on a single over. One boundary flurry favors {batting_team}, while a quick strike puts {bowling_team} ahead."
        )
    elif batting_prob >= 25:
        status = "Bowling Team in Control"
        badge_class = "orange"
        commentary = (
            f"{bowling_team} has tightened the screws significantly. With the required run rate rising to {rrr} RPO and "
            f"only {wickets_remaining} wickets left, {batting_team} is under acute scoreboard pressure."
        )
    else:
        status = "Dominant Defending Position"
        badge_class = "danger"
        commentary = (
            f"{bowling_team} is firmly in the driver's seat to defend their target. The steep required rate of {rrr} RPO "
            f"combined with depleted wickets makes this an extraordinarily uphill climb for {batting_team}."
        )

    # Pressure gauge indicator
    rate_gap = round(rrr - crr, 2)
    pressure_level = "High" if rate_gap > 3.0 else ("Moderate" if rate_gap > 0 else "Low")

    return {
        'status': status,
        'badge_class': badge_class,
        'commentary': commentary,
        'rate_gap': rate_gap,
        'pressure_level': pressure_level
    }

# =============================================================================
# Web Application Routes
# =============================================================================

@app.route('/')
def home():
    """
    Home page route displaying project overview, architecture,
    team cards, model performance metrics, and quick sample matches.
    """
    return render_template(
        'index.html',
        teams=TEAMS,
        cities=CITIES,
        team_metadata=TEAM_METADATA,
        city_venues=CITY_VENUES,
        sample_scenarios=SAMPLE_SCENARIOS,
        accuracy=ACCURACY,
        precision=PRECISION,
        recall=RECALL,
        roc_auc=ROC_AUC
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """
    Match predictor page route:
    - GET: Renders input form
    - POST: Server-side fallback for traditional form submission
    """
    prediction_result = None
    error_message = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        computed, error = validate_and_compute_features(form_data)

        if error:
            error_message = error
        else:
            # Prepare feature vector for ML Pipeline
            input_df = pd.DataFrame([{
                'batting_team': computed['batting_team'],
                'bowling_team': computed['bowling_team'],
                'city': computed['city'],
                'runs_left': computed['runs_left'],
                'balls_left': computed['balls_left'],
                'wickets_remaining': computed['wickets_remaining'],
                'target': computed['target'],
                'crr': computed['crr'],
                'rrr': computed['rrr']
            }])

            # Generate prediction probabilities
            probabilities = MODEL.predict_proba(input_df)[0]
            bowling_prob = round(probabilities[0] * 100, 1)
            batting_prob = round(probabilities[1] * 100, 1)

            # Ensure sum is strictly 100%
            if batting_prob + bowling_prob != 100.0:
                bowling_prob = round(100.0 - batting_prob, 1)

            analysis = generate_match_analysis(
                computed['batting_team'],
                computed['bowling_team'],
                batting_prob,
                bowling_prob,
                computed['crr'],
                computed['rrr'],
                computed['runs_left'],
                computed['balls_left'],
                computed['wickets_remaining']
            )

            prediction_result = {
                **computed,
                'batting_prob': batting_prob,
                'bowling_prob': bowling_prob,
                'analysis': analysis,
                'batting_meta': TEAM_METADATA.get(computed['batting_team'], {}),
                'bowling_meta': TEAM_METADATA.get(computed['bowling_team'], {})
            }

    return render_template(
        'prediction.html',
        teams=TEAMS,
        cities=CITIES,
        team_metadata=TEAM_METADATA,
        city_venues=CITY_VENUES,
        sample_scenarios=SAMPLE_SCENARIOS,
        accuracy=ACCURACY,
        prediction_result=prediction_result,
        error_message=error_message,
        form_data=form_data
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    RESTful JSON API endpoint for AJAX live predictions and interactive Chart.js visualization.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        computed, error = validate_and_compute_features(data)

        if error:
            return jsonify({'success': False, 'error': error}), 400

        # Construct input DataFrame for model
        input_df = pd.DataFrame([{
            'batting_team': computed['batting_team'],
            'bowling_team': computed['bowling_team'],
            'city': computed['city'],
            'runs_left': computed['runs_left'],
            'balls_left': computed['balls_left'],
            'wickets_remaining': computed['wickets_remaining'],
            'target': computed['target'],
            'crr': computed['crr'],
            'rrr': computed['rrr']
        }])

        # Predict winning probabilities
        probabilities = MODEL.predict_proba(input_df)[0]
        bowling_prob = round(float(probabilities[0]) * 100, 1)
        batting_prob = round(float(probabilities[1]) * 100, 1)

        # Normalize to 100%
        if batting_prob + bowling_prob != 100.0:
            bowling_prob = round(100.0 - batting_prob, 1)

        analysis = generate_match_analysis(
            computed['batting_team'],
            computed['bowling_team'],
            batting_prob,
            bowling_prob,
            computed['crr'],
            computed['rrr'],
            computed['runs_left'],
            computed['balls_left'],
            computed['wickets_remaining']
        )

        batting_meta = TEAM_METADATA.get(computed['batting_team'], {})
        bowling_meta = TEAM_METADATA.get(computed['bowling_team'], {})

        response_payload = {
            'success': True,
            'match_info': computed,
            'probabilities': {
                'batting_team': computed['batting_team'],
                'bowling_team': computed['bowling_team'],
                'batting_prob': batting_prob,
                'bowling_prob': bowling_prob
            },
            'metadata': {
                'batting': batting_meta,
                'bowling': bowling_meta,
                'venue': CITY_VENUES.get(computed['city'], computed['city'])
            },
            'analysis': analysis
        }

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f"Prediction error: {str(e)}"}), 500

@app.route('/api/sample-matches')
def get_sample_matches():
    """
    Returns curated thriller match scenarios.
    """
    return jsonify({'success': True, 'scenarios': SAMPLE_SCENARIOS})

@app.route('/api/teams')
def get_teams():
    """
    Returns team metadata and branding.
    """
    return jsonify({'success': True, 'teams': TEAM_METADATA})

if __name__ == '__main__':
    # Start local Flask development server
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting IPL Prediction Flask server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)

