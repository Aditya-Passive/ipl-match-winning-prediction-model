"""
=============================================================================
IPL Match Winner Prediction System - Automated Test Suite
=============================================================================
Executes integration tests on Flask routes, API endpoints, error handlers,
and machine learning inference logic.
=============================================================================
"""

import json
from app import app

def run_tests():
    client = app.test_client()
    print("=" * 60)
    print("   RUNNING AUTOMATED TEST SUITE FOR IPL PREDICTOR APP      ")
    print("=" * 60)

    # Test 1: Home Page GET
    print("[Test 1] Testing Home Page ('/')...")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"IPL" in res.data, "IPL keyword missing in home page"
    assert b"Prediction System" in res.data, "Prediction System missing in home page"
    print("         PASS: Home Page loaded with status 200 OK.")

    # Test 2: Predict Page GET
    print("[Test 2] Testing Predict Page ('/predict')...")
    res = client.get('/predict')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"Live 2nd Innings Predictor" in res.data
    assert b"Chennai Super Kings" in res.data
    print("         PASS: Predict Page loaded with status 200 OK.")

    # Test 3: API Live Prediction (Valid Scenario: CSK chasing 150 vs MI at Hyderabad)
    print("[Test 3] Testing '/api/predict' with valid payload...")
    valid_payload = {
        'batting_team': 'Chennai Super Kings',
        'bowling_team': 'Mumbai Indians',
        'city': 'Hyderabad',
        'target': 150,
        'current_score': 133,
        'overs': 18.2,
        'wickets_fallen': 5
    }
    res = client.post('/api/predict', data=json.dumps(valid_payload), content_type='application/json')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data['success'] is True, "Success should be True"
    assert 'probabilities' in data, "Probabilities field missing"
    bat_prob = data['probabilities']['batting_prob']
    bowl_prob = data['probabilities']['bowling_prob']
    print(f"         Output Probabilities -> Batting ({valid_payload['batting_team']}): {bat_prob}%, Bowling ({valid_payload['bowling_team']}): {bowl_prob}%")
    print(f"         CRR: {data['match_info']['crr']}, RRR: {data['match_info']['rrr']}")
    assert abs((bat_prob + bowl_prob) - 100.0) < 0.2, "Probabilities must sum to 100%"
    print("         PASS: Valid API prediction succeeded.")

    # Test 4: API Error Handling - Same Batting and Bowling Team
    print("[Test 4] Testing '/api/predict' with same batting and bowling team...")
    invalid_team_payload = {
        'batting_team': 'Mumbai Indians',
        'bowling_team': 'Mumbai Indians',
        'city': 'Mumbai',
        'target': 180,
        'current_score': 90,
        'overs': 10.0,
        'wickets_fallen': 2
    }
    res = client.post('/api/predict', data=json.dumps(invalid_team_payload), content_type='application/json')
    assert res.status_code == 400, f"Expected 400 for identical teams, got {res.status_code}"
    err_data = res.get_json()
    assert err_data['success'] is False
    print(f"         Captured Error message: '{err_data['error']}'")
    print("         PASS: Identical team validation passed.")

    # Test 5: API Error Handling - Score Already Exceeds Target
    print("[Test 5] Testing '/api/predict' when current score >= target...")
    invalid_score_payload = {
        'batting_team': 'Royal Challengers Bengaluru',
        'bowling_team': 'Kolkata Knight Riders',
        'city': 'Bengaluru',
        'target': 160,
        'current_score': 165,
        'overs': 17.0,
        'wickets_fallen': 3
    }
    res = client.post('/api/predict', data=json.dumps(invalid_score_payload), content_type='application/json')
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    err_data = res.get_json()
    assert err_data['success'] is False
    print(f"         Captured Error message: '{err_data['error']}'")
    print("         PASS: Score exceedance validation passed.")

    # Test 6: API Error Handling - Invalid Over Decimal Notation (e.g. 14.8)
    print("[Test 6] Testing '/api/predict' with invalid over decimal (14.8)...")
    invalid_over_payload = {
        'batting_team': 'Rajasthan Royals',
        'bowling_team': 'Gujarat Titans',
        'city': 'Jaipur',
        'target': 175,
        'current_score': 120,
        'overs': 14.8,
        'wickets_fallen': 4
    }
    res = client.post('/api/predict', data=json.dumps(invalid_over_payload), content_type='application/json')
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    err_data = res.get_json()
    assert err_data['success'] is False
    print(f"         Captured Error message: '{err_data['error']}'")
    print("         PASS: Invalid over decimal validation passed.")

    # Test 7: Form POST submission to '/predict' (Traditional Fallback)
    print("[Test 7] Testing traditional HTML Form POST to '/predict'...")
    form_payload = {
        'batting_team': 'Gujarat Titans',
        'bowling_team': 'Rajasthan Royals',
        'city': 'Ahmedabad',
        'target': '178',
        'current_score': '142',
        'overs': '17.0',
        'wickets_fallen': '6'
    }
    res = client.post('/predict', data=form_payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"ML Prediction Result" in res.data
    assert b"serverPredictionData" in res.data
    print("         PASS: Traditional form submit renders results correctly.")

    print("\n" + "=" * 60)
    print("   ALL 7 AUTOMATED INTEGRATION TESTS PASSED SUCCESSFULLY!  ")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()

