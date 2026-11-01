/**
 * IPL Match Winner Prediction System - Interactive JavaScript
 * Handles live team branding, instant metrics calculation,
 * scenario loading, AJAX prediction API calls, and Chart.js visualization.
 */

// Global Chart instances
let winProbChart = null;
let runRateChart = null;

// Team metadata map (injected from backend or defaults)
const teamColors = {
    'Chennai Super Kings': { primary: '#FDB913', secondary: '#005CA8', logo: 'csk.svg', code: 'CSK' },
    'Mumbai Indians': { primary: '#004BA0', secondary: '#D4AF37', logo: 'mi.svg', code: 'MI' },
    'Royal Challengers Bengaluru': { primary: '#EC1C24', secondary: '#111111', logo: 'rcb.svg', code: 'RCB' },
    'Kolkata Knight Riders': { primary: '#3A225D', secondary: '#F3CE52', logo: 'kkr.svg', code: 'KKR' },
    'Sunrisers Hyderabad': { primary: '#F26522', secondary: '#111111', logo: 'srh.svg', code: 'SRH' },
    'Delhi Capitals': { primary: '#004C97', secondary: '#EF3340', logo: 'dc.svg', code: 'DC' },
    'Rajasthan Royals': { primary: '#EA1A85', secondary: '#004BA0', logo: 'rr.svg', code: 'RR' },
    'Punjab Kings': { primary: '#DD1F2D', secondary: '#D4AF37', logo: 'pbks.svg', code: 'PBKS' },
    'Gujarat Titans': { primary: '#1C2A39', secondary: '#B89547', logo: 'gt.svg', code: 'GT' },
    'Lucknow Super Giants': { primary: '#00B4D8', secondary: '#F77F00', logo: 'lsg.svg', code: 'LSG' }
};

document.addEventListener('DOMContentLoaded', () => {
    initTeamSelectors();
    initLiveCalculation();
    initPredictionForm();
    initPresetScenarios();

    // If server rendered prediction results are present in DOM, render the chart immediately
    const serverResult = document.getElementById('serverPredictionData');
    if (serverResult) {
        try {
            const data = JSON.parse(serverResult.textContent);
            renderCharts(data);
        } catch (e) {
            console.error("Error parsing initial server prediction:", e);
        }
    }
});

/**
 * Updates team logo, badge, and color indicators when teams change
 */
function initTeamSelectors() {
    const battingSelect = document.getElementById('batting_team');
    const bowlingSelect = document.getElementById('bowling_team');

    if (!battingSelect || !bowlingSelect) return;

    function updateBattingTeam() {
        const team = battingSelect.value;
        const meta = teamColors[team] || { logo: 'ipl_trophy.svg', code: 'BAT' };
        
        const logoEl = document.getElementById('battingTeamLogo');
        const nameEl = document.getElementById('battingTeamName');
        const codeEl = document.getElementById('battingTeamCode');

        if (logoEl) logoEl.src = `/static/images/${meta.logo}`;
        if (nameEl) nameEl.textContent = team || 'Batting Team';
        if (codeEl) codeEl.textContent = meta.code || 'BAT';
    }

    function updateBowlingTeam() {
        const team = bowlingSelect.value;
        const meta = teamColors[team] || { logo: 'ipl_trophy.svg', code: 'BOWL' };

        const logoEl = document.getElementById('bowlingTeamLogo');
        const nameEl = document.getElementById('bowlingTeamName');
        const codeEl = document.getElementById('bowlingTeamCode');

        if (logoEl) logoEl.src = `/static/images/${meta.logo}`;
        if (nameEl) nameEl.textContent = team || 'Bowling Team';
        if (codeEl) codeEl.textContent = meta.code || 'BOWL';
    }

    battingSelect.addEventListener('change', () => {
        updateBattingTeam();
        if (battingSelect.value === bowlingSelect.value && battingSelect.value !== '') {
            showAlert('Batting and Bowling teams cannot be the same team.', 'warning');
        }
    });

    bowlingSelect.addEventListener('change', () => {
        updateBowlingTeam();
        if (battingSelect.value === bowlingSelect.value && bowlingSelect.value !== '') {
            showAlert('Batting and Bowling teams cannot be the same team.', 'warning');
        }
    });

    // Initial trigger
    updateBattingTeam();
    updateBowlingTeam();
}

/**
 * Real-time instant calculations as the user types
 */
function initLiveCalculation() {
    const targetInput = document.getElementById('target');
    const scoreInput = document.getElementById('current_score');
    const oversInput = document.getElementById('overs');
    const wicketsInput = document.getElementById('wickets_fallen');

    const runsPill = document.getElementById('liveRunsNeeded');
    const ballsPill = document.getElementById('liveBallsLeft');
    const crrPill = document.getElementById('liveCRR');
    const rrrPill = document.getElementById('liveRRR');

    function calculateLive() {
        const target = parseInt(targetInput?.value) || 0;
        const score = parseInt(scoreInput?.value) || 0;
        const overs = parseFloat(oversInput?.value) || 0.0;
        const wickets = parseInt(wicketsInput?.value) || 0;

        const runsNeeded = Math.max(0, target - score);

        const compOvers = Math.floor(overs);
        const ballsDecimal = Math.round((overs - compOvers) * 10);
        const ballsBowled = (compOvers * 6) + ballsDecimal;
        const ballsLeft = Math.max(0, 120 - ballsBowled);

        const crr = ballsBowled > 0 ? ((score * 6) / ballsBowled).toFixed(2) : '0.00';
        const rrr = ballsLeft > 0 ? ((runsNeeded * 6) / ballsLeft).toFixed(2) : (runsNeeded === 0 ? '0.00' : '99.00');

        if (runsPill) runsPill.textContent = runsNeeded;
        if (ballsPill) ballsPill.textContent = ballsLeft;
        if (crrPill) crrPill.textContent = crr;
        if (rrrPill) rrrPill.textContent = rrr;
    }

    [targetInput, scoreInput, oversInput, wicketsInput].forEach(input => {
        if (input) {
            input.addEventListener('input', calculateLive);
            input.addEventListener('change', calculateLive);
        }
    });

    // Initial run
    calculateLive();
}

/**
 * Curated preset scenarios loader
 */
function initPresetScenarios() {
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const batting = btn.getAttribute('data-batting');
            const bowling = btn.getAttribute('data-bowling');
            const city = btn.getAttribute('data-city');
            const target = btn.getAttribute('data-target');
            const score = btn.getAttribute('data-score');
            const overs = btn.getAttribute('data-overs');
            const wickets = btn.getAttribute('data-wickets');

            const battingSelect = document.getElementById('batting_team');
            const bowlingSelect = document.getElementById('bowling_team');
            const citySelect = document.getElementById('city');
            const targetInput = document.getElementById('target');
            const scoreInput = document.getElementById('current_score');
            const oversInput = document.getElementById('overs');
            const wicketsInput = document.getElementById('wickets_fallen');

            if (battingSelect) battingSelect.value = batting;
            if (bowlingSelect) bowlingSelect.value = bowling;
            if (citySelect) citySelect.value = city;
            if (targetInput) targetInput.value = target;
            if (scoreInput) scoreInput.value = score;
            if (oversInput) oversInput.value = overs;
            if (wicketsInput) wicketsInput.value = wickets;

            // Trigger events
            battingSelect?.dispatchEvent(new Event('change'));
            bowlingSelect?.dispatchEvent(new Event('change'));
            targetInput?.dispatchEvent(new Event('input'));

            // Smooth scroll to form
            document.getElementById('predictionForm')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
}

/**
 * Handles AJAX prediction form submit & visual results rendering
 */
function initPredictionForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const battingTeam = document.getElementById('batting_team')?.value;
        const bowlingTeam = document.getElementById('bowling_team')?.value;
        const city = document.getElementById('city')?.value;
        const target = parseInt(document.getElementById('target')?.value);
        const currentScore = parseInt(document.getElementById('current_score')?.value);
        const overs = parseFloat(document.getElementById('overs')?.value);
        const wickets = parseInt(document.getElementById('wickets_fallen')?.value);

        // Validation
        if (!battingTeam || !bowlingTeam) {
            showAlert("Please select both Batting and Bowling teams.", "danger");
            return;
        }
        if (battingTeam === bowlingTeam) {
            showAlert("Batting and Bowling teams cannot be identical. Please pick two different teams.", "danger");
            return;
        }
        if (!city) {
            showAlert("Please choose the match host city.", "danger");
            return;
        }
        if (isNaN(target) || target <= 0) {
            showAlert("Target score must be a positive integer greater than 0.", "danger");
            return;
        }
        if (isNaN(currentScore) || currentScore < 0) {
            showAlert("Current score cannot be negative.", "danger");
            return;
        }
        if (currentScore >= target) {
            showAlert(`Current score (${currentScore}) is already equal to or higher than the target (${target}). Batting team has already won!`, "warning");
            return;
        }
        if (isNaN(overs) || overs < 0.1 || overs > 19.5) {
            showAlert("Overs completed must be between 0.1 and 19.5 overs.", "danger");
            return;
        }
        const ballsDec = Math.round((overs - Math.floor(overs)) * 10);
        if (ballsDec > 5) {
            showAlert("Overs notation must be between .0 and .5 balls (e.g. 12.3).", "danger");
            return;
        }
        if (isNaN(wickets) || wickets < 0 || wickets > 9) {
            showAlert("Wickets fallen must be between 0 and 9 (match concludes at 10 wickets).", "danger");
            return;
        }

        // Show loading spinner overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) loadingOverlay.style.display = 'flex';

        try {
            const payload = {
                batting_team: battingTeam,
                bowling_team: bowlingTeam,
                city: city,
                target: target,
                current_score: currentScore,
                overs: overs,
                wickets_fallen: wickets
            };

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                showAlert(result.error || "Failed to calculate prediction.", "danger");
                if (loadingOverlay) loadingOverlay.style.display = 'none';
                return;
            }

            // Render prediction results to UI
            displayPredictionResults(result);

        } catch (err) {
            console.error("API error:", err);
            showAlert("Network or server error while predicting match winner.", "danger");
        } finally {
            if (loadingOverlay) {
                setTimeout(() => {
                    loadingOverlay.style.display = 'none';
                }, 400);
            }
        }
    });
}

/**
 * Updates UI with prediction data and charts
 */
function displayPredictionResults(data) {
    const emptyState = document.getElementById('resultsEmptyState');
    const contentState = document.getElementById('resultsContentState');
    const resultsCard = document.getElementById('resultsCard');

    if (emptyState) emptyState.style.display = 'none';
    if (contentState) contentState.style.display = 'block';
    if (resultsCard) resultsCard.classList.add('active');

    const battingProb = data.probabilities.batting_prob;
    const bowlingProb = data.probabilities.bowling_prob;
    const battingTeam = data.probabilities.batting_team;
    const bowlingTeam = data.probabilities.bowling_team;

    // Badges & percentages
    const batValEl = document.getElementById('resultBattingProb');
    const bowlValEl = document.getElementById('resultBowlingProb');
    const batNameEl = document.getElementById('resultBattingTeam');
    const bowlNameEl = document.getElementById('resultBowlingTeam');
    const batLogoEl = document.getElementById('resultBattingLogo');
    const bowlLogoEl = document.getElementById('resultBowlingLogo');

    if (batValEl) batValEl.textContent = `${battingProb}%`;
    if (bowlValEl) bowlValEl.textContent = `${bowlingProb}%`;
    if (batNameEl) batNameEl.textContent = battingTeam;
    if (bowlNameEl) bowlNameEl.textContent = bowlingTeam;

    if (batLogoEl) batLogoEl.src = `/static/images/${data.metadata.batting.logo || 'ipl_trophy.svg'}`;
    if (bowlLogoEl) bowlLogoEl.src = `/static/images/${data.metadata.bowling.logo || 'ipl_trophy.svg'}`;

    // Dual Progress bar
    const barBatting = document.getElementById('progressBarBatting');
    const barBowling = document.getElementById('progressBarBowling');

    if (barBatting) {
        barBatting.style.width = `${battingProb}%`;
        barBatting.style.backgroundColor = data.metadata.batting.primary_color || '#00d2ff';
    }
    if (barBowling) {
        barBowling.style.width = `${bowlingProb}%`;
        barBowling.style.backgroundColor = data.metadata.bowling.primary_color || '#ff3366';
    }

    // Match commentary and status pill
    const statusPill = document.getElementById('resultStatusPill');
    const commentaryBox = document.getElementById('resultCommentary');

    if (statusPill) {
        statusPill.textContent = data.analysis.status;
        statusPill.className = `match-status-pill status-pill-${data.analysis.badge_class}`;
    }
    if (commentaryBox) {
        commentaryBox.innerHTML = `<strong>Match Intelligence:</strong> ${data.analysis.commentary}`;
    }

    // 5 Key metrics tiles
    const tileRuns = document.getElementById('metricRunsLeft');
    const tileBalls = document.getElementById('metricBallsLeft');
    const tileWickets = document.getElementById('metricWicketsLeft');
    const tileCRR = document.getElementById('metricCRR');
    const tileRRR = document.getElementById('metricRRR');

    if (tileRuns) tileRuns.textContent = data.match_info.runs_left;
    if (tileBalls) tileBalls.textContent = data.match_info.balls_left;
    if (tileWickets) tileWickets.textContent = data.match_info.wickets_remaining;
    if (tileCRR) tileCRR.textContent = data.match_info.crr;
    if (tileRRR) tileRRR.textContent = data.match_info.rrr;

    // Render interactive charts
    renderCharts(data);

    // Scroll to results card smoothly
    resultsCard?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Initializes and updates Chart.js Donut and Bar charts
 */
function renderCharts(data) {
    const battingProb = data.probabilities.batting_prob;
    const bowlingProb = data.probabilities.bowling_prob;
    const battingTeam = data.probabilities.batting_team;
    const bowlingTeam = data.probabilities.bowling_team;

    const batColor = data.metadata?.batting?.primary_color || '#00d2ff';
    const bowlColor = data.metadata?.bowling?.primary_color || '#ff3366';

    // 1. Donut Win Probability Chart
    const donutCtx = document.getElementById('winProbDonutChart')?.getContext('2d');
    if (donutCtx) {
        if (winProbChart) {
            winProbChart.destroy();
        }

        winProbChart = new Chart(donutCtx, {
            type: 'doughnut',
            data: {
                labels: [battingTeam, bowlingTeam],
                datasets: [{
                    data: [battingProb, bowlingProb],
                    backgroundColor: [batColor, bowlColor],
                    borderColor: ['rgba(255,255,255,0.2)', 'rgba(255,255,255,0.2)'],
                    borderWidth: 2,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#cbd5e1',
                            font: { family: 'Inter', size: 12, weight: 600 },
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return ` ${context.label}: ${context.raw}% win probability`;
                            }
                        }
                    }
                }
            }
        });
    }

    // 2. Run Rate Pressure Bar Chart
    const barCtx = document.getElementById('runRateBarChart')?.getContext('2d');
    if (barCtx) {
        if (runRateChart) {
            runRateChart.destroy();
        }

        const crr = parseFloat(data.match_info.crr);
        const rrr = parseFloat(data.match_info.rrr);

        runRateChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Current Run Rate (CRR)', 'Required Run Rate (RRR)'],
                datasets: [{
                    label: 'Runs Per Over',
                    data: [crr, rrr],
                    backgroundColor: [
                        'rgba(0, 210, 255, 0.75)',
                        rrr > crr ? 'rgba(255, 51, 102, 0.75)' : 'rgba(0, 230, 118, 0.75)'
                    ],
                    borderColor: [
                        '#00d2ff',
                        rrr > crr ? '#ff3366' : '#00e676'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#cbd5e1', font: { weight: 600 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return ` ${context.raw} RPO`;
                            }
                        }
                    }
                }
            }
        });
    }
}

/**
 * Utility helper to show temporary alert banners
 */
function showAlert(message, type = 'danger') {
    const alertBox = document.getElementById('formAlertBox');
    if (!alertBox) {
        alert(message);
        return;
    }

    alertBox.className = `alert alert-${type} alert-dismissible fade show`;
    alertBox.innerHTML = `
        <div class="d-flex align-items-center gap-2">
            <strong>Notice:</strong> ${message}
        </div>
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertBox.style.display = 'block';

    alertBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

