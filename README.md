# 🏀 NBA Game Outcome Predictor

## 📌 Script: `nba_predict.py`

This project predicts the outcome of NBA games using player performance, team statistics, and temporal data. It leverages **XGBoost** and an advanced feature engineering pipeline to deliver high prediction accuracy.

---

## 📁 Data Sources

- `games.csv`: Game metadata and final scores  
- `games_details.csv`: Player-level stats per game  
- `ranking.csv`: Team standings and win percentages  
- `teams.csv`: Team ID-to-name mappings
- used the NBA Dataset from kaggle '(https://www.kaggle.com/datasets/nathanlauga/nba-games/data?select=games_details.csv)'

---

## 🧪 Preprocessing Pipeline

### 1. Player Efficiency & Star Power
- Calculates a custom efficiency score (`EFF`)
- Identifies top 20% efficient players as “star players”
- Aggregates stats at the team-game level

### 2. Rankings & Momentum
- Adds conference rankings and win percentages
- Computes ranking momentum (`rank_change`)

### 3. Advanced Contextual Features
- Win rate over last 100 games
- Days of rest before each game
- 5-game win streak counts
- Home vs. away point differentials

---

## 🧠 Model

- **Algorithm**: `XGBClassifier`
- **Features**: 17+ advanced stats and contextual variables
- **Evaluation**:
  - Accuracy score
  - Classification report
  - Top 10 most important features printed

---

## 🧙 Prediction Function

The script includes a command-line prediction function:

```python
predict_winner(input("\nHome team: "), input("Away team: "))


## Bonus Code
- I added my first Machine Learning project as well.
