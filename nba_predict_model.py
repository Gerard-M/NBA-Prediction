import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
games = pd.read_csv("games.csv")
teams = pd.read_csv("teams.csv")

# Preprocess dates
games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
games = games.sort_values('GAME_DATE_EST')

# Calculate cumulative stats
def get_cumulative_stats(games_df, team_type='home'):
    if team_type == 'home':
        df = games_df[['GAME_DATE_EST', 'HOME_TEAM_ID', 'PTS_home', 'PTS_away']]
        df = df.rename(columns={'HOME_TEAM_ID': 'TEAM_ID', 'PTS_home': 'scored', 'PTS_away': 'allowed'})
    else:
        df = games_df[['GAME_DATE_EST', 'VISITOR_TEAM_ID', 'PTS_away', 'PTS_home']]
        df = df.rename(columns={'VISITOR_TEAM_ID': 'TEAM_ID', 'PTS_away': 'scored', 'PTS_home': 'allowed'})
    
    df = df.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
    df['avg_scored'] = df.groupby('TEAM_ID')['scored'].expanding().mean().reset_index(level=0, drop=True)
    df['avg_allowed'] = df.groupby('TEAM_ID')['allowed'].expanding().mean().reset_index(level=0, drop=True)
    return df

home_stats = get_cumulative_stats(games, 'home')
away_stats = get_cumulative_stats(games, 'away')

# Merge stats into main dataframe
games = pd.merge(
    games,
    home_stats[['GAME_DATE_EST', 'TEAM_ID', 'avg_scored', 'avg_allowed']],
    left_on=['GAME_DATE_EST', 'HOME_TEAM_ID'],
    right_on=['GAME_DATE_EST', 'TEAM_ID'],
    suffixes=('', '_home')
)
games = pd.merge(
    games,
    away_stats[['GAME_DATE_EST', 'TEAM_ID', 'avg_scored', 'avg_allowed']],
    left_on=['GAME_DATE_EST', 'VISITOR_TEAM_ID'],
    right_on=['GAME_DATE_EST', 'TEAM_ID'],
    suffixes=('', '_away')
)

# Prepare features/target
features = games[['avg_scored', 'avg_allowed', 'avg_scored_away', 'avg_allowed_away']]
target = (games['PTS_home'] > games['PTS_away']).astype(int)

# Drop NaN rows
features = features.dropna()
target = target.loc[features.index]

# Train model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

# Prediction function
def predict_winner(home_team_name, away_team_name):
    home_id = teams[teams['NICKNAME'].str.lower() == home_team_name.lower()]['TEAM_ID'].values[0]
    away_id = teams[teams['NICKNAME'].str.lower() == away_team_name.lower()]['TEAM_ID'].values[0]
    
    home_avg = home_stats[home_stats['TEAM_ID'] == home_id].iloc[-1][['avg_scored', 'avg_allowed']].values
    away_avg = away_stats[away_stats['TEAM_ID'] == away_id].iloc[-1][['avg_scored', 'avg_allowed']].values
    
    prob = model.predict_proba([[*home_avg, *away_avg]])[0][1]
    winner = home_team_name if prob > 0.5 else away_team_name
    print(f"{home_team_name} vs {away_team_name} → Winner: {winner} (Confidence: {max(prob, 1-prob):.1%})")

# Example
predict_winner(input("Home team: "), input("Away team: "))