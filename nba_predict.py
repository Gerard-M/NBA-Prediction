import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load data with proper dtype handling
games = pd.read_csv("games.csv")
teams = pd.read_csv("teams.csv")
ranking = pd.read_csv("ranking.csv")

# Load games_details with specific dtype for problematic columns
games_details = pd.read_csv("games_details.csv", 
                          dtype={'MIN': str, 'FG_PCT': float, 'FG3_PCT': float},
                          low_memory=False)

# Convert dates where available
games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
ranking['STANDINGSDATE'] = pd.to_datetime(ranking['STANDINGSDATE'])

# 1. Merge games_details with games to get dates and team IDs
def add_player_features(games_df, details_df):
    # Get necessary columns from games
    game_info = games_df[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    
    # Merge with games_details
    details_with_info = pd.merge(details_df, game_info, on='GAME_ID', how='left')
    
    # Calculate player efficiency
    details_with_info['EFF'] = (details_with_info['PTS'] + 
                               details_with_info['REB'] + 
                               details_with_info['AST'] + 
                               details_with_info['STL'] + 
                               details_with_info['BLK'] - 
                               details_with_info['TO'] -
                               (details_with_info['FGA'] - details_with_info['FGM']) - 
                               (details_with_info['FTA'] - details_with_info['FTM']))
    
    # Get star players (top 20% by efficiency)
    eff_threshold = details_with_info.groupby('PLAYER_ID')['EFF'].mean().quantile(0.8)
    star_players = details_with_info.groupby('PLAYER_ID')['EFF'].mean()[
        lambda x: x > eff_threshold
    ].index.tolist()
    
    # Aggregate to team-game level
    team_stats = details_with_info.groupby(['GAME_ID', 'TEAM_ID']).agg(
        TEAM_EFF=('EFF', 'mean'),
        STAR_PLAYERS=('PLAYER_ID', lambda x: len(set(x) & set(star_players))),
        TOTAL_PTS=('PTS', 'sum'),
        TOTAL_REB=('REB', 'sum'),
        TOTAL_AST=('AST', 'sum')
    ).reset_index()
    
    # First merge for home team
    merged = pd.merge(games_df, 
                     team_stats.rename(columns={'TEAM_ID': 'HOME_TEAM_ID'}),
                     left_on=['GAME_ID', 'HOME_TEAM_ID'],
                     right_on=['GAME_ID', 'HOME_TEAM_ID'],
                     suffixes=('', '_home'))
    
    # Second merge for away team
    merged = pd.merge(merged, 
                     team_stats.rename(columns={'TEAM_ID': 'VISITOR_TEAM_ID'}),
                     left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
                     right_on=['GAME_ID', 'VISITOR_TEAM_ID'],
                     suffixes=('_home', '_away'))
    
    # Clean up duplicate columns
    merged = merged.loc[:,~merged.columns.duplicated()]
    
    return merged

# 2. Add Team ranking with proper date handling
def add_rankings(games_df, rankings_df):
    # Create a copy to avoid SettingWithCopyWarning
    rankings_df = rankings_df.copy()
    
    # Convert win percentages directly (already numeric)
    rankings_df['W_PCT'] = pd.to_numeric(rankings_df['W_PCT'], errors='coerce')
    
    # Calculate conference rankings
    rankings_df = rankings_df.sort_values(['CONFERENCE', 'STANDINGSDATE', 'W_PCT'], 
                                        ascending=[True, True, False])
    rankings_df['CONFERENCE_RANK'] = rankings_df.groupby(
        ['CONFERENCE', 'STANDINGSDATE']
    ).cumcount() + 1

    # Calculate ranking momentum
    rankings_df['RANK_CHANGE'] = rankings_df.groupby('TEAM_ID')['CONFERENCE_RANK'].diff().fillna(0)
    
    # Merge home team rankings
    home_rankings = rankings_df.rename(columns={
        'TEAM_ID': 'HOME_TEAM_ID',
        'CONFERENCE_RANK': 'HOME_CONF_RANK',
        'RANK_CHANGE': 'HOME_RANK_CHANGE',
        'W_PCT': 'HOME_W_PCT'
    })
    
    merged_df = pd.merge_asof(
        games_df.sort_values('GAME_DATE_EST'),
        home_rankings.sort_values('STANDINGSDATE').rename(columns={'STANDINGSDATE': 'GAME_DATE_EST'}),
        left_on='GAME_DATE_EST',
        right_on='GAME_DATE_EST',
        by='HOME_TEAM_ID',
        direction='backward'
    )
    
    # Merge away team rankings
    away_rankings = rankings_df.rename(columns={
        'TEAM_ID': 'VISITOR_TEAM_ID',
        'CONFERENCE_RANK': 'AWAY_CONF_RANK',
        'RANK_CHANGE': 'AWAY_RANK_CHANGE',
        'W_PCT': 'AWAY_W_PCT'
    })
    
    merged_df = pd.merge_asof(
        merged_df.sort_values('GAME_DATE_EST'),
        away_rankings.sort_values('STANDINGSDATE').rename(columns={'STANDINGSDATE': 'GAME_DATE_EST'}),
        left_on='GAME_DATE_EST',
        right_on='GAME_DATE_EST',
        by='VISITOR_TEAM_ID',
        direction='backward'
    )
    
    return merged_df

# 3. Create advanced features
def create_advanced_features(df):
    # Home advantage calculation
    df['HOME_WIN_RATE'] = df.groupby('HOME_TEAM_ID')['HOME_TEAM_WINS'].transform(
        lambda x: x.rolling(100, min_periods=1).mean().shift(1)
    )
    
    # Days rest calculation
    for team_type in ['HOME_TEAM_ID', 'VISITOR_TEAM_ID']:
        df[f'DAYS_REST_{team_type}'] = df.groupby(team_type)['GAME_DATE_EST'].diff().dt.days
    
    # Win streak features
    for team_type in ['HOME_TEAM_ID', 'VISITOR_TEAM_ID']:
        df[f'{team_type}_WIN_STREAK'] = df.groupby(team_type)['HOME_TEAM_WINS'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        )
    
    # Point differential
    df['HOME_POINT_DIFF'] = df['PTS_home'] - df['PTS_away']
    df['AWAY_POINT_DIFF'] = df['PTS_away'] - df['PTS_home']
    
    return df

# Full preprocessing pipeline
def full_preprocessing():
    # Base data processing
    df = add_player_features(games, games_details)
    df = add_rankings(df, ranking)
    df = create_advanced_features(df)
    
    # Define features and target
    features = [
        # Player stats
        'TEAM_EFF_home', 'TEAM_EFF_away',
        'STAR_PLAYERS_home', 'STAR_PLAYERS_away',
        
        # ranking
        'HOME_CONF_RANK', 'AWAY_CONF_RANK',
        'HOME_RANK_CHANGE', 'AWAY_RANK_CHANGE',
        'HOME_W_PCT', 'AWAY_W_PCT',
        
        # Advanced features
        'HOME_WIN_RATE',
        'DAYS_REST_HOME_TEAM_ID', 'DAYS_REST_VISITOR_TEAM_ID',
        'HOME_TEAM_ID_WIN_STREAK', 'VISITOR_TEAM_ID_WIN_STREAK',
        'HOME_POINT_DIFF', 'AWAY_POINT_DIFF'
    ]
    
    target = 'HOME_TEAM_WINS'
    
    # Filter and clean
    df = df.dropna(subset=features + [target])
    return df, features, target

# Execute preprocessing
processed_df, feature_cols, target_col = full_preprocessing()

# Temporal split (80% train, 20% test)
split_date = processed_df['GAME_DATE_EST'].quantile(0.8)
train = processed_df[processed_df['GAME_DATE_EST'] < split_date]
test = processed_df[processed_df['GAME_DATE_EST'] >= split_date]

# Prepare data
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])
X_test = scaler.transform(test[feature_cols])
y_train = train[target_col]
y_test = test[target_col]

# Train optimized model
model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
fi = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features:")
print(fi.head(10))

# Prediction function
def predict_winner(home_team, away_team):
    try:
        home_id = teams[teams['NICKNAME'].str.lower() == home_team.lower()]['TEAM_ID'].values[0]
        away_id = teams[teams['NICKNAME'].str.lower() == away_team.lower()]['TEAM_ID'].values[0]
        
        # Get latest features for both teams
        home_data = processed_df[
            (processed_df['HOME_TEAM_ID'] == home_id) |
            (processed_df['VISITOR_TEAM_ID'] == home_id)
        ].iloc[-1][feature_cols].values.reshape(1, -1)
        
        away_data = processed_df[
            (processed_df['HOME_TEAM_ID'] == away_id) |
            (processed_df['VISITOR_TEAM_ID'] == away_id)
        ].iloc[-1][feature_cols].values.reshape(1, -1)
        
        # Average features for prediction
        combined_features = (home_data + away_data) / 2
        scaled_features = scaler.transform(combined_features)
        
        proba = model.predict_proba(scaled_features)[0][1]
        winner = home_team if proba > 0.5 else away_team
        confidence = max(proba, 1 - proba)
        
        print(f"\n{home_team} vs {away_team}")
        print(f"Predicted Winner: {winner} ({confidence:.1%} confidence)")
        print(f"Detailed Probabilities: Home {proba:.1%} - Away {(1-proba):.1%}")
        
    except IndexError:
        print("Error: Team name not recognized. Check team names and try again.")

# Example usage
predict_winner(input("\nHome team: "), input("Away team: "))