import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
import joblib

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# NBA API imports
try:
    from nba_api.stats.endpoints import (
        scoreboardv2, teamdashboardbygeneralsplits,
        leaguegamefinder, teamgamelogs
    )
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    print("NBA API not found. Install with: pip install nba-api")
    NBA_API_AVAILABLE = False

warnings.filterwarnings('ignore')

# It's good practice to provide headers to avoid being blocked by the API provider
API_HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

class NBADataCollector:
    def __init__(self):
        self.team_mapping = self._get_team_mapping()

    def _get_team_mapping(self):
        if not NBA_API_AVAILABLE:
            return {
                'LAL': {'id': 1610612747, 'name': 'Los Angeles Lakers'},
                'GSW': {'id': 1610612744, 'name': 'Golden State Warriors'},
            }
        team_list = teams.get_teams()
        return {team['abbreviation']: {'id': team['id'], 'name': team['full_name']} for team in team_list}

    def collect_team_stats(self, season='2024-25'):
        print(f"Collecting team stats for {season}...")
        if not NBA_API_AVAILABLE:
            return self._generate_dummy_team_stats()

        team_stats = []
        for abbrev, team_info in self.team_mapping.items():
            try:
                time.sleep(0.6) # Respect API rate limits
                dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                    team_id=team_info['id'],
                    season=season,
                    headers=API_HEADERS,
                    timeout=30
                )
                overall_stats = dashboard.get_data_frames()[0]

                if not overall_stats.empty:
                    stats = overall_stats.iloc[0]
                    team_data = {
                        'TEAM': abbrev, 'TEAM_ID': team_info['id'], 'GP': stats['GP'], 'W': stats['W'],
                        'L': stats['L'], 'WIN_PCT': stats['W_PCT'], 'MIN': stats['MIN'], 'FGM': stats['FGM'],
                        'FGA': stats['FGA'], 'FG_PCT': stats['FG_PCT'], 'FG3M': stats['FG3M'],
                        'FG3A': stats['FG3A'], 'FG3_PCT': stats['FG3_PCT'], 'FTM': stats['FTM'],
                        'FTA': stats['FTA'], 'FT_PCT': stats['FT_PCT'], 'OREB': stats['OREB'],
                        'DREB': stats['DREB'], 'REB': stats['REB'], 'AST': stats['AST'], 'TOV': stats['TOV'],
                        'STL': stats['STL'], 'BLK': stats['BLK'], 'BLKA': stats['BLKA'], 'PF': stats['PF'],
                        'PFD': stats['PFD'], 'PTS': stats['PTS'], 'PLUS_MINUS': stats['PLUS_MINUS']
                    }
                    team_data.update(self._calculate_advanced_stats(team_data))
                    team_stats.append(team_data)
            except Exception as e:
                print(f"Error collecting stats for {abbrev}: {e}")
                continue
        return pd.DataFrame(team_stats)

    def _calculate_advanced_stats(self, team_data):
        try:
            possessions = team_data['FGA'] - team_data['OREB'] + team_data['TOV'] + (0.44 * team_data['FTA'])
            if possessions == 0: return {}
            
            off_rating = 100 * (team_data['PTS'] / possessions)
            # Defensive Rating is complex, so we use an approximation based on opponent points,
            # which we estimate from the team's plus-minus.
            pts_allowed = team_data['PTS'] - team_data['PLUS_MINUS']
            def_rating = 100 * (pts_allowed / possessions)

            return {
                'PACE': (48 / (team_data['MIN'] / 5)) * possessions / team_data['GP'] if team_data['GP'] > 0 else 100,
                'OFF_RATING': off_rating,
                'DEF_RATING': def_rating,
                'NET_RATING': off_rating - def_rating,
                'TS_PCT': team_data['PTS'] / (2 * (team_data['FGA'] + 0.44 * team_data['FTA'])),
                'EFG_PCT': (team_data['FGM'] + 0.5 * team_data['FG3M']) / team_data['FGA'],
                'TOV_PCT': team_data['TOV'] / possessions,
                'OREB_PCT': team_data['OREB'] / (team_data['OREB'] + team_data['DREB']), # Approximation
                'FT_RATE': team_data['FTA'] / team_data['FGA']
            }
        except (ZeroDivisionError, KeyError):
            return {} # Return empty dict if calculation fails

    def collect_historical_games(self, seasons=['2023-24', '2022-23']):
        print("Collecting historical game data...")
        if not NBA_API_AVAILABLE:
            return self._generate_dummy_historical_data()

        all_games = []
        id_to_abbrev = {v['id']: k for k, v in self.team_mapping.items()}

        for season in seasons:
            print(f"Fetching games for {season} season...")
            try:
                time.sleep(1)
                game_finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable='Regular Season',
                    headers=API_HEADERS,
                    timeout=30
                )
                games = game_finder.get_data_frames()[0]
                
                # Filter to unique games and process matchups
                games = games[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ID', 'WL']]
                # Each game appears twice, once for each team. We use this to establish home/away.
                games['is_home'] = games['MATCHUP'].str.contains('vs.')
                
                # Create a DataFrame with one row per game
                processed_games = []
                for game_id in games['GAME_ID'].unique():
                    game_entries = games[games['GAME_ID'] == game_id]
                    if len(game_entries) != 2: continue # Skip if we don't have both teams
                    
                    home_entry = game_entries[game_entries['is_home']]
                    away_entry = game_entries[~game_entries['is_home']]
                    
                    if home_entry.empty or away_entry.empty: continue

                    home_entry = home_entry.iloc[0]
                    away_entry = away_entry.iloc[0]

                    processed_games.append({
                        'GAME_ID': game_id,
                        'SEASON': season,
                        'HOME_TEAM': id_to_abbrev.get(home_entry['TEAM_ID']),
                        'AWAY_TEAM': id_to_abbrev.get(away_entry['TEAM_ID']),
                        'HOME_WIN': 1 if home_entry['WL'] == 'W' else 0
                    })
                all_games.extend(processed_games)

            except Exception as e:
                print(f"Error collecting games for {season}: {e}")
                continue

        return pd.DataFrame(all_games) if all_games else self._generate_dummy_historical_data()
        
    def _generate_dummy_team_stats(self):
        # Fallback dummy data if API fails
        return pd.DataFrame([{
            'TEAM': team, 'GP': 82, 'W': np.random.randint(20, 60), 'L': 82 - np.random.randint(20, 60),
            'WIN_PCT': np.random.uniform(0.3, 0.7), 'OFF_RATING': np.random.uniform(105, 120),
            'DEF_RATING': np.random.uniform(105, 120), 'NET_RATING': np.random.uniform(-8, 8),
            'PACE': np.random.uniform(95, 105), 'TS_PCT': np.random.uniform(0.52, 0.62),
            'EFG_PCT': np.random.uniform(0.50, 0.58), 'TOV_PCT': np.random.uniform(0.12, 0.16),
            'OREB_PCT': np.random.uniform(0.20, 0.30), 'FT_RATE': np.random.uniform(0.20, 0.30)
        } for team in self.team_mapping.keys()])

    def _generate_dummy_historical_data(self):
        teams = list(self.team_mapping.keys())
        return pd.DataFrame([{
            'GAME_ID': f"002{np.random.randint(10000, 99999)}",
            'HOME_TEAM': np.random.choice(teams),
            'AWAY_TEAM': np.random.choice([t for t in teams if t != np.random.choice(teams)]),
            'HOME_WIN': np.random.choice([0, 1]), 'SEASON': '2023-24'
        } for _ in range(1000)])

class NBAFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_matchup_features(self, team_stats_df, games_df):
        print("Engineering matchup features...")
        if team_stats_df.empty or games_df.empty:
             print("Warning: Missing stats or games data. Cannot create features.")
             return pd.DataFrame()

        matchup_features = []
        for _, game in games_df.iterrows():
            try:
                home_team = game['HOME_TEAM']
                away_team = game['AWAY_TEAM']
                home_stats = team_stats_df[team_stats_df['TEAM'] == home_team].iloc[0]
                away_stats = team_stats_df[team_stats_df['TEAM'] == away_team].iloc[0]

                features = {
                    'GAME_ID': game.get('GAME_ID', ''), 'HOME_TEAM': home_team,
                    'AWAY_TEAM': away_team, 'HOME_WIN': game.get('HOME_WIN', 0),
                    'NET_RATING_DIFF': home_stats['NET_RATING'] - away_stats['NET_RATING'],
                    'OFF_RATING_DIFF': home_stats['OFF_RATING'] - away_stats['OFF_RATING'],
                    'DEF_RATING_DIFF': home_stats['DEF_RATING'] - away_stats['DEF_RATING'],
                    'PACE_DIFF': home_stats['PACE'] - away_stats['PACE'],
                    'TS_PCT_DIFF': home_stats['TS_PCT'] - away_stats['TS_PCT'],
                    'EFG_PCT_DIFF': home_stats['EFG_PCT'] - away_stats['EFG_PCT'],
                    'TOV_PCT_DIFF': home_stats['TOV_PCT'] - away_stats['TOV_PCT'],
                    'OREB_PCT_DIFF': home_stats['OREB_PCT'] - away_stats['OREB_PCT'],
                    'FT_RATE_DIFF': home_stats['FT_RATE'] - away_stats['FT_RATE'],
                    'WIN_PCT_DIFF': home_stats['WIN_PCT'] - away_stats['WIN_PCT']
                }
                matchup_features.append(features)
            except (IndexError, KeyError) as e:
                # This will skip games where one of the teams is missing from the stats data
                # print(f"Skipping game {game.get('GAME_ID')}: Missing team stats for {home_team} or {away_team}. Error: {e}")
                continue
        
        return pd.DataFrame(matchup_features)

    def prepare_features_for_training(self, features_df):
        feature_cols = [col for col in features_df.columns if col not in ['GAME_ID', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_WIN']]
        X = features_df[feature_cols].fillna(0) # Fill NaNs with 0
        y = features_df['HOME_WIN'] if 'HOME_WIN' in features_df.columns else None
        return X, y, feature_cols

class NBAModelTrainer:
    def __init__(self):
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def train_models(self, X, y):
        print("Training multiple models...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale data for Logistic Regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        best_score = 0
        
        for name, model in models.items():
            print(f"Training {name}...")
            # Use scaled data only for Logistic Regression
            X_train_data = X_train_scaled if name == 'Logistic Regression' else X_train
            X_test_data = X_test_scaled if name == 'Logistic Regression' else X_test
            
            model.fit(X_train_data, y_train)
            test_score = model.score(X_test_data, y_test)
            
            results[name] = {'model': model, 'test_score': test_score}
            print(f"{name} Test Score: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                self.best_model = model
        
        self._calculate_feature_importance(X.columns)
        return results

    def _calculate_feature_importance(self, feature_names):
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
        else: return

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def predict_with_confidence(self, X):
        if self.best_model is None: raise ValueError("No trained model available")

        # Use the correct data (scaled or not) based on model type
        if isinstance(self.best_model, LogisticRegression):
            X_data = self.scaler.transform(X)
        else:
            X_data = X

        probabilities = self.best_model.predict_proba(X_data)
        predictions = self.best_model.predict(X_data)
        confidence_scores = np.max(probabilities, axis=1) * 100
        return predictions, probabilities, confidence_scores

    def save_model(self, filepath='nba_model.pkl'):
        model_data = {'model': self.best_model, 'scaler': self.scaler, 'features': self.feature_importance}
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='nba_model.pkl'):
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('features')
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Error: Model file '{filepath}' not found.")
            raise

class NBAPredictionSystem:
    def __init__(self):
        self.data_collector = NBADataCollector()
        self.feature_engineer = NBAFeatureEngineer()
        self.model_trainer = NBAModelTrainer()
        self.team_stats = None

    def train_full_system(self, seasons=['2023-24', '2022-23']):
        print("Starting full system training...")
        # Use stats from the most recent season for training feature generation
        self.team_stats = self.data_collector.collect_team_stats(season=seasons[0])
        historical_games = self.data_collector.collect_historical_games(seasons)
        
        if self.team_stats.empty or historical_games.empty:
            print("Critical Error: Cannot proceed with training due to lack of data.")
            return None

        features_df = self.feature_engineer.create_matchup_features(self.team_stats, historical_games)
        
        if features_df.empty:
            print("Error: No features were generated. Aborting training.")
            return None

        X, y, _ = self.feature_engineer.prepare_features_for_training(features_df)
        
        if X.empty or y is None:
            print("Error: Feature set is empty. Cannot train.")
            return None
            
        return self.model_trainer.train_models(X, y)

    def predict_games(self, games_data):
        if self.model_trainer.best_model is None:
            raise ValueError("Model not trained or loaded. Please train the system first.")
        
        # Ensure we have current season team stats for prediction
        if self.team_stats is None:
             self.team_stats = self.data_collector.collect_team_stats()
             if self.team_stats.empty:
                 print("Could not fetch team stats for prediction.")
                 return pd.DataFrame()

        features_df = self.feature_engineer.create_matchup_features(self.team_stats, games_data)
        
        if features_df.empty:
            print("Warning: No features generated for prediction.")
            return pd.DataFrame()

        X, _, _ = self.feature_engineer.prepare_features_for_training(features_df)
        predictions, probabilities, confidence = self.model_trainer.predict_with_confidence(X)

        results = features_df[['GAME_ID', 'HOME_TEAM', 'AWAY_TEAM']].copy()
        results['PREDICTED_HOME_WIN'] = predictions
        results['HOME_WIN_PROBABILITY'] = probabilities[:, 1]
        results['CONFIDENCE_SCORE'] = confidence
        return results

    def visualize_predictions(self, predictions):
        if predictions.empty:
            print("No predictions to visualize.")
            return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('NBA Game Prediction Dashboard', fontsize=16, fontweight='bold')
        
        sns.histplot(predictions['CONFIDENCE_SCORE'], bins=15, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title('Prediction Confidence Distribution', fontsize=12)
        axes[0].set_xlabel('Confidence Score', fontsize=10)
        axes[0].set_ylabel('Number of Games', fontsize=10)
        
        home_wins = (predictions['PREDICTED_HOME_WIN'] == 1).sum()
        away_wins = len(predictions) - home_wins
        axes[1].pie([home_wins, away_wins], labels=['Predicted Home Wins', 'Predicted Away Wins'],
                    autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
        axes[1].set_title('Home vs. Away Win Predictions', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def display_feature_importance(self):
        if self.model_trainer.feature_importance is None:
            print("No feature importance data available.")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.model_trainer.feature_importance.head(15)
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 15 Most Important Features', fontsize=14)
        plt.tight_layout()
        plt.show()

    def save_system(self, filepath='nba_prediction_system.pkl'):
        self.model_trainer.save_model(filepath)
    
    def load_system(self, filepath='nba_prediction_system.pkl'):
        self.model_trainer.load_model(filepath)

class DailyNBAPredictor:
    def __init__(self, model_path='nba_prediction_system.pkl'):
        self.system = NBAPredictionSystem()
        try:
            self.system.load_system(model_path)
            print("Prediction system loaded successfully.")
        except (FileNotFoundError, EOFError):
            print("No valid trained model found. Training a new one...")
            self.system.train_full_system()
            self.system.save_system(model_path)

    def get_todays_games(self, target_date=None):
        if target_date is None: target_date = datetime.now().strftime('%m/%d/%Y')
        print(f"Getting games for {target_date}...")
        
        if not NBA_API_AVAILABLE: return self._get_dummy_games()
        
        try:
            time.sleep(0.6)
            scoreboard = scoreboardv2.ScoreboardV2(game_date=target_date, headers=API_HEADERS, timeout=30)
            games = scoreboard.game_header.get_data_frame()
            if games.empty:
                print("No games scheduled for today.")
                return pd.DataFrame()

            id_to_abbrev = {v['id']: k for k, v in self.system.data_collector.team_mapping.items()}
            return pd.DataFrame({
                'GAME_ID': games['GAME_ID'],
                'HOME_TEAM': games['HOME_TEAM_ID'].map(id_to_abbrev),
                'AWAY_TEAM': games['VISITOR_TEAM_ID'].map(id_to_abbrev)
            })
        except Exception as e:
            print(f"Error getting today's games: {e}. Falling back to dummy data.")
            return self._get_dummy_games()

    def _get_dummy_games(self):
        teams = list(self.system.data_collector.team_mapping.keys())
        return pd.DataFrame([{
            'GAME_ID': f"dummy_game_{i}",
            'HOME_TEAM': np.random.choice(teams),
            'AWAY_TEAM': np.random.choice([t for t in teams if t != np.random.choice(teams)])
        } for i in range(5)])

    def predict_todays_games(self):
        print("\n" + "="*50)
        print("    NBA DAILY GAME PREDICTIONS")
        print("="*50)

        todays_games = self.get_todays_games()
        if todays_games.empty: return None

        predictions = self.system.predict_games(todays_games)
        if predictions.empty:
            print("No predictions were generated.")
            return None

        self.display_predictions(predictions)
        try:
            self.system.visualize_predictions(predictions)
        except Exception as e:
            print(f"Visualization error: {e}")
        
        return predictions

    def display_predictions(self, predictions):
        print(f"\nPREDICTIONS FOR {datetime.now().strftime('%Y-%m-%d')}")
        print("-" * 75)
        print(f"{'MATCHUP':<20} | {'PREDICTED WINNER':<20} | {'WIN PROBABILITY':<20} | {'CONFIDENCE':<10}")
        print("-" * 75)

        for _, pred in predictions.iterrows():
            home_team = pred['HOME_TEAM']
            away_team = pred['AWAY_TEAM']
            home_prob = pred['HOME_WIN_PROBABILITY']
            
            winner = home_team if pred['PREDICTED_HOME_WIN'] else away_team
            win_prob = home_prob if winner == home_team else 1 - home_prob
            
            matchup_str = f"{away_team} @ {home_team}"
            print(f"{matchup_str:<20} | {winner:<20} | {f'{win_prob:.1%}':<20} | {f'{pred.CONFIDENCE_SCORE:.1f}':<10}")
        
        print("-" * 75)

def main():
    print("NBA Game Prediction System Initializing...")
    print("=" * 40)
    
    if not NBA_API_AVAILABLE:
        print("⚠️  WARNING: nba-api not installed. Using dummy data for demonstration.")

    try:
        predictor = DailyNBAPredictor()
        predictor.predict_todays_games()
        print("\n✅ Analysis Complete!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()