import pyodbc
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATABASE CONNECTION
# ============================================================================
def connect_to_db():
    """Establish connection to MSSQL Server"""
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DESKTOP-J9IV3OH;'
        'DATABASE=nhlDB;'
        'Trusted_Connection=yes;'  # Use this for Windows Authentication
    )
    return conn

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================
def load_xg_model(model_path='nhl_xg_model.pkl'):
    """Load the trained xG model and encoders"""
    print("Loading trained xG model...")
    
    with open(model_path, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    model = model_artifacts['model']
    le_shot = model_artifacts['label_encoder_shot']
    feature_cols = model_artifacts['feature_cols']
    
    print(f"Model loaded successfully with {len(feature_cols)} features")
    return model, le_shot, feature_cols

# ============================================================================
# EXTRACT DATA FOR PREDICTION
# ============================================================================
def extract_shot_data(conn, season=None):
    """Extract all shot data for xG prediction"""
    
    season_filter = f"AND p.season = {season}" if season else ""
    
    query = f"""
    SELECT 
        p.game_id,
        p.season,
        p.event_id,
        p.sort_order,
        p.period_number,
        p.period_type,
        p.time_in_period,
        p.time_remaining,
        p.type_code,
        p.type_desc_key,
        p.situation_code,
        p.event_owner_team_id,
        p.x_coord,
        p.y_coord,
        p.zone_code,
        p.shot_type,
        
        -- FIX 1: Create a single 'shooter_id' column
        -- COALESCE takes the first non-NULL value.
        -- For goals, it will be scoring_player_id. For other shots, it will be shooting_player_id.
        COALESCE(p.scoring_player_id, p.shooting_player_id) AS shooter_id,

        p.goalie_in_net_id,
        p.away_score,
        p.home_score,
        p.blocking_player_id,
        p.missed_shot_reason
    FROM nhlDB.playbyplay.PLAY_EVENTS_COMPLETE p
    WHERE p.type_desc_key IN ('shot-on-goal', 'goal', 'missed-shot', 'blocked-shot')
        AND p.x_coord IS NOT NULL 
        AND p.y_coord IS NOT NULL
        -- FIX 2: Check the new unified column to ensure we have a player
        AND COALESCE(p.scoring_player_id, p.shooting_player_id) IS NOT NULL
        {season_filter}
    ORDER BY p.game_id, p.sort_order
    """
    
    df = pd.read_sql(query, conn)
    
    # Rename the new column to match the old one for consistency downstream
    df = df.rename(columns={'shooter_id': 'shooting_player_id'})

    print(f"Extracted {len(df):,} shot events")
    return df

def get_game_team_mapping(conn, season=None):
    """Get home/away team mapping by joining goals with the game roster"""
    
    season_filter = f"WHERE g.season = {season}" if season else ""
    
    # Correctly join GAME_GOALS with GAME_ROSTER to get the numeric team ID
    query = f"""
    SELECT DISTINCT
        g.game_id,
        r.team_id,      -- The numeric team ID from the roster
        g.is_home_goal
    FROM nhldb.gamecenter.GAME_GOALS AS g
    JOIN nhldb.playbyplay.GAME_ROSTER AS r 
      ON g.game_id = r.game_id 
     AND g.goal_scorer_player_id = r.player_id -- Use the correct column name
    {season_filter}
    """
    
    goals_df = pd.read_sql(query, conn)
    
    # FIX: Convert game_id to int64 for consistent merging
    goals_df['game_id'] = goals_df['game_id'].astype('int64')
    
    # This logic now works perfectly because it's using the numeric 'team_id'
    home_teams = goals_df[goals_df['is_home_goal'] == 1][['game_id', 'team_id']].drop_duplicates()
    home_teams.columns = ['game_id', 'home_team_id']
    
    away_teams = goals_df[goals_df['is_home_goal'] == 0][['game_id', 'team_id']].drop_duplicates()
    away_teams.columns = ['game_id', 'away_team_id']
    
    game_teams = home_teams.merge(away_teams, on='game_id', how='outer')
    
    return game_teams

# ============================================================================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ============================================================================
def engineer_features_for_prediction(df):
    """Create the same features used in training"""
    
    print("Engineering features for prediction...")
    df = df.copy()
    
    # Target variable
    df['is_goal'] = (df['type_code'] == 505).astype(int)
    df['type_code'] = pd.to_numeric(df['type_code'], errors='coerce')

    # Basic shot features
    df['x_adjusted'] = df['x_coord'].abs()
    df['y_adjusted'] = df['y_coord']
    df['distance'] = np.sqrt((89 - df['x_adjusted'])**2 + df['y_adjusted']**2)
    df['angle'] = np.abs(np.degrees(np.arctan2(df['y_adjusted'], (89 - df['x_adjusted']))))
    df['shot_type'] = df['shot_type'].fillna('unknown')
    
    # Game situation features
    df['situation_code'] = df['situation_code'].fillna('1551')
    df['skaters_attacking'] = df['situation_code'].astype(str).str[1].astype(int)
    df['skaters_defending'] = df['situation_code'].astype(str).str[2].astype(int)
    df['skater_diff'] = df['skaters_attacking'] - df['skaters_defending']
    df['is_powerplay'] = (df['skater_diff'] > 0).astype(int)
    df['is_shorthanded'] = (df['skater_diff'] < 0).astype(int)
    df['is_even_strength'] = (df['skater_diff'] == 0).astype(int)
    df['is_empty_net'] = (
        df['goalie_in_net_id'].isna() & 
        (df['type_desc_key'] != 'blocked-shot')
    ).astype(int)    

    # Score differential (requires home/away team mapping)
    # Sort values to ensure chronological order for filling
    df = df.sort_values(['game_id', 'sort_order'])

    # Forward-fill the scores within each game
    df['home_score'] = df.groupby('game_id')['home_score'].ffill()
    df['away_score'] = df.groupby('game_id')['away_score'].ffill()

    # Back-fill any NaNs at the start of a game (before the first goal)
    df['home_score'] = df.groupby('game_id')['home_score'].bfill()
    df['away_score'] = df.groupby('game_id')['away_score'].bfill()

    # Fill any remaining NaNs (for games with no goals) with 0
    df['home_score'] = df['home_score'].fillna(0)
    df['away_score'] = df['away_score'].fillna(0)


    # Determine if shooting team is home or away
    df['is_home_shot'] = (df['event_owner_team_id'] == df['home_team_id']).astype(int)

    # *** Adjust the score to reflect the state BEFORE the event ***
    # This part is correct and should remain.
    df['shooter_score_pre_event'] = np.where(
        df['is_home_shot'] == 1,
        df['home_score'],
        df['away_score']
    ) - df['is_goal']

    df['opponent_score_pre_event'] = np.where(
        df['is_home_shot'] == 1,
        df['away_score'],
        df['home_score']
    )

    # Calculate score differential from shooter's perspective using the pre-event scores
    df['score_diff'] = df['shooter_score_pre_event'] - df['opponent_score_pre_event']

    # Score state bins (leading, tied, trailing)
    df['is_leading'] = (df['score_diff'] > 0).astype(int)
    df['is_trailing'] = (df['score_diff'] < 0).astype(int)
    df['is_tied'] = (df['score_diff'] == 0).astype(int)
    
    # Time features
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return 0
        parts = str(time_str).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    df['seconds_in_period'] = df['time_in_period'].apply(time_to_seconds)
    df['seconds_remaining'] = df['time_remaining'].apply(time_to_seconds)
    df['is_period_1'] = (df['period_number'] == 1).astype(int)
    df['is_period_2'] = (df['period_number'] == 2).astype(int)
    df['is_period_3'] = (df['period_number'] == 3).astype(int)
    df['is_overtime'] = (df['period_type'] == 'OT').astype(int)
    
    # Sequence features
    df = df.sort_values(['game_id', 'sort_order']).reset_index(drop=True)
    df['time_since_last'] = df.groupby('game_id')['seconds_in_period'].diff()
    df['time_since_last'] = df['time_since_last'].fillna(999)
    
    df['x_last'] = df.groupby('game_id')['x_adjusted'].shift(1)
    df['y_last'] = df.groupby('game_id')['y_adjusted'].shift(1)
    df['distance_from_last'] = np.sqrt(
        (df['x_adjusted'] - df['x_last'])**2 + 
        (df['y_adjusted'] - df['y_last'])**2
    )
    df['distance_from_last'] = df['distance_from_last'].fillna(0)
    
    df['last_event_type'] = df.groupby('game_id')['type_code'].shift(1)
    df['last_event_team_id'] = df.groupby('game_id')['event_owner_team_id'].shift(1)
    
    # Rebound indicators (same team, quick shot after previous attempt)
    df['is_rebound'] = (
        (df['time_since_last'] <= 3) & 
        # FIX #2: Use numeric codes instead of text for the check
        (df['last_event_type'].isin([505, 506, 507, 508])) &
        (df['event_owner_team_id'] == df['last_event_team_id'])
    ).astype(int)
    
    # Rush indicator (quick shot after zone entry)
    df['is_rush'] = (
        (df['time_since_last'] <= 5) & 
        (df['distance_from_last'] > 30) &
        (df['event_owner_team_id'] == df['last_event_team_id'])
    ).astype(int)
    
    # Angle change
    df['angle_last'] = df.groupby('game_id')['angle'].shift(1)
    df['angle_change'] = np.abs(df['angle'] - df['angle_last'])
    df['angle_change'] = df['angle_change'].fillna(0)
    df['angle_change_rebound'] = np.where(df['is_rebound'] == 1, df['angle_change'], 0)
    
    # Location features
    df['is_slot'] = (
        (df['x_adjusted'] >= 69) & 
        (df['x_adjusted'] <= 89) & 
        (df['y_adjusted'].abs() <= 15)
    ).astype(int)
    
    slot_x, slot_y = 79, 0
    df['distance_from_slot'] = np.sqrt(
        (df['x_adjusted'] - slot_x)**2 + 
        (df['y_adjusted'] - slot_y)**2
    )
    
    df['is_behind_net'] = (df['x_adjusted'] > 89).astype(int)
    df['is_sharp_angle'] = (df['angle'] > 60).astype(int)
    
    print(f"Feature engineering complete")
    return df

# ============================================================================
# GENERATE XG PREDICTIONS
# ============================================================================
def generate_xg_predictions(df, model, le_shot, feature_cols):
    """Generate xG predictions for all shots"""
    
    print("Generating xG predictions...")
    
    # Encode shot type
    df['shot_type_encoded'] = le_shot.transform(df['shot_type'])
    
    # Prepare features
    X = df[feature_cols].fillna(0)
    
    # Generate predictions
    df['xG'] = model.predict_proba(X)[:, 1]
    
    print(f"Generated xG for {len(df):,} shots")
    print(f"Mean xG: {df['xG'].mean():.4f}")
    print(f"Total xG: {df['xG'].sum():.1f}")
    print(f"Total Goals: {df['is_goal'].sum()}")
    
    return df

# ============================================================================
# CREATE SHOT-LEVEL XG TABLE
# ============================================================================
def create_shot_xg_table(df, conn):
    """Create detailed shot-level xG table in SQL"""
    
    print("\nCreating shot-level xG table...")
    
    # Select columns for output
    output_cols = [
        'game_id', 'season', 'event_id', 'sort_order',
        'period_number', 'period_type', 'time_in_period',
        'event_owner_team_id', 'shooting_player_id', 'goalie_in_net_id',
        'shot_type', 'type_code', 'is_goal',
        'x_coord', 'y_coord', 'distance', 'angle',
        'situation_code', 'is_powerplay', 'is_shorthanded', 'is_even_strength',
        'is_empty_net', 'score_diff',
        'is_rebound', 'is_rush', 'is_slot',
        'xG'
    ]
    
    shot_xg_df = df[output_cols].copy()
    shot_xg_df = shot_xg_df.replace({np.nan: None})
    # Add timestamp
    shot_xg_df['created_at'] = datetime.now()
    
    # Create table if not exists
    cursor = conn.cursor()
    
    create_table_sql = """
    IF OBJECT_ID('nhldb.playbyplay.shot_xG', 'U') IS NOT NULL
        DROP TABLE nhldb.playbyplay.shot_xG;
    
    CREATE TABLE nhldb.playbyplay.shot_xG (
        game_id BIGINT,
        season INT,
        event_id INT,
        sort_order INT,
        period_number INT,
        period_type VARCHAR(10),
        time_in_period VARCHAR(10),
        event_owner_team_id INT,
        shooting_player_id INT,
        goalie_in_net_id INT,
        shot_type VARCHAR(20),
        type_code VARCHAR(20),
        is_goal INT,
        x_coord INT,
        y_coord INT,
        distance FLOAT,
        angle FLOAT,
        situation_code VARCHAR(10),
        is_powerplay INT,
        is_shorthanded INT,
        is_even_strength INT,
        is_empty_net INT,
        score_diff INT,
        is_rebound INT,
        is_rush INT,
        is_slot INT,
        xG FLOAT,
        created_at DATETIME,
        PRIMARY KEY (game_id, event_id)
    );
    """
    
    cursor.execute(create_table_sql)
    conn.commit()
    print("Created shot_xG table")
    
    # Insert data in batches
    batch_size = 1000
    total_rows = len(shot_xg_df)
    
    for i in range(0, total_rows, batch_size):
        batch = shot_xg_df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            insert_sql = """
            INSERT INTO nhldb.playbyplay.shot_xG VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, tuple(row))
        
        conn.commit()
        print(f"Inserted {min(i+batch_size, total_rows):,} / {total_rows:,} rows")
    
    print(f"✓ Shot-level xG table created with {total_rows:,} shots")

# ============================================================================
# CREATE TEAM GAME-BY-GAME XG TABLE
# ============================================================================
def create_team_game_xg_table(df, conn):
    """Create team game-by-game xG summary table"""
    
    print("\nCreating team game-by-game xG table...")
    
    # Get unique games
    games = df[['game_id', 'season', 'home_team_id', 'away_team_id']].dropna().drop_duplicates()
    
    team_game_stats = []
    
    for _, game_row in games.iterrows():
        game_id = game_row['game_id']
        season = game_row['season']
        home_team = game_row['home_team_id']
        away_team = game_row['away_team_id']
        
        game_shots = df[df['game_id'] == game_id].copy()
        
        # Home team stats
        home_shots = game_shots[game_shots['event_owner_team_id'] == home_team]
        away_shots = game_shots[game_shots['event_owner_team_id'] == away_team]
        
        # Get goalies (most common goalie in net for each team's shots against)
        home_goalie = away_shots['goalie_in_net_id'].mode()[0] if len(away_shots) > 0 and not away_shots['goalie_in_net_id'].mode().empty else None
        away_goalie = home_shots['goalie_in_net_id'].mode()[0] if len(home_shots) > 0 and not home_shots['goalie_in_net_id'].mode().empty else None
        
        # Home team row
        team_game_stats.append({
            'game_id': game_id,
            'season': season,
            'team_id': home_team,
            'is_home': 1,
            'opponent_team_id': away_team,
            'goalie_id': home_goalie,
            'opponent_goalie_id': away_goalie,
            'shots_for': len(home_shots),
            'shots_against': len(away_shots),
            'goals_for': home_shots['is_goal'].sum(),
            'goals_against': away_shots['is_goal'].sum(),
            'xG_for': home_shots['xG'].sum(),
            'xG_against': away_shots['xG'].sum(),
            'xG_diff': home_shots['xG'].sum() - away_shots['xG'].sum(),
            'shooting_percentage': home_shots['is_goal'].sum() / len(home_shots) if len(home_shots) > 0 else 0,
            'save_percentage': 1 - (away_shots['is_goal'].sum() / len(away_shots)) if len(away_shots) > 0 else 0,
            'xG_for_5v5': home_shots[home_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_against_5v5': away_shots[away_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_for_PP': home_shots[home_shots['is_powerplay'] == 1]['xG'].sum(),
            'xG_against_PP': away_shots[away_shots['is_powerplay'] == 1]['xG'].sum(),
        })
        
        # Away team row
        team_game_stats.append({
            'game_id': game_id,
            'season': season,
            'team_id': away_team,
            'is_home': 0,
            'opponent_team_id': home_team,
            'goalie_id': away_goalie,
            'opponent_goalie_id': home_goalie,
            'shots_for': len(away_shots),
            'shots_against': len(home_shots),
            'goals_for': away_shots['is_goal'].sum(),
            'goals_against': home_shots['is_goal'].sum(),
            'xG_for': away_shots['xG'].sum(),
            'xG_against': home_shots['xG'].sum(),
            'xG_diff': away_shots['xG'].sum() - home_shots['xG'].sum(),
            'shooting_percentage': away_shots['is_goal'].sum() / len(away_shots) if len(away_shots) > 0 else 0,
            'save_percentage': 1 - (home_shots['is_goal'].sum() / len(home_shots)) if len(home_shots) > 0 else 0,
            'xG_for_5v5': away_shots[away_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_against_5v5': home_shots[home_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_for_PP': away_shots[away_shots['is_powerplay'] == 1]['xG'].sum(),
            'xG_against_PP': home_shots[home_shots['is_powerplay'] == 1]['xG'].sum(),
        })
    
    team_game_df = pd.DataFrame(team_game_stats)
    team_game_df = team_game_df.replace({np.nan: None})
    team_game_df['created_at'] = datetime.now()
    
    # Create table
    cursor = conn.cursor()
    
    create_table_sql = """
    IF OBJECT_ID('nhldb.playbyplay.team_game_xG', 'U') IS NOT NULL
        DROP TABLE nhldb.playbyplay.team_game_xG;
    
    CREATE TABLE nhldb.playbyplay.team_game_xG (
        game_id BIGINT,
        season INT,
        team_id VARCHAR(10),
        is_home INT,
        opponent_team_id VARCHAR(10),
        goalie_id INT,
        opponent_goalie_id INT,
        shots_for INT,
        shots_against INT,
        goals_for INT,
        goals_against INT,
        xG_for FLOAT,
        xG_against FLOAT,
        xG_diff FLOAT,
        shooting_percentage FLOAT,
        save_percentage FLOAT,
        xG_for_5v5 FLOAT,
        xG_against_5v5 FLOAT,
        xG_for_PP FLOAT,
        xG_against_PP FLOAT,
        created_at DATETIME,
        PRIMARY KEY (game_id, team_id)
    );
    """
    
    cursor.execute(create_table_sql)
    conn.commit()
    print("Created team_game_xG table")
    
    # Insert data
    for _, row in team_game_df.iterrows():
        insert_sql = """
        INSERT INTO nhldb.playbyplay.team_game_xG VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_sql, tuple(row))
    
    conn.commit()
    print(f"✓ Team game-by-game xG table created with {len(team_game_df):,} rows")
    
    return team_game_df

# ============================================================================
# INCREMENTAL UPDATE HELPERS
# ============================================================================

def insert_shot_xg_table(df, conn):
    """Insert new shot-level xG data without dropping table"""
    
    print("\nInserting new shots into shot_xG table...")
    
    # Select columns for output
    output_cols = [
        'game_id', 'season', 'event_id', 'sort_order',
        'period_number', 'period_type', 'time_in_period',
        'event_owner_team_id', 'shooting_player_id', 'goalie_in_net_id',
        'shot_type', 'type_code', 'is_goal',
        'x_coord', 'y_coord', 'distance', 'angle',
        'situation_code', 'is_powerplay', 'is_shorthanded', 'is_even_strength',
        'is_empty_net', 'score_diff',
        'is_rebound', 'is_rush', 'is_slot',
        'xG'
    ]
    
    shot_xg_df = df[output_cols].copy()
    shot_xg_df = shot_xg_df.replace({np.nan: None})
    shot_xg_df['created_at'] = datetime.now()
    
    cursor = conn.cursor()
    
    # Insert data in batches
    batch_size = 1000
    total_rows = len(shot_xg_df)
    
    for i in range(0, total_rows, batch_size):
        batch = shot_xg_df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            insert_sql = """
            INSERT INTO nhldb.playbyplay.shot_xG VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, tuple(row))
        
        conn.commit()
        print(f"Inserted {min(i+batch_size, total_rows):,} / {total_rows:,} rows")
    
    print(f"✓ Added {total_rows:,} new shots to shot_xG table")

def insert_team_game_xg_table(df, conn):
    """Insert new team game-by-game xG data without dropping table"""
    
    print("\nInserting new games into team_game_xG table...")
    
    # Get unique games
    games = df[['game_id', 'season', 'home_team_id', 'away_team_id']].dropna().drop_duplicates()
    
    team_game_stats = []
    
    for _, game_row in games.iterrows():
        game_id = game_row['game_id']
        season = game_row['season']
        home_team = game_row['home_team_id']
        away_team = game_row['away_team_id']
        
        game_shots = df[df['game_id'] == game_id].copy()
        
        # Home team stats
        home_shots = game_shots[game_shots['event_owner_team_id'] == home_team]
        away_shots = game_shots[game_shots['event_owner_team_id'] == away_team]
        
        # Get goalies
        home_goalie = away_shots['goalie_in_net_id'].mode()[0] if len(away_shots) > 0 and not away_shots['goalie_in_net_id'].mode().empty else None
        away_goalie = home_shots['goalie_in_net_id'].mode()[0] if len(home_shots) > 0 and not home_shots['goalie_in_net_id'].mode().empty else None
        
        # Home team row
        team_game_stats.append({
            'game_id': game_id,
            'season': season,
            'team_id': home_team,
            'is_home': 1,
            'opponent_team_id': away_team,
            'goalie_id': home_goalie,
            'opponent_goalie_id': away_goalie,
            'shots_for': len(home_shots),
            'shots_against': len(away_shots),
            'goals_for': home_shots['is_goal'].sum(),
            'goals_against': away_shots['is_goal'].sum(),
            'xG_for': home_shots['xG'].sum(),
            'xG_against': away_shots['xG'].sum(),
            'xG_diff': home_shots['xG'].sum() - away_shots['xG'].sum(),
            'shooting_percentage': home_shots['is_goal'].sum() / len(home_shots) if len(home_shots) > 0 else 0,
            'save_percentage': 1 - (away_shots['is_goal'].sum() / len(away_shots)) if len(away_shots) > 0 else 0,
            'xG_for_5v5': home_shots[home_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_against_5v5': away_shots[away_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_for_PP': home_shots[home_shots['is_powerplay'] == 1]['xG'].sum(),
            'xG_against_PP': away_shots[away_shots['is_powerplay'] == 1]['xG'].sum(),
        })
        
        # Away team row
        team_game_stats.append({
            'game_id': game_id,
            'season': season,
            'team_id': away_team,
            'is_home': 0,
            'opponent_team_id': home_team,
            'goalie_id': away_goalie,
            'opponent_goalie_id': home_goalie,
            'shots_for': len(away_shots),
            'shots_against': len(home_shots),
            'goals_for': away_shots['is_goal'].sum(),
            'goals_against': home_shots['is_goal'].sum(),
            'xG_for': away_shots['xG'].sum(),
            'xG_against': home_shots['xG'].sum(),
            'xG_diff': away_shots['xG'].sum() - home_shots['xG'].sum(),
            'shooting_percentage': away_shots['is_goal'].sum() / len(away_shots) if len(away_shots) > 0 else 0,
            'save_percentage': 1 - (home_shots['is_goal'].sum() / len(home_shots)) if len(home_shots) > 0 else 0,
            'xG_for_5v5': away_shots[away_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_against_5v5': home_shots[home_shots['is_even_strength'] == 1]['xG'].sum(),
            'xG_for_PP': away_shots[away_shots['is_powerplay'] == 1]['xG'].sum(),
            'xG_against_PP': home_shots[home_shots['is_powerplay'] == 1]['xG'].sum(),
        })
    
    team_game_df = pd.DataFrame(team_game_stats)
    team_game_df = team_game_df.replace({np.nan: None})
    team_game_df['created_at'] = datetime.now()
    
    cursor = conn.cursor()
    
    # Insert data
    for _, row in team_game_df.iterrows():
        insert_sql = """
        INSERT INTO nhldb.playbyplay.team_game_xG VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_sql, tuple(row))
    
    conn.commit()
    print(f"✓ Added {len(team_game_df):,} new rows to team_game_xG table")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def get_existing_games(conn, table_name='nhldb.playbyplay.shot_xG'):
    """Get list of game_ids that already have xG predictions"""
    try:
        query = f"SELECT DISTINCT game_id FROM {table_name}"
        existing = pd.read_sql(query, conn)
        # FIX: Ensure game_id is the correct data type (int64)
        return set(existing['game_id'].astype('int64').values)
    except Exception as e:
        # Table doesn't exist yet
        print(f"No existing xG tables found - will create new tables ({e})")
        return set()

def main(season=None, incremental=True):
    print("="*70)
    print("NHL xG PREDICTION PIPELINE - GENERATE SQL TABLES")
    print("="*70)
    
    # Load model
    model, le_shot, feature_cols = load_xg_model()
    
    # Connect to database
    print("\nConnecting to database...")
    conn = connect_to_db()
    
    # Check for existing games if incremental mode
    if incremental:
        existing_games = get_existing_games(conn)
        print(f"Found {len(existing_games)} games with existing xG data")
    else:
        existing_games = set()
        print("Running in FULL REFRESH mode - will recreate all tables")
    
    # Extract data
    df = extract_shot_data(conn, season=season)
    
    # FIX: Ensure game_id column is int64 for proper comparison
    df['game_id'] = df['game_id'].astype('int64')
    
    # Filter to only new games if incremental
    if incremental and len(existing_games) > 0:
        initial_count = len(df)
        initial_games = df['game_id'].nunique()
        
        # Debug: Check data types
        print(f"DataFrame game_id type: {df['game_id'].dtype}")
        print(f"Sample existing game IDs: {list(existing_games)[:5]}")
        print(f"Sample DataFrame game IDs: {df['game_id'].head().tolist()}")
        
        df = df[~df['game_id'].isin(existing_games)]
        
        print(f"Filtered from {initial_count:,} to {len(df):,} shots")
        print(f"Processing {df['game_id'].nunique()} new games (skipped {initial_games - df['game_id'].nunique()} existing)")
    
    # Check if there's any data to process
    if len(df) == 0:
        print("\n✓ No new games to process - database is up to date!")
        conn.close()
        return None
    
    # Get team mappings
    game_teams = get_game_team_mapping(conn, season=season)
    df = df.merge(game_teams, on='game_id', how='left')
    
    # Engineer features
    df = engineer_features_for_prediction(df)
    
    # Generate xG predictions
    df = generate_xg_predictions(df, model, le_shot, feature_cols)
    
    # Create or update SQL tables
    if incremental and len(existing_games) > 0:
        insert_shot_xg_table(df, conn)
        insert_team_game_xg_table(df, conn)
    else:
        create_shot_xg_table(df, conn)
        create_team_game_xg_table(df, conn)
    
    conn.close()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    if incremental and len(existing_games) > 0:
        print("\nTables updated:")
        print(f"  1. nhldb.playbyplay.shot_xG (+{len(df):,} new shots)")
        print(f"  2. nhldb.playbyplay.team_game_xG (+{df['game_id'].nunique() * 2} new rows)")
    else:
        print("\nTables created:")
        print("  1. nhldb.playbyplay.shot_xG (shot-level predictions)")
        print("  2. nhldb.playbyplay.team_game_xG (team game-by-game stats)")
    
    return df

if __name__ == "__main__":
    # Run for all seasons, or specify a season like: main(season=20232024)
    df = main()