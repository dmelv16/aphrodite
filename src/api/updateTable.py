import pyodbc
import pandas as pd
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
        'Trusted_Connection=yes;'
    )
    return conn

# ============================================================================
# GET EXISTING GAMES
# ============================================================================
def get_existing_games(conn, schema, table):
    """Get set of game_ids that already exist in a table"""
    try:
        query = f"SELECT DISTINCT game_id FROM {schema}.{table}"
        existing = pd.read_sql(query, conn)
        return set(existing['game_id'].values)
    except Exception as e:
        print(f"Table {schema}.{table} doesn't exist yet or error: {e}")
        return set()

# ============================================================================
# UPDATE PLAYBYPLAY TABLES
# ============================================================================
def update_playbyplay_tables(conn):
    """Update play-by-play tables with new games only"""
    
    print("\n" + "="*70)
    print("UPDATING PLAY-BY-PLAY TABLES")
    print("="*70)
    
    cursor = conn.cursor()
    
    # Check existing games
    existing_games = get_existing_games(conn, 'playbyplay', 'PLAY_BY_PLAY_GAME_MASTER')
    print(f"\nFound {len(existing_games)} existing games in PLAY_BY_PLAY_GAME_MASTER")
    
    # Build exclusion filter
    if len(existing_games) > 0:
        exclude_list = ','.join([f"'{g}'" for g in existing_games])
        where_clause = f"WHERE L.game_id NOT IN ({exclude_list})"
    else:
        where_clause = ""
        print("No existing games - will process all records")
    
    # 1. INSERT NEW GAMES INTO PLAY_BY_PLAY_GAME_MASTER
    print("\n1. Inserting new games into PLAY_BY_PLAY_GAME_MASTER...")
    insert_game_master = f"""
    INSERT INTO [nhlDB].[playbyplay].[PLAY_BY_PLAY_GAME_MASTER]
    SELECT 
        L.game_id,
        L.season,
        JSON_VALUE(L.data_json, '$.id') AS game_id_from_json,
        JSON_VALUE(L.data_json, '$.gameType') AS game_type,
        JSON_VALUE(L.data_json, '$.limitedScoring') AS limited_scoring,
        JSON_VALUE(L.data_json, '$.gameDate') AS game_date,
        JSON_VALUE(L.data_json, '$.gameState') AS game_state,
        JSON_VALUE(L.data_json, '$.gameScheduleState') AS game_schedule_state,
        JSON_VALUE(L.data_json, '$.venue.default') AS venue_name,
        JSON_VALUE(L.data_json, '$.venueLocation.default') AS venue_location,
        JSON_VALUE(L.data_json, '$.startTimeUTC') AS start_time_utc,
        JSON_VALUE(L.data_json, '$.easternUTCOffset') AS eastern_utc_offset,
        JSON_VALUE(L.data_json, '$.venueUTCOffset') AS venue_utc_offset,
        JSON_VALUE(L.data_json, '$.periodDescriptor.number') AS current_period,
        JSON_VALUE(L.data_json, '$.periodDescriptor.periodType') AS current_period_type,
        JSON_VALUE(L.data_json, '$.periodDescriptor.maxRegulationPeriods') AS max_regulation_periods,
        JSON_VALUE(L.data_json, '$.clock.timeRemaining') AS time_remaining,
        JSON_VALUE(L.data_json, '$.clock.secondsRemaining') AS seconds_remaining,
        JSON_VALUE(L.data_json, '$.clock.running') AS clock_running,
        JSON_VALUE(L.data_json, '$.clock.inIntermission') AS in_intermission,
        JSON_VALUE(L.data_json, '$.awayTeam.id') AS away_team_id,
        JSON_VALUE(L.data_json, '$.awayTeam.abbrev') AS away_team_abbrev,
        JSON_VALUE(L.data_json, '$.awayTeam.commonName.default') AS away_team_name,
        JSON_VALUE(L.data_json, '$.awayTeam.placeName.default') AS away_place_name,
        JSON_VALUE(L.data_json, '$.awayTeam.score') AS away_team_score,
        JSON_VALUE(L.data_json, '$.awayTeam.sog') AS away_team_shots,
        JSON_VALUE(L.data_json, '$.awayTeam.logo') AS away_team_logo,
        JSON_VALUE(L.data_json, '$.awayTeam.darkLogo') AS away_team_dark_logo,
        JSON_VALUE(L.data_json, '$.homeTeam.id') AS home_team_id,
        JSON_VALUE(L.data_json, '$.homeTeam.abbrev') AS home_team_abbrev,
        JSON_VALUE(L.data_json, '$.homeTeam.commonName.default') AS home_team_name,
        JSON_VALUE(L.data_json, '$.homeTeam.placeName.default') AS home_place_name,
        JSON_VALUE(L.data_json, '$.homeTeam.score') AS home_team_score,
        JSON_VALUE(L.data_json, '$.homeTeam.sog') AS home_team_shots,
        JSON_VALUE(L.data_json, '$.homeTeam.logo') AS home_team_logo,
        JSON_VALUE(L.data_json, '$.homeTeam.darkLogo') AS home_team_dark_logo,
        JSON_VALUE(L.data_json, '$.shootoutInUse') AS shootout_in_use,
        JSON_VALUE(L.data_json, '$.otInUse') AS ot_in_use,
        JSON_VALUE(L.data_json, '$.maxPeriods') AS max_periods,
        JSON_VALUE(L.data_json, '$.regPeriods') AS regulation_periods,
        JSON_VALUE(L.data_json, '$.displayPeriod') AS display_period,
        JSON_VALUE(L.data_json, '$.gameOutcome.lastPeriodType') AS last_period_type,
        L.scraped_at
    FROM [nhlDB].[playbyplay].[play_by_play] L
    {where_clause}
    """
    
    cursor.execute(insert_game_master)
    rows_inserted = cursor.rowcount
    conn.commit()
    print(f"✓ Inserted {rows_inserted} new games")
    
    if rows_inserted == 0:
        print("\n✓ No new games to process - all tables are up to date!")
        return
    
    # 2. INSERT PLAY EVENTS
    print("\n2. Inserting play events for new games...")
    insert_events = f"""
    INSERT INTO [nhlDB].[playbyplay].[PLAY_EVENTS_COMPLETE]
        (game_id, season, event_id, sort_order, period_number, period_type, max_reg_periods,
         time_in_period, time_remaining, type_code, type_desc_key, situation_code,
         event_owner_team_id, x_coord, y_coord, zone_code,
         shot_type, shooting_player_id, goalie_in_net_id, away_sog, home_sog,
         scoring_player_id, scoring_player_total, assist1_player_id, assist1_player_total,
         assist2_player_id, assist2_player_total, away_score, home_score,
         faceoff_winning_player_id, faceoff_losing_player_id,
         hitting_player_id, hittee_player_id, blocking_player_id,
         penalty_type_code, penalty_desc_key, penalty_duration,
         penalty_committed_by_player_id, penalty_drawn_by_player_id, penalty_served_by_player_id,
         stoppage_reason, stoppage_secondary_reason,
         giveaway_takeaway_player_id, missed_shot_reason)
    SELECT 
        L.game_id,
        L.season,
        JSON_VALUE(play.value, '$.eventId') AS event_id,
        JSON_VALUE(play.value, '$.sortOrder') AS sort_order,
        JSON_VALUE(play.value, '$.periodDescriptor.number') AS period_number,
        JSON_VALUE(play.value, '$.periodDescriptor.periodType') AS period_type,
        JSON_VALUE(play.value, '$.periodDescriptor.maxRegulationPeriods') AS max_reg_periods,
        JSON_VALUE(play.value, '$.timeInPeriod') AS time_in_period,
        JSON_VALUE(play.value, '$.timeRemaining') AS time_remaining,
        JSON_VALUE(play.value, '$.typeCode') AS type_code,
        JSON_VALUE(play.value, '$.typeDescKey') AS type_desc_key,
        JSON_VALUE(play.value, '$.situationCode') AS situation_code,
        JSON_VALUE(play.value, '$.details.eventOwnerTeamId') AS event_owner_team_id,
        JSON_VALUE(play.value, '$.details.xCoord') AS x_coord,
        JSON_VALUE(play.value, '$.details.yCoord') AS y_coord,
        JSON_VALUE(play.value, '$.details.zoneCode') AS zone_code,
        JSON_VALUE(play.value, '$.details.shotType') AS shot_type,
        JSON_VALUE(play.value, '$.details.shootingPlayerId') AS shooting_player_id,
        JSON_VALUE(play.value, '$.details.goalieInNetId') AS goalie_in_net_id,
        JSON_VALUE(play.value, '$.details.awaySOG') AS away_sog,
        JSON_VALUE(play.value, '$.details.homeSOG') AS home_sog,
        JSON_VALUE(play.value, '$.details.scoringPlayerId') AS scoring_player_id,
        JSON_VALUE(play.value, '$.details.scoringPlayerTotal') AS scoring_player_total,
        JSON_VALUE(play.value, '$.details.assist1PlayerId') AS assist1_player_id,
        JSON_VALUE(play.value, '$.details.assist1PlayerTotal') AS assist1_player_total,
        JSON_VALUE(play.value, '$.details.assist2PlayerId') AS assist2_player_id,
        JSON_VALUE(play.value, '$.details.assist2PlayerTotal') AS assist2_player_total,
        JSON_VALUE(play.value, '$.details.awayScore') AS away_score,
        JSON_VALUE(play.value, '$.details.homeScore') AS home_score,
        JSON_VALUE(play.value, '$.details.winningPlayerId') AS faceoff_winning_player_id,
        JSON_VALUE(play.value, '$.details.losingPlayerId') AS faceoff_losing_player_id,
        JSON_VALUE(play.value, '$.details.hittingPlayerId') AS hitting_player_id,
        JSON_VALUE(play.value, '$.details.hitteePlayerId') AS hittee_player_id,
        JSON_VALUE(play.value, '$.details.blockingPlayerId') AS blocking_player_id,
        JSON_VALUE(play.value, '$.details.typeCode') AS penalty_type_code,
        JSON_VALUE(play.value, '$.details.descKey') AS penalty_desc_key,
        JSON_VALUE(play.value, '$.details.duration') AS penalty_duration,
        JSON_VALUE(play.value, '$.details.committedByPlayerId') AS penalty_committed_by_player_id,
        JSON_VALUE(play.value, '$.details.drawnByPlayerId') AS penalty_drawn_by_player_id,
        JSON_VALUE(play.value, '$.details.servedByPlayerId') AS penalty_served_by_player_id,
        JSON_VALUE(play.value, '$.details.reason') AS stoppage_reason,
        JSON_VALUE(play.value, '$.details.secondaryReason') AS stoppage_secondary_reason,
        JSON_VALUE(play.value, '$.details.playerId') AS giveaway_takeaway_player_id,
        JSON_VALUE(play.value, '$.details.reason') AS missed_shot_reason
    FROM [nhlDB].[playbyplay].[play_by_play] L
    OUTER APPLY OPENJSON(L.data_json, '$.plays') AS play
    WHERE JSON_VALUE(play.value, '$.eventId') IS NOT NULL
    {where_clause.replace('WHERE', 'AND')}
    """
    
    cursor.execute(insert_events)
    rows_inserted = cursor.rowcount
    conn.commit()
    print(f"✓ Inserted {rows_inserted} new play events")
    
    # 3. INSERT GAME ROSTER
    print("\n3. Inserting game roster for new games...")
    insert_roster = f"""
    INSERT INTO [nhlDB].[playbyplay].[GAME_ROSTER]
        (game_id, season, player_id, team_id, first_name, last_name, 
         sweater_number, position_code, headshot_url)
    SELECT 
        L.game_id,
        L.season,
        JSON_VALUE(roster.value, '$.playerId') AS player_id,
        JSON_VALUE(roster.value, '$.teamId') AS team_id,
        JSON_VALUE(roster.value, '$.firstName.default') AS first_name,
        JSON_VALUE(roster.value, '$.lastName.default') AS last_name,
        JSON_VALUE(roster.value, '$.sweaterNumber') AS sweater_number,
        JSON_VALUE(roster.value, '$.positionCode') AS position_code,
        JSON_VALUE(roster.value, '$.headshot') AS headshot_url
    FROM [nhlDB].[playbyplay].[play_by_play] L
    OUTER APPLY OPENJSON(L.data_json, '$.rosterSpots') AS roster
    WHERE JSON_VALUE(roster.value, '$.playerId') IS NOT NULL
    {where_clause.replace('WHERE', 'AND')}
    """
    
    cursor.execute(insert_roster)
    rows_inserted = cursor.rowcount
    conn.commit()
    print(f"✓ Inserted {rows_inserted} roster records")
    
    # 4. INSERT TV BROADCASTS
    print("\n4. Inserting TV broadcasts for new games...")
    insert_broadcasts = f"""
    INSERT INTO [nhlDB].[playbyplay].[TV_BROADCASTS_PBP]
        (game_id, season, broadcast_id, market, country_code, network, sequence_number)
    SELECT 
        L.game_id,
        L.season,
        JSON_VALUE(broadcast.value, '$.id') AS broadcast_id,
        JSON_VALUE(broadcast.value, '$.market') AS market,
        JSON_VALUE(broadcast.value, '$.countryCode') AS country_code,
        JSON_VALUE(broadcast.value, '$.network') AS network,
        JSON_VALUE(broadcast.value, '$.sequenceNumber') AS sequence_number
    FROM [nhlDB].[playbyplay].[play_by_play] L
    OUTER APPLY OPENJSON(L.data_json, '$.tvBroadcasts') AS broadcast
    WHERE JSON_VALUE(broadcast.value, '$.network') IS NOT NULL
    {where_clause.replace('WHERE', 'AND')}
    """
    
    cursor.execute(insert_broadcasts)
    rows_inserted = cursor.rowcount
    conn.commit()
    print(f"✓ Inserted {rows_inserted} broadcast records")

# ============================================================================
# UPDATE GAMECENTER LANDING TABLES
# ============================================================================
def update_gamecenter_landing_tables(conn):
    """Update gamecenter landing tables with new games only"""
    
    print("\n" + "="*70)
    print("UPDATING GAMECENTER LANDING TABLES")
    print("="*70)
    
    cursor = conn.cursor()
    
    # Check existing games
    existing_games = get_existing_games(conn, 'gamecenter', 'GAME_MASTER')
    print(f"\nFound {len(existing_games)} existing games in GAME_MASTER")
    
    # Build exclusion filter
    if len(existing_games) > 0:
        exclude_list = ','.join([f"'{g}'" for g in existing_games])
        where_clause = f"WHERE L.game_id NOT IN ({exclude_list})"
    else:
        where_clause = ""
        print("No existing games - will process all records")
    
    # 1. INSERT GAME_MASTER
    print("\n1. Inserting new games into GAME_MASTER...")
    insert_game_master = f"""
    INSERT INTO [nhlDB].[gamecenter].[GAME_MASTER]
    SELECT 
        L.game_id,
        L.season,
        JSON_VALUE(L.data_json, '$.id') AS game_id_from_json,
        JSON_VALUE(L.data_json, '$.gameType') AS game_type,
        JSON_VALUE(L.data_json, '$.limitedScoring') AS limited_scoring,
        JSON_VALUE(L.data_json, '$.gameDate') AS game_date,
        JSON_VALUE(L.data_json, '$.gameState') AS game_state,
        JSON_VALUE(L.data_json, '$.gameScheduleState') AS game_schedule_state,
        JSON_VALUE(L.data_json, '$.venue.default') AS venue_name,
        JSON_VALUE(L.data_json, '$.venueLocation.default') AS venue_location,
        JSON_VALUE(L.data_json, '$.venueTimezone') AS venue_timezone,
        JSON_VALUE(L.data_json, '$.startTimeUTC') AS start_time_utc,
        JSON_VALUE(L.data_json, '$.easternUTCOffset') AS eastern_utc_offset,
        JSON_VALUE(L.data_json, '$.venueUTCOffset') AS venue_utc_offset,
        JSON_VALUE(L.data_json, '$.periodDescriptor.number') AS current_period,
        JSON_VALUE(L.data_json, '$.periodDescriptor.periodType') AS current_period_type,
        JSON_VALUE(L.data_json, '$.periodDescriptor.maxRegulationPeriods') AS max_regulation_periods,
        JSON_VALUE(L.data_json, '$.clock.timeRemaining') AS time_remaining,
        JSON_VALUE(L.data_json, '$.clock.secondsRemaining') AS seconds_remaining,
        JSON_VALUE(L.data_json, '$.clock.running') AS clock_running,
        JSON_VALUE(L.data_json, '$.clock.inIntermission') AS in_intermission,
        JSON_VALUE(L.data_json, '$.awayTeam.id') AS away_team_id,
        JSON_VALUE(L.data_json, '$.awayTeam.abbrev') AS away_team_abbrev,
        JSON_VALUE(L.data_json, '$.awayTeam.commonName.default') AS away_team_name,
        JSON_VALUE(L.data_json, '$.awayTeam.placeName.default') AS away_place_name,
        JSON_VALUE(L.data_json, '$.awayTeam.placeNameWithPreposition.default') AS away_place_with_prep,
        JSON_VALUE(L.data_json, '$.awayTeam.placeNameWithPreposition.fr') AS away_place_with_prep_fr,
        JSON_VALUE(L.data_json, '$.awayTeam.score') AS away_team_score,
        JSON_VALUE(L.data_json, '$.awayTeam.sog') AS away_team_shots,
        JSON_VALUE(L.data_json, '$.awayTeam.logo') AS away_team_logo,
        JSON_VALUE(L.data_json, '$.awayTeam.darkLogo') AS away_team_dark_logo,
        JSON_VALUE(L.data_json, '$.homeTeam.id') AS home_team_id,
        JSON_VALUE(L.data_json, '$.homeTeam.abbrev') AS home_team_abbrev,
        JSON_VALUE(L.data_json, '$.homeTeam.commonName.default') AS home_team_name,
        JSON_VALUE(L.data_json, '$.homeTeam.placeName.default') AS home_place_name,
        JSON_VALUE(L.data_json, '$.homeTeam.placeNameWithPreposition.default') AS home_place_with_prep,
        JSON_VALUE(L.data_json, '$.homeTeam.placeNameWithPreposition.fr') AS home_place_with_prep_fr,
        JSON_VALUE(L.data_json, '$.homeTeam.score') AS home_team_score,
        JSON_VALUE(L.data_json, '$.homeTeam.sog') AS home_team_shots,
        JSON_VALUE(L.data_json, '$.homeTeam.logo') AS home_team_logo,
        JSON_VALUE(L.data_json, '$.homeTeam.darkLogo') AS home_team_dark_logo,
        JSON_VALUE(L.data_json, '$.shootoutInUse') AS shootout_in_use,
        JSON_VALUE(L.data_json, '$.otInUse') AS ot_in_use,
        JSON_VALUE(L.data_json, '$.tiesInUse') AS ties_in_use,
        JSON_VALUE(L.data_json, '$.maxPeriods') AS max_periods,
        JSON_VALUE(L.data_json, '$.regPeriods') AS regulation_periods,
        L.scraped_at
    FROM [nhlDB].[gamecenter].[landing] L
    {where_clause}
    """
    
    cursor.execute(insert_game_master)
    rows_inserted = cursor.rowcount
    conn.commit()
    print(f"✓ Inserted {rows_inserted} new games")
    
    if rows_inserted == 0:
        print("\n✓ No new games to process - all tables are up to date!")
        return
    
    # 2-7: Insert related tables (goals, assists, penalties, stars, broadcasts, shootout)
    tables_to_update = [
        ("GAME_GOALS", """
            INSERT INTO [nhlDB].[gamecenter].[GAME_GOALS]
                (game_id, season, goal_event_id, period_number, period_type, max_reg_periods, 
                 time_in_period, situation_code, strength, shot_type, goal_modifier, 
                 goal_scorer_player_id, goal_scorer_first_name, goal_scorer_last_name, 
                 goal_scorer_name, goal_scorer_headshot, goal_team_abbrev, is_home_goal, 
                 goal_scorer_goals_to_date, away_score_after, home_score_after, leading_team_abbrev)
            SELECT 
                L.game_id, L.season,
                JSON_VALUE(goal.value, '$.eventId') AS goal_event_id,
                JSON_VALUE(period.value, '$.periodDescriptor.number') AS period_number,
                JSON_VALUE(period.value, '$.periodDescriptor.periodType') AS period_type,
                JSON_VALUE(period.value, '$.periodDescriptor.maxRegulationPeriods') AS max_reg_periods,
                JSON_VALUE(goal.value, '$.timeInPeriod') AS time_in_period,
                JSON_VALUE(goal.value, '$.situationCode') AS situation_code,
                JSON_VALUE(goal.value, '$.strength') AS strength,
                JSON_VALUE(goal.value, '$.shotType') AS shot_type,
                JSON_VALUE(goal.value, '$.goalModifier') AS goal_modifier,
                JSON_VALUE(goal.value, '$.playerId') AS goal_scorer_player_id,
                JSON_VALUE(goal.value, '$.firstName.default') AS goal_scorer_first_name,
                JSON_VALUE(goal.value, '$.lastName.default') AS goal_scorer_last_name,
                JSON_VALUE(goal.value, '$.name.default') AS goal_scorer_name,
                JSON_VALUE(goal.value, '$.headshot') AS goal_scorer_headshot,
                JSON_VALUE(goal.value, '$.teamAbbrev.default') AS goal_team_abbrev,
                JSON_VALUE(goal.value, '$.isHome') AS is_home_goal,
                JSON_VALUE(goal.value, '$.goalsToDate') AS goal_scorer_goals_to_date,
                JSON_VALUE(goal.value, '$.awayScore') AS away_score_after,
                JSON_VALUE(goal.value, '$.homeScore') AS home_score_after,
                JSON_VALUE(goal.value, '$.leadingTeamAbbrev.default') AS leading_team_abbrev
            FROM [nhlDB].[gamecenter].[landing] L
            OUTER APPLY OPENJSON(L.data_json, '$.summary.scoring') AS period
            OUTER APPLY OPENJSON(period.value, '$.goals') AS goal
            WHERE (JSON_VALUE(goal.value, '$.eventId') IS NOT NULL 
               OR JSON_VALUE(goal.value, '$.playerId') IS NOT NULL)
        """),
        ("GAME_ASSISTS", """
            INSERT INTO [nhlDB].[gamecenter].[GAME_ASSISTS]
                (game_id, season, goal_event_id, assist_number, assist_player_id, 
                 assist_first_name, assist_last_name, assist_player_name, 
                 assist_sweater_number, assists_to_date, period_number, 
                 time_in_period, team_abbrev)
            SELECT 
                L.game_id, L.season,
                JSON_VALUE(goal.value, '$.eventId') AS goal_event_id,
                CAST(assist.[key] AS INT) + 1 AS assist_number,
                JSON_VALUE(assist.value, '$.playerId') AS assist_player_id,
                JSON_VALUE(assist.value, '$.firstName.default') AS assist_first_name,
                JSON_VALUE(assist.value, '$.lastName.default') AS assist_last_name,
                JSON_VALUE(assist.value, '$.name.default') AS assist_player_name,
                JSON_VALUE(assist.value, '$.sweaterNumber') AS assist_sweater_number,
                JSON_VALUE(assist.value, '$.assistsToDate') AS assists_to_date,
                JSON_VALUE(period.value, '$.periodDescriptor.number') AS period_number,
                JSON_VALUE(goal.value, '$.timeInPeriod') AS time_in_period,
                JSON_VALUE(goal.value, '$.teamAbbrev.default') AS team_abbrev
            FROM [nhlDB].[gamecenter].[landing] L
            OUTER APPLY OPENJSON(L.data_json, '$.summary.scoring') AS period
            OUTER APPLY OPENJSON(period.value, '$.goals') AS goal
            OUTER APPLY OPENJSON(goal.value, '$.assists') AS assist
            WHERE JSON_VALUE(assist.value, '$.playerId') IS NOT NULL
        """),
        ("GAME_PENALTIES", """
            INSERT INTO [nhlDB].[gamecenter].[GAME_PENALTIES]
                (game_id, season, period_number, time_in_period, penalized_player, 
                 penalty_type, duration_minutes, penalty_description, 
                 penalty_team_abbrev, drawn_by_player, period_type, max_reg_periods)
            SELECT 
                L.game_id, L.season,
                JSON_VALUE(pen.value, '$.periodDescriptor.number') AS period_number,
                JSON_VALUE(penalty.value, '$.timeInPeriod') AS time_in_period,
                JSON_VALUE(penalty.value, '$.committedByPlayer.default') AS penalized_player,
                JSON_VALUE(penalty.value, '$.type') AS penalty_type,
                JSON_VALUE(penalty.value, '$.duration') AS duration_minutes,
                JSON_VALUE(penalty.value, '$.descKey') AS penalty_description,
                JSON_VALUE(penalty.value, '$.teamAbbrev.default') AS penalty_team_abbrev,
                JSON_VALUE(penalty.value, '$.drawnBy.default') AS drawn_by_player,
                JSON_VALUE(pen.value, '$.periodDescriptor.periodType') AS period_type,
                JSON_VALUE(pen.value, '$.periodDescriptor.maxRegulationPeriods') AS max_reg_periods
            FROM [nhlDB].[gamecenter].[landing] L
            OUTER APPLY OPENJSON(L.data_json, '$.summary.penalties') AS pen
            OUTER APPLY OPENJSON(pen.value, '$.penalties') AS penalty
            WHERE JSON_VALUE(penalty.value, '$.type') IS NOT NULL
        """),
        ("GAME_THREE_STARS", """
            INSERT INTO [nhlDB].[gamecenter].[GAME_THREE_STARS]
                (game_id, season, star_number, player_id, player_name, 
                 team_abbrev, player_headshot, sweater_number, position, 
                 goals, assists, points)
            SELECT 
                L.game_id, L.season,
                JSON_VALUE(star.value, '$.star') AS star_number,
                JSON_VALUE(star.value, '$.playerId') AS player_id,
                JSON_VALUE(star.value, '$.name.default') AS player_name,
                JSON_VALUE(star.value, '$.teamAbbrev') AS team_abbrev,
                JSON_VALUE(star.value, '$.headshot') AS player_headshot,
                JSON_VALUE(star.value, '$.sweaterNo') AS sweater_number,
                JSON_VALUE(star.value, '$.position') AS position,
                JSON_VALUE(star.value, '$.goals') AS goals,
                JSON_VALUE(star.value, '$.assists') AS assists,
                JSON_VALUE(star.value, '$.points') AS points
            FROM [nhlDB].[gamecenter].[landing] L
            OUTER APPLY OPENJSON(L.data_json, '$.summary.threeStars') AS star
            WHERE JSON_VALUE(star.value, '$.playerId') IS NOT NULL
        """),
        ("GAME_TV_BROADCASTS", """
            INSERT INTO [nhlDB].[gamecenter].[GAME_TV_BROADCASTS]
                (game_id, season, broadcast_id, market, country_code, network, sequence_number)
            SELECT 
                L.game_id, L.season,
                JSON_VALUE(broadcast.value, '$.id') AS broadcast_id,
                JSON_VALUE(broadcast.value, '$.market') AS market,
                JSON_VALUE(broadcast.value, '$.countryCode') AS country_code,
                JSON_VALUE(broadcast.value, '$.network') AS network,
                JSON_VALUE(broadcast.value, '$.sequenceNumber') AS sequence_number
            FROM [nhlDB].[gamecenter].[landing] L
            OUTER APPLY OPENJSON(L.data_json, '$.tvBroadcasts') AS broadcast
            WHERE JSON_VALUE(broadcast.value, '$.network') IS NOT NULL
        """),
        ("GAME_SHOOTOUT", """
            INSERT INTO [nhlDB].[gamecenter].[GAME_SHOOTOUT]
                (game_id, season, attempt_number, player_id, player_name, team_abbrev, result, shot_type)
            SELECT 
                L.game_id, L.season,
                CAST(attempt.[key] AS INT) + 1 AS attempt_number,
                JSON_VALUE(attempt.value, '$.playerId') AS player_id,
                JSON_VALUE(attempt.value, '$.playerName.default') AS player_name,
                JSON_VALUE(attempt.value, '$.teamAbbrev.default') AS team_abbrev,
                JSON_VALUE(attempt.value, '$.result') AS result,
                JSON_VALUE(attempt.value, '$.shotType') AS shot_type
            FROM [nhlDB].[gamecenter].[landing] L
            OUTER APPLY OPENJSON(L.data_json, '$.summary.shootout') AS attempt
            WHERE JSON_QUERY(L.data_json, '$.summary.shootout') IS NOT NULL
              AND JSON_VALUE(attempt.value, '$.playerId') IS NOT NULL
        """)
    ]
    
    for table_name, insert_query in tables_to_update:
        print(f"\n{tables_to_update.index((table_name, insert_query)) + 2}. Inserting into {table_name}...")
        full_query = insert_query + f"\n{where_clause.replace('WHERE', 'AND')}"
        cursor.execute(full_query)
        rows_inserted = cursor.rowcount
        conn.commit()
        print(f"✓ Inserted {rows_inserted} records")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("NHL DATABASE INCREMENTAL UPDATE")
    print("="*70)
    
    conn = connect_to_db()
    
    try:
        # Update play-by-play tables
        update_playbyplay_tables(conn)
        
        # Update gamecenter landing tables
        update_gamecenter_landing_tables(conn)
        
        print("\n" + "="*70)
        print("ALL UPDATES COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()