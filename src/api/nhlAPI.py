"""
NHL API Scraper for Current Season (2025-26 Only)
Only scrapes historical data up to current date
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Set, Tuple
import logging
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NHLCurrentSeasonScraper:
    def __init__(self, 
                 server='localhost', 
                 database='NHL_DB', 
                 username=None, 
                 password=None, 
                 use_windows_auth=True,
                 reset_schedule_tables=False):
        """
        Initialize scraper for CURRENT SEASON ONLY (2025-26)
        Only scrapes data up to current date (excludes future games)
        """
        # HARDCODED TO CURRENT SEASON
        self.current_year = 2025
        self.season_str = "20252026"
        
        self.base_url = "https://api.nhle.com/stats/rest/en"
        self.web_api_url = "https://api-web.nhle.com/v1"
        self.session = requests.Session()
        
        # SQL Server connection setup
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.use_windows_auth = use_windows_auth
        self.engine = self.create_db_engine()
        self.create_database_schema(reset_schedule_tables=reset_schedule_tables)
        
        # Define endpoints
        self.skater_endpoints = {
            'bios': 'bios',
            'faceoffpercentages': 'faceoffpercentages', 
            'summary': 'summary',
            'faceoffwins': 'faceoffwins',
            'goalsForAgainst': 'goalsForAgainst',
            'realtime': 'realtime',
            'penalties': 'penalties',
            'penaltykill': 'penaltykill',
            'penaltyShots': 'penaltyShots',
            'powerplay': 'powerplay',
            'puckPossessions': 'puckPossessions',
            'summaryshooting': 'summaryshooting',
            'percentages': 'percentages',
            'scoringRates': 'scoringRates',
            'scoringpergame': 'scoringpergame',
            'shootout': 'shootout',
            'shottype': 'shottype',
            'timeonice': 'timeonice'
        }
        
        self.goalie_endpoints = {
            'summary': 'summary',
            'advanced': 'advanced',
            'bios': 'bios',
            'daysrest': 'daysrest',
            'penaltyShots': 'penaltyShots',
            'savesByStrength': 'savesByStrength',
            'shootout': 'shootout',
            'startedVsRelieved': 'startedVsRelieved'
        }
        
        self.team_endpoints = {
            'summary': 'summary',
            'daysbetweengames': 'daysbetweengames',
            'faceoffpercentages': 'faceoffpercentages',
            'faceoffwins': 'faceoffwins',
            'goalsagainstbystrength': 'goalsagainstbystrength',
            'goalsbyperiod': 'goalsbyperiod',
            'goalsforbystrength': 'goalsforbystrength',
            'goalsforbystrengthgoaliepull': 'goalsforbystrengthgoaliepull',
            'leadingtrailing': 'leadingtrailing',
            'realtime': 'realtime',
            'outshootoutshotby': 'outshootoutshotby',
            'penalties': 'penalties',
            'penaltykill': 'penaltykill',
            'penaltykilltime': 'penaltykilltime',
            'powerplay': 'powerplay',
            'powerplaytime': 'powerplaytime',
            'summaryshooting': 'summaryshooting',
            'percentages': 'percentages',
            'scoretrailfirst': 'scoretrailfirst',
            'shootout': 'shootout',
            'shottype': 'shottype',
            'goalgames': 'goalgames'
        }
        
        logger.info(f"Initialized scraper for {self.current_year}-{self.current_year+1} season")
        logger.info(f"Will only scrape data up to {datetime.now().date()}")
        
    def create_db_engine(self):
        """Create SQLAlchemy engine for SQL Server"""
        if self.use_windows_auth:
            connection_string = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={self.server};'
                f'DATABASE={self.database};'
                f'Trusted_Connection=yes;'
            )
        else:
            connection_string = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={self.server};'
                f'DATABASE={self.database};'
                f'UID={self.username};'
                f'PWD={self.password};'
            )
        
        connection_url = f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}"
        engine = create_engine(connection_url, fast_executemany=True, pool_pre_ping=True)
        
        logger.info(f"Connected to SQL Server: {self.server}/{self.database}")
        return engine
    
    def create_database_schema(self, reset_schedule_tables=False):
        """Create database tables if they don't exist - NEVER drops existing data"""
        logger.info("Checking database schema...")
        
        with self.engine.connect() as conn:
            # Create schemas if they don't exist
            for schema in ['skater', 'goalie', 'team', 'draft', 'metadata', 'schedule', 'playbyplay', 'gamecenter', 'shifts']:
                conn.execute(text(f"IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = '{schema}') EXEC('CREATE SCHEMA {schema}')"))
            
            # IMPORTANT: We NEVER drop tables - only append to existing tables
            # reset_schedule_tables parameter is ignored to prevent accidental data loss
            if reset_schedule_tables:
                logger.warning("⚠️  reset_schedule_tables=True was passed but is IGNORED for safety")
                logger.warning("⚠️  This scraper only APPENDS to existing tables, never drops them")
            
            conn.commit()
        
        logger.info("✓ Database schema ready (existing tables preserved)")
    
    def fetch_data(self, url: str, params: dict = None, max_retries: int = 3) -> Optional[dict]:
        """Fetch data from API with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.debug(f"404 Not Found: {url}")
                    return None
                else:
                    logger.warning(f"Status {response.status_code} for {url}")
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None
    
    def fetch_stats_data(self, endpoint_type: str, stat_type: str, date: str, limit: int = 100) -> List[dict]:
        """Fetch paginated stats data for a specific date"""
        all_data = []
        start = 0
        
        while True:
            url = f"{self.base_url}/{endpoint_type}/{stat_type}"
            
            params = {
                "isAggregate": "true",
                "isGame": "true",
                "start": start,
                "limit": limit,
                "cayenneExp": f'gameDate<="{date} 23:59:59" and gameDate>="{date}" and gameTypeId=2'
            }
            
            data = self.fetch_data(url, params)
            
            if data and "data" in data and len(data["data"]) > 0:
                all_data.extend(data["data"])
                
                if len(data["data"]) < limit:
                    break
                    
                start += limit
                time.sleep(0.1)
            else:
                break
        
        return all_data
    
    def save_to_sql(self, df: pd.DataFrame, table_name: str, schema: str = 'dbo', if_exists: str = 'append') -> bool:
        """Save DataFrame to SQL Server"""
        try:
            chunk_size = min(1000, max(1, 2000 // len(df.columns)))
            
            df.to_sql(
                name=table_name,
                con=self.engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                chunksize=chunk_size,
                method='multi'
            )
            
            logger.info(f"Saved {len(df)} rows to {schema}.{table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to {schema}.{table_name}: {e}")
            return False
    
    def check_existing_data(self, table_name: str, schema: str, season: str = None, date: str = None, game_id: int = None) -> bool:
        """Check if data already exists - handles missing tables gracefully"""
        try:
            if game_id:
                query = f"SELECT COUNT(*) as cnt FROM {schema}.{table_name} WHERE game_id = {game_id}"
            elif season and date:
                query = f"SELECT COUNT(*) as cnt FROM {schema}.{table_name} WHERE season = '{season}' AND gameDate = '{date}'"
            elif season:
                query = f"SELECT COUNT(*) as cnt FROM {schema}.{table_name} WHERE season = '{season}'"
            elif date:
                query = f"SELECT COUNT(*) as cnt FROM {schema}.{table_name} WHERE gameDate = '{date}'"
            else:
                return False
                
            result = pd.read_sql(query, self.engine)
            return result['cnt'].iloc[0] > 0
            
        except Exception as e:
            # Table doesn't exist yet - that's fine, we'll create it on first insert
            logger.debug(f"Table {schema}.{table_name} doesn't exist yet or query failed: {e}")
            return False
    
    def get_last_scraped_date(self, table_name: str, schema: str) -> Optional[str]:
        """Get the last date that was scraped for incremental updates"""
        try:
            query = f"""
                SELECT MAX(gameDate) as last_date 
                FROM {schema}.{table_name} 
                WHERE season = '{self.season_str}'
            """
            result = pd.read_sql(query, self.engine)
            last_date = result['last_date'].iloc[0]
            
            if pd.notna(last_date):
                logger.info(f"  Last scraped date in {schema}.{table_name}: {last_date}")
                return str(last_date)
            return None
            
        except Exception as e:
            logger.debug(f"Could not get last date from {schema}.{table_name}: {e}")
            return None
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        Get date range for current season, ending at current date
        
        Returns:
            (start_date, end_date) where end_date is yesterday
        """
        start_date = datetime(self.current_year, 10, 1)
        # End date is YESTERDAY (to avoid future games)
        end_date = datetime.now() - timedelta(days=1)
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        return start_date, end_date
    
    def scrape_skater_data(self):
        """Scrape skater game log data for current season up to today"""
        logger.info("Starting skater data collection (current season only)...")
        logger.info("  Mode: APPEND to existing data (no deletions)")
        
        start_date, end_date = self.get_date_range()
        
        for endpoint_name, endpoint_path in self.skater_endpoints.items():
            logger.info(f"  Scraping {endpoint_name}...")
            
            # Check last scraped date for this endpoint
            last_date = self.get_last_scraped_date(endpoint_name, 'skater')
            
            all_data = []
            current_date = start_date
            dates_processed = 0
            dates_with_data = 0
            dates_skipped = 0
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Check if we already have this data
                if self.check_existing_data(endpoint_name, 'skater', self.season_str, date_str):
                    dates_skipped += 1
                    current_date += timedelta(days=1)
                    continue
                
                # Fetch data for this date
                data = self.fetch_stats_data('skater', endpoint_path, date_str)
                
                if data:
                    dates_with_data += 1
                    for record in data:
                        record['gameDate'] = date_str
                        record['season'] = self.season_str
                    all_data.extend(data)
                    
                    # Save in batches
                    if len(all_data) >= 10000:
                        df = pd.DataFrame(all_data)
                        self.save_to_sql(df, endpoint_name, schema='skater')
                        all_data = []
                
                dates_processed += 1
                if dates_processed % 30 == 0:
                    logger.info(f"    Progress: {dates_processed} days checked, {dates_with_data} with new data, {dates_skipped} already in DB")
                
                current_date += timedelta(days=1)
                time.sleep(0.1)
            
            # Save remaining data
            if all_data:
                df = pd.DataFrame(all_data)
                self.save_to_sql(df, endpoint_name, schema='skater')
            
            logger.info(f"    ✓ Completed {endpoint_name}: {dates_with_data} new dates, {dates_skipped} already existed")
    
    def scrape_goalie_data(self):
        """Scrape goalie game log data for current season up to today"""
        logger.info("Starting goalie data collection (current season only)...")
        
        start_date, end_date = self.get_date_range()
        
        for endpoint_name, endpoint_path in self.goalie_endpoints.items():
            logger.info(f"  Scraping {endpoint_name}...")
            all_data = []
            
            current_date = start_date
            dates_processed = 0
            dates_with_data = 0
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                if self.check_existing_data(endpoint_name, 'goalie', self.season_str, date_str):
                    current_date += timedelta(days=1)
                    continue
                
                data = self.fetch_stats_data('goalie', endpoint_path, date_str)
                
                if data:
                    dates_with_data += 1
                    for record in data:
                        record['gameDate'] = date_str
                        record['season'] = self.season_str
                    all_data.extend(data)
                    
                    if len(all_data) >= 10000:
                        df = pd.DataFrame(all_data)
                        self.save_to_sql(df, endpoint_name, schema='goalie')
                        all_data = []
                
                dates_processed += 1
                if dates_processed % 30 == 0:
                    logger.info(f"    Progress: {dates_processed} days processed, {dates_with_data} with data")
                
                current_date += timedelta(days=1)
                time.sleep(0.1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                self.save_to_sql(df, endpoint_name, schema='goalie')
            
            logger.info(f"    Completed {endpoint_name}: {dates_with_data} dates with data")
    
    def scrape_team_data(self):
        """Scrape team game log data for current season up to today"""
        logger.info("Starting team data collection (current season only)...")
        
        start_date, end_date = self.get_date_range()
        
        for endpoint_name, endpoint_path in self.team_endpoints.items():
            logger.info(f"  Scraping {endpoint_name}...")
            all_data = []
            
            current_date = start_date
            dates_processed = 0
            dates_with_data = 0
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                if self.check_existing_data(endpoint_name, 'team', self.season_str, date_str):
                    current_date += timedelta(days=1)
                    continue
                
                data = self.fetch_stats_data('team', endpoint_path, date_str)
                
                if data:
                    dates_with_data += 1
                    for record in data:
                        record['gameDate'] = date_str
                        record['season'] = self.season_str
                    all_data.extend(data)
                    
                    if len(all_data) >= 10000:
                        df = pd.DataFrame(all_data)
                        self.save_to_sql(df, endpoint_name, schema='team')
                        all_data = []
                
                dates_processed += 1
                if dates_processed % 30 == 0:
                    logger.info(f"    Progress: {dates_processed} days processed, {dates_with_data} with data")
                
                current_date += timedelta(days=1)
                time.sleep(0.1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                self.save_to_sql(df, endpoint_name, schema='team')
            
            logger.info(f"    Completed {endpoint_name}: {dates_with_data} dates with data")
    
    def scrape_schedule(self):
        """Scrape schedule for current season up to today (excludes future games)"""
        logger.info(f"Starting schedule collection for {self.current_year}-{self.current_year+1}...")
        logger.info("  Mode: APPEND to existing schedule data")
        
        start_date, end_date = self.get_date_range()
        
        all_games = []
        current_date = start_date
        dates_processed = 0
        games_added = 0
        dates_skipped = 0
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if we already have data for this date
            if self.check_existing_data('schedule', 'schedule', self.season_str, date_str):
                dates_skipped += 1
                current_date += timedelta(days=1)
                continue
            
            url = f"{self.web_api_url}/schedule/{date_str}"
            data = self.fetch_data(url)
            
            if data and 'gameWeek' in data:
                for day in data['gameWeek']:
                    if 'games' in day:
                        for game in day['games']:
                            # FILTER: Only include completed games
                            game_state = game.get('gameState', '')
                            
                            if game_state == 'OFF':  # Only finished games
                                game_record = {
                                    'game_id': game.get('id'),
                                    'season': game.get('season'),
                                    'gameType': game.get('gameType'),
                                    'gameDate': date_str,
                                    'venue': game.get('venue', {}).get('default'),
                                    'neutralSite': game.get('neutralSite'),
                                    'startTimeUTC': game.get('startTimeUTC'),
                                    'gameState': game_state,
                                    'gameScheduleState': game.get('gameScheduleState'),
                                    
                                    # Away team info
                                    'awayTeam_id': game.get('awayTeam', {}).get('id'),
                                    'awayTeam_abbrev': game.get('awayTeam', {}).get('abbrev'),
                                    'awayTeam_placeName_default': game.get('awayTeam', {}).get('placeName', {}).get('default'),
                                    'awayTeam_score': game.get('awayTeam', {}).get('score'),
                                    'awayTeam_logo': game.get('awayTeam', {}).get('logo'),
                                    
                                    # Home team info
                                    'homeTeam_id': game.get('homeTeam', {}).get('id'),
                                    'homeTeam_abbrev': game.get('homeTeam', {}).get('abbrev'),
                                    'homeTeam_placeName_default': game.get('homeTeam', {}).get('placeName', {}).get('default'),
                                    'homeTeam_score': game.get('homeTeam', {}).get('score'),
                                    'homeTeam_logo': game.get('homeTeam', {}).get('logo'),
                                    
                                    # Store full JSON for complex fields
                                    'venue_default': game.get('venue', {}).get('default'),
                                    'tvBroadcasts_json': json.dumps(game.get('tvBroadcasts', [])),
                                    'periodDescriptor_json': json.dumps(game.get('periodDescriptor', {})),
                                    'gameOutcome_json': json.dumps(game.get('gameOutcome', {})),
                                    'full_data_json': json.dumps(game)
                                }
                                all_games.append(game_record)
                                games_added += 1
            
            dates_processed += 1
            if dates_processed % 30 == 0:
                logger.info(f"    Progress: {dates_processed} days checked, {games_added} new games, {dates_skipped} dates already in DB")
            
            # Save in batches
            if len(all_games) >= 500:
                df = pd.DataFrame(all_games)
                self.save_to_sql(df, 'schedule', schema='schedule', if_exists='append')
                all_games = []
            
            current_date += timedelta(days=1)
            time.sleep(0.1)
        
        # Save remaining games
        if all_games:
            df = pd.DataFrame(all_games)
            self.save_to_sql(df, 'schedule', schema='schedule', if_exists='append')
        
        logger.info(f"    ✓ Completed: {games_added} new completed games added, {dates_skipped} dates already existed")
    
    def scrape_rosters(self):
        """Scrape team rosters for current season"""
        logger.info(f"Starting roster collection for {self.current_year}-{self.current_year+1}...")
        
        # Get unique team abbreviations from schedule
        try:
            query = f"""
                SELECT DISTINCT awayTeam_abbrev as team_abbrev FROM schedule.schedule WHERE season = {self.season_str}
                UNION
                SELECT DISTINCT homeTeam_abbrev as team_abbrev FROM schedule.schedule WHERE season = {self.season_str}
            """
            teams_df = pd.read_sql(query, self.engine)
            teams = teams_df['team_abbrev'].dropna().unique().tolist()
            logger.info(f"    Found {len(teams)} teams: {', '.join(teams)}")
        except Exception as e:
            logger.warning(f"    Could not get teams from schedule: {e}")
            logger.info("    Skipping rosters - run schedule scrape first")
            return
        
        roster_data = []
        
        for team in teams:
            url = f"{self.web_api_url}/roster/{team}/{self.season_str}"
            data = self.fetch_data(url)
            
            if data:
                for position_group in ['forwards', 'defensemen', 'goalies']:
                    if position_group in data:
                        for player in data[position_group]:
                            player_record = {
                                'player_id': player.get('id'),
                                'teamAbbrev': team,
                                'season': self.season_str,
                                'positionGroup': position_group,
                                'firstName': player.get('firstName', {}).get('default'),
                                'lastName': player.get('lastName', {}).get('default'),
                                'sweaterNumber': player.get('sweaterNumber'),
                                'positionCode': player.get('positionCode'),
                                'shootsCatches': player.get('shootsCatches'),
                                'heightInInches': player.get('heightInInches'),
                                'weightInPounds': player.get('weightInPounds'),
                                'heightInCentimeters': player.get('heightInCentimeters'),
                                'weightInKilograms': player.get('weightInKilograms'),
                                'birthDate': player.get('birthDate'),
                                'birthCity': player.get('birthCity', {}).get('default'),
                                'birthCountry': player.get('birthCountry'),
                                'birthStateProvince': player.get('birthStateProvince', {}).get('default'),
                                'headshot': player.get('headshot'),
                                'full_data_json': json.dumps(player)
                            }
                            roster_data.append(player_record)
            
            time.sleep(0.1)
        
        if roster_data:
            df = pd.DataFrame(roster_data)
            self.save_to_sql(df, 'rosters', schema='team', if_exists='append')
            logger.info(f"    Saved {len(roster_data)} roster entries")
    
    def get_completed_game_ids(self) -> List[Tuple[int, str]]:
        """Get all completed game IDs from schedule for current season"""
        try:
            query = f"""
                SELECT DISTINCT game_id, season 
                FROM schedule.schedule 
                WHERE gameState = 'OFF' 
                    AND game_id IS NOT NULL
                    AND season = '{self.season_str}'
                ORDER BY game_id
            """
            games_df = pd.read_sql(query, self.engine)
            return list(zip(games_df['game_id'], games_df['season']))
        except Exception as e:
            logger.error(f"Could not get game IDs from schedule: {e}")
            return []
    
    def scrape_play_by_play(self):
        """Scrape play-by-play data for completed games in current season"""
        logger.info("Starting play-by-play collection (current season only)...")
        
        games = self.get_completed_game_ids()
        
        if not games:
            logger.warning("  No completed games found. Run scrape_schedule first.")
            return
        
        logger.info(f"  Found {len(games)} completed games")
        
        # Check what we already have
        try:
            existing_query = f"SELECT DISTINCT game_id FROM playbyplay.play_by_play WHERE season = '{self.season_str}'"
            existing_df = pd.read_sql(existing_query, self.engine)
            existing_ids = set(existing_df['game_id'].tolist())
        except:
            existing_ids = set()
        
        games_to_fetch = [(gid, season) for gid, season in games if gid not in existing_ids]
        logger.info(f"  Need to fetch {len(games_to_fetch)} games")
        
        pbp_data = []
        for idx, (game_id, season) in enumerate(games_to_fetch, 1):
            url = f"{self.web_api_url}/gamecenter/{game_id}/play-by-play"
            data = self.fetch_data(url)
            
            if data:
                pbp_record = {
                    'game_id': game_id,
                    'season': season,
                    'data_json': json.dumps(data),
                    'scraped_at': datetime.now()
                }
                pbp_data.append(pbp_record)
            
            if idx % 10 == 0:
                logger.info(f"    Progress: {idx}/{len(games_to_fetch)} games")
            
            time.sleep(0.3)
            
            if len(pbp_data) >= 50:
                df = pd.DataFrame(pbp_data)
                self.save_to_sql(df, 'play_by_play', schema='playbyplay')
                pbp_data = []
        
        if pbp_data:
            df = pd.DataFrame(pbp_data)
            self.save_to_sql(df, 'play_by_play', schema='playbyplay')
        
        logger.info("  Completed play-by-play collection")
    
    def scrape_landing(self):
        """Scrape landing data for completed games in current season"""
        logger.info("Starting landing data collection (current season only)...")
        
        games = self.get_completed_game_ids()
        
        if not games:
            logger.warning("  No completed games found. Run scrape_schedule first.")
            return
        
        try:
            existing_query = f"SELECT DISTINCT game_id FROM gamecenter.landing WHERE season = '{self.season_str}'"
            existing_df = pd.read_sql(existing_query, self.engine)
            existing_ids = set(existing_df['game_id'].tolist())
        except:
            existing_ids = set()
        
        games_to_fetch = [(gid, season) for gid, season in games if gid not in existing_ids]
        logger.info(f"  Need to fetch {len(games_to_fetch)} games")
        
        landing_data = []
        for idx, (game_id, season) in enumerate(games_to_fetch, 1):
            url = f"{self.web_api_url}/gamecenter/{game_id}/landing"
            data = self.fetch_data(url)
            
            if data:
                landing_record = {
                    'game_id': game_id,
                    'season': season,
                    'data_json': json.dumps(data),
                    'scraped_at': datetime.now()
                }
                landing_data.append(landing_record)
            
            if idx % 10 == 0:
                logger.info(f"    Progress: {idx}/{len(games_to_fetch)} games")
            
            time.sleep(0.3)
            
            if len(landing_data) >= 50:
                df = pd.DataFrame(landing_data)
                self.save_to_sql(df, 'landing', schema='gamecenter')
                landing_data = []
        
        if landing_data:
            df = pd.DataFrame(landing_data)
            self.save_to_sql(df, 'landing', schema='gamecenter')
        
        logger.info("  Completed landing data collection")
    
    def scrape_boxscore(self):
        """Scrape boxscore data for completed games in current season"""
        logger.info("Starting boxscore collection (current season only)...")
        
        games = self.get_completed_game_ids()
        
        if not games:
            logger.warning("  No completed games found. Run scrape_schedule first.")
            return
        
        try:
            existing_query = f"SELECT DISTINCT game_id FROM gamecenter.boxscore WHERE season = '{self.season_str}'"
            existing_df = pd.read_sql(existing_query, self.engine)
            existing_ids = set(existing_df['game_id'].tolist())
        except:
            existing_ids = set()
        
        games_to_fetch = [(gid, season) for gid, season in games if gid not in existing_ids]
        logger.info(f"  Need to fetch {len(games_to_fetch)} games")
        
        boxscore_data = []
        for idx, (game_id, season) in enumerate(games_to_fetch, 1):
            url = f"{self.web_api_url}/gamecenter/{game_id}/boxscore"
            data = self.fetch_data(url)
            
            if data:
                boxscore_record = {
                    'game_id': game_id,
                    'season': season,
                    'data_json': json.dumps(data),
                    'scraped_at': datetime.now()
                }
                boxscore_data.append(boxscore_record)
            
            if idx % 10 == 0:
                logger.info(f"    Progress: {idx}/{len(games_to_fetch)} games")
            
            time.sleep(0.3)
            
            if len(boxscore_data) >= 50:
                df = pd.DataFrame(boxscore_data)
                self.save_to_sql(df, 'boxscore', schema='gamecenter')
                boxscore_data = []
        
        if boxscore_data:
            df = pd.DataFrame(boxscore_data)
            self.save_to_sql(df, 'boxscore', schema='gamecenter')
        
        logger.info("  Completed boxscore collection")
    
    def scrape_shift_charts(self):
        """Scrape shift chart data for completed games in current season"""
        logger.info("Starting shift charts collection (current season only)...")
        
        games = self.get_completed_game_ids()
        
        if not games:
            logger.warning("  No completed games found. Run scrape_schedule first.")
            return
        
        try:
            existing_query = f"SELECT DISTINCT gameId FROM shifts.shift_charts WHERE season = '{self.season_str}'"
            existing_df = pd.read_sql(existing_query, self.engine)
            existing_ids = set(existing_df['gameId'].tolist())
        except:
            existing_ids = set()
        
        games_to_fetch = [(gid, season) for gid, season in games if gid not in existing_ids]
        logger.info(f"  Need to fetch {len(games_to_fetch)} games")
        
        all_shifts = []
        for idx, (game_id, season) in enumerate(games_to_fetch, 1):
            url = f"{self.base_url}/shiftcharts"
            params = {"cayenneExp": f"gameId={game_id}"}
            data = self.fetch_data(url, params)
            
            if data and 'data' in data and len(data['data']) > 0:
                for shift in data['data']:
                    shift['season'] = season
                all_shifts.extend(data['data'])
            
            if idx % 10 == 0:
                logger.info(f"    Progress: {idx}/{len(games_to_fetch)} games, {len(all_shifts)} shifts collected")
            
            time.sleep(0.2)
            
            if len(all_shifts) >= 10000:
                df = pd.DataFrame(all_shifts)
                self.save_to_sql(df, 'shift_charts', schema='shifts')
                all_shifts = []
        
        if all_shifts:
            df = pd.DataFrame(all_shifts)
            self.save_to_sql(df, 'shift_charts', schema='shifts')
        
        logger.info("  Completed shift charts collection")
    
    def run_full_scrape(self, include_gamecenter: bool = True):
        """Run complete scraping process for current season only"""
        logger.info("=" * 80)
        logger.info(f"NHL API SCRAPER - CURRENT SEASON ONLY ({self.current_year}-{self.current_year+1})")
        logger.info(f"Scraping data up to: {datetime.now().date()}")
        logger.info("=" * 80)
        
        try:
            # Core game log data
            self.scrape_skater_data()
            self.scrape_goalie_data()
            self.scrape_team_data()
            
            # Schedule MUST come first
            self.scrape_schedule()
            
            # Rosters
            self.scrape_rosters()
            
            # Game center data
            if include_gamecenter:
                self.scrape_play_by_play()
                self.scrape_landing()
                self.scrape_boxscore()
                self.scrape_shift_charts()
            
            logger.info("=" * 80)
            logger.info("SCRAPING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise


def main():
    """
    Run scraper for current season only
    SAFE MODE: Only appends new data, never deletes existing data
    """
    logger.info("=" * 80)
    logger.info("NHL CURRENT SEASON INCREMENTAL SCRAPER")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("  • Season: 2025-26")
    logger.info("  • Mode: APPEND ONLY (safe for daily runs)")
    logger.info("  • Data retention: ALL existing data preserved")
    logger.info("  • Duplicate prevention: Active")
    logger.info("=" * 80)
    
    scraper = NHLCurrentSeasonScraper(
        server='DESKTOP-J9IV3OH',
        database='nhlDB',
        use_windows_auth=True,
        reset_schedule_tables=False  # ⚠️ ALWAYS False - never drop tables
    )
    
    # Run the scrape (will only add new data)
    scraper.run_full_scrape(include_gamecenter=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Scraping completed safely - all existing data preserved")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()