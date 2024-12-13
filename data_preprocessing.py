import pandas as pd

def preprocess_participants(participants_path):
    
    try:
        participants_df = pd.read_json(participants_path)
    except ValueError as e:
        print(f"Participants 데이터 로드 중 오류 발생: {e}")
        return None
    
    columns_to_keep = [
        'championName', 'kills', 'deaths', 'assists', 'totalDamageDealtToChampions',
        'goldEarned', 'teamId', 'win', 'role', 'lane', 'totalMinionsKilled',
        'timePlayed', 'damageDealtToBuildings', 'damageDealtToObjectives',
        'wardsPlaced', 'visionScore', 'spell1Casts', 'spell2Casts', 'teamPosition'
    ]
    participants_df = participants_df[columns_to_keep]
    return participants_df


def preprocess_teams(teams_path):

    try:
        teams_df = pd.read_json(teams_path)
    except ValueError as e:
        print(f"Teams 데이터 로드 중 오류 발생: {e}")
        return None

    teams_df['baronKills'] = teams_df['objectives'].apply(lambda x: x['baron']['kills'] if isinstance(x, dict) else 0)
    teams_df['dragonKills'] = teams_df['objectives'].apply(lambda x: x['dragon']['kills'] if isinstance(x, dict) else 0)
    teams_df['towerKills'] = teams_df['objectives'].apply(lambda x: x['tower']['kills'] if isinstance(x, dict) else 0)

    columns_to_keep = ['teamId', 'win', 'baronKills', 'dragonKills', 'towerKills']
    teams_df = teams_df[columns_to_keep]
    return teams_df

participants_file = "C:/Users/junse/Documents/Python Project/League of Legends Machine Learning/Data/processed/all_participants.json"
teams_file = "C:/Users/junse/Documents/Python Project/League of Legends Machine Learning/Data/processed/all_teams.json"

participants_data = preprocess_participants(participants_file)
teams_data = preprocess_teams(teams_file)

if participants_data is not None:
    print("Participants 데이터 전처리 결과:")
    print(participants_data.head())

if teams_data is not None:
    print("Teams 데이터 전처리 결과:")
    print(teams_data.head())

if participants_data is not None:
    X_participants = participants_data.drop(columns=['championName'])  
    y_participants = participants_data['championName']  
    print("Participants X 데이터:")
    print(X_participants.head())
    print("Participants y 데이터:")
    print(y_participants.head())

if teams_data is not None:
    X_teams = teams_data.drop(columns=['win'])  
    y_teams = teams_data['win']  
    print("Teams X 데이터:")
    print(X_teams.head())
    print("Teams y 데이터:")
    print(y_teams.head())
