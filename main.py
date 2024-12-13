from data_preprocessing import preprocess_participants, preprocess_teams

def main():
    # 파일 경로
    participants_file = "C:/Users/junse/Documents/Python Project/League of Legends Machine Learning/Data/processed/all_participants.json"
    teams_file = "C:/Users/junse/Documents/Python Project/League of Legends Machine Learning/Data/processed/all_teams.json"

    # Participants 데이터 전처리
    participants_data = preprocess_participants(participants_file)
    if participants_data is not None:
        print("Participants 데이터 전처리 결과:")
        print(participants_data.head())
        X_participants = participants_data.drop(columns=['championName'])  # 타겟 변수 제외
        y_participants = participants_data['championName']  # 타겟 변수
        print("Participants X 데이터:")
        print(X_participants.head())
        print("Participants y 데이터:")
        print(y_participants.head())

    # Teams 데이터 전처리
    teams_data = preprocess_teams(teams_file)
    if teams_data is not None:
        print("Teams 데이터 전처리 결과:")
        print(teams_data.head())
        X_teams = teams_data.drop(columns=['win'])  # 타겟 변수 제외
        y_teams = teams_data['win']  # 타겟 변수
        print("Teams X 데이터:")
        print(X_teams.head())
        print("Teams y 데이터:")
        print(y_teams.head())

if __name__ == "__main__":
    main()
