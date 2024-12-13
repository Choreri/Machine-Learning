import pandas as pd
import os

# 데이터 폴더 경로 설정
data_folder = 'data/processed/'
output_folder = 'data/filtered/'

# 출력 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 필요한 열 지정
columns_to_keep = [
    'gameId', 'teamId', 'championId', 'individualPosition', 'kills', 'deaths', 
    'assists', 'goldEarned', 'totalDamageDealtToChampions', 'win'
]

# 필터링된 데이터를 읽고 저장하는 함수
def filter_csv_files(input_folder, output_folder, columns_to_keep):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_name}")
            
            # CSV 파일 읽기
            try:
                df = pd.read_csv(file_path, usecols=lambda col: col in columns_to_keep, low_memory=False)
                output_file_path = os.path.join(output_folder, f"filtered_{file_name}")

                # 필터링된 데이터 저장
                df.to_csv(output_file_path, index=False)
                print(f"Filtered data saved to {output_file_path}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# 필터링 실행
filter_csv_files(data_folder, output_folder, columns_to_keep)

print("필터링 완료!")
