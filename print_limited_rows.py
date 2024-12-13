import pandas as pd
import os

# 데이터 폴더 설정
processed_folder = 'data/processed/'

# 출력할 최대 행 수 설정
MAX_PRINT_ROWS = 10

# CSV 파일 읽고 출력하는 함수
def print_limited_rows(folder_path, max_rows=10):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"\nReading file: {file_name}")
            try:
                # 데이터 읽기
                data = pd.read_csv(file_path)
                
                # 출력 제한
                print(f"Displaying first {max_rows} rows of {file_name}:")
                print(data.head(max_rows))  # 처음 max_rows 행만 출력
                
                print("\n--- End of Output ---\n")
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

# 데이터 출력 실행
print_limited_rows(processed_folder, MAX_PRINT_ROWS)
