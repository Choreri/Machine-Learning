import pandas as pd
import xgboost as xgb

def predict_with_model(model_path, input_data):

    model = xgb.Booster()
    model.load_model(model_path)

    
    test_dmatrix = xgb.DMatrix(input_data)

   
    prediction = (model.predict(test_dmatrix) > 0.5).astype(int)
    return prediction

if __name__ == "__main__":
    # 사용자 입력 데이터 예시
    input_data = pd.DataFrame([{
        'total_blue_barons': 1, 'total_blue_drakes': 2, 'total_blue_heralds': 1, 'total_blue_inhibs': 1,
        'total_blue_kills': 30, 'total_blue_turrets': 7, 'blue_first_blood': 1, 'blue_first_herald': 1,
        'blue_first_drake': 1, 'blue_first_baron': 1, 'blue_first_inhib': 1, 'blue_first_turret': 1,
        'total_blue_vs': 75, 'total_blue_gold': 60000, 'med_blue_kills': 6, 'med_blue_xp': 1500,
        'med_blue_gold': 12000, 'med_blue_dmg': 20000, 'blue_got_soul': 1, 'total_blue_elders': 1, 'med_blue_vs': 25,
        'total_red_barons': 0, 'total_red_drakes': 2, 'total_red_heralds': 1, 'total_red_inhibs': 0,
        'total_red_kills': 15, 'total_red_turrets': 4, 'red_first_blood': 0, 'red_first_herald': 0,
        'red_first_drake': 1, 'red_first_baron': 0, 'red_first_inhib': 0, 'red_first_turret': 0,
        'total_red_vs': 50, 'total_red_gold': 50000, 'med_red_kills': 3, 'med_red_xp': 1200, 'med_red_gold': 10000,
        'med_red_dmg': 15000, 'red_got_soul': 0, 'total_red_elders': 0, 'med_red_vs': 20
    }])

    # 예측 수행
    print("모델 로드 및 예측 중...")
    model_path = "xgboost_winning_team_model.json"
    prediction = predict_with_model(model_path, input_data)
    print(f"예측 결과 (0=Red, 1=Blue): {prediction[0]}")
