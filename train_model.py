import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

def preprocess_data(data):
    features = [
        'total_blue_barons', 'total_blue_drakes', 'total_blue_heralds', 'total_blue_inhibs',
        'total_blue_kills', 'total_blue_turrets', 'blue_first_blood', 'blue_first_herald',
        'blue_first_drake', 'blue_first_baron', 'blue_first_inhib', 'blue_first_turret',
        'total_blue_vs', 'total_blue_gold', 'med_blue_kills', 'med_blue_xp', 'med_blue_gold',
        'med_blue_dmg', 'blue_got_soul', 'total_blue_elders', 'med_blue_vs', 'total_red_barons',
        'total_red_drakes', 'total_red_heralds', 'total_red_inhibs', 'total_red_kills',
        'total_red_turrets', 'red_first_blood', 'red_first_herald', 'red_first_drake',
        'red_first_baron', 'red_first_inhib', 'red_first_turret', 'total_red_vs',
        'total_red_gold', 'med_red_kills', 'med_red_xp', 'med_red_gold', 'med_red_dmg',
        'red_got_soul', 'total_red_elders', 'med_red_vs'
    ]
    target = 'winning_team'

    X = data[features]
    y = data[target]

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

   
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    
    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softmax",  
        "num_class": len(set(y_train)),  
        "max_depth": 6,
        "eta": 0.1,
        "eval_metric": "mlogloss",
        "tree_method": "gpu_hist",  
        "predictor": "gpu_predictor",
        "nthread": -1,
    }

    #
    evals_result = {}
    model = xgb.train(
        params,
        train_data,
        evals=[(test_data, "validation")],
        num_boost_round=1000,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=10,
    )

    
    y_pred = model.predict(test_data).astype(int)
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    return model


if __name__ == "__main__":
 
    data_file = "data/processed/ML_Data.csv"
    print("데이터 로드 및 전처리 중...")
    data = pd.read_csv(data_file)

    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    print("모델 학습 및 평가 중...")
    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("모델 저장 중...")
    model.save_model("xgboost_winning_team_model.json")
    print("모든 작업 완료!")
