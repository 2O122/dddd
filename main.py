import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CSV 파일 경로와 컬럼명에 맞게 수정
FILE_PATH = "annual_temperature.csv"
YEAR_COL = "year"
TEMP_COL = "temperature"

# 데이터 불러오기
df = pd.read_csv(FILE_PATH)

df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce")
df = df[[YEAR_COL, TEMP_COL]].dropna().sort_values(YEAR_COL)

# 공통 테스트 데이터: 2006~2025
test = df[(df[YEAR_COL] >= 2006) & (df[YEAR_COL] <= 2025)]

# 학습 기간
train_50 = df[(df[YEAR_COL] >= 1956) & (df[YEAR_COL] <= 2005)]
train_100 = df[(df[YEAR_COL] >= 1906) & (df[YEAR_COL] <= 2005)]


def train_and_evaluate(train, test, name):
    X_train = train[[YEAR_COL]]
    y_train = train[TEMP_COL]

    X_test = test[[YEAR_COL]]
    y_test = test[TEMP_COL]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")
    print(f"학습 기간 : {train[YEAR_COL].min()}~{train[YEAR_COL].max()}")
    print(f"테스트 기간 : {test[YEAR_COL].min()}~{test[YEAR_COL].max()}")
    print(f"기울기(slope) : {model.coef_[0]:.6f}")
    print(f"절편(intercept) : {model.intercept_:.6f}")
    print(f"회귀식 : temperature = {model.coef_[0]:.6f} × year + {model.intercept_:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MSE : {mse:.6f}")
    print(f"R² : {r2:.6f}")

    return {
        "model": model,
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
    }


# 50년 학습
result_50 = train_and_evaluate(
    train_50,
    test,
    "최근 50년 학습 (1956~2005)"
)

# 100년 학습
result_100 = train_and_evaluate(
    train_100,
    test,
    "최근 100년 학습 (1906~2005)"
)


# 결과 비교
comparison = pd.DataFrame({
    "학습기간": ["1956~2005", "1906~2005"],
    "기울기": [
        result_50["slope"],
        result_100["slope"],
    ],
    "MAE": [
        result_50["MAE"],
        result_100["MAE"],
    ],
    "MSE": [
        result_50["MSE"],
        result_100["MSE"],
    ],
    "R²": [
        result_50["R2"],
        result_100["R2"],
    ],
})

print("\n\n[회귀선 및 테스트 성능 비교]")
print(comparison.to_string(index=False))

print("\n[성능 차이: 50년 - 100년]")
print(f"기울기 차이 : {result_50['slope'] - result_100['slope']:.6f}")
print(f"MAE 차이 : {result_50['MAE'] - result_100['MAE']:.6f}")
print(f"MSE 차이 : {result_50['MSE'] - result_100['MSE']:.6f}")
print(f"R² 차이 : {result_50['R2'] - result_100['R2']:.6f}")


# 테스트 기간 실제값과 두 회귀선의 예측값 저장
predictions = test[[YEAR_COL, TEMP_COL]].copy()
predictions["pred_50y"] = result_50["model"].predict(test[[YEAR_COL]])
predictions["pred_100y"] = result_100["model"].predict(test[[YEAR_COL]])

predictions.to_csv(
    "regression_test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

comparison.to_csv(
    "regression_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
