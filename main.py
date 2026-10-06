import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 기본 설정
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

DATA_START_YEAR = 1908
DATA_END_YEAR = 2025
MIN_DAYS = 300

TRAIN_END_YEAR = 2004
TEST_START_YEAR = 2005

RECENT_START_YEAR = 2006
RECENT_END_YEAR = 2025


st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ 서울 기온 예측기")

st.write(
    "서울의 연평균기온 데이터를 이용해 장기적인 기온 변화와 "
    "다양한 회귀모델의 예측 성능을 비교합니다."
)


# ============================================================
# 데이터 불러오기
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    df["연도"] = df["날짜"].dt.year

    # 2025년까지 사용
    df = df[
        df["연도"] <= DATA_END_YEAR
    ]

    # 연도별 평균기온과 관측일수
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 해 제외
    yearly = yearly[
        yearly["관측일수"] >= MIN_DAYS
    ]

    # 1908년부터 사용
    yearly = yearly[
        yearly["연도"] >= DATA_START_YEAR
    ]

    return yearly.sort_values(
        "연도"
    ).reset_index(drop=True)


df = load_data()


if len(df) < 2:
    st.error(
        "분석할 연도별 데이터가 충분하지 않습니다."
    )
    st.stop()


# ============================================================
# 1. 전체 기간 선형회귀
# ============================================================

df["경과연수"] = (
    df["연도"] - DATA_START_YEAR
)

X_all = df[
    ["경과연수"]
]

y_all = df["평균기온"]


model_all = LinearRegression()

model_all.fit(
    X_all,
    y_all
)


slope_all = model_all.coef_[0]

intercept_all = model_all.intercept_

# 100년에 몇 도 상승하는가
slope_all_100 = slope_all * 100

correlation = df["연도"].corr(
    df["평균기온"]
)


# ============================================================
# 2. 최근 20년 선형회귀
# ============================================================

recent_df = df[
    (df["연도"] >= RECENT_START_YEAR)
    & (df["연도"] <= RECENT_END_YEAR)
].copy()


recent_df["경과연수"] = (
    recent_df["연도"]
    - RECENT_START_YEAR
)


X_recent = recent_df[
    ["경과연수"]
]

y_recent = recent_df[
    "평균기온"
]


model_recent = LinearRegression()

model_recent.fit(
    X_recent,
    y_recent
)


slope_recent = model_recent.coef_[0]

intercept_recent = model_recent.intercept_

slope_recent_100 = (
    slope_recent * 100
)


# ============================================================
# 3. 화면 - 전체 기간 상승률
# ============================================================

st.header("📈 장기적인 기온 상승 추세")

st.metric(
    label="100년에 기온이 얼마나 오르는가?",
    value=f"{slope_all_100:+.2f} °C",
)

st.caption(
    f"{int(df['연도'].min())}~{int(df['연도'].max())}년 "
    "연평균기온으로 계산한 선형회귀 기울기입니다."
)


# ============================================================
# 4. 전체 기간 vs 최근 20년 비교
# ============================================================

st.subheader(
    "전체 기간 vs 최근 20년 상승률 비교"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        label=(
            f"전체 기간 "
            f"({int(df['연도'].min())}~"
            f"{int(df['연도'].max())})"
        ),
        value=(
            f"{slope_all_100:+.2f} "
            "°C / 100년"
        ),
    )

    st.write(
        f"연간 기울기: "
        f"{slope_all:+.5f} °C/년"
    )


with col2:

    st.metric(
        label=(
            f"최근 20년 "
            f"({RECENT_START_YEAR}~"
            f"{RECENT_END_YEAR})"
        ),
        value=(
            f"{slope_recent_100:+.2f} "
            "°C / 100년"
        ),
    )

    st.write(
        f"연간 기울기: "
        f"{slope_recent:+.5f} °C/년"
    )


difference = (
    slope_recent_100
    - slope_all_100
)


if difference > 0:

    st.info(
        "최근 20년의 상승 기울기가 "
        f"전체 기간보다 100년 기준 "
        f"{difference:.2f}°C 더 큽니다."
    )

elif difference < 0:

    st.info(
        "최근 20년의 상승 기울기가 "
        f"전체 기간보다 100년 기준 "
        f"{abs(difference):.2f}°C 더 작습니다."
    )

else:

    st.info(
        "두 기간의 상승 기울기가 같습니다."
    )


# ============================================================
# 5. 전체 기간 기본 정보
# ============================================================

st.subheader("📊 전체 데이터 정보")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "회귀에 사용한 연도 수",
        f"{len(df)}개",
    )


with col2:

    st.metric(
        "시작 연도",
        f"{int(df['연도'].min())}년",
    )


with col3:

    st.metric(
        "끝 연도",
        f"{int(df['연도'].max())}년",
    )


with col4:

    st.metric(
        "상관계수",
        f"{correlation:.4f}",
    )


# ============================================================
# 6. 연도 슬라이더를 이용한 전체 선형회귀 예측
# ============================================================

st.header("🔮 연도별 기온 예측")


selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)


selected_elapsed = (
    selected_year - DATA_START_YEAR
)


selected_prediction = model_all.predict(
    np.array(
        [[selected_elapsed]]
    )
)[0]


st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{selected_prediction:.2f} °C",
)


# ============================================================
# 7. 선형회귀 그래프
# ============================================================

st.subheader(
    "서울 연평균기온과 선형회귀선"
)


fig_linear = go.Figure()


fig_linear.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#1f77b4",
            opacity=0.75,
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)


linear_years = np.arange(
    1900,
    2101,
)


linear_x = (
    linear_years
    - DATA_START_YEAR
).reshape(-1, 1)


linear_prediction = model_all.predict(
    linear_x
)


fig_linear.add_trace(
    go.Scatter(
        x=linear_years,
        y=linear_prediction,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            color="#e74c3c",
            width=3,
        ),
    )
)


fig_linear.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[selected_prediction],
        mode="markers",
        name="선택 연도 예측",
        marker=dict(
            size=15,
            color="#2ca02c",
            line=dict(
                color="white",
                width=2,
            ),
        ),
    )
)


fig_linear.update_layout(
    xaxis=dict(
        title="연도",
        range=[1900, 2100],
        dtick=10,
    ),
    yaxis=dict(
        title="연평균기온 (°C)",
    ),
    template="plotly_white",
    height=550,
)


st.plotly_chart(
    fig_linear,
    use_container_width=True,
)


# ============================================================
# 8. 다항회귀: 훈련 / 테스트 분리
# ============================================================

st.header(
    "🧪 1차·3차·9차 곡선 예측 성능 비교"
)


st.write(
    "2005년 이전 데이터를 훈련용으로만 사용하고, "
    "2005~2025년 데이터는 학습에 사용하지 않은 "
    "테스트용 데이터로만 채점합니다."
)


train = df[
    df["연도"] <= TRAIN_END_YEAR
].copy()


test = df[
    df["연도"] >= TEST_START_YEAR
].copy()


# 데이터 개수
col1, col2 = st.columns(2)


with col1:

    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개",
    )

    st.caption(
        f"{int(train['연도'].min())}~"
        f"{int(train['연도'].max())}년"
    )


with col2:

    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개",
    )

    st.caption(
        f"{int(test['연도'].min())}~"
        f"{int(test['연도'].max())}년"
    )


# ============================================================
# 9. 연도 스케일링
# ============================================================

# 고차 다항회귀의 수치 불안정을 줄이기 위해
# 1908년을 0으로 만들어 사용

X_train = (
    train["연도"]
    - DATA_START_YEAR
).to_numpy().reshape(-1, 1)


y_train = train[
    "평균기온"
].to_numpy()


X_test = (
    test["연도"]
    - DATA_START_YEAR
).to_numpy().reshape(-1, 1)


y_test = test[
    "평균기온"
].to_numpy()


# ============================================================
# 10. 1차 / 3차 / 9차 모델 학습
# ============================================================

degrees = [
    1,
    3,
    9,
]


models = {}

results = []


for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree
        ),
        LinearRegression()
    )

    # 오직 훈련 데이터로 학습
    model.fit(
        X_train,
        y_train
    )


    # 학습에 사용하지 않은 테스트 데이터로 평가
    test_prediction = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        test_prediction
    )


    mse = mean_squared_error(
        y_test,
        test_prediction
    )


    r2 = r2_score(
        y_test,
        test_prediction
    )


    # 2050년 예측
    X_2050 = np.array(
        [[2050 - DATA_START_YEAR]]
    )


    prediction_2050 = model.predict(
        X_2050
    )[0]


    models[degree] = model


    results.append(
        {
            "모델": f"{degree}차",
            "테스트 MAE (°C)": mae,
            "테스트 MSE": mse,
            "테스트 R²": r2,
            "2050년 예측 (°C)": prediction_2050,
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 11. 모델 비교표
# ============================================================

st.subheader(
    "📋 테스트 데이터 성능 및 2050년 예측"
)


display_df = results_df.copy()


display_df[
    "테스트 MAE (°C)"
] = display_df[
    "테스트 MAE (°C)"
].map(
    lambda x: f"{x:.2f}"
)


display_df[
    "테스트 MSE"
] = display_df[
    "테스트 MSE"
].map(
    lambda x: f"{x:.2f}"
)


display_df[
    "테스트 R²"
] = display_df[
    "테스트 R²"
].map(
    lambda x: f"{x:.4f}"
)


display_df[
    "2050년 예측 (°C)"
] = display_df[
    "2050년 예측 (°C)"
].map(
    lambda x: f"{x:.2f}"
)


st.table(
    display_df
)


st.info(
    "MAE·MSE·R²는 모두 학습에 사용하지 않은 "
    "2005~2025년 테스트 데이터만으로 계산했습니다."
)


# ============================================================
# 12. 다항회귀 그래프
# ============================================================

st.subheader(
    "📈 1차·3차·9차 회귀곡선"
)


fig_poly = go.Figure()


# 훈련 데이터
fig_poly.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련용 데이터",
        marker=dict(
            size=6,
            color="#1f77b4",
            opacity=0.65,
        ),
    )
)


# 테스트 데이터
fig_poly.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        name="테스트용 데이터",
        marker=dict(
            size=8,
            color="#ff7f0e",
            symbol="diamond",
        ),
    )
)


plot_years = np.arange(
    int(df["연도"].min()),
    2051,
)


plot_x = (
    plot_years
    - DATA_START_YEAR
).reshape(-1, 1)


colors = {
    1: "#e74c3c",
    3: "#2ca02c",
    9: "#8e44ad",
}


for degree in degrees:

    model = models[degree]


    prediction = model.predict(
        plot_x
    )


    fig_poly.add_trace(
        go.Scatter(
            x=plot_years,
            y=prediction,
            mode="lines",
            name=f"{degree}차 곡선",
            line=dict(
                color=colors[degree],
                width=3 if degree != 9 else 2,
                dash=(
                    "solid"
                    if degree != 9
                    else "dash"
                ),
            ),
        )
    )


# 2005년 기준선
fig_poly.add_vline(
    x=2005,
    line_width=2,
    line_dash="dot",
    line_color="gray",
)


fig_poly.add_annotation(
    x=2005,
    y=1,
    yref="paper",
    text="2005년: 테스트 시작",
    showarrow=False,
    xanchor="left",
)


fig_poly.update_layout(
    title="훈련 데이터와 다항회귀 곡선",
    xaxis=dict(
        title="연도",
    ),
    yaxis=dict(
        title="연평균기온 (°C)",
    ),
    template="plotly_white",
    height=650,
    hovermode="closest",
)


st.plotly_chart(
    fig_poly,
    use_container_width=True,
)


# ============================================================
# 13. 테스트 데이터 실제값 vs 모델 예측값
# ============================================================

st.subheader(
    "🔍 테스트 데이터 실제값과 예측값"
)


test_result = test[
    [
        "연도",
        "평균기온",
    ]
].copy()


for degree in degrees:

    test_result[
        f"{degree}차 예측"
    ] = models[degree].predict(
        X_test
    )


st.dataframe(
    test_result.style.format(
        {
            "평균기온": "{:.2f}",
            "1차 예측": "{:.2f}",
            "3차 예측": "{:.2f}",
            "9차 예측": "{:.2f}",
        }
    ),
    use_container_width=True,
)


# ============================================================
# 14. 회귀식 및 분석 정보
# ============================================================

with st.expander(
    "분석 방법 및 기준"
):

    st.markdown(
        f"""
### 데이터 처리

- 서울 일별 기온 데이터를 연도별 평균기온으로 변환
- 2025년 이후 데이터 제외
- 연간 관측일수가 {MIN_DAYS}일 미만인 연도 제외
- {DATA_START_YEAR}년 이후 데이터 사용

### 전체 기간 선형회귀

- 전체 유효 연도를 사용
- 독립변수는 `연도 - {DATA_START_YEAR}`
- 기울기는 `°C/년`에서 `°C/100년`으로 변환하여 표시

### 다항회귀

- 훈련 데이터: {DATA_START_YEAR}년~{TRAIN_END_YEAR}년
- 테스트 데이터: {TEST_START_YEAR}년~{DATA_END_YEAR}년
- 1차, 3차, 9차 다항회귀를 각각 별도로 학습
- 테스트 데이터는 학습 과정에 전혀 사용하지 않음
- MAE, MSE, R²로 테스트 성능 평가
- 2050년은 각 모델에 한 번도 학습시키지 않은 미래값으로 예측

### 수치 안정성

고차 다항회귀에서 큰 연도를 그대로 사용하면 수치적으로 불안정할 수 있으므로
회귀 계산에서는 실제 연도 대신 다음 값을 사용합니다.

`경과연수 = 연도 - {DATA_START_YEAR}`

따라서 {DATA_START_YEAR}년은 0, 2000년은
{2000 - DATA_START_YEAR}, 2025년은
{2025 - DATA_START_YEAR}로 변환됩니다.
        """
    )


# ============================================================
# 15. 연도별 데이터
# ============================================================

with st.expander(
    "연도별 평균기온 데이터 보기"
):

    st.dataframe(
        df[
            [
                "연도",
                "관측일수",
                "평균기온",
            ]
        ].style.format(
            {
                "평균기온": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
