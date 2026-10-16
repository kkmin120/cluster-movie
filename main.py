
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 스크린 수, 누적 관객, 10위권 일수, 롱런 지수를 이용해 "
    "비슷한 특징을 가진 영화 3개 유형으로 나눕니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    return df


try:
    original_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

total_movies = len(original_df)

# --------------------------------------------------
# 3. 숫자 데이터 변환 및 파생 속성 만들기
# --------------------------------------------------
df = original_df.copy()

numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col].astype("string").str.replace(",", "", regex=False),
        errors="coerce"
    )

# 로그 계산이 가능한 양수인지 확인
valid = (
    df["first_scrn"].gt(0)
    & df["total_audi"].gt(0)
    & df["first_week_audi"].gt(0)
    & df["days_in_top10"].notna()
    & df["first_scrn"].notna()
    & df["total_audi"].notna()
    & df["first_week_audi"].notna()
)

df = df.loc[valid].copy()

# 네 가지 분석 속성
df["스크린 수 (로그)"] = df["first_scrn"].map(
    lambda x: __import__("math").log10(x)
)

df["누적 관객 (로그)"] = df["total_audi"].map(
    lambda x: __import__("math").log10(x)
)

df["10위권 일수"] = df["days_in_top10"]

df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

feature_names = [
    "스크린 수 (로그)",
    "누적 관객 (로그)",
    "10위권 일수",
    "롱런 지수"
]

if len(df) < 3:
    st.error("분류할 수 있는 영화가 3편보다 적습니다.")
    st.stop()

# 전체 편수와 분석에 사용한 편수
st.markdown(
    f"**전체 영화: {total_movies:,}편　|　"
    f"묶은 영화: {len(df):,}편**"
)

st.caption(
    "결측값이 있거나 스크린 수·누적 관객이 0 이하인 영화, "
    "첫 주 관객이 0 이하인 영화는 분석에서 제외합니다. "
    "로그는 상용로그(log₁₀)를 사용합니다."
)

# --------------------------------------------------
# 4. 군집에 사용할 속성 선택
# --------------------------------------------------
st.subheader("1. 영화 유형을 나눌 속성 선택")

selected_features = st.multiselect(
    "두 개 이상 선택하세요.",
    options=feature_names,
    default=feature_names,
    key="selected_features"
)

if len(selected_features) < 2:
    st.warning("군집 분석을 위해 속성을 두 개 이상 선택하세요.")
    st.stop()

# --------------------------------------------------
# 5. 표준화 및 K-평균 군집 분석
# --------------------------------------------------
X = df[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["군집 번호"] = kmeans.fit_predict(X_scaled)

# --------------------------------------------------
# 6. 누적 관객 평균 순으로 ㉮, ㉯, ㉰ 지정
# --------------------------------------------------
cluster_order = (
    df.groupby("군집 번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

label_map = {
    cluster_id: ["㉮", "㉯", "㉰"][rank]
    for rank, cluster_id in enumerate(cluster_order)
}

df["영화 유형"] = df["군집 번호"].map(label_map)

cluster_colors = {
    "㉮": "#E76F51",
    "㉯": "#457B9D",
    "㉰": "#2A9D8F"
}

# --------------------------------------------------
# 7. 2차원 산점도
# --------------------------------------------------
st.subheader("2. 2차원 산점도")
st.write(
    "가로축과 세로축을 선택해 영화 유형별 분포를 확인하세요. "
    "점에 마우스를 올리면 영화 제목을 볼 수 있습니다."
)

col_x, col_y = st.columns(2)

with col_x:
    x_axis = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        key="x_axis"
    )

with col_y:
    y_options = [f for f in selected_features if f != x_axis]
    y_axis = st.selectbox(
        "세로축",
        options=y_options,
        index=0,
        key="y_axis"
    )

fig2d = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="영화 유형",
    color_discrete_map=cluster_colors,
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        x_axis: ":.3f",
        y_axis: ":.3f"
    },
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    labels={
        x_axis: x_axis,
        y_axis: y_axis,
        "영화 유형": "영화 유형"
    },
    title="영화 유형별 분포"
)

fig2d.update_traces(marker=dict(size=8, opacity=0.8))
fig2d.update_layout(
    height=550,
    legend_title_text="영화 유형"
)

st.plotly_chart(fig2d, use_container_width=True)

# --------------------------------------------------
# 8. 3차원 산점도
# --------------------------------------------------
st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 군집 분석에 사용할 속성을 "
        "세 개 이상 선택하세요."
    )
else:
    st.write(
        "세 축을 선택하세요. 그래프를 마우스로 드래그하면 "
        "시점을 회전할 수 있습니다."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축",
            options=selected_features,
            index=0,
            key="x3"
        )

    y3_options = [f for f in selected_features if f != x3]

    with c2:
        y3 = st.selectbox(
            "Y축",
            options=y3_options,
            index=0,
            key="y3"
        )

    z3_options = [
        f for f in selected_features
        if f != x3 and f != y3
    ]

    with c3:
        z3 = st.selectbox(
            "Z축",
            options=z3_options,
            index=0,
            key="z3"
        )

    fig3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        color_discrete_map=cluster_colors,
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            x3: ":.3f",
            y3: ":.3f",
            z3: ":.3f"
        },
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        labels={
            x3: x3,
            y3: y3,
            z3: z3,
            "영화 유형": "영화 유형"
        },
        title="영화 유형별 3차원 분포"
    )

    fig3d.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig3d.update_layout(
        height=650,
        legend_title_text="영화 유형",
        scene=dict(
            xaxis_title=x3,
            yaxis_title=y3,
            zaxis_title=z3
        )
    )

    st.plotly_chart(fig3d, use_container_width=True)

# --------------------------------------------------
# 9. 유형별 편수와 원래 단위의 평균
# --------------------------------------------------
st.subheader("4. 유형별 편수와 네 속성의 평균")

summary = (
    df.groupby("영화 유형")
    .agg(
        영화 편수=("movieNm", "size"),
        평균_스크린수=("first_scrn", "mean"),
        평균_누적관객=("total_audi", "mean"),
        평균_10위권일수=("days_in_top10", "mean"),
        평균_롱런지수=("롱런 지수", "mean")
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
)

summary = summary.rename(columns={
    "영화 유형": "유형",
    "영화 편수": "편수",
    "평균_스크린수": "평균 스크린 수 (개)",
    "평균_누적관객": "평균 누적 관객 (명)",
    "평균_10위권일수": "평균 10위권 일수 (일)",
    "평균_롱런지수": "평균 롱런 지수"
})

summary["평균 스크린 수 (개)"] = (
    summary["평균 스크린 수 (개)"].round(1)
)
summary["평균 누적 관객 (명)"] = (
    summary["평균 누적 관객 (명)"].round(0)
)
summary["평균 10위권 일수 (일)"] = (
    summary["평균 10위권 일수 (일)"].round(1)
)
summary["평균 롱런 지수"] = (
    summary["평균 롱런 지수"].round(2)
)

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "유형 기호는 누적 관객 평균이 높은 유형부터 ㉮, ㉯, ㉰ 순서입니다. "
    "표의 평균은 로그 변환 전 원래 단위를 사용합니다."
)

# --------------------------------------------------
# 10. 유형별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("5. 유형별 누적 관객 상위 영화 5편")

for label in ["㉮", "㉯", "㉰"]:
    cluster_df = (
        df[df["영화 유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    st.markdown(f"### {label} 유형")

    if cluster_df.empty:
        st.write("해당 유형에 영화가 없습니다.")
        continue

    top_movies = cluster_df[
        ["movieNm", "total_audi", "first_week_audi", "days_in_top10"]
    ].copy()

    top_movies.columns = [
        "영화 제목",
        "누적 관객 (명)",
        "첫 주 관객 (명)",
        "10위권 일수"
    ]

    top_movies["누적 관객 (명)"] = (
        top_movies["누적 관객 (명)"].map(lambda x: f"{x:,.0f}")
    )
    top_movies["첫 주 관객 (명)"] = (
        top_movies["첫 주 관객 (명)"].map(lambda x: f"{x:,.0f}")
    )

    st.dataframe(
        top_movies,
        hide_index=True,
        use_container_width=True
    )

# --------------------------------------------------
# 11. 분석 방법 안내
# --------------------------------------------------
with st.expander("분석 방법 자세히 보기"):
    st.markdown("""
    **① 결측값 처리**

    분석에 필요한 속성에 값이 없거나, 로그 계산에 필요한
    스크린 수·누적 관객이 양수가 아니거나, 첫 주 관객이
    양수가 아닌 영화는 제외합니다.

    **② 롱런 지수 계산**

    누적 관객을 첫 주 관객으로 나눈 값입니다.
    지수가 20을 넘으면 20으로 제한합니다.

    **③ 표준화**

    속성마다 단위와 값의 범위가 다르므로 평균 0, 표준편차 1을
    기준으로 표준화합니다.

    **④ K-평균 군집 분석**

    선택한 속성을 이용해 영화들을 세 군집으로 나눕니다.
    정답 유형을 미리 지정하지 않는 비지도 학습입니다.

    **⑤ 유형 이름 지정**

    분석 결과의 군집 번호는 임의적이므로, 누적 관객 평균이
    높은 군집부터 ㉮, ㉯, ㉰로 표시합니다.
    """)

st.caption(
    "같은 데이터와 같은 속성을 선택하면 random_state=42 설정으로 "
    "군집 결과를 재현할 수 있습니다."
)
