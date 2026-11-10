import streamlit as st
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="C형 간염 데이터 분석 (2차시)", layout="wide")

st.title("🏥 C형 간염 관련 간 질환군 분류 프로젝트 (2차시)")
st.write("데이터 전처리(인코딩·결측치 대치) 및 탐색적 데이터 분석(EDA) 과정입니다.")

csv_files = glob.glob("*.csv") + glob.glob("*/*.csv")
if not csv_files:
    st.error("CSV 파일을 찾을 수 없습니다.")
    st.stop()

df = pd.read_csv(csv_files[0])
df_clean = df.drop(columns=['Unnamed: 0'], errors='ignore')
df_clean = df_clean[df_clean['Category'] != '0s=suspect Blood Donor'].copy()
df_clean['Target'] = df_clean['Category'].apply(lambda x: 0 if x == '0=Blood Donor' else 1)

with st.expander("📌 1차시 요약 보기 (데이터 정제 및 Target 생성 결과)"):
    st.write(f"분석 대상 데이터: 총 {len(df_clean)}건 (정상 533건, 간 질환군 75건)")
    st.dataframe(df_clean.head())

st.divider()

# 1. 성별 인코딩 및 데이터 분리 / 결측치 대치
st.header("1. 성별 인코딩 및 결측치 대치 (데이터 누출 방지)")
df_clean['Sex'] = df_clean['Sex'].map({'m': 0, 'f': 1})

features = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
X = df_clean[features]
y = df_clean['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

impute_Medians = X_train.median()
X_train_imp = X_train.fillna(impute_Medians)
X_test_imp = X_test.fillna(impute_Medians)

col1, col2, col3 = st.columns(3)
col1.metric("학습 데이터(Train, 80%)", f"{len(X_train_imp)}건")
col2.metric("테스트 데이터(Test, 20%)", f"{len(X_test_imp)}건")
col3.metric("대치 후 남은 결측치", f"{X_train_imp.isnull().sum().sum() + X_test_imp.isnull().sum().sum()}개")

st.write("**학습 데이터 기준 변수별 중앙값(결측치 대치 기준값)**")
st.dataframe(pd.DataFrame(impute_Medians, columns=['중앙값(Median)']).T, use_container_width=True)

st.divider()

# 2. 상관관계 분석
st.header("2. Target(질환 여부)과 각 속성의 피어슨 상관계수")
train_df = X_train_imp.copy()
train_df['Target'] = y_train
corrs = train_df.corr()['Target'].drop('Target').sort_values(ascending=False)
corr_df = pd.DataFrame({'변수명': corrs.index, '상관계수': corrs.values.round(4)})

col4, col5 = st.columns([1, 2])
with col4:
    st.subheader("상관계수 순위표")
    st.dataframe(corr_df, use_container_width=True)
with col5:
    st.subheader("변수별 상관계수 막대그래프")
    st.bar_chart(corr_df.set_index('변수명'))

st.divider()

# 3. 정상군 vs 질환군 주요 변수 비교
st.header("3. 정상군(0) vs 간 질환군(1) 주요 생화학 지표 비교")
key_vars = ['AST', 'GGT', 'BIL', 'CHE', 'ALB', 'CHOL']
group_mean = train_df.groupby('Target')[key_vars].mean().T
group_mean.columns = ['정상군(0) 평균', '간 질환군(1) 평균']

col6, col7 = st.columns([1, 2])
with col6:
    st.subheader("집단별 주요 수치 평균 비교")
    st.dataframe(group_mean.round(2), use_container_width=True)
with col7:
    st.subheader("집단별 수치 차이 시각화")
    st.bar_chart(group_mean)
