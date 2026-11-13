import streamlit as st
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="C형 간염 간 질환 분류 AI 웹앱 (최종)", layout="wide")

st.title("🏥 C형 간염 관련 간 질환군 분류 예측 웹앱 (최종 완성본)")
st.write("혈액 생화학 검사 수치를 입력하면 로지스틱 회귀 모델이 정상군(0)과 간 질환군(1) 여부를 실시간으로 예측합니다.")

# 데이터 로드 및 전처리
csv_files = glob.glob("*.csv") + glob.glob("*/*.csv")
if not csv_files:
    st.error("CSV 파일을 찾을 수 없습니다.")
    st.stop()

df = pd.read_csv(csv_files[0])
df_clean = df.drop(columns=['Unnamed: 0'], errors='ignore')
df_clean = df_clean[df_clean['Category'] != '0s=suspect Blood Donor'].copy()
df_clean['Target'] = df_clean['Category'].apply(lambda x: 0 if x == '0=Blood Donor' else 1)
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 이전 차시 요약 (접어두기)
with st.expander("📌 1~3차시 데이터 전처리 및 모델 성능 평가 요약 보기"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("정확도 (Accuracy)", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    c2.metric("정밀도 (Precision)", f"{precision_score(y_test, y_pred)*100:.2f}%")
    c3.metric("재현율 (Recall)", f"{recall_score(y_test, y_pred)*100:.2f}%")
    c4.metric("F1-점수 (F1-score)", f"{f1_score(y_test, y_pred)*100:.2f}%")
    st.write("- 전체 608건 데이터 (학습 486건, 테스트 122건) / 학습 데이터 중앙값 결측치 대치 및 표준화 완료")

st.divider()

# 1. 실시간 간 질환 분류 예측 시뮬레이터
st.header("1. 환자 혈액 검사 수치 입력 및 실시간 질환 예측")
st.write("아래 샘플 선택 버튼을 누르거나 슬라이더를 조절하여 혈액 검사 수치를 입력해 보세요.")

preset = st.radio(
    "빠른 테스트를 위한 입력값 프리셋 선택:",
    ["직접 슬라이더로 조절 (기본값: 정상군 평균)", "정상 헌혈자 평균 수치 불러오기", "간 질환군(간염·섬유화·간경변) 평균 수치 불러오기"],
    horizontal=True
)

normal_means = X_train_imp[y_train == 0].mean()
disease_means = X_train_imp[y_train == 1].mean()

if preset == "간 질환군(간염·섬유화·간경변) 평균 수치 불러오기":
    default_vals = disease_means
else:
    default_vals = normal_means

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    st.subheader("기본 정보 및 핵심 효소")
    age = st.slider("Age (나이)", 18, 80, int(default_vals['Age']))
    sex_label = st.selectbox("Sex (성별)", ["남성 (m=0)", "여성 (f=1)"])
    sex = 0 if "남성" in sex_label else 1
    ast = st.slider("AST (아스파르테이트 아미노전이효소, U/L)", 10.0, 350.0, float(round(default_vals['AST'], 1)))
    ggt = st.slider("GGT (감마-글루타밀전이효소, U/L)", 5.0, 650.0, float(round(default_vals['GGT'], 1)))

with col_in2:
    st.subheader("간 손상 및 대사 지표")
    bil = st.slider("BIL (빌리루빈 농도, µmol/L)", 1.0, 260.0, float(round(default_vals['BIL'], 1)))
    alt = st.slider("ALT (알라닌 아미노전이효소, U/L)", 1.0, 350.0, float(round(default_vals['ALT'], 1)))
    alp = st.slider("ALP (알칼리성 인산분해효소, U/L)", 10.0, 420.0, float(round(default_vals['ALP'], 1)))
    crea = st.slider("CREA (크레아티닌 농도, µmol/L)", 8.0, 1100.0, float(round(default_vals['CREA'], 1)))

with col_in3:
    st.subheader("간 합성 기능 지표")
    alb = st.slider("ALB (알부민 농도, g/L)", 15.0, 85.0, float(round(default_vals['ALB'], 1)))
    che = st.slider("CHE (콜린에스테라아제, kU/L)", 1.0, 17.0, float(round(default_vals['CHE'], 1)))
    chol = st.slider("CHOL (콜레스테롤, mmol/L)", 1.5, 10.0, float(round(default_vals['CHOL'], 1)))
    prot = st.slider("PROT (총 단백질 농도, g/L)", 40.0, 95.0, float(round(default_vals['PROT'], 1)))

input_data = pd.DataFrame([[age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]], columns=features)
input_scaled = scaler.transform(input_data)

pred_class = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0]

st.divider()

# 2. 예측 결과 및 주요 수치 비교 시각화
st.header("2. AI 모델 예측 결과 및 주요 지표 비교 분석")

res_col1, res_col2 = st.columns([1, 1.3])

with res_col1:
    st.subheader("진단 분류 예측 결과")
    if pred_class == 1:
        st.error("🚨 예측 결과: **간 질환군 (Target = 1)** 가능성 높음")
    else:
        st.success("✅ 예측 결과: **정상 헌혈자군 (Target = 0)** 가능성 높음")

    st.metric("간 질환군(1) 예측 확률", f"{pred_proba[1]*100:.2f}%")
    st.metric("정상군(0) 예측 확률", f"{pred_proba[0]*100:.2f}%")

    st.write("**입력된 환자 데이터 요약표**")
    st.dataframe(input_data, use_container_width=True)

with res_col2:
    st.subheader("입력 수치 vs 정상군·질환군 주요 지표 평균 비교")
    compare_vars = ['AST', 'GGT', 'BIL', 'ALB', 'CHE', 'CHOL']
    comp_df = pd.DataFrame({
        'Normal Mean (0)': normal_means[compare_vars].values,
        'Patient Input': input_data[compare_vars].values[0],
        'Disease Mean (1)': disease_means[compare_vars].values
    }, index=compare_vars)

    fig, ax = plt.subplots(figsize=(7, 4))
    comp_df.plot(kind='bar', ax=ax, rot=0)
    ax.set_title("Key Biomarkers Comparison (Normal vs Input vs Disease)")
    ax.set_ylabel("Value")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
