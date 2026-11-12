import streamlit as st
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="C형 간염 분류 모델링 및 평가 (3차시)", layout="wide")

st.title("🏥 C형 간염 관련 간 질환군 분류 프로젝트 (3차시)")
st.write("데이터 표준화(StandardScaler), 로지스틱 회귀 모델 학습 및 성능 평가 과정입니다.")

# 데이터 로드 및 1~2차시 전처리
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

with st.expander("📌 1~2차시 전처리 요약 보기"):
    st.write(f"- 전체 분석 데이터: {len(df_clean)}건 (학습용 486건, 테스트용 122건)")
    st.write("- 결측치 31개 학습 데이터 중앙값 대치 완료 및 12개 독립변수 선정")

st.divider()

# 1. 데이터 표준화 (스케일링)
st.header("1. 데이터 표준화 (StandardScaler 적용)")
st.write("변수마다 단위와 범위가 다르므로, 학습 데이터(Train) 기준으로 평균 0, 표준편차 1이 되도록 표준화를 수행합니다.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

scaled_df = pd.DataFrame(X_train_scaled, columns=features)
st.subheader("표준화 변환 후 학습 데이터 상위 5행 확인")
st.dataframe(scaled_df.head().round(3), use_container_width=True)

st.divider()

# 2. 로지스틱 회귀 모델 학습 및 성능 평가 지표
st.header("2. 로지스틱 회귀 모델 학습 및 성능 평가")
st.write("정상군(533건)과 질환군(75건)의 클래스 불균형을 보정하기 위해 `class_weight='balanced'` 옵션을 적용하여 학습했습니다.")

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("정확도 (Accuracy)", f"{acc:.4f} ({acc*100:.2f}%)")
col2.metric("정밀도 (Precision)", f"{prec:.4f} ({prec*100:.2f}%)")
col3.metric("재현율 (Recall)", f"{rec:.4f} ({rec*100:.2f}%)")
col4.metric("F1-점수 (F1-score)", f"{f1:.4f} ({f1*100:.2f}%)")

st.divider()

# 3. 혼동 행렬 및 변수 중요도 시각화
st.header("3. 혼동 행렬(Confusion Matrix) 및 회귀 계수 분석")

col5, col6 = st.columns(2)

with col5:
    st.subheader("테스트 데이터 혼동 행렬 (122건)")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large', weight='bold')
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Pred Normal (0)', 'Pred Disease (1)'])
    ax_cm.set_yticklabels(['Actual Normal (0)', 'Actual Disease (1)'])
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('Actual Label')
    ax_cm.set_title('Confusion Matrix')
    plt.tight_layout()
    st.pyplot(fig_cm)
    st.write(f"- **정상 맞힘(TN)**: {cm[0,0]}건 / **정상 오분류(FP)**: {cm[0,1]}건")
    st.write(f"- **질환 놓침(FN)**: {cm[1,0]}건 / **질환 맞힘(TP)**: {cm[1,1]}건")

with col6:
    st.subheader("독립변수별 로지스틱 회귀 계수 (영향력)")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0].round(4)
    }).sort_values(by='Coefficient', ascending=False)
    st.dataframe(coef_df, use_container_width=True)
    st.bar_chart(coef_df.set_index('Feature'))
