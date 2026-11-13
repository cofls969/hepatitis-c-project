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

# 데이터 로드 및 전처리 (기존과 100% 동일)
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

# 이전 차시 요약 (접어두기 - 기존 유지)
with st.expander("📌 1~3차시 데이터 전처리 및 모델 성능 평가 요약 보기"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("정확도 (Accuracy)", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    c2.metric("정밀도 (Precision)", f"{precision_score(y_test, y_pred)*100:.2f}%")
    c3.metric("재현율 (Recall)", f"{recall_score(y_test, y_pred)*100:.2f}%")
    c4.metric("F1-점수 (F1-score)", f"{f1_score(y_test, y_pred)*100:.2f}%")
    st.write("- 전체 608건 데이터 (학습 486건, 테스트 122건) / 학습 데이터 중앙값 결측치 대치 및 표준화 완료")

st.divider()

# 1. 실시간 간 질환 분류 예측 시뮬레이터 (기존 유지)
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

# 2. 예측 결과 및 주요 수치 비교 시각화 (기존 유지)
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

st.divider()

# 3. [신규 추가] 로지스틱 회귀 예측 근거 분석 (변수별 개별 기여도)
st.header("3. 로지스틱 회귀 모델의 예측 근거 분석 (변수별 개별 기여도)")
st.info(
    "💡 **해석 및 주의 안내**: 아래 기여도는 예측 확률(%)의 직접적인 증감률이 아니라, **간 질환군(1)의 로그 오즈(Log-odds)에 대한 선형 기여도(표준화된 입력값 × 회귀계수)**입니다. "
    "양수(+)는 질환군(1) 예측 방향으로, 음수(-)는 정상군(0) 예측 방향으로 작용함을 뜻하며 절댓값이 클수록 예측 점수에 미친 영향이 큽니다. "
    "또한 이 분석 결과는 **본 머신러닝 모델의 수학적 예측 과정에 대한 설명**일 뿐, 실제 의학적 진단이나 질병 발생의 직접적인 원인을 의미하지 않습니다."
)

# 변수별 기여도 및 로짓(Logit), 시그모이드 검증 계산 (학습 당시 변수 순서 그대로 유지)
scaled_vals = input_scaled[0]
coefs = model.coef_[0]
intercept = model.intercept_[0]
contributions = scaled_vals * coefs

contrib_sum = np.sum(contributions)
logit_val = contrib_sum + intercept
sigmoid_prob = 1 / (1 + np.exp(-logit_val))

# 데이터프레임 구성
contrib_df = pd.DataFrame({
    '변수명': features,
    '입력 수치': input_data.iloc[0].values,
    '표준화 값': scaled_vals.round(4),
    '회귀 계수': coefs.round(4),
    '개별 기여도(Log-odds)': contributions.round(4),
    '절댓값': np.abs(contributions)
})

# 수평 막대그래프용 정렬 (기여도가 큰 변수부터 상단에 오도록 정렬)
contrib_sorted_plot = contrib_df.sort_values(by='개별 기여도(Log-odds)', ascending=True)

col_exp1, col_exp2 = st.columns([1.2, 1.1])

with col_exp1:
    st.subheader("12개 독립변수의 개별 기여도 (Log-odds 기준)")
    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    colors = ['#d62728' if v >= 0 else '#1f77b4' for v in contrib_sorted_plot['개별 기여도(Log-odds)']]
    bars = ax_c.barh(
        contrib_sorted_plot['변수명'],
        contrib_sorted_plot['개별 기여도(Log-odds)'],
        color=colors,
        alpha=0.85
    )
    ax_c.axvline(0, color='black', linewidth=1, linestyle='--')
    ax_c.set_xlabel("Log-odds Contribution (Standardized Input × Coefficient)")
    ax_c.set_title("Feature Contributions to Prediction Score (Red: Disease(1) / Blue: Normal(0))")
    ax_c.grid(axis='x', linestyle='--', alpha=0.5)

    # 막대 옆에 기여도 수치 함께 표시
    min_x = contrib_sorted_plot['개별 기여도(Log-odds)'].min()
    max_x = contrib_sorted_plot['개별 기여도(Log-odds)'].max()
    margin = max(abs(min_x), abs(max_x), 0.5) * 0.22
    ax_c.set_xlim(min(min_x - margin, -0.5), max(max_x + margin, 0.5))

    for bar, val in zip(bars, contrib_sorted_plot['개별 기여도(Log-odds)']):
        offset = margin * 0.08 if val >= 0 else -margin * 0.08
        ha_align = 'left' if val >= 0 else 'right'
        ax_c.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va='center',
            ha=ha_align,
            fontsize=9,
            weight='bold'
        )

    plt.tight_layout()
    st.pyplot(fig_c)

with col_exp2:
    st.subheader("예측에 가장 크게 영향을 미친 상위 5개 변수")
    top5_df = contrib_df.sort_values(by='절댓값', ascending=False).head(5).copy()
    top5_df['영향 방향 설명'] = top5_df['개별 기여도(Log-odds)'].apply(
        lambda x: "질환군(1) 예측 방향으로 작용 (+)" if x >= 0 else "정상군(0) 예측 방향으로 작용 (-)"
    )
    display_top5 = top5_df[['변수명', '입력 수치', '회귀 계수', '개별 기여도(Log-odds)', '영향 방향 설명']].reset_index(drop=True)
    st.dataframe(display_top5, use_container_width=True)

    st.write("**모델 예측 점수(Logit) 및 시그모이드(Sigmoid) 재현 검증**")
    v_col1, v_col2, v_col3 = st.columns(3)
    v_col1.metric("기여도 총합 + 절편(로짓)", f"{logit_val:.4f}", f"절편: {intercept:.4f}", delta_color="off")
    v_col2.metric("시그모이드 변환 확률", f"{sigmoid_prob*100:.2f}%")
    v_col3.metric("기존 predict_proba 확률", f"{pred_proba[1]*100:.2f}%")
    st.caption(
        f"✅ 검증 결과: 12개 변수 기여도 합({contrib_sum:.4f}) + 절편({intercept:.4f}) = 로짓({logit_val:.4f})이며, "
        f"이를 시그모이드 함수에 적용한 값({sigmoid_prob:.6f})이 기존 모델의 `predict_proba` 결과({pred_proba[1]:.6f})와 정확히 일치합니다."
    )

# 4. 입력값에 따라 동적으로 생성되는 모델 예측 과정 설명
st.subheader("📝 현재 입력값에 대한 모델의 예측 과정 설명")
pos_top = top5_df[top5_df['개별 기여도(Log-odds)'] > 0]
neg_top = top5_df[top5_df['개별 기여도(Log-odds)'] < 0]

pos_desc = [f"**{row['변수명']}**(입력값 {row['입력 수치']}, 기여도 {row['개별 기여도(Log-odds)']:+.3f})" for _, row in pos_top.iterrows()]
neg_desc = [f"**{row['변수명']}**(입력값 {row['입력 수치']}, 기여도 {row['개별 기여도(Log-odds)']:+.3f})" for _, row in neg_top.iterrows()]

final_label_str = "간 질환군(1)" if pred_class == 1 else "정상 헌혈자군(0)"

if len(pos_desc) > 0 and len(neg_desc) > 0:
    dynamic_explanation = (
        f"현재 입력된 혈액 검사 수치 중 예측에 큰 영향을 준 상위 5개 변수를 분석한 결과, "
        f"{', '.join(pos_desc)} 변수가 **간 질환군(1) 방향**으로 상대적으로 큰 영향을 주었으며, "
        f"{', '.join(neg_desc)} 변수는 **정상군(0) 방향**으로 영향을 주었습니다. "
        f"모델은 12개 변수의 로그 오즈 기여도 총합({contrib_sum:+.3f})과 절편({intercept:+.3f})을 종합하여 최종 로짓 점수({logit_val:+.3f})를 산출했고, "
        f"이에 따라 간 질환군 예측 확률을 **{pred_proba[1]*100:.2f}%**로 계산하여 최종적으로 **{final_label_str}**으로 예측했습니다."
    )
elif len(pos_desc) > 0:
    dynamic_explanation = (
        f"현재 입력된 혈액 검사 수치 중 주요 상위 변수인 {', '.join(pos_desc)} 등이 모두 **간 질환군(1) 방향**으로 강하게 영향을 주었습니다. "
        f"이러한 변수들의 양(+)의 기여도 총합({contrib_sum:+.3f})과 절편({intercept:+.3f})을 종합하여 최종 로짓 점수({logit_val:+.3f})가 산출되었으며, "
        f"모델이 간 질환군 예측 확률을 **{pred_proba[1]*100:.2f}%**로 계산하여 최종적으로 **{final_label_str}**으로 예측했습니다."
    )
else:
    dynamic_explanation = (
        f"현재 입력된 혈액 검사 수치 중 주요 상위 변수인 {', '.join(neg_desc)} 등이 모두 **정상군(0) 방향**으로 강하게 영향을 주었습니다. "
        f"이러한 변수들의 음(-)의 기여도 총합({contrib_sum:+.3f})과 절편({intercept:+.3f})을 종합하여 최종 로짓 점수({logit_val:+.3f})가 산출되었으며, "
        f"모델이 간 질환군 예측 확률을 **{pred_proba[1]*100:.2f}%**(정상군 확률 **{pred_proba[0]*100:.2f}%**)로 계산하여 최종적으로 **{final_label_str}**으로 예측했습니다."
    )

st.success(dynamic_explanation)
