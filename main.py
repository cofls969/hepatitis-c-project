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

# 측정 단위 사전 정의
feature_units = {
    'Age': '세', 'Sex': '0=m, 1=f', 'ALB': 'g/L', 'ALP': 'U/L',
    'ALT': 'U/L', 'AST': 'U/L', 'BIL': 'µmol/L', 'CHE': 'kU/L',
    'CHOL': 'mmol/L', 'CREA': 'µmol/L', 'GGT': 'U/L', 'PROT': 'g/L'
}

# [개선 1 & 3] 1~3차시 데이터 전처리 및 모델 성능 평가 요약 보기 (실제 계산값 기반 상세화)
with st.expander("📌 1~3차시 데이터 전처리 및 모델 성능 평가 요약 보기"):
    st.subheader("모델 최종 성능 평가 지표 (Test 데이터 기준)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("정확도 (Accuracy)", f"{accuracy_score(y_test, y_pred)*100:.2f}% ({accuracy_score(y_test, y_pred):.4f})")
    c2.metric("정밀도 (Precision)", f"{precision_score(y_test, y_pred)*100:.2f}% ({precision_score(y_test, y_pred):.4f})")
    c3.metric("재현율 (Recall)", f"{recall_score(y_test, y_pred)*100:.2f}% ({recall_score(y_test, y_pred):.4f})")
    c4.metric("F1-점수 (F1-score)", f"{f1_score(y_test, y_pred)*100:.2f}% ({f1_score(y_test, y_pred):.4f})")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ 데이터 정제·인코딩·분리 결과",
        "2️⃣ 결측치 대치 및 표준화 결과",
        "3️⃣ 상관관계 분석 및 박스플롯(Boxplot)",
        "4️⃣ 혼동 행렬 및 회귀 계수 분석"
    ])

    with tab1:
        st.markdown("#### 1. 데이터 개수 및 정제 결과 (실제 데이터 계산값)")
        raw_total = len(df)
        suspect_cnt = int((df['Category'] == '0s=suspect Blood Donor').sum())
        clean_total = len(df_clean)
        normal_cnt = int((y == 0).sum())
        disease_cnt = int((y == 1).sum())
        male_cnt = int((df_clean['Sex'] == 0).sum())
        female_cnt = int((df_clean['Sex'] == 1).sum())

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("**데이터 정제 및 Target 분포**")
            summary_target_df = pd.DataFrame({
                '구분': [
                    '원본 전체 데이터',
                    '제거된 의심 헌혈자(0s=suspect)',
                    '최종 분석 데이터',
                    '정상 헌혈자군 (Target = 0)',
                    '간 질환군 (Target = 1)'
                ],
                '데이터 수(건)': [raw_total, suspect_cnt, clean_total, normal_cnt, disease_cnt],
                '비율(%)': [
                    100.0,
                    round(suspect_cnt / raw_total * 100, 2),
                    round(clean_total / raw_total * 100, 2),
                    round(normal_cnt / clean_total * 100, 2),
                    round(disease_cnt / clean_total * 100, 2)
                ]
            })
            st.dataframe(summary_target_df, use_container_width=True, hide_index=True)

        with col_s2:
            st.markdown("**성별(Sex) 수치 인코딩 결과**")
            sex_df = pd.DataFrame({
                '성별 범주': ['남성 (m → 0)', '여성 (f → 1)', '합계'],
                '데이터 수(건)': [male_cnt, female_cnt, clean_total],
                '비율(%)': [
                    round(male_cnt / clean_total * 100, 2),
                    round(female_cnt / clean_total * 100, 2),
                    100.0
                ]
            })
            st.dataframe(sex_df, use_container_width=True, hide_index=True)

            st.markdown("**원본 Category 세부 분포**")
            cat_counts = df['Category'].value_counts().reset_index()
            cat_counts.columns = ['Category', '건수']
            st.dataframe(cat_counts, use_container_width=True, hide_index=True)

        with col_s3:
            st.markdown("**학습(Train) / 테스트(Test) 8:2 계층 분리 결과**")
            split_df = pd.DataFrame({
                '데이터셋 구분': ['학습 데이터 (Train, 80%)', '테스트 데이터 (Test, 20%)', '전체 합계'],
                '전체 건수': [len(X_train), len(X_test), clean_total],
                '정상군 (Target=0)': [int((y_train == 0).sum()), int((y_test == 0).sum()), normal_cnt],
                '질환군 (Target=1)': [int((y_train == 1).sum()), int((y_test == 1).sum()), disease_cnt],
                '질환군 비율(%)': [
                    round((y_train == 1).mean() * 100, 2),
                    round((y_test == 1).mean() * 100, 2),
                    round(disease_cnt / clean_total * 100, 2)
                ]
            })
            st.dataframe(split_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### 2. 결측치 탐색 및 학습 데이터(Train) 중앙값 대치 결과")
        total_missing_cells = int(X.isna().sum().sum())
        total_missing_rows = int(X.isna().any(axis=1).sum())
        after_missing_cells = int(X_train_imp.isna().sum().sum() + X_test_imp.isna().sum().sum())

        st.write(
            f"- 정제된 데이터(608건) 내 전체 결측값 개수: **{total_missing_cells}개** (결측치 포함 행: **{total_missing_rows}건**) → "
            f"데이터 누출 방지를 위해 **학습 데이터(Train)의 중앙값(Median)**으로 대치 완료 (대치 후 잔여 결측치: **{after_missing_cells}개**)"
        )

        na_summary = pd.DataFrame({
            '변수명': features,
            '측정 단위': [feature_units[f] for f in features],
            '전체 결측치 수': [int(X[f].isna().sum()) for f in features],
            'Train 결측치 수': [int(X_train[f].isna().sum()) for f in features],
            'Test 결측치 수': [int(X_test[f].isna().sum()) for f in features],
            '대치 기준값 (Train 중앙값)': [round(impute_Medians[f], 3) for f in features],
            '대치 후 결측치 수': [int(X_train_imp[f].isna().sum() + X_test_imp[f].isna().sum()) for f in features]
        })
        st.dataframe(na_summary, use_container_width=True, hide_index=True)

        st.markdown("#### 3. 데이터 표준화 (StandardScaler 변환 후 학습 데이터 상위 5행)")
        scaled_preview_df = pd.DataFrame(X_train_scaled[:5], columns=features).round(3)
        st.dataframe(scaled_preview_df, use_container_width=True)

    with tab3:
        st.markdown("#### 1. 변수별 Target(간 질환 여부)과의 피어슨 상관계수 분석")
        train_corr_df = X_train_imp.copy()
        train_corr_df['Target'] = y_train.values
        corr_matrix = train_corr_df.corr()
        target_corr = corr_matrix['Target'].drop('Target').sort_values(ascending=False)

        col_c1, col_c2 = st.columns([1, 1.4])
        with col_c1:
            corr_table = pd.DataFrame({
                '변수명': target_corr.index,
                'Target 상관계수(r)': target_corr.values.round(4),
                '상관 방향': ['양(+)의 상관관계' if v >= 0 else '음(-)의 상관관계' for v in target_corr.values]
            })
            st.dataframe(corr_table, use_container_width=True, hide_index=True)

        with col_c2:
            fig_corr, ax_corr = plt.subplots(figsize=(6.5, 4.2))
            corr_colors = ['#d62728' if v >= 0 else '#1f77b4' for v in target_corr.values]
            ax_corr.barh(target_corr.index[::-1], target_corr.values[::-1], color=corr_colors[::-1], alpha=0.85)
            ax_corr.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax_corr.set_title("Correlation with Target (Train Data)")
            ax_corr.set_xlabel("Pearson Correlation Coefficient (r)")
            ax_corr.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_corr)

        st.divider()
        st.markdown("#### 2. 정상군(0) vs 간 질환군(1) 주요 혈액 지표 분포 박스플롯 (Boxplot)")
        box_vars = ['AST', 'GGT', 'BIL', 'ALB', 'CHE', 'CHOL']
        fig_box, axes_box = plt.subplots(2, 3, figsize=(13, 6.5))
        axes_box = axes_box.flatten()

        for idx, var in enumerate(box_vars):
            data_0 = X_train_imp[y_train == 0][var]
            data_1 = X_train_imp[y_train == 1][var]
            bp = axes_box[idx].boxplot(
                [data_0, data_1],
                labels=['Normal (0)', 'Disease (1)'],
                patch_artist=True,
                widths=0.45
            )
            bp['boxes'][0].set_facecolor('#a6cee3')
            bp['boxes'][1].set_facecolor('#fb9a99')
            axes_box[idx].set_title(f"{var} ({feature_units[var]}) Distribution")
            axes_box[idx].set_ylabel(f"{var} ({feature_units[var]})")
            axes_box[idx].grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig_box)

    with tab4:
        st.markdown("#### 테스트 데이터(122건) 혼동 행렬 및 로지스틱 회귀 계수 분석")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
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
            ax_cm.set_title('Confusion Matrix (Test Set, n=122)')
            plt.tight_layout()
            st.pyplot(fig_cm)
            st.write(f"- **정상 맞힘(TN)**: {cm[0,0]}건 / **정상 오분류(FP)**: {cm[0,1]}건")
            st.write(f"- **질환 놓침(FN)**: {cm[1,0]}건 / **질환 맞힘(TP)**: {cm[1,1]}건")

        with col_m2:
            coef_summary_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_[0].round(4)
            }).sort_values(by='Coefficient', ascending=False)
            st.dataframe(coef_summary_df, use_container_width=True, hide_index=True)

st.divider()

# 1. 실시간 간 질환 분류 예측 시뮬레이터 (기존과 100% 동일)
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

# 2. 예측 결과 및 주요 수치 비교 시각화 ([개선 2] 6개 변수 개별 그래프 및 단위 분리)
st.header("2. AI 모델 예측 결과 및 주요 지표 비교 분석")

res_col1, res_col2 = st.columns([1, 1.4])

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
    st.subheader("입력 수치 vs 정상군·질환군 주요 지표 평균 비교 (단위별 개별 비교)")
    st.caption("각 혈액 지표는 측정 단위와 수치 범위가 다르므로 변수별 개별 그래프로 비교합니다.")

    compare_vars = ['AST', 'GGT', 'BIL', 'ALB', 'CHE', 'CHOL']
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    axes = axes.flatten()

    bar_labels = ['Normal(0)', 'Input', 'Disease(1)']
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, var in enumerate(compare_vars):
        unit_str = feature_units[var]
        vals = [
            float(normal_means[var]),
            float(input_data[var].iloc[0]),
            float(disease_means[var])
        ]
        bars = axes[idx].bar(bar_labels, vals, color=bar_colors, alpha=0.88, width=0.55)
        axes[idx].set_title(f"{var} ({unit_str})", fontsize=11, weight='bold')
        axes[idx].set_ylabel(f"Value ({unit_str})", fontsize=9)
        axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
        axes[idx].tick_params(axis='x', labelsize=8.5)

        max_v = max(vals) if max(vals) > 0 else 1.0
        axes[idx].set_ylim(0, max_v * 1.22)

        for bar, v in zip(bars, vals):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                v + max_v * 0.03,
                f"{v:.1f}",
                ha='center',
                va='bottom',
                fontsize=8.5,
                weight='bold'
            )

    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# 3. 로지스틱 회귀 예측 근거 분석 (변수별 개별 기여도 - 기존과 100% 동일 유지)
st.header("3. 로지스틱 회귀 모델의 예측 근거 분석 (변수별 개별 기여도)")
st.info(
    "💡 **해석 및 주의 안내**: 아래 기여도는 예측 확률(%)의 직접적인 증감률이 아니라, **간 질환군(1)의 로그 오즈(Log-odds)에 대한 선형 기여도(표준화된 입력값 × 회귀계수)**입니다. "
    "양수(+)는 질환군(1) 예측 방향으로, 음수(-)는 정상군(0) 예측 방향으로 작용함을 뜻하며 절댓값이 클수록 예측 점수에 미친 영향이 큽니다. "
    "또한 이 분석 결과는 **본 머신러닝 모델의 수학적 예측 과정에 대한 설명**일 뿐, 실제 의학적 진단이나 질병 발생의 직접적인 원인을 의미하지 않습니다."
)

scaled_vals = input_scaled[0]
coefs = model.coef_[0]
intercept = model.intercept_[0]
contributions = scaled_vals * coefs

contrib_sum = np.sum(contributions)
logit_val = contrib_sum + intercept
sigmoid_prob = 1 / (1 + np.exp(-logit_val))

contrib_df = pd.DataFrame({
    '변수명': features,
    '입력 수치': input_data.iloc[0].values,
    '표준화 값': scaled_vals.round(4),
    '회귀 계수': coefs.round(4),
    '개별 기여도(Log-odds)': contributions.round(4),
    '절댓값': np.abs(contributions)
})

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
