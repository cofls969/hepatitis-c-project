import streamlit as st
import pandas as pd

# 페이지 기본 설정
st.set_page_config(page_title="C형 간염 데이터 분석 (1차시)", layout="wide")

st.title("🏥 C형 간염 관련 간 질환군 분류 프로젝트 (1차시)")
st.write("데이터 구조 확인, 결측치 탐색 및 이진 분류 타깃(Target) 생성 과정입니다.")

# 1. 데이터 구조 및 기초 정보 확인
st.header("1. 데이터 구조 및 기초 정보 확인")
df = pd.read_csv("HepatitisCdata.csv")

col1, col2 = st.columns(2)
with col1:
    st.metric("전체 행(Row) 개수", f"{df.shape[0]}개")
with col2:
    st.metric("전체 열(Column) 개수", f"{df.shape[1]}개")

st.subheader("데이터 상위 5행 미리보기")
st.dataframe(df.head())

st.subheader("컬럼별 자료형(Data Type)")
dtype_df = pd.DataFrame({'컬럼명': df.columns, '자료형': df.dtypes.astype(str).values})
st.dataframe(dtype_df, use_container_width=True)

st.divider()

# 2. 결측치 현황 분석
st.header("2. 결측치(Missing Value) 현황 분석")
missing_counts = df.isnull().sum()
missing_df = pd.DataFrame({'컬럼명': missing_counts.index, '결측치 개수': missing_counts.values})
missing_df = missing_df[missing_df['결측치 개수'] > 0]

st.write(f"- **전체 결측치 총합**: {df.isnull().sum().sum()}개")
st.write(f"- **결측치가 1개 이상 포함된 행의 개수**: {df.isnull().any(axis=1).sum()}개")
st.dataframe(missing_df, use_container_width=True)

st.divider()

# 3. 범주별 데이터 분포 탐색 및 시각화
st.header("3. Category 범주별 데이터 분포 탐색")
cat_counts = df['Category'].value_counts()

col3, col4 = st.columns(2)
with col3:
    st.subheader("범주별 데이터 개수")
    st.dataframe(cat_counts)
with col4:
    st.subheader("범주별 분포 막대그래프")
    st.bar_chart(cat_counts)

st.divider()

# 4. 불필요한 컬럼 제거 및 의심군 제외
st.header("4. 불필요한 컬럼 제거 및 의심 헌혈자 제외")
df_clean = df.drop(columns=['Unnamed: 0'])
df_clean = df_clean[df_clean['Category'] != '0s=suspect Blood Donor'].copy()

st.success(f"✅ 'Unnamed: 0' 컬럼 제거 및 '0s=suspect Blood Donor'(7건) 제외 완료! (남은 데이터: {len(df_clean)}개)")
st.dataframe(df_clean.head())

st.divider()

# 5. 이진 분류를 위한 Target 변수 생성
st.header("5. 이진 분류를 위한 Target 변수 생성")
df_clean['Target'] = df_clean['Category'].apply(lambda x: 0 if x == '0=Blood Donor' else 1)

target_counts = df_clean['Target'].value_counts().rename(index={0: '0 (정상 헌혈자)', 1: '1 (간 질환군)'})

col5, col6 = st.columns(2)
with col5:
    st.subheader("Target 변수 분포 (0: 정상, 1: 질환군)")
    st.dataframe(target_counts)
with col6:
    st.subheader("Target 분포 시각화")
    st.bar_chart(target_counts)
