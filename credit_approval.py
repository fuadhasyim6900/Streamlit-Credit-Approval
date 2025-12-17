import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from streamlit_lottie import st_lottie_spinner

# ---------------------------
# Load Dataset
# ---------------------------
train_original = pd.read_csv(
    "https://raw.githubusercontent.com/fuadhasyim6900/Portofolio-Fuad/main/Final%20Project%20Data%20Science/dataset/train.csv"
)
test_original = pd.read_csv(
    "https://raw.githubusercontent.com/fuadhasyim6900/Portofolio-Fuad/main/Final%20Project%20Data%20Science/dataset/test.csv"
)
full_data = pd.concat([train_original, test_original], axis=0).sample(frac=1).reset_index(drop=True)

# Split function
def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_original, test_original = data_split(full_data, 0.2)
train_copy = train_original.copy()
test_copy = test_original.copy()

# ---------------------------
# Helper Functions
# ---------------------------
def value_cnt_norm_cal(df, feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ["Count", "Frequency (%)"]
    return ftr_value_cnt_concat

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading_an = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json")

# ---------------------------
# Transformer Classes
# ---------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=["Family member count", "Income", "Employment length"]):
        self.feat_with_outliers = feat_with_outliers
    def fit(self, df): return self
    def transform(self, df):
        if set(self.feat_with_outliers).issubset(df.columns):
            Q1 = df[self.feat_with_outliers].quantile(0.25)
            Q3 = df[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[self.feat_with_outliers] < (Q1 - 3*IQR)) | (df[self.feat_with_outliers] > (Q3 + 3*IQR))).any(axis=1)]
            return df
        return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=["Has a mobile phone","Children count","Job title","Account age"]):
        self.feature_to_drop = feature_to_drop
    def fit(self, df): return self
    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            df = df.drop(self.feature_to_drop, axis=1)
        return df

class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days=["Employment length","Age"]):
        self.feat_with_days = feat_with_days
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        if set(self.feat_with_days).issubset(X.columns):
            X[self.feat_with_days] = np.abs(X[self.feat_with_days])
        return X

class RetireeHandler(BaseEstimator, TransformerMixin):
    def fit(self, df): return self
    def transform(self, df):
        if "Employment length" in df.columns:
            df.loc[df["Employment length"]==365243, "Employment length"]=0
        return df

class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=["Income","Age"]):
        self.feat_with_skewness = feat_with_skewness
    def fit(self, df): return self
    def transform(self, df):
        if set(self.feat_with_skewness).issubset(df.columns):
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
        return df

class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=["Has a work phone","Has a phone","Has an email"]):
        self.feat_with_num_enc = feat_with_num_enc
    def fit(self, df): return self
    def transform(self, df):
        if set(self.feat_with_num_enc).issubset(df.columns):
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1:"Y",0:"N"})
        return df

class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=["Gender","Marital status","Dwelling","Employment status","Has a car","Has a property","Has a work phone","Has a phone","Has an email"]):
        self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self, df): return self
    def transform(self, df):
        if set(self.one_hot_enc_ft).issubset(df.columns):
            ohe = OneHotEncoder()
            ohe.fit(df[self.one_hot_enc_ft])
            feat_names = ohe.get_feature_names_out(self.one_hot_enc_ft)
            one_hot_df = pd.DataFrame(ohe.transform(df[self.one_hot_enc_ft]).toarray(), columns=feat_names, index=df.index)
            rest_features = [col for col in df.columns if col not in self.one_hot_enc_ft]
            df = pd.concat([one_hot_df, df[rest_features]], axis=1)
        return df

class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=["Education level"]):
        self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self, df): return self
    def transform(self, df):
        if set(self.ordinal_enc_ft).issubset(df.columns):
            enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = enc.fit_transform(df[self.ordinal_enc_ft])
        return df

class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=["Age","Income","Employment length"]):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self, df): return self
    def transform(self, df):
        if set(self.min_max_scaler_ft).issubset(df.columns):
            scaler = MinMaxScaler()
            df[self.min_max_scaler_ft] = scaler.fit_transform(df[self.min_max_scaler_ft])
        return df

class ChangeToNumTarget(BaseEstimator, TransformerMixin):
    def fit(self, df): return self
    def transform(self, df):
        if "Is high risk" in df.columns:
            df["Is high risk"] = pd.to_numeric(df["Is high risk"])
        return df

class OversampleSMOTE(BaseEstimator, TransformerMixin):
    def fit(self, df): return self
    def transform(self, df):
        if "Is high risk" in df.columns:
            smote = SMOTE()
            X_bal, y_bal = smote.fit_resample(df.iloc[:,:-1], df.iloc[:,-1])
            df = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
        return df

# ---------------------------
# Full Pipeline
# ---------------------------
def full_pipeline(df):
    pipeline = Pipeline([
        ("outlier_remover", OutlierRemover()),
        ("feature_dropper", DropFeatures()),
        ("time_conversion_handler", TimeConversionHandler()),
        ("retiree_handler", RetireeHandler()),
        ("skewness_handler", SkewnessHandler()),
        ("binning_num_to_yn", BinningNumToYN()),
        ("one_hot_with_feat_names", OneHotWithFeatNames()),
        ("ordinal_feat_names", OrdinalFeatNames()),
        ("min_max_with_feat_names", MinMaxWithFeatNames()),
        ("change_to_num_target", ChangeToNumTarget()),
        ("oversample_smote", OversampleSMOTE())
    ])
    return pipeline.fit_transform(df)

# ---------------------------
# Streamlit Inputs
# ---------------------------
st.title("Credit Card Approval Prediction")
st.write("Predict whether an applicant will be approved for a credit card.")

# Gender
input_gender = st.radio("Select your gender", ["Male","Female"], index=0)
# Age
input_age = np.negative(st.slider("Select your age", 18, 70, 42) * 365.25)
# Marital status
marital_status_values = list(value_cnt_norm_cal(full_data, "Marital status").index)
marital_status_key = ["Married","Single/not married","Civil marriage","Separated","Widowed"]
marital_status_dict = dict(zip(marital_status_key, marital_status_values))
input_marital_status_val = marital_status_dict[st.selectbox("Select marital status", marital_status_key)]
# Family member
fam_member_count = float(st.selectbox("Family member count", [1,2,3,4,5,6]))
# Dwelling type
dwelling_type_values = list(value_cnt_norm_cal(full_data, "Dwelling").index)
dwelling_type_key = ["House / apartment","Live with parents","Municipal apartment ","Rented apartment","Office apartment","Co-op apartment"]
dwelling_type_dict = dict(zip(dwelling_type_key, dwelling_type_values))
input_dwelling_type_val = dwelling_type_dict[st.selectbox("Dwelling type", dwelling_type_key)]
# Income
input_income = int(st.text_input("Enter your income (USD)", 0))
# Employment status
employment_status_values = list(value_cnt_norm_cal(full_data, "Employment status").index)
employment_status_key = ["Working","Commercial associate","Pensioner","State servant","Student"]
employment_status_dict = dict(zip(employment_status_key, employment_status_values))
input_employment_status_val = employment_status_dict[st.selectbox("Employment status", employment_status_key)]
# Employment length
input_employment_length = np.negative(st.slider("Employment length (years)", 0,30,6) * 365.25)
# Education level
edu_level_values = list(value_cnt_norm_cal(full_data, "Education level").index)
edu_level_key = ["Secondary school","Higher education","Incomplete higher","Lower secondary","Academic degree"]
edu_level_dict = dict(zip(edu_level_key, edu_level_values))
input_edu_level_val = edu_level_dict[st.selectbox("Education level", edu_level_key)]
# Car ownership
input_car_ownship = st.radio("Own a car?", ["Yes","No"], index=0)
# Property ownership
input_prop_ownship = st.radio("Own a property?", ["Yes","No"], index=0)
# Work phone
work_phone_val = {"Yes":1,"No":0}[st.radio("Work phone?", ["Yes","No"], index=0)]
# Phone
phone_val = {"Yes":1,"No":0}[st.radio("Phone?", ["Yes","No"], index=0)]
# Email
email_val = {"Yes":1,"No":0}[st.radio("Email?", ["Yes","No"], index=0)]

# ---------------------------
# Create profile DataFrame
# ---------------------------
profile_to_predict = [
    0, # ID
    input_gender[:1],
    input_car_ownship[:1],
    input_prop_ownship[:1],
    0, # Children count (dropped)
    input_income,
    input_employment_status_val,
    input_edu_level_val,
    input_marital_status_val,
    input_dwelling_type_val,
    input_age,
    input_employment_length,
    1, # Has mobile phone (dropped)
    work_phone_val,
    phone_val,
    email_val,
    "to_be_droped", # Job title (dropped)
    fam_member_count,
    0.0, # Account age (dropped)
    0 # placeholder target
]

profile_to_predict_df = pd.DataFrame([profile_to_predict], columns=train_copy.columns)
train_copy_with_profile_to_pred = pd.concat([train_copy, profile_to_predict_df], ignore_index=True)
train_copy_with_profile_to_pred_prep = full_pipeline(train_copy_with_profile_to_pred)
profile_to_pred_prep = train_copy_with_profile_to_pred_prep.loc[train_copy_with_profile_to_pred_prep["ID"]==0].drop(columns=["ID","Is high risk"])

# ---------------------------
# Load model lokal
# ---------------------------
import os
import requests

MODEL_URL = "https://raw.githubusercontent.com/fuadhasyim6900/Streamlit-Credit-Approval/main/file/gradient_boosting_tuned_new.pkl"
MODEL_PATH = "gradient_boosting_tuned_new.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return joblib.load(MODEL_PATH)

def make_prediction():
    try:
        model = load_model()
        return model.predict(profile_to_pred_prep)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
    
# ---------------------------
# Predict button
# ---------------------------
if st.button("Predict"):
    with st_lottie_spinner(lottie_loading_an, quality="high", height=200, width=200):
        final_pred = make_prediction()
    if final_pred is not None:
        if final_pred[0] == 0:
            st.success("## You have been approved for a credit card 🎉")
            st.balloons()
        else:
            st.error("## Unfortunately, you have not been approved for a credit card ❌")



