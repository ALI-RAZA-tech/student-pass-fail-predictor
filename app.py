# ============================================================
# PROJECT: Student Pass/Fail Predictor (Streamlit Web App)
# Model  : Logistic Regression (scikit-learn)
# Kaam   : Student ki maloomat le kar batata hai ke wo Pass hoga ya Fail
# Chalane ka tareeqa (terminal mein):  streamlit run app.py
# ============================================================

# ---- Libraries import karna ----
import streamlit as st                                   # web app banane ke liye
import pandas as pd                                      # data ko table mein rakhne ke liye
from sklearn.model_selection import train_test_split     # data ko Train/Test mein baantne ke liye
from sklearn.linear_model import LogisticRegression      # hamara model
from sklearn.metrics import accuracy_score               # model ko napne ke liye

# ---- Page ki basic settings (browser tab ka title waghera) ----
st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓")


# ------------------------------------------------------------
# HISSA 1: Data tayyar karna
# ------------------------------------------------------------
def load_data():
    """20 students ka chhota sample data (sirf seekhne ke liye, asli nahi)."""
    data = {
        # Rozana parhai ke ghante
        "study_hours":    [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 6, 1, 4, 7, 8, 3, 2, 6, 5],
        # Hazri ka percentage (0 se 100)
        "attendance":     [50, 60, 65, 70, 80, 85, 90, 95, 55, 70, 75, 88, 45, 72, 92, 96, 60, 58, 82, 78],
        # Pichhle imtihan ke marks (100 mein se)
        "previous_marks": [35, 40, 50, 55, 60, 70, 75, 85, 38, 48, 58, 68, 30, 52, 80, 88, 45, 42, 65, 62],
        # Rozana neend ke ghante
        "sleep_hours":    [4, 5, 6, 6, 7, 7, 8, 8, 5, 6, 7, 7, 4, 6, 8, 7, 5, 5, 7, 6],
        # RESULT: 1 = Pass, 0 = Fail
        "result":         [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    }
    return pd.DataFrame(data)


# ------------------------------------------------------------
# HISSA 2: Model train karna
# @st.cache_resource ka matlab: model sirf EK baar train ho,
# har button dabane par dobara train na ho (app tez rehti hai).
# ------------------------------------------------------------
@st.cache_resource
def train_model():
    df = load_data()

    # X = input (result ke ilawa sab columns), y = output (result)
    X = df.drop(columns=["result"])
    y = df["result"]

    # 80% Train, 20% Test (random_state=42 se har baar same split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model banana aur train karna
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Test data par accuracy nikalna
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, X.columns, accuracy, df


# Model tayyar karwana
model, feature_columns, accuracy, df = train_model()


# ------------------------------------------------------------
# HISSA 3: App ki shakal (User Interface)
# ------------------------------------------------------------
st.title("🎓 Student Pass/Fail Predictor")
st.write(
    "Apni maloomat daalein aur **Logistic Regression** model batayega "
    "ke student **Pass** hoga ya **Fail**."
)

# Sliders: user se 4 inputs lena
# st.slider(label, min, max, shuru ki value)
study_hours = st.slider("Rozana parhai ke ghante", 0, 12, 4)
attendance = st.slider("Hazri (%)", 0, 100, 70)
previous_marks = st.slider("Pichhle imtihan ke marks (100 mein se)", 0, 100, 50)
sleep_hours = st.slider("Rozana neend ke ghante", 0, 12, 6)

# Button dabane par prediction hogi
if st.button("Predict karein"):

    # Naye student ki maloomat ko table mein badalna.
    # Columns ke naam wohi hone chahiye jo training mein the.
    new_student = pd.DataFrame(
        [[study_hours, attendance, previous_marks, sleep_hours]],
        columns=feature_columns,
    )

    # Model ki raaye: 1 (Pass) ya 0 (Fail)
    prediction = model.predict(new_student)[0]

    # Pass hone ki probability (percent mein)
    pass_chance = model.predict_proba(new_student)[0][1] * 100

    # Nateeja dikhana
    if prediction == 1:
        st.success(f"✅ PASS hone ka imkaan hai (Prediction = 1)")
    else:
        st.error(f"❌ FAIL hone ka khatra hai (Prediction = 0)")

    st.write(f"Pass hone ka chance: **{pass_chance:.1f}%**")
    st.progress(int(pass_chance))   # progress bar 0 se 100 tak


# ------------------------------------------------------------
# HISSA 4: Model ke baare mein maloomat
# ------------------------------------------------------------
st.divider()
st.caption(
    f"Test accuracy: {accuracy * 100:.0f}% (sirf 4 test students par). "
    "Ye ek seekhne wala project hai, asli faislon ke liye istemal na karein."
)

# Expander: click karne par data dikhta hai
with st.expander("Training data dekhein"):
    st.dataframe(df)
