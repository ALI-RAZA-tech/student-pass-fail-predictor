# 🎓 Student Pass/Fail Predictor

A beginner-friendly machine learning web app that predicts whether a student will **pass or fail** using **Logistic Regression**.

Built with Python, scikit-learn, pandas and Streamlit.

## Features

- Enter study hours, attendance, previous marks and sleep hours using sliders
- Get a Pass (1) / Fail (0) prediction
- See the model's probability of passing
- View the training data inside the app

## How it works

1. A small sample dataset of 20 students is loaded with pandas
2. Data is split into 80% training and 20% testing (`train_test_split`)
3. A `LogisticRegression` model is trained on the training data
4. The model is evaluated on the test data using accuracy
5. The Streamlit interface sends new student inputs to the model and shows the result

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Programming language |
| pandas | Data handling |
| scikit-learn | Logistic Regression model |
| Streamlit | Web app interface |

## Run it locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/student-pass-fail-predictor.git
cd student-pass-fail-predictor

# 2. Install the libraries
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Limitations

- The dataset is a small, hand-made sample (20 rows) for learning purposes
- The test set has only 4 students, so the accuracy number is not statistically reliable
- This is an educational project and must not be used for real decisions

## Future improvements

- Train on a real dataset (for example Kaggle "Student Performance")
- Add more features and proper data cleaning
- Compare with other models (Decision Tree, Random Forest)
- Deploy online with Streamlit Community Cloud

## Author

**Ali**, BS Artificial Intelligence student, University of Management and Technology (UMT), Lahore.
