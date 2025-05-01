# Credit Risk Prediction App

This project is a machine learning web application that predicts whether a customer is at risk of defaulting on a loan. It uses a Random Forest classifier trained on real financial data and includes methods for handling imbalanced data using SMOTE.

## Features

- Real-time credit risk prediction
- SMOTE-applied training set to address class imbalance
- Streamlit-based interactive interface
- Feature importance visualization
- Deployable on Streamlit Cloud

## Dataset

- Source: "Give Me Some Credit" - Kaggle Dataset
- Contains 150,000 rows of anonymized customer credit behavior
- Target variable: SeriousDlqin2yrs (1 = default within 2 years)

## Machine Learning

- Model: Random Forest Classifier (Scikit-learn)
- Data split: 80% train / 20% test
- Imbalance handled using SMOTE from imbalanced-learn
- Evaluation metrics: accuracy, precision, recall, f1-score

## Getting Started

To run the project locally:

1. Clone the repository:

```
git clone https://github.com/your-username/credit-risk-app.git
cd credit-risk-app
```

2. Install dependencies:

```
streamlit
pandas
scikit-learn
joblib
imbalanced-learn
```

3. Run the app:

```
streamlit run app.py
```

## Project Structure

```
credit-risk-app/
├── app.py                # Streamlit interface
├── train_model.py        # Model training script
├── credit_model.pkl      # Trained and saved model
└── README.md             # Project documentation
```

## Technologies Used

- Python 3
- Scikit-learn
- Imbalanced-learn
- Pandas
- Streamlit
- Matplotlib

## Author

Tristan Louis  