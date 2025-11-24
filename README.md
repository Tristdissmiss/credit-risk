# Credit Risk Prediction App

This repository contains a machine learning application for predicting the likelihood of a customer defaulting on a loan. The project includes data preprocessing, class imbalance handling, model training, evaluation, and an interactive Streamlit-based interface for real-time predictions.

---

### Overview  

This project uses financial features from the Kaggle “Give Me Some Credit” dataset to classify whether a borrower is likely to default within two years. It applies an end-to-end ML workflow including feature engineering, SMOTE oversampling, model evaluation, and deployment through a lightweight web interface. 

---
### Dataset 
**Source:**
Give Me Some Credit - Kaggle 

**Details:** 
- 150,000 anonymized customer records 
- Target variable
  - `SeriousDlqin2yrs` (1 = default within 2 years, 0 = no default)
- Features include:
  - Revolving utilization
  - Credit lines
  - Debt ratio
  - Delinquencies
  - Monthly income
  - Dependent count
    
---

### Modeling Approach 

**Approach Used:** 
`Random Forest Classifier (Scikit-learn)` 

**Training Setup** 
- Train/test split: 80% / 20%
- Class imbalance addressed using SMOTE from imbalanced-learn
- Standard pre-processing and handling of missing values
- Feature importance extracted for model interpretation

 **Evaluation Metrics:** 
 -Accuracy 
 -Precision
 -Recall
 -F1- score 
 -Confusion matrix

---
  
### Key Features

- Real-time credit-risk predictions via Streamlit
-SMOTE-enhanced training for balanced classification
- Feature importance visualization
- Saved model for reproducibility (credit_model.pkl)
- Easy local deployment

----

### Project Structure 
``` bash
credit-risk-app/
├── app.py                # Streamlit interface
├── train_model.py        # Model training script
├── credit_model.pkl      # Trained and saved model
└── README.md             # Project documentation
```

---

### Tech Stack 
Languages & Libraries 
- `Python 3`
- `pandas`
- `scikit-learn`
- `imbalanced-learn(SMOTE)`
- `Streamlit`
- `matplotlib(optional visualizations)`
- `joblib(model serializations)`

---
### Running the Project Locally 

**1. Clone the Repository**  
```bash
git clone <your_repo_url>
cd credit-risk
```
**2. Install Dependencies** 
Creates a virtual environment if desired, then installs requirements:
```bash
pip install -r requirements.txt
```
**3. Train the Model(Optional)** 
If you want to retrain the model 
```bash
python train_model.py
```
This will regenerate `credit_model.pkl` 

**4. Launch the StreamlitApp** 
```bash
streamlit run app.py
```
Streamlit will open the interface in your browser for testing credit-risk prediction scenarios

----

### Future Improvements 
- Add logistic regression or XGBoost comparison models
- Integrate SHAP for model explainability
- Add unit tests for feature handling
- Deploy a full backend API(FAST API) for REST-based predicitons
- Create a Docker container for easy deployment
  
