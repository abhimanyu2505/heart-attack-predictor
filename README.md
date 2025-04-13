# Heart Attack Predictor

This is a data science project that predicts the likelihood of a heart attack using machine learning. It includes data preprocessing, modeling, and a Streamlit-based web app.

## Features
- Uses SMOTE for class balancing.
- Trained with Random Forest Classifier.
- Rule-based risk categorization fallback logic for better prediction reliability.
- Generates personalized PDF reports with charts for each user.

## Steps to Run

1. **Install dependencies**  
   pip install -r requirements.txt


2. **Run the training script**  
   python heart_attack_predictor.py


3. **Explore data and model insights**  
Open `EDA_and_Modeling.ipynb` to visualize feature importance, distributions, etc.

4. **Launch the web app**  
   python -m streamlit run app.py