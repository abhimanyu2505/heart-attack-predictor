import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import os

# Load dataset
df = pd.read_csv('data/heart_attack_prediction_dataset_cleaned.csv')
print("Original shape:", df.shape)

# Drop irrelevant columns
df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere'], errors='ignore')

# Define expected features
features = ['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
            'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
            'Sedentary Hours Per Day', 'bmi', 'Triglycerides']

df = df[features + ['Heart Attack Risk']]

# Map values (handle lowercase or unexpected values)
df['Sex'] = df['Sex'].str.strip().str.lower().map({'male': 1, 'female': 0})
binary_map = {'yes': 1, 'no': 0}

for col in ['Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption']:
    df[col] = df[col].astype(str).str.strip().str.lower().map(binary_map)

# Drop rows with any NA values
print("Before dropping NAs:", df.shape)

df = df.dropna()
print("✅ After cleaning, dataset shape:", df.shape)

# Split features/target
X = df[features]
y = df['Heart Attack Risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to models/model.pkl")

# Risk calculation function based on predefined thresholds
def calculate_risk_score(age, sex, cholesterol, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption, exercise, sedentary, bmi, triglycerides):
    score = 0

    # Age risk
    if age >= 80:
        score += 25
    elif age >= 75:
        score += 20
    elif age >= 70:
        score += 18
    elif age >= 65:
        score += 16
    elif age >= 60:
        score += 14
    elif age >= 55:
        score += 12
    elif age >= 50:
        score += 10
    elif age >= 45:
        score += 8
    elif age >= 40:
        score += 6
    elif age >= 35:
        score += 4
    else:
        score += 0

    # Cholesterol risk (mg/dL)
    if cholesterol >= 280:
        score += 8
    elif cholesterol >= 240:
        score += 6
    elif cholesterol >= 200:
        score += 5
    elif cholesterol >= 160:
        score += 3
    else:
        score += 0

    # Smoking risk
    if smoking == 1:
        score += 5

    # Diabetes, Family History, Obesity, Alcohol Consumption as binary values (yes = 1, no = 0)
    if diabetes == 1:
        score += 5
    if family_history == 1:
        score += 5
    if obesity == 1:
        score += 5
    if alcohol_consumption == 1:
        score += 3

    # Physical Activity (Inverse risk: less activity = higher risk)
    if exercise < 3:
        score += 3

    # BMI (Body Mass Index)
    if bmi >= 30:
        score += 5  # Obese category

    # Risk Category based on score
    if score < 10:
        risk = "Low Risk"
    elif score < 20:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return risk, score

# Example input: You would replace this with actual user inputs
example_input = {
    'age': 45,
    'sex': 1,  # Male
    'cholesterol': 210,
    'heart_rate': 80,
    'diabetes': 1,
    'family_history': 1,
    'smoking': 1,
    'obesity': 0,
    'alcohol_consumption': 0,
    'exercise': 2,  # Less than 3 hours per week
    'sedentary': 8,
    'bmi': 32,
    'triglycerides': 180
}

risk_category, score = calculate_risk_score(
    example_input['age'],
    example_input['sex'],
    example_input['cholesterol'],
    example_input['heart_rate'],
    example_input['diabetes'],
    example_input['family_history'],
    example_input['smoking'],
    example_input['obesity'],
    example_input['alcohol_consumption'],
    example_input['exercise'],
    example_input['sedentary'],
    example_input['bmi'],
    example_input['triglycerides']
)

print(f"Risk Category: {risk_category}, Risk Score: {score}")
