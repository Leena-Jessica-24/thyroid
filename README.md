Introduction
The Thyroid Disease Detection System is a machine learning application designed to predict the recurrence of thyroid disease based on various clinical and demographic features. Utilizing patient data, the model aims to assist healthcare professionals in diagnosing and managing thyroid conditions effectively.

Features
Predicts the recurrence of thyroid disease based on input features.
Utilizes Random Forest Classifier for prediction.
Encodes categorical variables for improved model performance.
Standardizes feature values for better accuracy.

Technologies Used
Programming Language: Python
Libraries:
pandas for data manipulation and analysis.
numpy for numerical operations.
scikit-learn for machine learning and data preprocessing.
matplotlib and seaborn for data visualization.

Dataset
The dataset used in this project includes various attributes related to thyroid disease, including:
Age
Gender
Smoking history
History of radiotherapy
Thyroid function results
Physical examination findings
Adenopathy
Pathology results
Focality
Risk assessment
Recurrence status (target variable)
Ensure you have the dataset in a CSV format (e.g., thyroid_data.csv) placed in the same directory as the project script.

Installation
Clone the repository:
git clone https://github.com/Leena-Jessica-24/thyroid-disease-detection.git
cd thyroid-disease-detection

Install the required packages:
pip install pandas numpy scikit-learn matplotlib seaborn

Load the dataset by modifying the path in the code:
df = pd.read_csv('your_data.csv')

Provide input values for prediction in the format:
input_values = [Age, Gender, Smoking, Hx Smoking, Hx Radiotherapy, Thyroid Function, Physical Examination, Adenopathy, Pathology, Focality, Risk]

Call the prediction function:
result = predict_thyroid_recurrence(input_values)
print(result)

Model Details
Model Used: Random Forest Classifier
Training: The model is trained on a scaled version of the dataset to improve accuracy and performance.
Evaluation Metrics: Classification report and confusion matrix are used to evaluate the model's performance.

Results
The model's performance metrics (accuracy, precision, recall, F1-score) will be displayed upon running the code.
Example output of predictions will indicate whether there is a risk of recurrence or not.

Future Work
Explore additional machine learning algorithms (e.g., Logistic Regression, SVM) to compare performance.
Enhance the model with more features or a larger dataset.
