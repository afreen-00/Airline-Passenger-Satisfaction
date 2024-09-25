# Airline Passenger Satisfaction

This project is centered on analyzing airline passenger satisfaction data to identify factors influencing customer satisfaction and predicting whether a passenger was satisfied with their flight experience. By leveraging machine learning models, this project provides valuable insights into passenger preferences and helps airlines enhance customer satisfaction.

## Project Objective
The primary goal of this project was to predict passenger satisfaction based on various factors such as service quality, seat comfort, in-flight entertainment, and more. The dataset was analyzed and preprocessed to ensure accurate predictions and actionable insights. Several machine learning models were developed and tested to determine the most accurate predictor of passenger satisfaction.

## Data Preprocessing and Exploration
- **Data Cleaning**: Missing values were handled using appropriate imputation methods, and categorical features were transformed using OneHotEncoder.
- **Exploratory Data Analysis (EDA)**: EDA was conducted to discover trends and relationships within the dataset. Features such as flight distance, class of travel, and in-flight entertainment had a significant impact on customer satisfaction.

## Machine Learning Models
The following machine learning algorithms were applied to build predictive models:

1. **Random Forest Classifier**
   - A robust ensemble method that creates multiple decision trees and merges them to get a more accurate and stable prediction.
   - **Accuracy**: 92%
   - **Metrics**: Precision, Recall, and F1-score were also evaluated, showing balanced results with the best precision for satisfied passengers.

2. **K-Nearest Neighbors (KNN)**
   - A simple, instance-based learning algorithm that classifies a data point based on how its neighbors are classified.
   - **Accuracy**: 58%
   - KNN performed decently but was computationally expensive and showed lower accuracy compared to other models due to the high dimensionality of the data.

3. **Decision Tree Classifier**
   - A simple but powerful model for classification problems. It splits the dataset into subsets based on the most significant feature at each node.
   - **Accuracy**: 88%
   - While Decision Trees provided interpretability and fast execution, they were prone to overfitting.

4. **Logistic Regression**
   - A linear model used for binary classification, predicting the probability of a passenger being satisfied or not.
   - **Accuracy**: 66%
   - Logistic Regression offered strong baseline performance but lacked the flexibility to model complex, non-linear relationships in the data.

## Model Evaluation
Each model was evaluated using cross-validation and hyperparameter tuning to ensure optimal performance. The primary evaluation metric was accuracy, but precision, recall, and F1-score were also computed to give a more comprehensive assessment of the model's performance.

- **Best Model**: Random Forest Classifier
   - The Random Forest model outperformed other algorithms, achieving the highest accuracy of 92%. It provided the most balanced performance in terms of precision and recall, making it the best choice for predicting passenger satisfaction.

## Model Tracking
- MLFlow was used throughout the project to track models, experiments, hyperparameters, and metrics. This helped streamline the workflow, allowing for easy comparison and selection of the best-performing model.


## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: pandas, scikit-learn, MLFlow
- **Machine Learning Algorithms**: Random Forest, K-Nearest Neighbors, Decision Tree, Logistic Regression
- **Model Metrics**: Accuracy, Precision, Recall, F1-score
- **Data Analysis Techniques**: Exploratory Data Analysis, Feature Engineering, Hyperparameter Tuning, Model Evaluation

## Conclusion
This project successfully built a machine learning model that predicts airline passenger satisfaction with an accuracy of 92%. The Random Forest model demonstrated the best performance, providing valuable insights into factors affecting passenger satisfaction. Airlines can utilize this model to improve their services based on data-driven predictions, enhancing the overall customer experience.
