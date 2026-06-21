# FitNation Customer Retention & Revenue Prediction (Machine Learning) 

## Project Overview
This project was developed for **FitNation**, a major French fitness chain with over 400,000 members across 50 cities. Following the post-COVID recovery, the company faced increased customer volatility and wanted to shift towards a data-driven strategy.

The objective of this project was to explore customer data and build two predictive models to guide commercial decisions: estimate the monthly revenue generated per customer, and anticipate the risk of cancellation (churn) to deploy targeted retention actions.

## The Data Pipeline
1. **Data Cleaning & Engineering (Python):**
   - Processed a dataset of 10,000 customer records from 2018 to 2023.
   - Handled missing values using statistical imputation (e.g., median for satisfaction scores, and group means based on premium subscriptions for age).
   - Detected and removed structural outliers using the IQR method (e.g., recorded ages of 150, or 100 sessions per month) to preserve model accuracy.

## Machine Learning Models
1. **Revenue Estimation (Linear Regression):**
   - Built a regression model to predict the `revenu_mensuel` (monthly revenue).
   - The model achieved a strong R-squared ($R^2$) score of 0.76, meaning it successfully explains 76% of the revenue variance.
2. **Churn Prediction (Logistic Regression):**
   - Developed a classification model to predict whether a customer will cancel their membership (`churn`).
   - Addressed a highly imbalanced dataset (93.5% retained vs. 6.4% churned) by implementing balanced class weights (`class_weight='balanced'`) during model training.
   - Applied the finalized model to new 2024 data to extract probabilities and identify the top customers at risk.

## Key Business Insights
- **Revenue Drivers:** The regression analysis revealed that `abonnement_premium` (Premium Subscription) and `coaching` are the most influential variables. Holding other factors constant, a premium subscription increases monthly revenue by an average of 25 units, while coaching adds 19.4 units.
- **Customer Engagement:** The number of monthly sessions also positively impacts revenue, showing that active customers contribute more to the bottom line. 
- **Non-Influential Factors:** Interestingly, customer age and loyalty/tenure (`anciennete`) showed almost zero correlation with the monthly revenue generated.
- **Strategic Recommendations:** The findings provide a clear roadmap for the marketing department to bundle coaching with premium offers and deploy preventive targeting for users identified by the churn probability algorithm.

## Technical Highlights
- **Languages & Libraries:** Python (Pandas, NumPy), Scikit-Learn (LinearRegression, LogisticRegression, RandomForestClassifier).
- **Model Evaluation:** R-squared, Mean Squared Error (MSE), Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- **Data Visualization:** Seaborn and Matplotlib for correlation heatmaps and boxplots.

---

## Project Resources
- **Jupyter Notebook:** `ML_Fitnation.ipynb` (End-to-End Pipeline & Models)
- **Predictions File:** `pred_clients_2024.csv` (Final Churn Probabilities)
- **INPUT AND DATA** : `INPUT` (Data and Input_2024)
