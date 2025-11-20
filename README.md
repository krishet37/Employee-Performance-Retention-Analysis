# Employee Performance and Retention Analysis

## Overview
This project develops an **Employee Performance and Retention Analysis** system using a real-world dataset. The goal is to apply concepts from probability, statistics, machine learning, and deep learning to analyze employee data and predict performance and retention trends.

## Objective
Analyze employee performance metrics and predict attrition using advanced data science techniques including:
- Statistical Analysis
- Machine Learning (Random Forest, Linear Regression)
- Deep Learning (Neural Networks)
- Exploratory Data Analysis (EDA)

## Libraries Required
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **Scikit-learn** - Machine learning models
- **TensorFlow/Keras** - Deep learning

## Dataset Features
The employee dataset includes:
- Employee ID
- Name
- Age
- Department
- Salary
- Years at Company
- Performance Score
- Attrition (Yes/No)

## Project Phases

### Phase 1: Data Collection and Exploratory Data Analysis (EDA)
**Step 1 - Data Collection and Preprocessing**
- Load employee data from CSV
- Handle missing values and duplicates
- Clean inconsistent data entries

**Step 2 - Exploratory Data Analysis**
- Calculate descriptive statistics (mean, median, mode, variance, standard deviation)
- Create visualizations using Matplotlib and Seaborn:
  - Pairplot for feature relationships
  - Heatmap for correlation analysis
  - Boxplots for outlier identification

**Step 3 - Probability and Statistical Analysis**
- Calculate probability of employee attrition
- Apply Bayes' Theorem for conditional probabilities
- Perform ANOVA test to compare performance across departments

### Phase 2: Predictive Modeling
**Step 4 - Feature Engineering and Encoding**
- Label encoding for categorical variables
- Min-Max scaling for numerical features

**Step 5 - Employee Attrition Prediction Model**
- Random Forest Classifier
- Model evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization

**Step 6 - Employee Performance Prediction Model**
- Linear Regression for performance score prediction
- Evaluation metrics: R² Score, MSE, RMSE, MAE

### Phase 3: Deep Learning Models
**Step 7 - Deep Learning for Performance Prediction**
- Feedforward Neural Network architecture
- Layers: Dense (64, 32, 16) with ReLU activation
- Dropout layers for regularization
- Adam optimizer with MSE loss

**Step 8 - Employee Attrition Analysis with Deep Learning**
- Binary classification neural network
- Sigmoid activation for output layer
- Metrics: Accuracy, Precision, Recall

### Phase 4: Reporting and Insights
**Step 9 - Insights and Recommendations**
- Feature importance analysis
- High-risk departments identification
- Performance score distribution
- Actionable recommendations for HR

**Step 10 - Data Visualization and Reporting**
- Line plots for performance trends
- Bar charts for attrition by department
- Scatter plots for salary vs performance
- Model performance comparison

## Key Findings
- Identification of top factors contributing to employee performance
- High-risk departments for attrition
- Statistical insights using probability theory and hypothesis testing
- Predictive models with high accuracy for attrition and performance forecasting

## Model Performance
- **Random Forest (Attrition)**: High accuracy classification
- **Linear Regression (Performance)**: R² score indicating good fit
- **Deep Learning (Performance)**: Competitive R² with neural networks
- **Deep Learning (Attrition)**: Binary classification with optimized metrics

## How to Run
1. Clone this repository
2. Install required libraries: `pip install pandas numpy matplotlib seaborn scikit-learn tensorflow`
3. Upload your employee dataset (employee_data.csv) to Google Drive
4. Open the notebook in Google Colab
5. Mount Google Drive and adjust the file path
6. Run all cells sequentially

## Project Structure
```
Employee-Performance-Retention-Analysis/
├── Employee_Performance_Retention_Analysis.ipynb
├── README.md
└── employee_data.csv (not included - use your own dataset)
```

## Technologies Used
- Python 3.x
- Google Colab
- Jupyter Notebook
- Statistical Analysis
- Machine Learning
- Deep Learning

## Author
**krishet37**  
B.Tech CSE Student | Data Science Enthusiast

## License
This project is open source and available for educational purposes.

---
*Project completed as part of data science coursework focusing on employee analytics and predictive modeling.*
