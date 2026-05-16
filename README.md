# Pima Indians Diabetes Statistical Analysis

## Project Overview
This project presents a complete statistical analysis of the Pima Indians Diabetes Dataset using Python. It demonstrates data preprocessing, feature engineering, functional programming, recursive computation, and statistical validation using both manual implementations and pandas.

The project was developed as part of the **CSE 205 – Hands On With Python** course.

---

## Objectives
- Load and validate the dataset with exception handling  
- Explore and analyze medical data  
- Handle missing values using manual median calculation  
- Create derived health-risk features  
- Identify high-risk diabetic patients  
- Implement functional programming concepts  
- Use recursion for risk aggregation  
- Compute manual statistics and compare with pandas  
- Generate an automated analysis report  

---

## Dataset Information
- **Dataset:** Pima Indians Diabetes Dataset  
- **Records:** 768 patients  
- **Features:** 9 medical attributes  
- **Target Variable:** Outcome (0 = Non-Diabetic, 1 = Diabetic)  

### Features Included
- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome  

---

## Technologies Used
- Python  
- Pandas  
- Functional Programming (`filter`, `map`, `reduce`)  
- Recursive Functions  

---

## Project Workflow

### 1. Data Loading
- CSV loading using pandas  
- Exception handling for:
  - FileNotFoundError  
  - PermissionError  
  - Other I/O errors  

### 2. Data Exploration
- Display first 10 rows and last 5 rows  
- Generate statistical summaries  
- Perform loop-based filtering using relational operators  

### 3. Missing Value Handling
Zero values in medical columns were treated as missing data and replaced using manually calculated medians.

### 4. Feature Engineering
Derived features created:
- BMI_Glucose_Index  
- Insulin_Glucose_Ratio  
- Age_Risk_Level  

### 5. High-Risk Analysis
Patients were classified as high-risk using the following condition:

```python
(Glucose > 140) or (BMI > 30) or (Age > 50)
```

### 6. Functional Programming
Implemented:
- `filter()` for insulin filtering  
- `map()` for ratio calculation  
- `reduce()` for aggregation  

### 7. Recursive Analysis
Used recursion to calculate total risk scores and validate iterative results.

### 8. Statistical Computation
Manual computation of:
- Mean  
- Median  
- Mode  
- Minimum  
- Maximum  

Results were validated against pandas statistics.

### 9. Report Generation
Generated an automated text report containing:
- Dataset overview  
- Statistical results  
- High-risk analysis  
- Key insights  

---

## Key Findings
- Glucose, BMI, and Age are strong indicators of diabetes risk  
- High-risk patients showed significantly higher diabetes prevalence  
- Manual statistical computations matched pandas outputs, validating implementation accuracy  

---

## Future Improvements
- Add visualizations and dashboards  
- Build machine learning classification models  
- Deploy using Streamlit or Flask  
- Apply advanced feature selection techniques  

---

## How to Run

```bash
python main.py
```

Ensure that the dataset file (`diabetes.csv`) is placed in the same directory as the script.

---

## Output
The project automatically generates:

```text
diabetes_analysis_report.txt
```

---

## Author
**Pattela Hemanth**  
CSE 205 – Hands On With Python  
SRM University AP
