import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\Zainab\Desktop\diabetes_prediction_dataset.csv')
    return data

data = load_data()

st.markdown(
    """
    <style>
        .main { background-color: #f9f9f9; padding: 20px; font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
    """,
    unsafe_allow_html=True
)

st.title("🧑‍⚕️💊 Diabetes Risk Prediction")
st.write("Analyzing diabetes risks based on medical data and lifestyle habits")
st.sidebar.header("Explore Analysis")

options = [
    "What is Diabetes?",
    "Dataset Overview",
    "Age Distribution",
    "Gender Distribution",
    "Diabetes and Smoking",
    "BMI and Diabetes",
    "HbA1c Level and Diabetes",
    "Heart Disease and Diabetes",
    "Diabetes and Hypertension",
    "Blood Glucose Level Across Age Groups",
    "Model Training & Evaluation"
]
selected_option = st.sidebar.radio("Choose an analysis:", options)

if selected_option == "What is Diabetes?":
    st.subheader("Introduction to Diabetes 🩺🩸")

    st.write("""
    ### What is Diabetes?
    
    Diabetes is a chronic medical condition where the body either **does not produce enough insulin** or **cannot effectively use the insulin it produces**. Insulin is a hormone that helps regulate **blood glucose levels**, and when its production or effectiveness is impaired, it leads to elevated blood sugar. The two main types of diabetes are:
    
    - **Type 1 Diabetes**: An autoimmune condition where the body attacks the insulin-producing cells in the pancreas.
    - **Type 2 Diabetes**: A metabolic disorder where the body becomes resistant to insulin, leading to higher blood sugar levels.
    
    Diabetes can cause various **long-term health complications**, including heart disease, kidney failure, nerve damage, and vision impairment. It is a significant public health challenge, affecting millions of people globally. 

    ### Key Objectives of the Analysis 🧑‍⚕️
    
    - **Investigate the effect of lifestyle habits**: Lifestyle choices such as body weight and smoking habits can significantly influence diabetes risk. Through this analysis, we aim to identify patterns and correlations between these factors and diabetes.
    
    - **Assess the role of healthcare factors**: Healthcare-related factors like **BMI**, **blood glucose levels**, and **hypertension** are critical in diabetes risk assessment. We will analyze these parameters to better understand their contribution to the disease.
    
    - **Data Visualization**: Using visualizations, we will illustrate trends, patterns, and outliers within the dataset, helping to uncover relationships between diabetes and various risk factors.

    - **Build a Predictive Model**: We will also develop a **machine learning model** to predict the likelihood of an individual developing diabetes based on key health and lifestyle variables. This model will be evaluated using classification metrics to determine its effectiveness.

    ### Methodology 🧑‍💻

    The analysis will proceed in the following steps:

    1. **Exploratory Data Analysis (EDA)**: 
        - We begin by performing a comprehensive EDA to explore the data and identify potential trends, outliers, and missing values. 
        - Visualizations such as **histograms**, **boxplots**, and **scatter plots** will be used to highlight important patterns in the data.

    2. **Correlation Analysis**: 
        - We will calculate the **correlation** between different variables and diabetes. Correlation helps us understand the strength and direction of relationships between variables. For example, how BMI and blood glucose levels are correlated with diabetes.

    3. **Model Training & Evaluation**:
        - After preprocessing and feature engineering, we will build a **Random Forest Classifier** model to predict diabetes risk. The model will be trained using a training set and evaluated on a test set.
        - The evaluation will include **classification metrics** such as **accuracy**, **precision**, **recall**, and **F1-score** to measure the model's performance.

    ### Expected Outcomes 🎯
    
    By the end of this analysis, we aim to:
    
    - **Identify key risk factors** for diabetes and their interactions.
    - **Visualize the relationships** between various lifestyle habits, healthcare factors, and the likelihood of developing diabetes.
    - **Predict diabetes risk** using a machine learning model, with high accuracy and precision.
    
    This analysis will provide valuable insights for **healthcare professionals** and **policy makers** working to prevent and manage diabetes. Additionally, it can help individuals understand how their lifestyle choices might impact their health and diabetes risk.

    ### Why Is This Important? 🤔
    
    Diabetes is one of the leading causes of **chronic disease** and **death** globally. By identifying the key factors that contribute to the development of diabetes, we can:
    
    - Promote **early detection** and intervention to improve health outcomes.
    - Develop **prevention strategies**, including lifestyle changes, to reduce the incidence of diabetes.
    - Educate the public on the importance of maintaining a **healthy lifestyle** to prevent the onset of diabetes and related health conditions.
    
    This analysis, therefore, serves as an important tool not only for understanding diabetes but also for taking actionable steps to reduce its burden on society.

    Stay tuned as we explore the dataset and uncover the relationships between health and lifestyle factors with diabetes!
    """)

elif selected_option == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.write("Summary Statistics:")
    st.write(data.describe())

    # Unique values in categorical columns
    st.subheader("Unique Values by Column")
    for column in data.select_dtypes(include=['object']):
        st.write(f"{column}: {data[column].unique()}")

    # Missing value analysis
    st.subheader("Missing Values by Column")
    st.write(data.isnull().sum())

if selected_option == "Gender Distribution":
    st.subheader("Gender-wise distribution of the dataset 👩‍🦱👨‍🦱")
    gender_counts = data['gender'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    ax.set_title('Gender Distribution')
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig)  # Display the plot in Streamlit

# Age Distribution - Histplot
elif selected_option == "Age Distribution":
    st.subheader("Distribution of ages in the dataset 📊")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['age'], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)  # Display the plot in Streamlit

elif selected_option == "Diabetes and Smoking":
    st.subheader("Correlation of Diabetes with Smoking 🚬")

    st.write("""
    **Smoking** is a well-known risk factor for various health conditions, including cardiovascular diseases and respiratory issues. However, its relationship with **diabetes** is also of great interest. Smoking can increase inflammation, stress on the body, and worsen insulin resistance, potentially contributing to the development of Type 2 diabetes. 

    In this section, we will explore the correlation between **smoking history** and **diabetes status**. Specifically, we will examine how different smoking habits, such as being a current smoker, former smoker, or never smoker, relate to the likelihood of developing diabetes.
    """)
    
    # Plot the distribution of diabetes and smoking history using Plotly
    fig = px.histogram(data, x='smoking_history', color='diabetes', barmode='group', 
                        labels={'smoking_history': 'Smoking History', 'diabetes': 'Diabetes'}
                        )
    st.plotly_chart(fig)
    
    # Create a contingency table (Smoking History vs Diabetes) using Pandas
    contingency_table = pd.crosstab(data['smoking_history'], data['diabetes'])
    
    # Plotting the contingency table as a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=False, 
                xticklabels=["No Diabetes", "Diabetes"], yticklabels=["Never", "Former", "Current", "No Info"])
    plt.title("Heatmap of Smoking History vs Diabetes")
    st.pyplot(plt)
    
    # Perform the Chi-Square Test for correlation between diabetes and smoking history
    from scipy.stats import chi2_contingency
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    st.write(f"Chi-Square Statistic: {chi2}")
    st.write(f"P-Value: {p}")
    
    if p < 0.05:
        st.write("""
        There is a **significant correlation** between smoking history and diabetes (p < 0.05). This suggests that smoking, whether current or former, may contribute to the development or worsening of diabetes. Individuals with a history of smoking might be at a higher risk of developing diabetes, and this relationship warrants further investigation for targeted prevention strategies.
        """)
    else:
        st.write("""
        There is **no significant correlation** between smoking history and diabetes (p >= 0.05). This indicates that smoking history, on its own, may not be a strong predictor of diabetes. While smoking is known to affect overall health, other factors might play a more significant role in the development of diabetes in this dataset.
        """)


elif selected_option == "BMI and Diabetes":
    st.subheader("Correlation of BMI and Diabetes 🍏🥗🍩")

    # Explanation of BMI
    st.write("""
    **Body Mass Index (BMI)** is a value derived from an individual's height and weight. It provides an estimate of whether a person is underweight, normal weight, overweight, or obese. 
             
    A normal BMI range is considered to be between **18.5 and 24.9**, whereas:
    - **Underweight**: BMI less than 18.5
    - **Normal weight**: BMI from 18.5 to 24.9
    - **Overweight**: BMI from 25 to 29.9
    - **Obesity**: BMI greater than or equal to 30

    The **BMI value** is a quick screening tool to assess whether a person is at a higher risk for diseases such as heart disease, diabetes, and certain cancers.
    """)

    # Plotting BMI vs Diabetes using a Boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='diabetes', y='bmi', data=data, ax=ax, palette="Set2")
    ax.set_title("BMI Distribution by Diabetes Status")
    ax.set_xlabel("Diabetes (0 = No, 1 = Yes)")
    ax.set_ylabel("Body Mass Index (BMI)")
    st.pyplot(fig)

    # Perform T-Test to check if the difference in BMI between people with and without diabetes is significant
    from scipy.stats import ttest_ind

    # Separate data into two groups: with diabetes (1) and without diabetes (0)
    diabetes_group = data[data['diabetes'] == 1]['bmi']
    no_diabetes_group = data[data['diabetes'] == 0]['bmi']

    # Perform a t-test
    t_stat, p_val = ttest_ind(diabetes_group, no_diabetes_group, nan_policy='omit')

    # Show t-test result
    st.write(f"**T-Statistic**: {t_stat:.2f}")
    st.write(f"**P-Value**: {p_val:.4f}")

    # Interpretation of the results
    if p_val < 0.05:
        if diabetes_group.mean() > no_diabetes_group.mean():
            st.write("""
            Individuals with diabetes tend to have a **higher BMI**, suggesting that **higher BMI** is associated with a greater risk of developing diabetes.
            Higher BMI levels can contribute to insulin resistance, which plays a major role in the development of Type 2 diabetes.
            """)
        else:
            st.write("""
            Individuals with diabetes tend to have a **lower BMI**, suggesting no direct correlation between higher BMI and diabetes risk in this dataset.
            However, this may be influenced by other factors, and further analysis is needed to fully understand the relationship between BMI and diabetes.
            """)
    else:
        st.write("""
        There is **no statistically significant difference** in BMI between individuals with and without diabetes, indicating that BMI alone may not be a strong predictor of diabetes risk.
        However, BMI remains an important factor to consider alongside other health metrics and lifestyle factors for understanding diabetes risk.
        """)


elif selected_option == "HbA1c Level and Diabetes":
    st.subheader("Correlation of HbA1c Level and Diabetes 💉")

    # Explanation of HbA1c
    st.write("""
    **Hemoglobin A1c (HbA1c)** is a form of hemoglobin that is bound to glucose. It is an important indicator of long-term blood sugar control in individuals with diabetes. 
    The HbA1c test measures the average blood sugar levels over the past two to three months. It is typically used to diagnose and monitor diabetes.

    **The HbA1c levels are classified as follows:**
    - **Normal**: HbA1c less than 5.7%
    - **Pre-diabetes**: HbA1c from 5.7% to 6.4%
    - **Diabetes**: HbA1c 6.5% or higher

    Elevated HbA1c levels indicate poor blood sugar control and can increase the risk of diabetes-related complications.
    """)

    # Plotting HbA1c level vs Diabetes using a Violin Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='diabetes', y='HbA1c_level', data=data, ax=ax, palette="Set2")
    ax.set_title("HbA1c Level Distribution by Diabetes Status")
    ax.set_xlabel("Diabetes (0 = No, 1 = Yes)")
    ax.set_ylabel("HbA1c Level")
    st.pyplot(fig)

    # Conclusion based on the visual
    st.write("""
    From the **violin plot**, we can observe the following:
    - **Individuals with diabetes (1)** tend to have higher **HbA1c levels** compared to individuals without diabetes (0). 
    - The distribution for the diabetic group is more spread out, indicating variability in HbA1c levels among those with diabetes.
    - The non-diabetic group has a more concentrated distribution of lower HbA1c levels, indicating better long-term blood sugar control.

    This suggests that elevated HbA1c levels are commonly associated with diabetes, and monitoring HbA1c is crucial for managing and diagnosing diabetes effectively.
    """)


elif selected_option == "Heart Disease and Diabetes":
    st.subheader("Correlation of Heart Disease and Diabetes 🫀")

    # Brief Introduction
    st.write("""
    Diabetes and heart disease are interconnected health conditions with significant impacts on public health.  
    Studies show that individuals with diabetes are at a higher risk of developing heart disease, primarily due to elevated blood sugar levels, increased cholesterol, and hypertension.  
    This analysis investigates their relationship using **statistical measures** and **visual insights**.
    """)

    # Visual: Bar Plot for Heart Disease occurrence within Diabetes Categories
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=data, x='diabetes', hue='heart_disease', palette='coolwarm')
    plt.title('Heart Disease Distribution by Diabetes Status')
    plt.xlabel('Diabetes (0 = No, 1 = Yes)')
    plt.ylabel('Frequency')
    plt.legend(title='Heart Disease', labels=['No', 'Yes'])
    st.pyplot(fig)

    # Visual: Heatmap for Correlation between Diabetes and Heart Disease
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_matrix = data[['diabetes', 'heart_disease']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', center=0, cbar=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap: Diabetes vs Heart Disease")
    st.pyplot(fig)

    # Chi-Square Test for Independence
    from scipy.stats import chi2_contingency
    contingency_table = pd.crosstab(data['diabetes'], data['heart_disease'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Display Chi-Square Test Results
    st.write(f"**Chi-Square Test Results**")
    st.write(f"- Chi-Square Statistic: {chi2:.2f}")
    st.write(f"- Degrees of Freedom: {dof}")
    st.write(f"- P-Value: {p_value:.4f}")

    # Conclusion based on Chi-Square Test
    if p_value < 0.05:
        st.write("""
        **Conclusion**:  
        There is a **statistically significant relationship** between diabetes and heart disease (p < 0.05).  
        This indicates that having diabetes increases the likelihood of heart disease.  
        Preventative measures such as regular health monitoring and a healthy lifestyle are crucial.
        """)
    else:
        st.write("""
        **Conclusion**:  
        There is **no statistically significant relationship** between diabetes and heart disease (p ≥ 0.05).  
        While this dataset may not show a strong correlation, other risk factors (e.g., age, obesity, physical inactivity) should be explored in more detail.
        """)

elif selected_option == "Diabetes and Hypertension":
    st.subheader("Hypertension and Diabetes 🧬🩹")

    # Introduction
    st.write("""
    **Hypertension** (high blood pressure) is a common comorbidity in individuals with diabetes. Both hypertension and diabetes are major risk factors for cardiovascular diseases. 
    It is important to examine whether there is an association between hypertension and diabetes, as individuals with both conditions may be at an increased risk for other health complications.
    The following visualization and statistical test will explore the relationship between hypertension and diabetes in this dataset.
    """)

    # Using seaborn's catplot for better visual representation
    fig = sns.catplot(x='diabetes', hue='hypertension', data=data, kind='count', height=6, aspect=1.5)
    fig.set_axis_labels("Diabetes", "Count")
    fig.set_titles("Hypertension vs Diabetes")

    # Perform Chi-Square Test for association between Hypertension and Diabetes
    from scipy.stats import chi2_contingency
    contingency_table = pd.crosstab(data['diabetes'], data['hypertension'])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Show results of the Chi-Square test
    st.write(f"Chi-Square Statistic: {chi2}")
    st.write(f"P-Value: {p}")

    # Interpretation of the results
    if p < 0.05:
        st.write("There is a significant correlation between hypertension which implies that indivisuals with diabetes are more likely to have hypertension")
    else:
        st.write("There is no significant correlation between hypertension and diabetes.")
    st.pyplot(fig)

elif selected_option == "Blood Glucose Level Across Age Groups":
    st.subheader("Average Blood Glucose Level Across Age Groups 🩸")

    # Introduction
    st.write("""
    **Blood glucose levels** can vary significantly across different age groups due to factors like metabolism, lifestyle, and the presence of chronic conditions such as diabetes. 
    It is important to understand how blood glucose levels fluctuate as individuals age, and whether age groups with diabetes show different trends compared to those without. 
    The following box plot and statistical test will help us understand these differences more clearly.
    """)

    # Categorize age into different groups
    data['age_group'] = pd.cut(data['age'], bins=[0, 18, 30, 50, 70, 100], labels=['0-18', '19-30', '31-50', '51-70', '71+'])

    # Plotting using Plotly for better interactive visualization
    fig = px.box(data, x='age_group', y='blood_glucose_level', color='diabetes', 
                 title="Blood Glucose Level Across Age Groups")
    st.plotly_chart(fig)

    # Perform ANOVA to test if blood glucose levels differ significantly across age groups
    from scipy.stats import f_oneway

    # Separate data into age groups for ANOVA test
    age_groups = [data[data['age_group'] == group]['blood_glucose_level'] for group in data['age_group'].unique()]
    f_stat, p_val = f_oneway(*age_groups)

    # Show results of ANOVA test
    st.write(f"F-Statistic: {f_stat}")
    st.write(f"P-Value: {p_val}")

    # Interpretation of the results
    if p_val < 0.05:
        st.write("There is a significant difference in blood glucose levels across age groups (p < 0.05).")
    else:
        st.write("There is no significant difference in blood glucose levels across age groups (p >= 0.05).")

    # Conclusion
    st.write("""
    - If the p-value is below 0.05, this indicates that blood glucose levels significantly differ across the age groups, and this could suggest that age may be a contributing factor to variations in blood glucose levels.
    - If the p-value is above 0.05, this suggests that there is no significant difference in blood glucose levels across the age groups, implying that age alone may not strongly influence blood glucose levels.
    These findings can be useful in understanding the potential impact of age on blood glucose control, particularly in the context of diabetes management.
    """)

elif selected_option == "Model Training & Evaluation":
    st.subheader("Model Training & Evaluation 🏥")
    
    # Displaying the columns used for training
    st.write("Columns used for training the model:")
    columns_used = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 
                    'bmi', 'HbA1c_level', 'blood_glucose_level']
    st.write(columns_used)

    # Encoding categorical features
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])
    data['smoking_history'] = le.fit_transform(data['smoking_history'])
    
    # Features (X) and Target (y)
    X = data[columns_used]
    y = data['diabetes']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Training the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluating the model's performance
    st.write("### Model Evaluation")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    # Visualizing the confusion matrix using Seaborn heatmap
    st.write("### Confusion Matrix Heatmap:")
    fig, ax = plt.subplots(figsize=(8, 6))  # Explicit figure creation
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
    ax.set_title("Confusion Matrix Heatmap")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # Plotting the feature importance
    st.write("### Feature Importance (Random Forest)")
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))  # Explicit figure creation
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

    # Model performance visual (ROC curve)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    st.write("### ROC Curve")
    fig, ax = plt.subplots(figsize=(8, 6))  # Explicit figure creation
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    # Calculating accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy Score (in percentage):")
    st.write(f"{accuracy * 100:.2f}%")
