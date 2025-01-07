# Diabetes-Risk-Predictor 👩‍⚕️💊

This project aims to predict the likelihood of diabetes in individuals using machine learning techniques. It involves Exploratory Data Analysis (EDA), data preprocessing, modeling, and deploying an interactive web application using **Streamlit**.

## Project Structure

The repository contains the following components:

### 1. **Notebooks**
- **Kaggle Notebook**: A complete notebook with data analysis, visualization, and model training.
- Includes libraries such as `Matplotlib`, `Seaborn`, and `Plotly` for visualization.

### 2. **Streamlit Application**
- **`app.py`**: The main script for the interactive web application.
- Provides a user-friendly interface for predicting diabetes based on input features.

### 3. **Supporting Files**
- **Data**: The dataset used for training and evaluation.
- **Preprocessing scripts**: Scripts for cleaning and preparing the data.

## Setup and Usage

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Running on Kaggle
1. Install necessary packages:
   ```python
   !pip install streamlit pyngrok cloudflared
   ```

2. Run the application with tunneling:
   ```python
   !streamlit run app.py & npx cloudflared tunnel --url http://localhost:8501
   ```

3. Access the provided public URL to interact with the application.

## Features
- **EDA and Visualization**: Gain insights into the dataset with charts and graphs.
- **Data Preprocessing**: Handle missing values, scaling, and feature selection.
- **Machine Learning Models**: Train and evaluate models for diabetes prediction.
- **Interactive Interface**: A user-friendly app built with Streamlit for predictions.

## Technologies Used
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn, Plotly**
- **Scikit-learn**
- **Streamlit**

## Future Improvements
- Add more advanced machine learning models.
- Improve UI/UX of the Streamlit app.
- Integrate additional features like model explainability.

## License
This project is licensed under the MIT License.

---

### Author
Zainab Naveed

Feel free to contribute by submitting issues or pull requests!
