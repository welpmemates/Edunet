import streamlit as st
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.title("Solar Panel Performance Analysis UI")

# Step 1: Generate CSV
if st.button("Generate CSV (Run gen.py)"):
    result = subprocess.run(["python", "gen.py"], capture_output=True, text=True)
    st.success("CSV generated!")
    st.text(result.stdout)
    # Set a session state variable to indicate that the CSV has been generated
    st.session_state.csv_generated = True

# Check if the CSV has been generated
if 'csv_generated' not in st.session_state or not st.session_state.csv_generated:
    st.warning("Please generate the CSV file first to see the analysis options.")
else:
    # Step 2: Choose Analysis
    option = st.selectbox(
        "Choose what you want to do:",
        ("Show Boxplot by Season", "Show Day-wise Bar Chart", "Linear Regression", "Logistic Regression")
    )

    if option == "Show Boxplot by Season":
        df = pd.read_pickle('df_all_seasons.pkl')
        fig, ax = plt.subplots()
        df.boxplot(column='kwh', by='season', grid=False, figsize=(8,6), ax=ax)
        plt.title('Solar Panel Energy Output (kWh) by Season')
        plt.suptitle('')
        plt.xlabel('Season')
        plt.ylabel('Energy Output (kWh)')
        st.pyplot(fig)

    elif option == "Show Day-wise Bar Chart":
        df = pd.read_pickle('df_all_seasons.pkl')
        fig, ax = plt.subplots(figsize=(14,6))
        ax.bar(df.index, df['kwh'], color='orange')
        plt.xlabel('Day Index')
        plt.ylabel('Energy Output (kWh)')
        plt.title('Day-wise Solar Panel Energy Output (Unaveraged)')
        st.pyplot(fig)

    elif option == "Linear Regression":
        result = subprocess.run(["python", "LinearRegression.py"], capture_output=True, text=True)
        st.text(result.stdout)
        # Load and plot y_test vs y_pred
        y_test = np.load('y_test.npy')
        y_pred = np.load('y_pred.npy')
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r--')
        plt.xlabel("Actual kWh")
        plt.ylabel("Predicted kWh")
        plt.title("Actual vs. Predicted kWh")
        st.pyplot(fig)

    elif option == "Logistic Regression":
        result = subprocess.run(["python", "LogisticRegression.py"], capture_output=True, text=True)
        st.text(result.stdout)
        # Load and plot confusion matrix
        y_test = np.load('y_test_2.npy')
        y_pred = np.load('y_pred_2.npy')
        class_labels = np.load('class_labels.npy', allow_pickle=True)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
