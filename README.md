# **Interactive Data Analyzer & Visualizer**

## **Project Overview**

This Streamlit application provides an intuitive and interactive platform for basic data analysis, visualization, and cleaning. Users can upload their datasets (CSV, JSON, Excel), perform exploratory data analysis, apply various data cleaning operations, and generate a range of dynamic plots to understand their data better.

This tool is designed to empower users with quick insights into their datasets without requiring deep programming knowledge, making data exploration accessible.

Live Demo: Your Streamlit App URL Here  
(Please replace this placeholder with the actual URL after deploying your app to Streamlit Community Cloud.)  
**Key Features:**

* **Flexible Data Upload:** Supports uploading datasets in CSV, JSON, and Excel (XLS, XLSX) formats.  
* **Exploratory Data Analysis (EDA):**  
  * View head and tail of the dataset.  
  * Inspect column names, unique values, and data types.  
  * Generate descriptive statistics.  
* **Data Cleaning Operations:**  
  * Remove duplicate rows.  
  * Handle missing (NULL) values by removing rows or filling with mean, median, mode, standard deviation, or a custom value.  
  * (Placeholder for "Remove Garbage Values" functionality, which appears to handle non-numeric NaNs/empty strings by dropping rows or custom fill)  
* **Interactive Data Visualization:** Generate various plots using Plotly and Seaborn:  
  * Scatter Plot  
  * Bar Graph  
  * Line Graph  
  * Histogram  
  * Pie Chart (for columns with few unique values)  
  * Correlation Heatmap  
  * Pairplot (for numerical columns)  
* **Data Filtering:** Filter data based on unique values in selected columns for focused analysis.

## **Project Structure**

data-analyzer-app/  
├── .gitignore             \# Specifies intentionally untracked files and directories to ignore by Git.  
├── LICENSE                \# Contains the licensing information for the project.  
├── README.md              \# This file; provides a comprehensive overview of the project.  
├── requirements.txt       \# Lists all required Python packages for the project.  
└── app.py                 \# The main Streamlit application script.

## **Installation (Local Setup)**

To run this application on your local machine, follow these steps:

1. **Clone the repository:**  
   git clone https://github.com/pokhriyal-anmol/data-analyzer-app.git  
   cd data-analyzer-app

   *(Remember to replace pokhriyal-anmol with your GitHub username and data-analyzer-app with your actual repository name.)*  
2. Create and activate a Python virtual environment:  
   It's highly recommended to use a virtual environment to manage project dependencies.  
   python \-m venv venv

   * **On Linux/macOS:**  
     source venv/bin/activate

   * **On Windows (Command Prompt):**  
     venv\\Scripts\\activate.bat

   * **On Windows (PowerShell):**  
     .\\venv\\Scripts\\Activate.ps1

3. Install Python dependencies:  
   With your virtual environment activated, install all required libraries:  
   pip install \-r requirements.txt

## **Usage (Local)**

Once the installation is complete and your virtual environment is active, you can run the Streamlit application:

streamlit run app.py

This command will open the application in your default web browser (usually at http://localhost:8501).

## **Usage (Web)**

To use the live deployed version of the app, simply visit the live demo link provided at the top of this README.md.

### **How to Interact with the App:**

1. **Upload Your Dataset:** On the "Home" or "Visualization" or "Processing" tab, use the file uploader to load your CSV, JSON, or Excel file.  
2. **Explore Data:** Navigate to the "Processing" tab and select "Exploratory Data Analysis" to view basic statistics, head, tail, unique values, and data types.  
3. **Clean Data:** On the "Processing" tab, select "Data Cleaning" to access options for removing duplicates, handling NULL values, or addressing "garbage values".  
4. **Visualize Data:** Switch to the "Visualization" tab. Use the sidebar to filter your data and select from various plot types (Scatter, Bar, Line, Pie, Histogram, Correlation Heatmap, Pairplot) to generate interactive charts.

## **Technical Stack**

* **Streamlit:** For building the interactive web application.  
* **Pandas:** For data manipulation and analysis.  
* **Plotly Express & Plotly Graph Objects:** For creating rich, interactive data visualizations.  
* **Seaborn & Matplotlib:** For statistical data visualization (used in Pairplot).  
* **streamlit-option-menu:** For the navigation menu.  
* **openpyxl:** A dependency for Pandas to handle Excel files.

## **Future Enhancements**

* Implement more advanced data preprocessing techniques (e.g., feature scaling, encoding categorical variables).  
* Add more sophisticated outlier detection methods.  
* Integrate machine learning model training and prediction capabilities.  
* Allow users to download the cleaned or processed dataset.  
* Improve error handling and user feedback for various operations.  
* Enhance UI/UX for a more seamless user experience.

## **Contributing**

Contributions are highly welcome\! If you have suggestions for improvements, find a bug, or want to add new features, please:

1. **Open an Issue:** Before starting work, please open an issue to discuss the bug or feature you'd like to address.  
2. **Fork the Repository:** Create your own fork of this project on GitHub.  
3. **Create a New Branch:**  
   git checkout \-b feature/your-feature-name  
   \# or: git checkout \-b bugfix/your-bug-fix-description

4. **Make Your Changes:** Implement your changes, ensuring your code adheres to existing style and conventions.  
5. **Write Clear Commit Messages:** Use descriptive commit messages.  
6. **Push Your Branch:** Push your new branch to your forked repository.  
   git push origin feature/your-feature-name

7. **Open a Pull Request:** Submit a pull request from your branch to the main (or master) branch of this repository. Provide a clear description.

## **License**

This project is licensed under the **MIT License**. See the LICENSE file for full details.

## **Copyright (c) 2024 Anmol Pokhriyal**

## **Acknowledgments**

* **Streamlit Community:** For the fantastic framework that made this app possible.  
* **Pandas, Plotly, Seaborn, Matplotlib:** For powerful data handling and visualization libraries.  
* **streamlit-option-menu:** For the customizable navigation.  
* The open-source community for their invaluable contributions.