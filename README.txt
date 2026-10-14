MANUFACTURING EQUIPMENT OUTPUT PREDICTION
================================================

Project Type:
Supervised Learning - Regression

Application:
Streamlit web application for predicting hourly manufacturing output
(Parts Per Hour) from machine operating parameters.

------------------------------------------------
SYSTEM REQUIREMENTS
------------------------------------------------

1. Python 3.10.x is recommended.
2. Internet connection is required during the first installation
   so Python packages can be downloaded.
3. The complete project folder should be kept together.

------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------

Manufacturing_Output_Predictor/
|
|-- app.py
|-- requirements.txt
|-- README.txt
|-- run_project.bat
|
|-- data/
|   |-- manufacturing_dataset_1000_samples.csv
|
|-- models/
|   |-- manufacturing_output_prediction_model.pkl
|   |-- model_metrics.csv
|   |-- feature_config.json
|
|-- assets/
    |-- factory_background.jpg
    |-- icons/
        |-- building-factory-2.svg
        |-- home.svg
        |-- tool.svg
        |-- chart-dots-2.svg
        |-- database-search.svg
        |-- info-circle.svg
        |-- chevron-left.svg
        |-- settings-2.svg

------------------------------------------------
INSTALLATION
------------------------------------------------

STEP 1 - Install Python

Install Python 3.10.x if it is not already installed.

During Python installation on Windows, enable:
"Add Python to PATH"

Verify the installation:

    python --version

It should show Python 3.10.x.

------------------------------------------------
STEP 2 - Open the Project Folder
------------------------------------------------

Open Command Prompt or PowerShell and navigate to the
project folder.

Example:

    cd "D:\Manufacturing_Output_Predictor"

------------------------------------------------
STEP 3 - Install Required Libraries
------------------------------------------------

Run:

    python -m pip install -r requirements.txt

This installs all Python libraries required by the application.

------------------------------------------------
STEP 4 - Run the Application
------------------------------------------------

Run:

    python -m streamlit run app.py

The Streamlit application will start and normally open
automatically in a web browser.

If it does not open automatically, copy the Local URL shown
in the terminal and open it in a browser.

Example:

    http://localhost:8501

------------------------------------------------
EASIER METHOD - RUN_PROJECT.BAT
------------------------------------------------

Instead of manually installing the requirements and starting
the application, double-click:

    run_project.bat

The batch file will:
1. Install the required packages from requirements.txt.
2. Start the Streamlit application.

------------------------------------------------
IMPORTANT
------------------------------------------------

Do not move or rename the following folders/files after
installation unless the corresponding paths in app.py are
also updated:

- data/
- models/
- assets/
- app.py

The trained machine learning model is stored in:

    models/manufacturing_output_prediction_model.pkl

The application loads this saved model at runtime.

------------------------------------------------
APPLICATION FEATURES
------------------------------------------------

- Manufacturing output prediction
- Interactive prediction interface
- Model performance metrics
- Actual vs Predicted visualization
- Feature influence visualization
- Dataset exploration
- Parts Per Hour distribution
- Numerical feature exploration
- Responsive Streamlit interface

------------------------------------------------
MODEL PERFORMANCE
------------------------------------------------

Final refined Linear Regression model:

R² Score : 0.9042
MAE      : 2.8119
RMSE     : 3.5354
MSE      : 12.4990

------------------------------------------------
TROUBLESHOOTING
------------------------------------------------

If "python" is not recognized:

Try:

    py --version

Then install the requirements using:

    py -m pip install -r requirements.txt

And run:

    py -m streamlit run app.py

If Streamlit is not recognized, use:

    python -m streamlit run app.py

If a previous Streamlit application is already running,
stop it with Ctrl+C in its terminal before starting another
instance.

================================================
