# IBM Applied Data Science Capstone: SpaceX Falcon 9 Landing Prediction

This repository contains the final capstone presentation and supporting code for the IBM Applied Data Science Capstone project. The project predicts whether a SpaceX Falcon 9 first stage will land successfully using data collection, data wrangling, exploratory data analysis, SQL, Folium maps, Plotly Dash visualization, and machine learning.

## Final Deliverables

- `Data Science Capstone Project Report.pdf` - final upload-ready PDF presentation
- `IBM_SpaceX_Capstone_Final_Report.pdf` - premium PDF copy
- `IBM_SpaceX_Capstone_Final_Report.pptx` - PowerPoint version
- `spacex_dash_app.py` - Plotly Dash dashboard application
- `generate_premium_capstone_report.py` - reproducible report generator

## Project Workflow

1. Collect launch data from SpaceX API endpoints.
2. Scrape historical Falcon 9 and Falcon Heavy launch tables from Wikipedia.
3. Clean and transform the launch records.
4. Perform EDA with visualization and SQL queries.
5. Build interactive visual analytics with Folium and Plotly Dash.
6. Train and compare Logistic Regression, SVM, Decision Tree, and KNN models.
7. Summarize findings in a professional presentation report.

## Key Results

- The Falcon 9 modeling dataset contains 90 launches.
- The overall landing success rate is approximately 66.7%.
- KSC LC 39A and VAFB SLC 4E show the strongest site-level landing success rates in the cleaned dataset.
- The best model in the report is selected using test accuracy and 10-fold cross-validation accuracy.

## Reproduce the Report

Install dependencies:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

Regenerate the final report:

```bash
python generate_premium_capstone_report.py
```

To replace the placeholder GitHub URL in every slide, set the environment variable before running the generator:

```bash
set CAPSTONE_GITHUB_URL=https://github.com/your-username/your-repository
python generate_premium_capstone_report.py
```

