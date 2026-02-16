# IT Resource Optimizer – Forecasting and Scaling Recommendation System

## Live Application

Deployed App:
https://it-resource-optimizer-tvlkpmes8f2y5qtwxl4hph.streamlit.app/

---

## Project Overview

IT Resource Optimizer is a machine learning–based web application that analyzes historical IT resource usage data and generates forecasts to support infrastructure optimization decisions. The system is designed to help organizations plan capacity efficiently, reduce operational costs, and make informed scaling decisions based on predicted demand patterns.

The application uses time-series forecasting techniques to model resource utilization trends and provides automated recommendations on whether infrastructure resources should be scaled up, scaled down, or maintained.

---

## Key Features

* Upload custom IT resource usage datasets
* Time-series forecasting using the Prophet model
* Interactive visualizations of historical and predicted usage
* Automated scaling recommendations
* Estimated cost analysis based on predicted utilization
* Support for multiple resource metrics

---

## Technology Stack

Frontend:

* Streamlit

Backend and Machine Learning:

* Python
* Prophet (time-series forecasting)
* Pandas and NumPy
* Scikit-learn

Visualization:

* Plotly

---

## Dataset Requirements

The application requires a CSV file with a specific structure. The dataset must follow the same format as the file provided in the repository named:

enhanced_resource_usage_2023.csv

For Ease of testing users can download this dataset from this repository and test the application on the Deployed App

The expected columns include:

* timestamp (date-time values)
* CPU usage
* Memory usage
* Disk usage
* Network usage

Users should ensure that the dataset structure matches this format for the application to function correctly.

---

## How to Use the Application

1. Open the deployed application using the provided link.
2. Upload a CSV file that follows the required dataset format.
3. Select the resource metric to analyze.
4. The system will automatically generate forecasts and display interactive visualizations.
5. Review the scaling recommendations and cost estimates provided by the system.

---

## Running the Project Locally

### Clone the Repository

```bash
git clone https://github.com/MruganKulkarni/IT-Resource-Optimizer.git
cd IT-Resource-Optimizer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

---

## Use Cases

* Infrastructure capacity planning
* Cloud resource optimization
* IT operations cost management
* Resource demand forecasting
* Data-driven scaling strategy development

---

## Future Enhancements

* Real-time monitoring integration
* Multi-resource joint forecasting
* Automated alert system for scaling decisions
* Integration with cloud infrastructure APIs
* Advanced anomaly detection capabilities

---

## Author

Mrugan Kulkarni
GitHub: https://github.com/MruganKulkarni

---

This project is intended for academic and learning purposes.
