## Buying Power Project
Overview:
The Buying Power Project analyzes historical economic data to forecast the future purchasing power of individuals in the United States.
It leverages Python for data preprocessing, visualization, and machine learning, while C++ is used for performance-critical computations and data transformation.
This project also examines how changing economic conditions will affect younger generations, whose future financial stability may be influenced by inflation trends,
wage growth, and shifting market forces. The goal is to provide short-term and long-term forecasts that can help individuals, businesses, and policymakers make informed financial decisions.
## Why This Matters
Economic shifts do not affect all generations equally, but younger generations often face unique challenges:
- Wage Stagnation: compared to rising living costs.
- Student debt: That limits long-term financial growth.
- Housing market barriers: Making homeownership more difficult.
- Inflation-driven erosion: of savings and retirement planning.
  
By forecasting buying power trends, this project aims to highlight potential future risks
and provide data-driven insights to help the younger population prepare for the economic realities
ahead.

Features:
## Data Preprocessing (Python & C++):

Cleans raw CSV datasets (inflation rates, wages, commodity prices).

Normalizes and structures data for machine learning models.

## Machine Learning Forecasting (Python):

Uses regression-based ML models to predict future buying power.

Supports custom forecast horizons (e.g., 6 months, 1 year).

## C++ Performance Module:

Optimized calculations for large datasets.

Fast CSV parsing and transformation.

## Data Visualization (Python/Matplotlib):

Historical vs. predicted trends plotted.

Outputs charts and saves them as PNGs.

Tech Stack
## Languages:

Python 3.13

C++17

## Libraries & Tools:

Python: Pandas, NumPy, Matplotlib, Scikit-learn

C++: Standard Template Library (STL), file I/O for CSV handling

Git for version control

PyCharm & Visual Studio Code for development

Setup Instructions
## 1. Clone the repository
bash
Copy
Edit
git clone https://github.com/<YOUR_USERNAME>/Buying_Power_Project.git
cd Buying_Power_Project

## 2. Python environment setup
bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## 3.Build the C++ module
bash
Copy
Edit
cd cpp_module
g++ -std=c++17 -O2 main.cpp -o data_transformer
./data_transformer

## 4. Run the Python ML pipeline
bash
Copy
Edit
python train.py

## Usage
You can configure the forecast horizon by editing the FUTURE_MONTHS variable inside train.py.
Results will be saved as:

future_forecast.csv — Predicted values in table format.

future_forecast.png — Chart showing historical and forecasted values.

Example Output

## Future Improvements
Add support for neural networks (LSTM) for time series prediction.

Automate dataset updates from public APIs.

Integrate a web dashboard for interactive visualization.

## Reference/Source
- Federal Reserve Bank of St.Louis(FRED)
https://fred.stlouisfed.org/series/CUUR0000SA0R











