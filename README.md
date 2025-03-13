# Food Famine Forecasting System

## About the Project

This project implements an advanced food famine forecasting system using Artificial Neural Networks (ANN) as described in the research paper "Food Famine Forecasting using Artificial Neural Network" . The system analyzes multiple factors including crop yield, rainfall patterns, pesticide usage, seasonal variations, and population data to predict potential famine scenarios in different regions of India.

## Key Features

- **Crop Vulnerability Analysis**: Identifies the most vulnerable crops in specific regions based on yield data
- **Regional Vulnerability Assessment**: Determines regions most susceptible to food shortages for particular crops
- **Famine Year Prediction**: Forecasts potential years when famine might occur and identifies major contributing factors
- **Multi-Factor Analysis**: Considers environmental, social, and economic dimensions for comprehensive prediction

## Technology Stack

- **Backend**: Python, Flask
- **Data Analysis**: Scikit-learn, Pandas, NumPy
- **Machine Learning**: Artificial Neural Networks, Random Forest (fallback model)
- **Frontend**: HTML, CSS, Bootstrap
- **Visualization**: Matplotlib, Seaborn (for model development)

## Model Performance

The Artificial Neural Network model implemented in this system demonstrates superior performance compared to traditional forecasting methods:

- **Accuracy**: 92.65%
- **RMSE**: 36.60
- **MAD**: 28.10
- **MAPE**: 21.51%
- **R² Value**: 0.86


## Dataset

The system uses a comprehensive dataset (famine.csv) that includes:

- Historical crop yield data from India (1997-2023)
- Environmental factors like rainfall and temperature
- Agricultural inputs such as pesticide usage
- Population data for different regions
- Seasonal information (kharif, rabi, summer, saith)

## Project Structure

```
food-famine-forecasting/
│
├── app.py                    # Flask application with routes and model implementation
├── famine.csv                # Main dataset
├── templates/                
│   └── index.html            # Frontend interface
├── static/                   
│   └── styles.css            # Additional styles (if needed)
├── models/                   
│   ├── model.pkl             # Saved trained model
│   └── preprocessor.pkl      # Saved data preprocessor
├── Food_Famine_Forecasting.pdf  # Research paper
└── README.md                 # Project documentation
```





![Screenshot 2025-03-13 145053](https://github.com/user-attachments/assets/6a613a20-7520-457b-9426-a09732c6069e)

![Screenshot 2025-03-13 144746](https://github.com/user-attachments/assets/2c2347c1-4ecf-4d2c-921d-564aea7edf21)


![Screenshot 2025-03-13 145126](https://github.com/user-attachments/assets/7fddbe51-05d5-425e-8cdc-121caa7978fb)

## Research Background
This implementation is based on the research paper "Food Famine Forecasting using Artificial Neural Network". The paper introduced a novel approach using a Multi-Dimensional Factor Index (MFI) for classifying factors that influence food famine. The ANN model was trained on data from 1997 to 2023 and demonstrates higher accuracy (92.65%) compared to traditional methods like FEWS NET (70%) and K-Nearest Neighbor (87%).


## Acknowledgements

- Department of Computer Science and Engineering, Sona College of Technology
- The research is based on comprehensive data analysis of agricultural patterns in India
