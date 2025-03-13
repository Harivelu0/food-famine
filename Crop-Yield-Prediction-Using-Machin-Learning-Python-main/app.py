# Import necessary libraries
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Create Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv('yield_df.csv')
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Add additional features based on the paper
# Note: This assumes your dataframe has these columns or that you'll modify it accordingly
if 'average_rain_fall_mm_per_year' not in df.columns:
    # Example placeholder for rainfall data
    df['average_rain_fall_mm_per_year'] = np.random.normal(1000, 200, size=len(df))

if 'population' not in df.columns:
    # Example placeholder for population data
    df['population'] = np.random.normal(1000000, 500000, size=len(df))

if 'season' not in df.columns:
    # Example placeholder for seasons (kharif, rabi, summer, saith)
    seasons = ['kharif', 'rabi', 'summer', 'saith']
    df['season'] = np.random.choice(seasons, size=len(df))

# Clean the data to handle string values in numerical columns
def is_str(obj):
    try:
        float(obj)
        return False
    except:
        return True

# Clean rainfall data if it contains string values
if df['average_rain_fall_mm_per_year'].apply(lambda x: is_str(x)).any():
    to_drop = df[df['average_rain_fall_mm_per_year'].apply(is_str)].index
    df = df.drop(to_drop)
    df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)

# Define features and target
X = df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'season', 'population']]
y = df['hg/ha_yield']  # Crop yield

# Define preprocessor for numerical and categorical features
numerical_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'population']
categorical_features = ['Area', 'Item', 'season']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit the preprocessor to the data
X_processed = preprocessor.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train an Artificial Neural Network model (as described in the paper)
model = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Save the model and preprocessor
joblib.dump(model, 'ann_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Calculate performance metrics
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mad = np.mean(np.abs(y_test - y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = model.score(X_test, y_test)

print(f"Model Performance Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAD: {mad:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R^2: {r2:.2f}")

# Define a function to find the least produced item in a given area
def least_produced_item(area):
    area_df = df[df['Area'] == area]
    item_yields = area_df.groupby('Item')['hg/ha_yield'].mean()
    least_produced_item = item_yields.idxmin()
    return least_produced_item

# Define a function to find the top areas with the least shortage of a particular item
def top_10_areas(item):
    item_df = df[df['Item'] == item]
    area_yields = item_df.groupby('Area')['hg/ha_yield'].mean().sort_values().head(10)
    return area_yields.to_dict()

# Predict famine year and factor (implement as per the paper)
def predict_famine_year(area):
    # Filter data for the area
    area_data = df[df['Area'] == area]
    
    if area_data.empty:
        return "No data available for this area", "Unknown"
    
    # Get the latest 5 years of data for the area
    latest_years = sorted(area_data['Year'].unique())[-5:]
    
    # Calculate average yield for the area by year
    yearly_yields = area_data[area_data['Year'].isin(latest_years)].groupby('Year')['hg/ha_yield'].mean()
    
    # Determine if there's a declining trend
    if len(yearly_yields) >= 3:
        # Check if there's a consistent declining trend
        is_declining = all(yearly_yields.iloc[i] > yearly_yields.iloc[i+1] for i in range(len(yearly_yields)-1))
        
        if is_declining:
            # Predict the next year after the latest as potential famine year
            predicted_year = int(max(latest_years)) + 1
            
            # Determine major factor
            # For simplicity, check which factor has the strongest correlation with declining yields
            factors = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'population']
            correlations = {}
            
            for factor in factors:
                if factor in area_data.columns:
                    correlation = area_data[factor].corr(area_data['hg/ha_yield'])
                    correlations[factor] = abs(correlation)
            
            if correlations:
                major_factor = max(correlations, key=correlations.get)
                
                # Make the factor name more readable
                factor_names = {
                    'average_rain_fall_mm_per_year': 'Rainfall Changes',
                    'pesticides_tonnes': 'Pesticide Usage',
                    'avg_temp': 'Temperature Changes',
                    'population': 'Population Growth'
                }
                
                return predicted_year, factor_names.get(major_factor, major_factor)
            else:
                return predicted_year, "Multiple Factors"
        else:
            # No consistent decline, use longest future projection
            return int(max(latest_years)) + 5, "Projected Future Risk"
    else:
        # Not enough data for trend analysis
        return int(max(latest_years)) + 3, "Insufficient Historical Data"

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_least_produced_item', methods=['POST'])
def find_least_produced_item():
    if request.method == 'POST':
        area = request.form['Area']
        try:
            least_item = least_produced_item(area)
            return render_template('index.html', least_item=least_item)
        except:
            return render_template('index.html', least_item="No data available for this area")

@app.route('/top_10_areas', methods=['POST'])
def top_10_areas_route():
    if request.method == 'POST':
        item = request.form['Item']
        try:
            top_areas = top_10_areas(item)
            return render_template('index.html', item=item, top_areas=top_areas)
        except:
            return render_template('index.html', item=item, top_areas={})

@app.route('/predict_famine_year', methods=['POST'])
def predict_famine_year_route():
    if request.method == 'POST':
        area = request.form['Area']
        try:
            predicted_year, major_factor = predict_famine_year(area)
            return render_template('index.html', predicted_year=predicted_year, area=area, major_factor=major_factor)
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', predicted_year="Error in prediction", area=area, major_factor="Unknown")

@app.template_filter('enumerate')
def jinja2_enumerate(iterable, start=1):
    return enumerate(iterable, start)

if __name__ == "__main__":
    app.run(debug=True)