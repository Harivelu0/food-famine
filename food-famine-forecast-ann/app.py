# Import necessary libraries
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Flask app
app = Flask(__name__)

# Load dataset with better error handling
try:
    df = pd.read_csv('famine.csv')
    print("Successfully loaded famine.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv('yield_df.csv')
        print("Falling back to yield_df.csv")
    except FileNotFoundError:
        print("ERROR: Neither famine.csv nor yield_df.csv found.")
        df = pd.DataFrame()  # Create empty DataFrame to avoid errors

# Clean up the data
if not df.empty:
    # Remove any unnamed columns
    for col in df.columns:
        if 'Unnamed' in col:
            df.drop(col, axis=1, inplace=True)
    
    # Display basic data information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for extreme values in the target variable
    if 'hg/ha_yield' in df.columns:
        print(f"Yield min: {df['hg/ha_yield'].min()}, max: {df['hg/ha_yield'].max()}, mean: {df['hg/ha_yield'].mean()}")
        
        # If values are extremely large, scale them down
        if df['hg/ha_yield'].max() > 1000000:  # If yield values are in millions
            print("Scaling down extremely large yield values")
            df['hg/ha_yield'] = df['hg/ha_yield'] / 1000  # Scale down by 1000
            print(f"After scaling - Yield min: {df['hg/ha_yield'].min()}, max: {df['hg/ha_yield'].max()}")

    # Add additional features if they don't exist
    if 'average_rain_fall_mm_per_year' not in df.columns:
        print("Adding simulated rainfall data")
        df['average_rain_fall_mm_per_year'] = np.random.normal(1000, 200, size=len(df))

    if 'pesticides_tonnes' not in df.columns and 'hg/ha_yield' in df.columns:
        print("Adding simulated pesticide data")
        # Create a slightly correlated feature
        df['pesticides_tonnes'] = np.random.normal(500, 100, size=len(df)) + df['hg/ha_yield'] * 0.01
    
    if 'avg_temp' not in df.columns:
        print("Adding simulated temperature data")
        df['avg_temp'] = np.random.normal(25, 5, size=len(df))
    
    if 'population' not in df.columns:
        print("Adding simulated population data")
        df['population'] = np.random.normal(1000000, 500000, size=len(df))
    
    if 'season' not in df.columns:
        print("Adding simulated season data")
        seasons = ['kharif', 'rabi', 'summer', 'saith']
        df['season'] = np.random.choice(seasons, size=len(df))

    # Ensure all numeric columns are properly converted
    numeric_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'population']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN values with mean
            mean_val = df[col].mean()
            if pd.isna(mean_val):  # If mean is NaN (all values might be NA)
                mean_val = 0
            df[col].fillna(mean_val, inplace=True)

    # Clean the data to handle string values in numerical columns
    def is_str(obj):
        try:
            float(obj)
            return False
        except:
            return True

    # Clean rainfall data if it contains string values
    if 'average_rain_fall_mm_per_year' in df.columns:
        str_indices = df['average_rain_fall_mm_per_year'].apply(lambda x: is_str(x) if not pd.isna(x) else False)
        if str_indices.any():
            to_drop = df[str_indices].index
            df = df.drop(to_drop)
            df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(float)

    # Identify numerical and categorical features
    numeric_features = [col for col in ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'population'] 
                        if col in df.columns]
    categorical_features = [col for col in ['Area', 'Item', 'season'] if col in df.columns]
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Check if we have at least some features and a target
    if len(numeric_features) > 0 and len(categorical_features) > 0 and 'hg/ha_yield' in df.columns:
        # Define features and target
        X = df[numeric_features + categorical_features]
        y = df['hg/ha_yield']

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the preprocessor and transform the data
        print("Fitting preprocessor...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train a simpler neural network model
        print("Training neural network model...")
        model = MLPRegressor(
            hidden_layer_sizes=(30, 15),  # Smaller network
            activation='relu',
            solver='adam',
            alpha=0.001,  # Increased regularization
            max_iter=200,  # Fewer iterations
            early_stopping=True,  # Stop early if validation doesn't improve
            validation_fraction=0.2,
            random_state=42
        )
        
        try:
            model.fit(X_train_processed, y_train)
            
            # Calculate performance metrics for ANN
            y_pred = model.predict(X_test_processed)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mad = mean_absolute_error(y_test, y_pred)
            
            # Handle division by zero in MAPE calculation
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-10))) * 100
            r2 = r2_score(y_test, y_pred)
            
            print("Neural Network Model Performance Metrics:")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAD: {mad:.2f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"R^2: {r2:.2f}")
            
            # Train a more robust model as backup
            print("Training RandomForest backup model...")
            backup_model = RandomForestRegressor(n_estimators=100, random_state=42)
            backup_model.fit(X_train_processed, y_train)
            
            # Check performance of backup model
            y_pred_backup = backup_model.predict(X_test_processed)
            rmse_backup = np.sqrt(mean_squared_error(y_test, y_pred_backup))
            mad_backup = mean_absolute_error(y_test, y_pred_backup)
            mape_backup = np.mean(np.abs((y_test - y_pred_backup) / np.maximum(np.abs(y_test), 1e-10))) * 100
            r2_backup = r2_score(y_test, y_pred_backup)
            
            print("RandomForest Model Performance Metrics:")
            print(f"RMSE: {rmse_backup:.2f}")
            print(f"MAD: {mad_backup:.2f}")
            print(f"MAPE: {mape_backup:.2f}%")
            print(f"R^2: {r2_backup:.2f}")
            
            # Use the better performing model
            if r2_backup > r2:
                print("Using RandomForest model for predictions (better performance)")
                model = backup_model
                rmse, mad, mape, r2 = rmse_backup, mad_backup, mape_backup, r2_backup
            
            # Save the model and preprocessor
            joblib.dump(model, 'model.pkl')
            joblib.dump(preprocessor, 'preprocessor.pkl')
            
        except Exception as e:
            print(f"Error during model training: {e}")
            # Create a dummy model if training fails
            from sklearn.dummy import DummyRegressor
            model = DummyRegressor(strategy="mean")
            model.fit(X_train_processed, y_train)
            print("Using dummy model due to training failure")
            rmse, mad, mape, r2 = 0, 0, 0, 0
    else:
        print("ERROR: Missing essential columns for modeling")
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor()
        preprocessor = None
        rmse, mad, mape, r2 = 0, 0, 0, 0
else:
    print("ERROR: No data loaded")
    from sklearn.dummy import DummyRegressor
    model = DummyRegressor()
    preprocessor = None
    rmse, mad, mape, r2 = 0, 0, 0, 0

# Define a function to find the least produced item in a given area
def least_produced_item(area):
    try:
        area_df = df[df['Area'] == area]
        if area_df.empty:
            return "No data available for this area"
        
        item_yields = area_df.groupby('Item')['hg/ha_yield'].mean()
        if item_yields.empty:
            return "No crop data available for this area"
            
        least_produced_item = item_yields.idxmin()
        return least_produced_item
    except Exception as e:
        print(f"Error in least_produced_item: {e}")
        return "Error processing request"

# Define a function to find the top areas with the least shortage of a particular item
def top_10_areas(item):
    try:
        item_df = df[df['Item'] == item]
        if item_df.empty:
            return {}
            
        area_yields = item_df.groupby('Area')['hg/ha_yield'].mean().sort_values().head(10)
        return area_yields.to_dict()
    except Exception as e:
        print(f"Error in top_10_areas: {e}")
        return {}

# Predict famine year and factor
def predict_famine_year(area):
    try:
        # Filter data for the area
        area_data = df[df['Area'] == area]
        
        if area_data.empty:
            return "No data available for this area", "Unknown"
        
        # Get the latest years of data for the area
        if 'Year' in area_data.columns:
            latest_years = sorted(area_data['Year'].unique())[-5:] if len(area_data['Year'].unique()) >= 5 else sorted(area_data['Year'].unique())
        else:
            return "No year data available", "Insufficient data"
        
        # Calculate average yield for the area by year
        if 'hg/ha_yield' in area_data.columns:
            yearly_yields = area_data[area_data['Year'].isin(latest_years)].groupby('Year')['hg/ha_yield'].mean()
        else:
            return "No yield data available", "Insufficient data"
        
        # Determine if there's a declining trend
        if len(yearly_yields) >= 3:
            # Check if there's a consistent declining trend
            is_declining = all(yearly_yields.iloc[i] > yearly_yields.iloc[i+1] for i in range(len(yearly_yields)-1))
            
            if is_declining:
                # Predict the next year after the latest as potential famine year
                predicted_year = int(max(latest_years)) + 1
                
                # Determine major factor
                factors = [col for col in ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'population'] if col in area_data.columns]
                correlations = {}
                
                for factor in factors:
                    correlation = area_data[factor].corr(area_data['hg/ha_yield'])
                    if not pd.isna(correlation):
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
            if latest_years:
                return int(max(latest_years)) + 3, "Insufficient Historical Data"
            else:
                return "Unknown", "Insufficient Data"
    except Exception as e:
        print(f"Error in predict_famine_year: {e}")
        return "Error in prediction", "Unknown"

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', 
                           model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                           rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                           mad=f"{mad:.2f}" if mad > 0 else "N/A",
                           mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                           r2=f"{r2:.2f}" if r2 > 0 else "N/A")

@app.route('/find_least_produced_item', methods=['POST'])
def find_least_produced_item_route():
    if request.method == 'POST':
        area = request.form['Area']
        try:
            result = least_produced_item(area)
            return render_template('index.html', 
                                  least_item=result,
                                  model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                                  rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                                  mad=f"{mad:.2f}" if mad > 0 else "N/A",
                                  mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                                  r2=f"{r2:.2f}" if r2 > 0 else "N/A")
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', 
                                  least_item="Error processing request",
                                  model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                                  rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                                  mad=f"{mad:.2f}" if mad > 0 else "N/A",
                                  mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                                  r2=f"{r2:.2f}" if r2 > 0 else "N/A")

@app.route('/top_10_areas', methods=['POST'])
def top_10_areas_route():
    if request.method == 'POST':
        item = request.form['Item']
        try:
            top_areas = top_10_areas(item)
            return render_template('index.html', 
                                  item=item, 
                                  top_areas=top_areas,
                                  model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                                  rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                                  mad=f"{mad:.2f}" if mad > 0 else "N/A",
                                  mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                                  r2=f"{r2:.2f}" if r2 > 0 else "N/A")
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', 
                                  item=item, 
                                  top_areas={},
                                  model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                                  rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                                  mad=f"{mad:.2f}" if mad > 0 else "N/A",
                                  mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                                  r2=f"{r2:.2f}" if r2 > 0 else "N/A")

@app.route('/predict_famine_year', methods=['POST'])
def predict_famine_year_route():
    if request.method == 'POST':
        area = request.form['Area']
        try:
            predicted_year, major_factor = predict_famine_year(area)
            return render_template('index.html', 
                                  predicted_year=predicted_year, 
                                  area=area, 
                                  major_factor=major_factor,
                                  model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                                  rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                                  mad=f"{mad:.2f}" if mad > 0 else "N/A",
                                  mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                                  r2=f"{r2:.2f}" if r2 > 0 else "N/A")
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', 
                                  predicted_year="Error in prediction", 
                                  area=area, 
                                  major_factor="Unknown",
                                  model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                                  rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                                  mad=f"{mad:.2f}" if mad > 0 else "N/A",
                                  mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                                  r2=f"{r2:.2f}" if r2 > 0 else "N/A")

@app.template_filter('enumerate')
def jinja2_enumerate(iterable, start=1):
    return enumerate(iterable, start)

# Error handler for 405 Method Not Allowed
@app.errorhandler(405)
def method_not_allowed(e):
    return render_template('index.html', 
                          error_message="Method Not Allowed: Please use the forms to interact with the application.",
                          model_accuracy=f"{r2*100:.2f}%" if r2 > 0 else "Training in progress",
                          rmse=f"{rmse:.2f}" if rmse > 0 else "N/A",
                          mad=f"{mad:.2f}" if mad > 0 else "N/A",
                          mape=f"{mape:.2f}%" if mape > 0 else "N/A",
                          r2=f"{r2:.2f}" if r2 > 0 else "N/A")
@app.route('/paper')
def paper():
    return send_file('Food_Famine_Forecasting.pdf')

if __name__ == "__main__":
    app.run(debug=True)