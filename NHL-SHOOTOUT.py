import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load
donnes = pd.read_excel("C:\\Users\\david\\OneDrive\\Desktop\\NHL_DONNES.xlsx")
donnes.columns = donnes.columns.str.strip()

# Generale Stats
donnes['G_per_Gp'] = donnes['G'] / donnes['GP']
donnes['A_per_Gp'] = donnes['A'] / donnes['GP']
donnes['S_per_Gp'] = donnes['S'] / donnes['GP']
donnes['PTS_per_Gp'] = donnes['PTS'] / donnes['GP']
donnes['GWG_per_Gp'] = donnes['GWG'] / donnes['GP']
donnes['PPG_per_Gp'] = donnes['PPG'] / donnes['GP']
donnes['PPA_per_Gp'] = donnes['PPA'] / donnes['GP']

# Shootout
donnes['SO_Experience'] = donnes['SOA'] / donnes['GP']
donnes['SO_Efficiency'] = donnes['SOG'] / donnes['SOA'].replace(0, 0.001)

#Experimentale
donnes['ClutchIndex'] = donnes['GWG'] / donnes['G'].replace(0, 0.001)
donnes['FinishingRate'] = donnes['G'] / donnes['S'].replace(0, 0.001)
donnes['Consistency'] = donnes['G_per_Gp'] / donnes['PTS_per_Gp']


donnes['StressManagement'] = ( # The idea of pressure
    donnes['GWG_per_Gp'] * 0.4 +      # pure clutch goals
    donnes['PPG_per_Gp'] * 0.2 +      # power-play scoring
    donnes['PPA_per_Gp'] * 0.1 +      # assists under pressure
    donnes['FinishingRate'] * 0.2 +   # Actaully being able to Score
    donnes['SO_Efficiency'].replace(0, 0.001) * 0.05 +  # shootouts attempts (if he was chosen often)
    donnes['SO_Experience'].replace(0, 0.001) * 0.05  # shootouts goals (if he scored often)
)

donnes['StressManagement'] = donnes['StressManagement'] / donnes['StressManagement'].max() * 100 # tHIS HELPS SCALE TO LOOK LIKE PERCENTAGE


donnes['ClutchScore'] = ( # The idea of being able to score when it matters
    (donnes['ClutchIndex'] * 0.4) +   # Game-winners are pure pressure goals
    (donnes['PPG'] * 0.2) +          # Power-play performance
    (donnes['PPA'] * 0.1) +        # Power-play assists
    (donnes['StressManagement'] * 0.4)
)
donnes['ClutchScore'] = donnes['ClutchScore'] / donnes['ClutchScore'].max() * 100

# Drop the unnecessssary
donnes = donnes.drop(columns=['RK','Name','GP','G','A','S','PPG','PPA','PTS','FL','FW','PIM','FO%','TOI/G','SHFT'], errors='ignore')

# Features & target
x = donnes[[
    'G_per_Gp', 'A_per_Gp', 'S_per_Gp', # Normal
    'Consistency', 'FinishingRate', # Experiemntale
    'SO_Experience', 'SO_Efficiency' # Shootouts

]]


y = donnes['ClutchScore']

# Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"Cross-validation R²: {cv.mean():.3f} ± {cv.std():.3f}")

importances = pd.Series(model.feature_importances_, index=x.columns)
print("\nTop features:\n", importances.sort_values(ascending=False).head(10))
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load
donnes = pd.read_excel("C:\\Users\\david\\OneDrive\\Desktop\\NHL_DONNES.xlsx")
donnes.columns = donnes.columns.str.strip()

# Generale Stats
donnes['G_per_Gp'] = donnes['G'] / donnes['GP']
donnes['A_per_Gp'] = donnes['A'] / donnes['GP']
donnes['S_per_Gp'] = donnes['S'] / donnes['GP']
donnes['PTS_per_Gp'] = donnes['PTS'] / donnes['GP']
donnes['GWG_per_Gp'] = donnes['GWG'] / donnes['GP']
donnes['PPG_per_Gp'] = donnes['PPG'] / donnes['GP']
donnes['PPA_per_Gp'] = donnes['PPA'] / donnes['GP']

# Shootout
donnes['SO_Experience'] = donnes['SOA'] / donnes['GP']
donnes['SO_Efficiency'] = donnes['SOG'] / donnes['SOA'].replace(0, 0.001)

#Experimentale
donnes['ClutchIndex'] = donnes['GWG'] / donnes['G'].replace(0, 0.001)
donnes['FinishingRate'] = donnes['G'] / donnes['S'].replace(0, 0.001)
donnes['Consistency'] = donnes['G_per_Gp'] / donnes['PTS_per_Gp']


donnes['StressManagement'] = ( # The idea of pressure
    donnes['GWG_per_Gp'] * 0.4 +      # pure clutch goals
    donnes['PPG_per_Gp'] * 0.2 +      # power-play scoring
    donnes['PPA_per_Gp'] * 0.1 +      # assists under pressure
    donnes['FinishingRate'] * 0.2 +   # Actaully being able to Score
    donnes['SO_Efficiency'].replace(0, 0.001) * 0.05 +  # shootouts attempts (if he was chosen often)
    donnes['SO_Experience'].replace(0, 0.001) * 0.05  # shootouts goals (if he scored often)
)

donnes['StressManagement'] = donnes['StressManagement'] / donnes['StressManagement'].max() * 100 # tHIS HELPS SCALE TO LOOK LIKE PERCENTAGE


donnes['ClutchScore'] = ( # The idea of being able to score when it matters
    (donnes['ClutchIndex'] * 0.4) +   # Game-winners are pure pressure goals
    (donnes['PPG'] * 0.2) +          # Power-play performance
    (donnes['PPA'] * 0.1) +        # Power-play assists
    (donnes['StressManagement'] * 0.4)
)
donnes['ClutchScore'] = donnes['ClutchScore'] / donnes['ClutchScore'].max() * 100

# Drop the unnecessssary
donnes = donnes.drop(columns=['RK','Name','GP','G','A','S','PPG','PPA','PTS','FL','FW','PIM','FO%','TOI/G','SHFT'], errors='ignore')

# Features & target
x = donnes[[
    'G_per_Gp', 'A_per_Gp', 'S_per_Gp', # Normal
    'Consistency', 'FinishingRate', # Experiemntale
    'SO_Experience', 'SO_Efficiency' # Shootouts

]]


y = donnes['ClutchScore']

# Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"Cross-validation R²: {cv.mean():.3f} ± {cv.std():.3f}")

importances = pd.Series(model.feature_importances_, index=x.columns)
print("\nTop features:\n", importances.sort_values(ascending=False).head(10))
