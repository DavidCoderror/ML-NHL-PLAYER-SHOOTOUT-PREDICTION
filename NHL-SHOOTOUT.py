""""

Project AI: PREDICTION POURCENTAGE SHOOTOUTS
Createur: DAVID CODERRE
Description: Etre Capable de predire en forme de pourcentage, leur chance quand il sont dans Shootout

"""

# Importations
import pandas as pd
from sklearn.model_selection import train_test_split  # Split nos donnes
from sklearn.ensemble import RandomForestRegressor  # Notre Model de choix
from sklearn.metrics import mean_absolute_error, r2_score  # Evaluation
from sklearn.preprocessing import StandardScaler  # Standard
from sklearn.preprocessing import MinMaxScaler #Stress

# Cherchez nos donnes
donnes = pd.read_excel("C:\\Users\\david\\OneDrive\Desktop\\NHL_DONNES.xlsx")
donnes.columns = donnes.columns.str.strip()  # removes leading/trailing spaces

player_names = donnes['Name']

# Stats Generale / Games Played
donnes['G_per_Gp'] = donnes['G'] / donnes['GP']
donnes['A_per_Gp'] = donnes['A'] / donnes['GP']
donnes['S_per_Gp'] = donnes['S'] / donnes['GP']
donnes['PTS_per_Gp'] = donnes['PTS'] / donnes['GP']

# Stats Sous Stress / Games Played
donnes['GWG_per_Gp'] = donnes['GWG'] / donnes['GP']
donnes['PPG_per_Gp'] = donnes['PPG'] / donnes['GP']
donnes['PPA_per_Gp'] = donnes['PPA'] / donnes['GP']

# Experiemntation
donnes['Accuracy'] = donnes['S_per_Gp'] * donnes['G_per_Gp'] * donnes['S%']
donnes['Stress_Management'] = donnes['GWG_per_Gp'] * donnes['PPG_per_Gp'] * donnes['PPA_per_Gp']
donnes['Consistency'] = donnes['G_per_Gp'] / donnes['PTS_per_Gp']
donnes['SOA_Gp'] = donnes['SOA'] / donnes['GP']

scaler_stress = MinMaxScaler()

donnes['Accuracy_Sous Stress'] = donnes['Accuracy'] * donnes['Stress_Management'].replace(0, 0.001)


donnes = donnes.drop(columns=['RK', 'Name', 'GP', 'G', 'A', 'S', 'PPG', 'PPA','PTS', 'FL', 'FW', 'PIM', 'FO%','TOI/G', 'SHFT']) # General
donnes = donnes.drop(columns=['SOA', 'SOG', 'SO%'])


x = donnes.drop(columns=['SO%', 'Accuracy_Sous Stress'], errors='ignore')  # Optional safety in case 'SO%' still exists
y = donnes['Accuracy_Sous Stress']
y = y / y.max() * 100  # scale so the max = 100%

# Split les donnes
X_train, X_temporaire, Y_train, Y_temporaire = train_test_split(x, y, test_size=0.2, random_state=42)  # Training Set
X_val, X_test, Y_val, Y_test = train_test_split(X_temporaire, Y_temporaire, test_size=0.5, random_state=42)  # Validation Set

# Standard
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Model Random Foret et Fit
model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(Y_test, Y_pred)
mean = mean_absolute_error(Y_test, Y_pred)

# Print
print(f"Mean Absolute Error: {mean:.3f}")
print(f"R² Score: {r2:.3f}")

# Results table with player names
Y_test = Y_test.reset_index(drop=True)
Y_pred = pd.Series(Y_pred)
results = pd.DataFrame({
    'Player': player_names.iloc[Y_test.index],
    'Actual SO%': Y_test,
    'Predicted SO%': Y_pred
})

# Sort by predicted SO% descending to see top shootout players
results = results.sort_values(by='Predicted SO%', ascending=False)
print(results.head(15))

# Feature importance
importances = pd.Series(model.feature_importances_, index=x.columns)
print("\nTop features:\n", importances.sort_values(ascending=False).head(10))



"""""

GRAPHIQUE

"""
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
plt.scatter(results['Actual SO%'], results['Predicted SO%'], alpha=0.6, color='blue', label='Players')

# Ligne de tendancee
z = np.polyfit(results['Actual SO%'], results['Predicted SO%'], 1)
p = np.poly1d(z)
plt.plot(results['Actual SO%'], p(results['Actual SO%']), color='green', linestyle='--', label='Trend line')

plt.title("Relation entre pourcentage réel et prédit (Shootout %)")
plt.xlabel("Pourcentage réel ")
plt.ylabel("Pourcentage prédit ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
