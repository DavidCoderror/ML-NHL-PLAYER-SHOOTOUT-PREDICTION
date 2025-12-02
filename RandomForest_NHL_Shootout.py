"""
Project AI: PREDICTION POURCENTAGE SHOOTOUTS
Createur: DAVID CODERRE
Description: Etre Capable de predire en forme de pourcentage, leur chance quand il sont dans Shootout
"""

# ------------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------------------------
# Cherchez mes donnes
# ------------------------------------------------------------------------------------

donnes = pd.read_excel("C:\\Users\\david\\OneDrive\\Desktop\\NHL_DONNES.xlsx")
player_names = donnes['Name']  # Noms des joeurs (Utiliser plus tard durant l'evaluation)

# ------------------------------------------------------------------------------------
# Statistiques Generale
# ------------------------------------------------------------------------------------

# Besoin de diviser tous les statistiques par nombre de jeux jouer
donnes['G_per_Gp'] = donnes['G'] / donnes['GP']
donnes['A_per_Gp'] = donnes['A'] / donnes['GP']
donnes['S_per_Gp'] = donnes['S'] / donnes['GP']
donnes['PTS_per_Gp'] = donnes['PTS'] / donnes['GP']
donnes['GWG_per_Gp'] = donnes['GWG'] / donnes['GP']
donnes['PPG_per_Gp'] = donnes['PPG'] / donnes['GP']
donnes['PPA_per_Gp'] = donnes['PPA'] / donnes['GP']

# ------------------------------------------------------------------------------------
# Statistique de Stress
# ------------------------------------------------------------------------------------

donnes['SO_Experience'] = donnes['SOA'] / donnes['GP']
donnes['SO_Efficiency'] = donnes['SOG'] / donnes['SOA'].replace(0, 0.001)
donnes['FinishingRate'] = donnes['G'] / donnes['S'].replace(0, 0.001)  # Nombre de but par Shot

donnes['StressManagement'] = (                             # L'idee de pression
        donnes['GWG_per_Gp'] * 0.4 +                       # Buts gagnant
        donnes['PPG_per_Gp'] * 0.2 +                       # power-play buts
        donnes['PPA_per_Gp'] * 0.1 +                       # power-play assists
        donnes['FinishingRate'] * 0.2 +                    # Nombre de but par Shot
        donnes['SO_Efficiency'].replace(0, 0.001) * 0.05 + # Esseyes de shootouts
        donnes['SO_Experience'].replace(0, 0.001) * 0.05   # shootouts goals (
)
donnes['StressManagement'] = donnes['StressManagement'] / donnes['StressManagement'].max() * 100  # Le faire entre 0-100

# ------------------------------------------------------------------------------------
# Experimentale 
# ------------------------------------------------------------------------------------

donnes['ClutchIndex'] = donnes['GWG'] / donnes['G'].replace(0, 0.001)  # Nombre de buts gagnante diviser par but normale
donnes['Consistency'] = donnes['G_per_Gp'] / donnes['PTS_per_Gp']      # Nombre de but et de points total par jeux

donnes['ClutchScore'] = (                     # L'idee de faire des points quand il compte
        (donnes['ClutchIndex'] * 0.4) +       # Buts gagnant  est un facteur majeurs
        (donnes['StressManagement'] * 0.4) +  # Gestion de Stress
        (donnes['PPG'] * 0.2) +               # Power-play Buts
        (donnes['PPA'] * 0.1)                 # Power-play assists
)
donnes['ClutchScore'] = donnes['ClutchScore'] / donnes['ClutchScore'].max() * 100  # Le faire entre 0-100 (Si non, donnes seront trop large, techniquement normalisation)

# ------------------------------------------------------------------------------------
# Inputs/Outputs
# ------------------------------------------------------------------------------------

# Drop Tous les donnes qu'ont utilise pas
donnes = donnes.drop(
    columns=['RK', 'Name', 'GP', 'G', 'A', 'S', 'PPG', 'PPA', 'PTS', 'FL', 'FW', 'PIM', 'FO%', 'TOI/G', 'SHFT'], errors='ignore')

# Inputs
x = donnes[[
    'G_per_Gp', 'A_per_Gp', 'S_per_Gp',  # Normale
    'Consistency', 'FinishingRate',      # Experiemntale
    'SO_Experience', 'SO_Efficiency'     # Stress
]]
# Output
y = donnes['ClutchScore']

# Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model et Train
model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"Cross-validation R²: {cv.mean():.3f} ± {cv.std():.3f}")

# ------------------------------------------------------------------------------------
# Features Liste - J'ai besoin de voir quel sont les plus forts
# ------------------------------------------------------------------------------------

# Montrez Nos Features
importances = pd.Series(model.feature_importances_, index=x.columns)
print("\nTop features:\n", importances.sort_values(ascending=False).head(10))

# Results table with player names
Y_test = y_test.reset_index(drop=True)
Y_pred = pd.Series(y_pred)

# ------------------------------------------------------------------------------------
# Joeurs liste - J'ai besoin de voir qui est le meilleur pour valider facilement logiquement
# ------------------------------------------------------------------------------------

results = pd.DataFrame({ 'Player': player_names.iloc[Y_test.index], 'Actual SO%': Y_test, 'Predicted SO%': Y_pred})

# Sort by predicted SO% descending to see top shootout players
results = results.sort_values(by='Predicted SO%', ascending=False)
print(results.head(15))

# ------------------------------------------------------------------------------------
# Section Graphique
# ------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(results['Actual SO%'], results['Predicted SO%'], alpha=0.6)

# Ligne de tendance
coef = np.polyfit(results['Actual SO%'], results['Predicted SO%'], 1)
trend = np.poly1d(coef)
plt.plot(results['Actual SO%'], trend(results['Actual SO%']), linestyle='--')

plt.title("Réel vs Prédit (Shootout %)")
plt.xlabel("SO% réel")
plt.ylabel("SO% prédit")
plt.grid(alpha=0.3)
plt.show()

