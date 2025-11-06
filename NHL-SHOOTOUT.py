""""
Project AI: PREDICTION POURCENTAGE SHOOTOUTS
Createur: DAVID CODERRE
Description: Etre Capable de predire en forme de pourcentage, leur chance quand il sont dans Shootout
"""

# Importations
import pandas as pd
from sklearn.model_selection import train_test_split  # Split nos donnes
from sklearn.ensemble import RandomForestRegressor  # Notre Model de choix
from sklearn.metrics import mean_absolute_error, r2_score # Evaluation

# 1. Cherchez nos donnes
donnes = pd.read_excel("C:\\Users\\david\\OneDrive\Desktop\\NHL_DONNES.xlsx")
donnes.columns = donnes.columns.str.strip()  # removes leading/trailing spaces
donnes = donnes.drop(columns=['Name', 'RK', 'POS', 'TOI/G'])   #Besoin de faire adjustement  a POS, TOI/G

# 2. Input et output
x = donnes.drop("SO%", axis=1)  # Input
y = donnes["SO%"]  # Output

# 3. Split les donnes
X_train, X_temporaire, Y_train, Y_temporaire = train_test_split(x, y, test_size=0.2, random_state=42)  # Training Set
X_val, X_test, Y_val, Y_test = train_test_split(X_temporaire, Y_temporaire, test_size=0.5, random_state=42)  # Validation Set

# 4. Model Random Foret et Fit
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, Y_train)

# 5. Predictions
Y_pred = model.predict(X_test)

# 6. Evaluation
r2 = r2_score(Y_test, Y_pred)
mean = mean_absolute_error(Y_test, Y_pred)

print(f"Mean Absolute Error: {mean:.3f}")
print(f"R² Score: {r2:.3f}")
