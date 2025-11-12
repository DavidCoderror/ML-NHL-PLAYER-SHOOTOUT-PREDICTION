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
