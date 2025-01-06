'''# Paramètres
alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
x, y = 40, 9  # Populations initiales
dt = 0.1  # Pas de temps
T = 50  # Durée totale
times = []  # Temps
prey = []  # Population de proies
predators = []  # Population de prédateurs

for t in range(int(T/dt)):
    dx = (alpha * x - beta * x * y) * dt
    dy = (delta * x * y - gamma * y) * dt
    x += dx
    y += dy
    times.append(t * dt)
    prey.append(x)
    predators.append(y)

# Tracer les résultats
import matplotlib.pyplot as plt
plt.plot(times, prey, label="Proies")
plt.plot(times, predators, label="Prédateurs")
plt.legend()
plt.show()
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialisation des données et paramètres
time = [0]
lapin = [1000]
renard = [2000]

alpha, beta, delta, gamma = 1/3, 1/3, 1/3, 1/3
step = 0.001

# Simulation des populations avec la méthode d'Euler
for _ in range(0, 1000):
    new_val_time = time[-1] + step
    new_val_lapin = (lapin[-1] * (alpha - beta * renard[-1])) * step + lapin[-1]
    new_val_renard = (renard[-1] * (delta * lapin[-1] - gamma)) * step + renard[-1]

    time.append(new_val_time)
    lapin.append(new_val_lapin)
    renard.append(new_val_renard)

# Tracé des résultats simulés
plt.figure(figsize=(15, 6))
plt.plot(time, lapin, "b-", label="Lapins (Simulation)")
plt.plot(time, renard, "r-", label="Renards (Simulation)")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Simulation des populations (Modèle Lotka-Volterra)")
plt.legend()
plt.show()

# Chargement des données réelles
data = pd.read_csv('populations_lapins_renards.csv')
print(data.head())

# Extraction des colonnes pour les données réelles
time_real = data['time']
lapin_real = data['lapin']
renard_real = data['renard']

# Fonction pour calculer l'erreur quadratique moyenne (MSE)
def mse_loss(lapin_sim, renard_sim, lapin_real, renard_real):
    mse_lapin = np.mean((np.array(lapin_sim) - np.array(lapin_real)) ** 2)
    mse_renard = np.mean((np.array(renard_sim) - np.array(renard_real)) ** 2)
    return mse_lapin + mse_renard

# Calcul du MSE entre la simulation et les données réelles
mse = mse_loss(lapin, renard, lapin_real, renard_real)
print(f"Erreur quadratique moyenne (MSE) : {mse}")

# Tracé des données réelles pour comparaison
plt.figure(figsize=(15, 6))
plt.plot(time_real, lapin_real, "b--", label="Lapins (Données réelles)")
plt.plot(time_real, renard_real, "r--", label="Renards (Données réelles)")
plt.plot(time, lapin, "b-", label="Lapins (Simulation)")
plt.plot(time, renard, "r-", label="Renards (Simulation)")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Comparaison entre données réelles et simulation")
plt.legend()
plt.show()
