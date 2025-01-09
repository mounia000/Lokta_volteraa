import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Charger les données du fichier CSV
file_path = 'C:/Users/user/Documents/projet master1/mathemat/Lokta_volteraa/populations_lapins_renards.csv'
data = pd.read_csv(file_path)
real_time = data['date'].values
real_lapins = data['lapin'].values
real_renards = data['renard'].values

# Définir la fonction de simulation du modèle Lotka-Volterra
def simulate_lotka_volterra(alpha, beta, delta, gamma, step=0.001, num_steps=100000):
    time = [0]
    lapin = [1]  # Initialisation de la population de lapins
    renard = [2]  # Initialisation de la population de renards
    
    for _ in range(num_steps):
        new_val_time = time[-1] + step
        new_val_lapin = (lapin[-1] * (alpha - beta * renard[-1])) * step + lapin[-1]
        new_val_renard = (renard[-1] * (delta * lapin[-1] - gamma)) * step + renard[-1]
        
        time.append(new_val_time)
        lapin.append(new_val_lapin)
        renard.append(new_val_renard)
    
    lapin = np.array(lapin) * 1000  # Ajustement de l'échelle
    renard = np.array(renard) * 1000  # Ajustement de l'échelle
    
    return time, lapin, renard

# Définir la fonction d'erreur (MSE)
def mse(real_lapins, real_renards, predicted_lapins, predicted_renards):
    mse_lapins = mean_squared_error(real_lapins, predicted_lapins)
    mse_renards = mean_squared_error(real_renards, predicted_renards)
    return mse_lapins + mse_renards  # Somme des erreurs pour les deux populations

# Paramètres à tester (grid search)
alpha_values = [1/3, 2/3, 1, 4/3]
beta_values = [1/3, 2/3, 1, 4/3]
delta_values = [1/3, 2/3, 1, 4/3]
gamma_values = [1/3, 2/3, 1, 4/3]

best_mse = float('inf')
best_params = None

# Grid search pour trouver les meilleurs paramètres
for alpha in alpha_values:
    for beta in beta_values:
        for delta in delta_values:
            for gamma in gamma_values:
                # Simuler les populations avec les paramètres actuels
                time, lapin_sim, renard_sim = simulate_lotka_volterra(alpha, beta, delta, gamma)
                
                # Truncate or align the time for MSE calculation
                min_length = min(len(lapin_sim), len(real_lapins))
                lapin_sim = lapin_sim[:min_length]
                renard_sim = renard_sim[:min_length]
                
                # Calculer l'erreur MSE
                current_mse = mse(real_lapins[:min_length], real_renards[:min_length], lapin_sim, renard_sim)
                
                # Mettre à jour les meilleurs paramètres si l'erreur est plus faible
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_params = (alpha, beta, delta, gamma)

# Afficher les meilleurs paramètres trouvés
print(f"Meilleurs paramètres : alpha={best_params[0]}, beta={best_params[1]}, delta={best_params[2]}, gamma={best_params[3]}")
print(f"Erreur quadratique moyenne minimale : {best_mse}")

# Simuler avec les meilleurs paramètres trouvés
alpha, beta, delta, gamma = best_params
time, lapin_sim, renard_sim = simulate_lotka_volterra(alpha, beta, delta, gamma)

# Afficher les résultats
plt.figure(figsize=(15, 6))
plt.plot(time, lapin_sim, "b-", label="Lapins simulés")
plt.plot(time, renard_sim, "r-", label="Renards simulés")
plt.plot(real_time, real_lapins, "bo", label="Lapins réels")
plt.plot(real_time, real_renards, "ro", label="Renards réels")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Dynamique des populations de lapins et renards")
plt.legend()
plt.show()