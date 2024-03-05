import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import models, layers
import time
import tkinter as tk
frequences=np.linspace(0,30,201)


directory_to_save_CRV3 = "/home/noureddine/Digital-Twin/Synthétiques/Données_Synthétiques-test1/Neural-Network/CRV3/"
model_vx_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_vx")
model_vy_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_vy")
model_vz_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_vz")
model_ax_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_ax")
model_ay_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_ay")
model_az_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_az")
model_ux_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_ux")
model_uy_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_uy")
model_uz_CRV3 = tf.keras.models.load_model(directory_to_save_CRV3 + "/model_uz")


########################"CRA2



directory_to_save_CRA2 = "/home/noureddine/Digital-Twin/Synthétiques/Données_Synthétiques-test1/Neural-Network/CRA2"
model_vx_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_vx")
model_vy_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_vy")
model_vz_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_vz")
model_ax_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_ax")
model_ay_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_ay")
model_az_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_az")
model_ux_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_ux")
model_uy_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_uy")
model_uz_CRA2 = tf.keras.models.load_model(directory_to_save_CRA2 + "/model_uz")

########################CRA4



directory_to_save_CRA4 = "/home/noureddine/Digital-Twin/Synthétiques/Données_Synthétiques-test1/Neural-Network/CRA4"
model_vx_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_vx")
model_vy_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_vy")
model_vz_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_vz")
model_ax_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_ax")
model_ay_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_ay")
model_az_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_az")
model_ux_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_ux")
model_uy_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_uy")
model_uz_CRA4 = tf.keras.models.load_model(directory_to_save_CRA4 + "/model_uz")

##########################DALO
directory_to_save_DALO = "/home/noureddine/Digital-Twin/Synthétiques/Données_Synthétiques-test1/Neural-Network/DALO"
model_vx_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_vx")
model_vy_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_vy")
model_vz_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_vz")
model_ax_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_ax")
model_ay_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_ay")
model_az_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_az")
model_ux_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_ux")
model_uy_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_uy")
model_uz_DALO = tf.keras.models.load_model(directory_to_save_DALO + "/model_uz")
from tkinter import ttk
# Définir une liste de modèles par station
station_models = {
    "DALO": [model_vx_DALO, model_vy_DALO, model_vz_DALO, model_ax_DALO, model_ay_DALO, model_az_DALO, model_ux_DALO, model_uy_DALO, model_uz_DALO],
    "CRV3": [model_vx_CRV3, model_vy_CRV3, model_vz_CRV3, model_ax_CRV3, model_ay_CRV3, model_az_CRV3, model_ux_CRV3, model_uy_CRV3, model_uz_CRV3],
    "CRA2": [model_vx_CRA2, model_vy_CRA2, model_vz_CRA2, model_ax_CRA2, model_ay_CRA2, model_az_CRA2, model_ux_CRA2, model_uy_CRA2, model_uz_CRA2],
    "CRA4": [model_vx_CRA4, model_vy_CRA4, model_vz_CRA4, model_ax_CRA4, model_ay_CRA4, model_az_CRA4, model_ux_CRA4, model_uy_CRA4, model_uz_CRA4]
}
min_max_Dalo = {
    'model_1': 2.0861474325784002e-08,
    'model_2': 1.1366082297570301e-08,
    'model_3':4.01825051010718e-09,
    'model_4':7.331080098538223e-08,
    'model_5':3.9814572971863527e-08,
    'model_6':7.806625346162832e-09,
    'model_7':7.906411174221151e-09,
    'model_8':4.235892858694967e-09,
    'model_9':3.2392622170925766e-09
}
min_Dalo = {
    'model_1': -1.0741604938857563e-08,
    'model_2': -5.231755029200258e-09,
    'model_3':-1.2628177392670636e-09,
    'model_4':-4.2111150122536856e-08,
    'model_5':-1.689131856608128e-08,
    'model_6':-4.962245725437242e-09,
    'model_7':-7.906409749125487e-09,
    'model_8':-1.484672496054884e-09,
    'model_9':-6.686618693094691e-15
}
min_max_CRV3 = {
    'model_1': 2.0861474325784002e-08,
    'model_2': 1.1366082297570301e-08,
    'model_3':4.01825051010718e-09,
    'model_4':7.331080098538223e-08,
    'model_5':3.9814572971863527e-08,
    'model_6':7.806625346162832e-09,
    'model_7':7.906411174221151e-09,
    'model_8':4.235892858694967e-09,
    'model_9':3.2392622170925766e-09
}
min_CRV3 = {
    'model_1': -1.0741604938857563e-08,
    'model_2': -5.231755029200258e-09,
    'model_3':-1.2628177392670636e-09,
    'model_4':-4.2111150122536856e-08,
    'model_5':-1.689131856608128e-08,
    'model_6':-4.962245725437242e-09,
    'model_7':-7.906409749125487e-09,
    'model_8':-1.484672496054884e-09,
    'model_9':-6.686618693094691e-15
}
min_max_CRA4 = {
    'model_1': 2.094058793034037e-08,
    'model_2': 1.136940319668156e-08,
    'model_3':3.931634240394999e-09,
    'model_4':7.312065974929283e-08,
    'model_5':3.954446547993484e-08,
    'model_6':7.651200562008853e-09,
    'model_7':7.903053929827898e-09,
    'model_8':4.223920990753527e-09,
    'model_9':3.149291793304159e-09
}
min_CRA4 = {
    'model_1': -1.0779119818948857e-08,
    'model_2': -5.234517264085525e-09,
    'model_3':-1.2461193188428865e-09,
    'model_4':-4.202717462931105e-08,
    'model_5':-1.673988236916557e-08,
    'model_6':-4.87884621591661e-09,
    'model_7':-7.90305243469902e-09,
    'model_8':-1.432957752456332e-09,
    'model_9':-7.848567478709826e-15
}
min_max_CRA2 = {
    'model_1': 2.101875384852292e-08,
    'model_2': 1.1440271396878643e-08,
    'model_3':3.936489578748592e-09,
    'model_4':7.337694896136782e-08,
    'model_5':3.9698473841554005e-08,
    'model_6':7.660056589031683e-09,
    'model_7':7.91189574730055e-09,
    'model_8':4.241215156852718e-09,
    'model_9':3.150641959798676e-09
}
min_CRA2 = {
    'model_1': -1.083096989873411e-08,
    'model_2': -5.264123803527809e-09,
    'model_3':-1.2497440859959852e-09,
    'model_4':-4.2163758706692533e-08,
    'model_5':-1.684997563700108e-08,
    'model_6':-4.886871352027811e-09,
    'model_7':-7.911894250867135e-09,
    'model_8':-1.4481471577454386e-09,
    'model_9':-7.539774841524693e-15
}
# Fonction pour prédire le mouvement
def predict_movement():
    # Récupérer les valeurs saisies par l'utilisateur
    Xsource = float(entry_Xsource.get())
    Ysource = float(entry_Ysource.get())
    Zsource = float(entry_Zsource.get())
    Vs1 = float(entry_Vs1.get())
    Vs2 = float(entry_Vs2.get())
    Vs3 = float(entry_Vs3.get())

    # Normaliser les entrées
    Vs1 = (Vs1 - 1150) / (1650 - 1150)
    Vs2 = (Vs2 - 2150) / (2650 - 2150)
    Vs3 = (Vs3 - 3750) / (4250 - 3750)
    Xsource = (Xsource - 14068) / (15068 - 14068)
    Ysource = (Ysource - 17553) / (18553 - 17553)
    Zsource = (Zsource + 28520) / (-27520 + 28520)

    input_data = np.array([[Vs1, Vs2, Vs3, Xsource, Ysource, Zsource]])  # Créer une entrée pour le modèle

    # Obtenir la valeur de la station sélectionnée
    selected_station = station_combobox.get()
    selected_model_index = model_combobox.current()

    # Charger le modèle correspondant à la station et au modèle sélectionnés
    model = station_models[selected_station][selected_model_index]

    # Récupérer les valeurs de minimum et de maximum pour le modèle sélectionné
    if selected_station == "DALO":
        min_max_station = min_max_Dalo
        min_station = min_Dalo
    elif selected_station == "CRV3":
        min_max_station = min_max_CRV3
        min_station = min_CRV3
    elif selected_station == "CRA2":
        min_max_station = min_max_CRA2
        min_station = min_CRA2
    elif selected_station == "CRA4":
        min_max_station = min_max_CRA4
        min_station = min_CRA4

    model_constant = min_max_station[f'model_{selected_model_index + 1}']
    model_min = min_station[f'model_{selected_model_index + 1}']

    # Faire la prédiction avec le modèle sélectionné
    result = model.predict(input_data)
    result = result.flatten() * model_constant + model_min

    plt.figure()
    plt.plot(frequences, result)  # Utilisez vos données de fréquence appropriées ici
    plt.xlabel('Fréquences')
    plt.ylabel('Mouvement du sol')
    plt.title('Série temporelle du mouvement du sol prédit')
    plt.show()

# Créer une fenêtre principale
root = tk.Tk()
root.geometry("1000x700")
root.title("Prédiction du mouvement du sol")

# Créer un LabelFrame pour les propriétés géophysiques
geophysical_frame = tk.LabelFrame(root, text="Propriétés géophysiques")
geophysical_frame.pack(padx=20, pady=10)

# Créer des Labels et des Entries pour chaque propriété géophysique
label_Vs1 = tk.Label(geophysical_frame, text="Vs1 :")
label_Vs1.pack()
entry_Vs1 = tk.Entry(geophysical_frame)
entry_Vs1.pack()

label_Vs2 = tk.Label(geophysical_frame, text="Vs2 :")
label_Vs2.pack()
entry_Vs2 = tk.Entry(geophysical_frame)
entry_Vs2.pack()

label_Vs3 = tk.Label(geophysical_frame, text="Vs3 :")
label_Vs3.pack()
entry_Vs3 = tk.Entry(geophysical_frame)
entry_Vs3.pack()

# Créer un LabelFrame pour la position de la source
seismic_frame = tk.LabelFrame(root, text="Position de la source")
seismic_frame.pack(padx=20, pady=10)

# Créer des Labels et des Entries pour la position de la source
label_Xsource = tk.Label(seismic_frame, text="Est-West :")
label_Xsource.pack()
entry_Xsource = tk.Entry(seismic_frame)
entry_Xsource.pack()

label_Ysource = tk.Label(seismic_frame, text="North-South :")
label_Ysource.pack()
entry_Ysource = tk.Entry(seismic_frame)
entry_Ysource.pack()

label_Zsource = tk.Label(seismic_frame, text="Depth :")
label_Zsource.pack()
entry_Zsource = tk.Entry(seismic_frame)
entry_Zsource.pack()

# Créer un LabelFrame pour le choix de la station
station_frame = tk.LabelFrame(root, text="Choix de la station")
station_frame.pack(padx=20, pady=10)

# Créer une Combobox pour choisir la station
station_label = tk.Label(station_frame, text="Station :")
station_label.pack()
station_combobox = ttk.Combobox(station_frame, values=["DALO", "CRV3","CRA2", "CRA4"])
station_combobox.pack()

# Créer un LabelFrame pour le choix du modèle
model_frame = tk.LabelFrame(root, text="Choix du modèle")
model_frame.pack(padx=20, pady=10)

# Créer une Combobox pour choisir le modèle
model_label = tk.Label(model_frame, text="Modèle :")
model_label.pack()
model_combobox = ttk.Combobox(model_frame, values=["Est-West-velocity", "North-south velocity", "Vertical-veloctiy", "Est-West-acceleration", "North-South-acceleration", "Vertical-acceleration", "Est-West-deplacement", "North-South-deplacement", "Vertical-deplacement"])
model_combobox.pack()

# Créer un bouton pour exécuter la prédiction
button_predict = tk.Button(root, text="Prédire", command=predict_movement)
button_predict.pack()

# Démarrer la boucle principale
root.mainloop()


