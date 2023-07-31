# modelo para calcular area de circulo

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import seaborn as sb

# Importando datos
medidas_df = pd.read_csv("area_circulo.csv")

# Visualizando datos del excel
# sb.scatterplot(
#     data=medidas_df,
#     x="Radio",
#     y="Area"
# )
# plt.show()

# Cargando set de datos
x_train = medidas_df["Radio"]
y_train = medidas_df["Area"]

# División en conjuntos de entrenamiento y prueba
radios_train, radios_test, areas_train, areas_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Aumento de datos
radios_train_augmented = []
areas_train_augmented = []

for x_train, y_train in zip(radios_train, areas_train):
    radios_train_augmented.append(x_train)
    areas_train_augmented.append(y_train)
    # Aumento de datos mediante rotaciones y traslaciones
    radios_train_augmented.append(x_train + 0.5)
    areas_train_augmented.append(y_train + 3.14 * (x_train + 0.5)**2)
    radios_train_augmented.append(x_train - 0.5)
    areas_train_augmented.append(y_train + 3.14 * (x_train - 0.5)**2)


# Definición del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
#     tf.keras.layers.Dense(1)
# ])

# Regularización
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

# Compilación del modelo
model.compile(optimizer='adam', loss='mse')

# Entrenamiento del modelo
epochs_hist = model.fit(radios_train_augmented, areas_train_augmented, epochs=100, validation_data=(radios_test, areas_test))

# epochs_hist = model.fit(x_train, y_train, epochs=1000)

# Evaluación del modelo en el conjunto de prueba
# Evaluando el modelo
epochs_hist.history.keys()

loss = model.evaluate(radios_test, areas_test)
print("Pérdida en el conjunto de prueba:", loss)

# Grafico
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de Perdida durante Entrenamiento del Modelo')
plt.xlabel('Epoch')
plt.ylabel('Perdida')
plt.legend('Training Loss')
plt.show()

model.get_weights()

# Predicción del área para un radio dado
radio = 6
area_predicha = model.predict([radio])

print("El área predicha para un radio de", radio, "es:", area_predicha)

radio = 10
area_predicha = model.predict([radio])

print("El área predicha para un radio de", radio, "es:", area_predicha)

radio = 102
area_predicha = model.predict([radio])

print("El área predicha para un radio de", radio, "es:", area_predicha)

# # Optimizador
# model.compile(optimizer=tf.keras.optimizers.Adam(100),loss="mean_squared_error")

