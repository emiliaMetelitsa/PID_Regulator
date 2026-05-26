#Импорты
import numpy as np
import matplotlib.pyplot as plt
from knp_ann2snn.altainn import TernaryDense, heaviside, Clip
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

#Параметры ДПТ
dt = 0.01 #Шаг по времени
N = 4000 #Количество элементов внаборе данных

#Электрические параметры
ce = 1.0 #Коэффициент ЭДС
phi = 1.0 #Магнитный поток
Ra = 1.0 #Сопротивление якоря
Rd = 0.2 #Дополнительное сопротивление
R = Ra + Rd #Общее сопротивление

#Механические параметры
J = 0.1 #Момент инерции ротора
cm = 1.0 #Коэффициент момента
B = 0.3 #Коэффициент вязкого трения
M_load = 0.0 #Момент нагрузки

#Ограничение напряжения
u_max = 10.0

#"Учитель" ПИД
Kp_true = 2.0 #Пропорциональный коэффициент
Ki_true = 0.5 #Интегральный коэффициент
Kd_true = 0.01 #Дифференциальный коэффициент

#Генерация данных
omega = 0.0 #Текущая угловая скорость
integral = 0.0 #Интеграл ошибки
prev_error = 0.0 #Предыдущая ошибка

r = 0.0  #Задание скорости

#Массивы данных
data_X = []
data_Y = []

#Генерация массива данных
for i in range(N):

    #Смена скорости каждые 100 шагов
    if i % 100 == 0:
        r = np.random.uniform(-2, 2)

    #Ошибка
    error = r - omega
    d_error = (error - prev_error) / dt
    integral += error * dt

    #ПИД (учитель)
    u = Kp_true * error + Ki_true * integral + Kd_true * d_error
    u = np.clip(u, -u_max, u_max)

    #ДПТ
    #Ток
    Ia = (u - ce * phi * omega) / R

    #Динамика скорости
    domega = (cm * phi * Ia - M_load - B * omega) / J
    omega = omega + dt * domega

    #Данные для сети
    data_X.append([error, d_error, integral, omega, r])
    data_Y.append([u])

    prev_error = error

data_X = np.array(data_X)
data_Y = np.array(data_Y)

#Разделение на выборки для обучения и теста
train_idx = 3000

x_train = data_X[:train_idx]
y_train = data_Y[:train_idx]

x_test = data_X[train_idx:]
y_test = data_Y[train_idx:]

#Модель нейросети
# Encoder (функция активации сигмоида, размер входа - 5, размер выхода - 16)
encoder = Sequential([
    Dense(16, activation="sigmoid", input_shape=(5,))
])

# SNN (функция активации хевисайда, размер входа - 16, размер выхода - 16)
snn = Sequential([
    TernaryDense(
        16,
        activation=heaviside,
        input_shape=(16,),
        use_bias=False
    )
])

# Decoder (функция активации , размер входа - 16, размер выхода - 1)
decoder = Sequential([
    Dense(1, activation="linear", input_shape=(16,))
])

model = Sequential([encoder, snn, decoder])

#Оптимизатор - Adam, функция потерь - среднеквадратическая ошибка
model.compile(
    optimizer="adam",
    loss="mse"
)

#Обучение
checkpoint = ModelCheckpoint(
    "sin_model.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min"
)

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_data=(x_test, y_test)
)

#Тест
predict_test = model.predict(x_test)
predict_test = predict_test.flatten()

#График ошибки обучения
plt.figure()

plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')

plt.xlabel('Эпоха')
plt.ylabel('MSE ошибка')
plt.title('Ошибка обучения нейросети')
plt.legend()
plt.grid(True)

plt.show()

#График управляющего сигнала
plt.figure()
plt.plot(predict_test, label="NN output (u)")
plt.plot(y_test.flatten(), label="Teacher PID (u)", alpha=0.5)
plt.legend()
plt.title("Сравнение управления (нейросеть и ПИД)")

# Сохранение кодирующего блока в файл encoder_sin.keras.
model.layers[0].save("encoder_sin.keras")
# Сохранение нейронной сети в файл snn_sin.keras.
model.layers[1].save("snn_sin.keras")
# Сохранение декодирующего блока в файл decoder_sin.keras.
model.layers[2].save("decoder_sin.keras")

plt.show()
