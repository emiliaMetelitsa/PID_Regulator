#Импорты
import numpy as np
import matplotlib.pyplot as plt
from knp_ann2snn.altainn import TernaryDense, heaviside, Clip
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import deque

def main():
    #Параметры ДПТ
    dt = 0.01 #Шаг по времени
    N = 400000 #Количество элементов внаборе данных

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

    #Генерация данных
    omega = 0.0 #Текущая угловая скорость
    integral = 0.0 #Интеграл ошибки
    prev_error = 0.0 #Предыдущая ошибка

    r = 0.0  #Задание скорости

    #Массивы данных
    data_X = []
    data_Y = []

    #Уровень шума
    noise_std = 0.0

    #Окно признаков
    HISTORY = 10

    def adaptive_pid(error, d_error, integral, Kp0 = 2.0, Ki0 = 0.5, Kd0 = 0.01):
        abs_e = abs(error)
        abs_de = abs(d_error)
        abs_i = abs(integral)

        # Пропорциональный коэффициент
        # Большая ошибка -> увеличить Kp
        kp = Kp0 * (1.0 + 0.7 * np.tanh(abs_e))

        # Если начинается колебание, немного уменьшаем Kp
        kp *= (1.0 + 0.3 * np.tanh(abs_e))
        kp *= (1.0 - 0.2 * np.tanh(abs_de / 50))

        # уменьшаем Kp возле уставки
        if error * d_error < 0:
            kp *= 0.7

        # Интегральный коэффициент
        # Чем больше накопленная ошибка, тем сильнее интегратор.
        ki = Ki0 * (1.0 + 0.5 * np.tanh(abs_i))

        # При быстром изменении ошибки уменьшаем интегратор
        ki /= (1.0 + 0.4 * np.tanh(abs_de))

        # защита от разгона интегратора
        if abs(error) > 0.5:
            ki *= 0.5

        # Дифференциальный коэффициент
        # Если ошибка быстро меняется, увеличиваем демпфирование.
        kd = Kd0 * (1.0 + 1.2 * np.tanh(abs_de))

        # Если ошибка практически исчезла, дифференциальная часть почти не нужна.
        kd *= (0.5 + 0.5 * np.tanh(abs_e))

        # Ограничения
        kp = np.clip(kp, 0.5 * Kp0, 2.0 * Kp0)

        ki = np.clip(ki, 0.2 * Ki0, 2.5 * Ki0)

        kd = np.clip(kd, 0.3 * Kd0, 3.0 * Kd0)

        return kp, ki, kd

    #Генерация массива данных
    for i in range(N):

        #Смена скорости каждые 4000 шагов
        if i % 4000 == 0:
            r = np.random.uniform(0.2, 2)

            # Шум датчика
            noise_std = np.random.uniform(0.0, 0.05)

            integral = 0.0
            prev_error = 0.0
            error_hist = deque([0.0] * HISTORY, maxlen=HISTORY)
            control_hist = deque([0.0] * HISTORY, maxlen=HISTORY)

        # Измеренная скорость
        omega_meas = omega + np.random.normal(0, noise_std)

        # Ошибка
        error = r - omega_meas

        # Производная ошибки
        d_error = (error - prev_error) / dt

        # Интеграл
        integral += error * dt
        integral = np.clip(integral, -10, 10)

        # Учитель-ПИД
        Kp, Ki, Kd = adaptive_pid(error, d_error, integral)

        u = (Kp * error + Ki * integral + Kd * d_error)
        u = np.clip(u, -u_max, u_max)

        #Электрическая часть
        Ia = (u - ce * phi * omega) / R
        domega = (cm * phi * Ia - B * omega - M_load) / J
        omega += dt * domega

        data_X.append([error, d_error, integral, r, *error_hist, *control_hist])

        data_Y.append([u])

        error_hist.append(error)
        control_hist.append(u)

        prev_error = error

    data_X = np.array(data_X)
    data_Y = np.array(data_Y)

    #Разделение на выборки для обучения и теста
    train_idx = int(0.8*N)

    x_train = data_X[:train_idx]
    y_train = data_Y[:train_idx]

    x_test = data_X[train_idx:]
    y_test = data_Y[train_idx:]

    #Нормализация
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)

    x_std[x_std < 1e-8] = 1.0

    x_train_norm = (x_train - x_mean) / x_std
    x_test_norm = (x_test - x_mean) / x_std

    #Сохраняем параметры нормализации
    np.save("x_mean.npy", x_mean)
    np.save("x_std.npy", x_std)

    #Модель нейросети
    # Encoder (функция активации сигмоида, размер входа - 5, размер выхода - 16)
    encoder = Sequential([
        Dense(64, activation="sigmoid", input_shape=(24,))
    ])

    # SNN (функция активации хевисайда, размер входа - 16, размер выхода - 16)
    snn = Sequential([
        TernaryDense(
            64,
            activation=heaviside,
            input_shape=(64,),
            use_bias=False
        )
    ])

    # Decoder (функция активации , размер входа - 16, размер выхода - 1)
    decoder = Sequential([
        Dense(1, activation="linear", input_shape=(64,))
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

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        x_train_norm,
        y_train,
        epochs=100,
        validation_data=(x_test_norm, y_test),
        callbacks = [checkpoint, early_stop]
    )

    best_model = load_model(
        "sin_model.keras",
        custom_objects={
            "TernaryDense": TernaryDense,
            "heaviside_mod": heaviside,
            "Clip": Clip
        }
    )

    #Тест
    predict_test = best_model.predict(x_test_norm)
    predict_test = predict_test.flatten()

    # Моделирование двигателя на тестовой выборке

    omega_pid = []
    omega_nn = []
    r_history = []

    # Начальные условия
    omega_pid_curr = 0.0
    omega_nn_curr = 0.0

    integral_pid = 0.0
    prev_error_pid = 0.0

    for i in range(len(x_test)):
        r = x_test[i, 3]  # целевая скорость

        # ПИД
        error_pid = r - omega_pid_curr
        d_error_pid = (error_pid - prev_error_pid) / dt
        integral_pid += error_pid * dt

        Kp_true, Ki_true, Kd_true = adaptive_pid(error_pid, d_error_pid, integral_pid)
        u_pid = Kp_true * error_pid + Ki_true * integral_pid + Kd_true * d_error_pid

        u_pid = np.clip(u_pid, -u_max, u_max)

        Ia_pid = (u_pid - ce * phi * omega_pid_curr) / R

        domega_pid = (cm * phi * Ia_pid - M_load - B * omega_pid_curr) / J

        omega_pid_curr += dt * domega_pid

        prev_error_pid = error_pid

        # Нейросеть
        u_nn = predict_test[i]

        Ia_nn = (u_nn - ce * phi * omega_nn_curr) / R

        domega_nn = (cm * phi * Ia_nn - M_load - B * omega_nn_curr) / J

        omega_nn_curr += dt * domega_nn

        omega_pid.append(omega_pid_curr)
        omega_nn.append(omega_nn_curr)
        r_history.append(r)

    omega_pid = np.array(omega_pid)
    omega_nn = np.array(omega_nn)
    r_history = np.array(r_history)

    #График ошибки обучения
    plt.figure()

    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')

    plt.xlabel('Эпоха')
    plt.ylabel('MSE ошибка')
    plt.title('Ошибка обучения нейросети')
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_err.png", dpi=300)
    plt.show()

    #График управляющего сигнала
    plt.figure()
    plt.plot(predict_test, label="NN output (u)")
    plt.plot(y_test.flatten(), label="Teacher PID (u)", alpha=0.5)
    plt.legend()
    plt.title("Сравнение управления (нейросеть и ПИД)")
    plt.savefig("learning_control.png", dpi=300)
    plt.show()

    # График угловой скорости
    plt.figure(figsize=(12, 6))
    plt.plot(r_history,label="Целевая скорость r",linewidth=2)
    plt.plot(omega_pid,label="Скорость с ПИД")
    plt.plot(omega_nn,label="Скорость с нейросетью")
    plt.xlabel("Шаг")
    plt.ylabel("Угловая скорость")
    plt.title("Сравнение качества регулирования")
    plt.grid(True)
    plt.legend()
    plt.savefig("learning_tracking.png", dpi=300)
    plt.show()

    # Сохранение кодирующего блока в файл encoder_sin.keras.
    best_model.layers[0].save("encoder_sin.keras")
    # Сохранение нейронной сети в файл snn_sin.keras.
    best_model.layers[1].save("snn_sin.keras")
    # Сохранение декодирующего блока в файл decoder_sin.keras.
    best_model.layers[2].save("decoder_sin.keras")

if __name__ == "__main__":
    main()
