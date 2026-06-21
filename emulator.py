from knp_ann2snn.altainn import TernaryDense
from knp_ann2snn.altainn import heaviside
from knp_ann2snn.altainn import Clip
from knp_ann2snn.python_altai import Altai
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # Параметры ДПТ
    dt = 0.01
    Tsim = 20.0
    N = int(Tsim / dt)

    # Электрическая часть
    ce = 1.0
    phi = 1.0
    Ra = 1.0
    Rd = 0.2
    R = Ra + Rd

    # Механическая часть
    J = 0.1
    cm = 1.0
    B = 0.3
    M_load = 0.0

    # Ограничение управления
    u_max = 10.0

    # Параметры ПИД
    Kp = 2.0
    Ki = 0.5
    Kd = 0.01

    # Параметры эксперимента
    N_EXPERIMENTS = 10

    # Шум измерения
    NOISE_STD = 0.05

    # Величины уставки
    references = np.linspace(0.2, 2.0, N_EXPERIMENTS)

    # Размер выхода SNN
    SNN_OUTPUT_SHAPE = 32

    # Загрузка Altai
    altai = Altai()

    # Программный эмулятор
    altai.build("sin.json", "gm")

    # Загрузка модели
    encoder = load_model(
        "encoder_sin.keras",
        custom_objects={
            "heaviside_mod": heaviside
        }
    )
    snn_model = load_model(
        "snn_sin.keras",
        custom_objects={
            "TernaryDense": TernaryDense,
            "heaviside_mod": heaviside,
            "Clip": Clip
        }
    )
    decoder = load_model(
        "decoder_sin.keras"
    )

    #Загрузка параметров нормализации
    x_mean = np.load("x_mean.npy")
    x_std = np.load("x_std.npy")

    # Функция RMSE
    def calculate_rmse(reference, response):

        reference = np.array(reference)
        response = np.array(response)

        return np.sqrt(np.mean((reference - response) ** 2))

    # Анализ метрик
    def transition_metrics(signal, reference, dt):
        signal = np.array(signal)
        final_value = reference

        # Перерегулирование
        peak = np.max(signal)

        if abs(final_value) > 1e-6:
            overshoot = max(0, ((peak - final_value) / abs(final_value)) * 100)
        else:
            overshoot = np.nan

        # Время нарастания
        try:
            idx10 = np.where(signal >=0.1 * final_value)[0][0]
            idx90 = np.where(signal >=0.9 * final_value)[0][0]

            rise_time = (idx90 - idx10) * dt

        except:
            rise_time = np.nan

        # Время установления
        band = (0.05 *abs(final_value))
        outside = np.where(np.abs(signal - final_value) > band)[0]

        if len(outside):
            settling_time = (outside[-1]* dt)
        else:
            settling_time = 0.0

        return (overshoot,rise_time,settling_time)

    # Анализ эксперимента
    def evaluate_controller(response,reference,dt):
        target = reference[-1]

        overshoot, rise_time, settling_time = transition_metrics(response, target, dt)

        rmse = calculate_rmse(reference, response)

        return {"Overshoot": overshoot, "RiseTime": rise_time, "SettlingTime": settling_time, "RMSE": rmse}

    # Таблицы результатов
    pid_clean_results = []
    pid_noise_results = []
    snn_clean_results = []
    snn_noise_results = []

    # Данные для графиков экспериментов
    all_clean_results = []
    all_noise_results = []

    # Один эксперимент
    def run_experiment(experiment_id, use_noise=False):
        np.random.seed(experiment_id)

        # Состояния ПИД
        omega_pid = 0.0
        integral_pid = 0.0
        prev_error_pid = 0.0

        # Сосотояния SNN
        omega_snn = 0.0
        integral_snn = 0.0
        prev_error_snn = 0.0

        # Массивы
        time_arr = []
        reference_arr = []
        omega_pid_arr = []
        omega_snn_arr = []
        u_pid_arr = []
        u_snn_arr = []

        # Начальная уставка
        r = references[experiment_id]

        # Главный цикл
        for k in range(N):
            t = k * dt

            # ПИД
            if use_noise:
                omega_pid_meas = (omega_pid + np.random.normal(0, NOISE_STD))
            else:
                omega_pid_meas = omega_pid
            error_pid = (r - omega_pid_meas)
            d_error_pid = (error_pid - prev_error_pid) / dt
            integral_pid += (error_pid * dt)

            u_pid = (Kp * error_pid + Ki * integral_pid + Kd * d_error_pid)
            u_pid = np.clip(u_pid,-u_max,u_max)

            Ia_pid = (u_pid - ce * phi * omega_pid) / R
            domega_pid = (cm * phi * Ia_pid - B * omega_pid - M_load) / J
            omega_pid += (dt * domega_pid)
            prev_error_pid = (error_pid)

            # SNN
            if use_noise:
                omega_snn_meas = (omega_snn + np.random.normal(0,NOISE_STD))
            else:
                omega_snn_meas = omega_snn
            error_snn = (r - omega_snn_meas)
            d_error_snn = (error_snn - prev_error_snn) / dt
            integral_snn += (error_snn * dt)

            x_nn = np.array([[error_snn, d_error_snn, integral_snn, omega_snn_meas, r]])
            x_nn_norm = (x_nn - x_mean) / x_std

            # Encoder
            encoded = (encoder.predict_on_batch(x_nn_norm))

            # Altai
            altai.prepare_spikes((encoded >= 0.5).astype(np.int32))

            altai.start_ticks(1)
            spikes_idx = (altai.get_spikes())
            spikes_idx = spikes_idx[spikes_idx != -2147483647]
            spikes = np.zeros(SNN_OUTPUT_SHAPE)
            spikes[spikes_idx] = 1

            # Decoder
            decoded = (decoder.predict_on_batch(spikes.reshape(1,-1)))

            altai.clear_input()

            u_snn = float(decoded[0][0])

            u_snn = np.clip(u_snn,-u_max,u_max)
            Ia_snn = (u_snn - ce * phi * omega_snn) / R
            domega_snn = (cm * phi * Ia_snn - B * omega_snn - M_load) / J
            omega_snn += (dt * domega_snn)
            prev_error_snn = (error_snn)

            # Сохранение
            time_arr.append(t)
            reference_arr.append(r)
            omega_pid_arr.append(omega_pid)
            omega_snn_arr.append(omega_snn)
            u_pid_arr.append(u_pid)
            u_snn_arr.append(u_snn)

        # Метрики
        pid_metrics = (evaluate_controller(omega_pid_arr, reference_arr,dt))
        snn_metrics = (evaluate_controller(omega_snn_arr,reference_arr,dt))

        return {
            "time": time_arr,
            "reference": reference_arr,
            "omega_pid": omega_pid_arr,
            "omega_snn": omega_snn_arr,
            "u_pid": u_pid_arr,
            "u_snn": u_snn_arr,
            "pid_metrics": pid_metrics,
            "snn_metrics":snn_metrics
        }

    # Эксперименты без шума
    print()
    print("RUNNING CLEAN EXPERIMENTS")
    print()

    for exp_id in tqdm(range(N_EXPERIMENTS)):
        result = run_experiment(experiment_id=exp_id,use_noise=False)

        pid_clean_results.append([
            exp_id + 1,
            result["pid_metrics"]["Overshoot"],
            result["pid_metrics"]["RiseTime"],
            result["pid_metrics"]["SettlingTime"],
            result["pid_metrics"]["RMSE"]
        ])

        snn_clean_results.append([
            exp_id + 1,
            result["snn_metrics"]["Overshoot"],
            result["snn_metrics"]["RiseTime"],
            result["snn_metrics"]["SettlingTime"],
            result["snn_metrics"]["RMSE"]
        ])

        # Сохраняем запуск
        all_clean_results.append(result)

    # Эксперименты с шумом
    print()
    print("RUNNING NOISY EXPERIMENTS")
    print()
    for exp_id in tqdm(range(N_EXPERIMENTS)):
        result = run_experiment(experiment_id=exp_id,use_noise=True)

        pid_noise_results.append([
            exp_id + 1,
            result["pid_metrics"]["Overshoot"],
            result["pid_metrics"]["RiseTime"],
            result["pid_metrics"]["SettlingTime"],
            result["pid_metrics"]["RMSE"]
        ])

        snn_noise_results.append([
            exp_id + 1,
            result["snn_metrics"]["Overshoot"],
            result["snn_metrics"]["RiseTime"],
            result["snn_metrics"]["SettlingTime"],
            result["snn_metrics"]["RMSE"]])

        # Сохраняем запуск
        all_noise_results.append(result)

    # Dataframe
    columns = ["Experiment", "Overshoot (%)", "Rise Time (s)", "Settling Time (s)", "RMSE"]

    pid_clean_df = pd.DataFrame(pid_clean_results, columns=columns)
    snn_clean_df = pd.DataFrame(snn_clean_results,columns=columns)
    pid_noise_df = pd.DataFrame(pid_noise_results,columns=columns)
    snn_noise_df = pd.DataFrame(snn_noise_results,columns=columns)

    # Средние значения
    def add_mean_row(df):
        mean_row = {
            "Experiment": "Mean",
            "Overshoot (%)": df["Overshoot (%)"].mean(),
            "Rise Time (s)": df["Rise Time (s)"].mean(),
            "Settling Time (s)": df["Settling Time (s)"].mean(),
            "RMSE":df["RMSE"].mean()
        }

        return pd.concat([df,pd.DataFrame([mean_row])],ignore_index=True)

    pid_clean_df = (add_mean_row(pid_clean_df))
    snn_clean_df = (add_mean_row(snn_clean_df))
    pid_noise_df = (add_mean_row(pid_noise_df))
    snn_noise_df = (add_mean_row(snn_noise_df))

    # Вывод таблиц
    print()
    print("=" * 60)
    print("PID WITHOUT NOISE")
    print("=" * 60)
    print(pid_clean_df)

    print()
    print("=" * 60)
    print("SNN WITHOUT NOISE")
    print("=" * 60)
    print(snn_clean_df)

    print()
    print("=" * 60)
    print("PID WITH NOISE")
    print("=" * 60)
    print(pid_noise_df)

    print()
    print("=" * 60)
    print("SNN WITH NOISE")
    print("=" * 60)
    print(snn_noise_df)

    # Сохранение в CSV
    pid_clean_df.to_csv("pid_clean_results.csv", index=False)
    snn_clean_df.to_csv("snn_clean_results.csv",index=False)
    pid_noise_df.to_csv("pid_noise_results.csv",index=False)
    snn_noise_df.to_csv("snn_noise_results.csv",index=False)

    # График скоростей без шума
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()
    for i, result in enumerate(all_clean_results):
        axes[i].plot(
            result["time"],
            result["reference"],
            "k--",
            label="Reference"
        )
        axes[i].plot(
            result["time"],
            result["omega_pid"],
            label="PID"
        )
        axes[i].plot(
            result["time"],
            result["omega_snn"],
            label="SNN"
        )
        axes[i].set_title(
            f"Experiment {i + 1} (r={references[i]:.1f})"
        )
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    plt.savefig("tracking.png", dpi=300)
    plt.show()

    # График управляющих сигналов без шума
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()
    for i, result in enumerate(all_clean_results):
        axes[i].plot(
            result["time"],
            result["u_pid"],
            label="PID"
        )
        axes[i].plot(
            result["time"],
            result["u_snn"],
            label="SNN"
        )
        axes[i].set_title(
            f"Experiment {i + 1} (r={references[i]:.1f})"
        )
        axes[i].grid(True)

        if i == 0:
            axes[i].legend()
    plt.tight_layout()
    plt.savefig("all_control.png", dpi=300)
    plt.show()

    # График скоростей с шумом
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()
    for i, result in enumerate(all_noise_results):
        axes[i].plot(
            result["time"],
            result["reference"],
            "k--",
            label="Reference"
        )
        axes[i].plot(
            result["time"],
            result["omega_pid"],
            label="PID"
        )
        axes[i].plot(
            result["time"],
            result["omega_snn"],
            label="SNN"
        )
        axes[i].set_title(
            f"Experiment {i + 1} (r={references[i]:.1f})"
        )
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()

    # График управляющих сигналов с шумом
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()
    for i, result in enumerate(all_noise_results):
        axes[i].plot(
            result["time"],
            result["u_pid"],
            label="PID"
        )
        axes[i].plot(
            result["time"],
            result["u_snn"],
            label="SNN"
        )
        axes[i].set_title(
            f"Experiment {i + 1} (r={references[i]:.1f})"
        )
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()
    plt.tight_layout()
    plt.savefig("all_control.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
