from NN_Learning import main as train_model
from emulator import main as run_emulator
from knp_ann2snn import Placer

def place_model():

    placer = Placer(
        "snn_sin.keras",
        hw_config_path="/usr/local/lib/python3.12/dist-packages/"
                       "knp_ann2snn/placer_build/resources/"
                       "hw_config.yaml",
        log_placer_="sin_placer.log"
    )

    placer.run_placement("sin.json")

def main():

    #Обучение нейросети
    print("TRAINING...")
    train_model()

    #Размещение нейросети
    print("PLACEMENT...")
    place_model()

    #Запуск инференса
    print("EMULATION...")
    run_emulator()

    print("FINISHED")

if __name__ == "__main__":
    main()