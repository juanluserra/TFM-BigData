from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from Event_detection_Toellke import *
FILE_PATH: Path = Path(__file__).resolve()


def read_results(file_path: Path) -> np.ndarray:
    """
    Lee los resultados de un archivo de texto y los convierte en un array de numpy.
    Args:
        file_path (Path): La ruta del archivo de texto que contiene los datos.
    Returns:
        np.ndarray: Un array de numpy que contiene los datos leídos y convertidos a float.
    El archivo de texto debe contener números en notación científica separados por comas,
    y los números deben estar entre corchetes. Por ejemplo: "[1.23E4, 5.67E8, ...]".
    """

    # Leemos el archivo de texto
    with open(file_path, 'r') as file:
        # Leemos el contenido y lo separamos por comas
        data = file.read().strip()
        data = data.strip('[]').split(',')
        
        # Convertimos los datos a float
        for idx,num in enumerate(data):
            # En el caso de no ser strings vacíos separamos la base y el exponente
            # y convertimos el número a float
            if num != '':
                base, exp = num.split('E')
                data[idx] = float(base) * 10**int(exp)
            # Si el string es vacío lo eliminamos
            else:
                data = np.delete(data, idx)
        
        # Devolver los datos en un array de numpy
        return np.array(data.astype(float))

def main() -> None:
    # Obtenemos la carpeta principal
    main_folder = FILE_PATH.parent.parent / 'TFM-LennartTollke' / 'LennartTollke-repository' / 'hippoSimUpdated' / 'sorted_output'
    
    # Obtenemos la dirección de la carpeta del TFM de Lennart
    tfm_folder = FILE_PATH.parent.parent / 'TFM-LennartTollke' / 'LennartTollke-repository'
    
    # Creamos un diccionario para guardar todos los archivos de texto
    txt_files = {}
    for txt_file in main_folder.rglob('*.txt'):
        txt_files[txt_file.name] = txt_file

    # Recorremos los archivos de texto con un bucle
    for key, file_path in txt_files.items():
        print(key)
        results = read_results(file_path)
        
        plt.plot(results, ".", alpha=0.25)
        plt.show()
        print(FILE_PATH.parent / f"{key}.png")
        
        # Aplicamos la funciónn de detección de eventos
        detection = event_detection(results)
        swrs = detection[1][0]
        print(len(swrs))
        
        # Pintamos cada uno de los SWR detectados
        for swr in swrs:
            plt.plot(swr)
        plt.show()
        
        break
    
if __name__ == '__main__':
    main()