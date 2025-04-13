from plotting_Toellke import *
from pathlib import Path
import numpy as np
from typing import Tuple, List, Dict
import re

FILE_PATH: Path = Path(__file__).resolve()


def read_lfp(file_path: Path) -> np.ndarray:
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
        for idx, num in enumerate(data):
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


def read_folders(folders_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Lee archivos 'LFP.txt' en cada subdirectorio dentro de la ruta especificada y los une en un arreglo de NumPy.

    Parámetros:
        folders_path (Path): Ruta al directorio que contiene subdirectorios donde se buscará el archivo 'LFP.txt'. 

    Retorna:
        np.ndarray: Matriz de valores donde cada fila corresponde a los valores contenidos en un archivo 'LFP.txt' encontrado.
    """

    # Numpy array vacío para almacenar los datos
    data = np.array([])
    folders_name = []

    # Recorremos cada subdirectorio para buscar el archivo 'LFP.txt'
    for folder in folders_path.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.name == "LFP.txt":
                    # Guardamos el nombre de la carpeta
                    folders_name.append(folder.name)
                    lfp_values = read_lfp(file)
                    # Guardamos los valores en el array:
                    # Si el array está vacío, inicializamos con los valores leídos
                    # Si no está vacío, apilamos los nuevos valores
                    if data.size == 0:
                        data = lfp_values
                    else:
                        data = np.vstack((data, lfp_values))

    return data, folders_name


def main():
    print("Leyendo archivos LFP...")
    data_folders = FILE_PATH.parent.parent / "results" / "all_parameters"
    lfps, folders_name = read_folders(data_folders)

    print("Guardando resultados...")
    events_list = []
    all_spectrum_peaks_list = []
    sharp_wave_ripples_list = []
    band_spectra_list = []
    for lfp in lfps:
        # Se elimina la media del array de resultados para centrar la señal en cero
        lfp = lfp - np.mean(lfp)

        # Aplicamos la función de detección de eventos
        events_data, ripples_data, band_spectra = event_detection(lfp)
        events, _, all_spectrum_peaks, _ = events_data
        sharp_wave_ripples, _, _ = ripples_data

        # Guardamos los resultados en listas
        events_list.append(events)
        all_spectrum_peaks_list.append(all_spectrum_peaks)
        sharp_wave_ripples_list.append(sharp_wave_ripples)
        band_spectra_list.append(band_spectra)

    # Creamos un diccionario para guardar las direcciones de los archivos según
    # el valor de los parámetros
    parameters_groups: Dict[List[str]] = {}
    for idx in range(16):
        parameters_groups[idx] = []

    # Iteremos un bucle para agrupar los archivos
    healthy_parameters = [10000, 60, 1, 0.05]
    for idx, name in enumerate(folders_name):
        parameters: List[str] = re.split(r"[-_]", name)
        parameters: List[float] = [float(p) for p in parameters if '0' in p]
        if parameters == healthy_parameters:
            parameters_groups[0].append(idx)
        elif [parameters[i] for i in (1, 2, 3)] == [healthy_parameters[i] for i in (1, 2, 3)]:
            parameters_groups[1].append(idx)
        elif [parameters[i] for i in (0, 2, 3)] == [healthy_parameters[i] for i in (0, 2, 3)]:
            parameters_groups[2].append(idx)
        elif [parameters[i] for i in (0, 1, 3)] == [healthy_parameters[i] for i in (0, 1, 3)]:
            parameters_groups[3].append(idx)
        elif [parameters[i] for i in (0, 1, 2)] == [healthy_parameters[i] for i in (0, 1, 2)]:
            parameters_groups[4].append(idx)
        elif [parameters[i] for i in (0, 1)] == [healthy_parameters[i] for i in (0, 1)]:
            parameters_groups[5].append(idx)
        elif [parameters[i] for i in (0, 2)] == [healthy_parameters[i] for i in (0, 2)]:
            parameters_groups[6].append(idx)
        elif [parameters[i] for i in (0, 3)] == [healthy_parameters[i] for i in (0, 3)]:
            parameters_groups[7].append(idx)
        elif [parameters[i] for i in (1, 2)] == [healthy_parameters[i] for i in (1, 2)]:
            parameters_groups[8].append(idx)
        elif [parameters[i] for i in (1, 3)] == [healthy_parameters[i] for i in (1, 3)]:
            parameters_groups[9].append(idx)
        elif [parameters[i] for i in (2, 3)] == [healthy_parameters[i] for i in (2, 3)]:
            parameters_groups[10].append(idx)
        elif parameters[0] == healthy_parameters[0]:
            parameters_groups[11].append(idx)
        elif parameters[1] == healthy_parameters[1]:
            parameters_groups[12].append(idx)
        elif parameters[2] == healthy_parameters[2]:
            parameters_groups[13].append(idx)
        elif parameters[3] == healthy_parameters[3]:
            parameters_groups[14].append(idx)
        else:
            parameters_groups[15].append(idx)

    # Organizamos los resultados para poder graficarlos
    all_spectrum_peaks = {}
    band_spectra = {}
    for group, idx in parameters_groups.items():
        all_spectrum_peaks[group] = []
        band_spectra[group] = []
        if group != 0:
            spectrum_values = (f"{23}", all_spectrum_peaks_list[23])
            all_spectrum_peaks[group].append(spectrum_values)
            band_values = (f"{23}", band_spectra_list[23])
            band_spectra[group].append(band_values)
        for i in idx:
            spectrum_values = (f"{i}", all_spectrum_peaks_list[i])
            all_spectrum_peaks[group].append(spectrum_values)
            band_values = (f"{i}", band_spectra_list[i])
            band_spectra[group].append(band_values)

    # Graficamos los resultados
    print("Realizando gráficas...")
    for group in parameters_groups.keys():
        plot_peak_frequencies(all_spectrum_peaks[group], f"group_{group}")
        plot_power_spectral_density_bands(
            band_spectra[group], f"group_{group}")


if __name__ == "__main__":
    main()
