from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from Event_detection_Toellke import *
import scipy.signal as signal
from plotting_Toellke import *
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


def test_plots(txt_files: dict) -> None:
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
            print(len(swr))
        plt.show()
        
        # Se elimina la media del array de resultados para centrar la señal en cero
        results = results - np.mean(results)
        
        # Se aplica un filtro Savitzky-Golay para suavizar la señal
        filtered_results = signal.savgol_filter(results, window_length=51, polyorder=3)
        # Se grafica la señal original y la señal filtrada para comparación
        plt.plot(results, label='Original', alpha=0.5)
        plt.plot(filtered_results, label='Filtered', alpha=0.75)
        plt.legend()
        plt.title('Original and Savitzky-Golay Filtered Results')
        plt.show()
           
        # Se sustrae la señal filtrada de la señal original para resaltar las diferencias
        results = results - filtered_results
        plt.plot(results)
        plt.title('Filtered Results')
        plt.show()
         
        # Se calcula el espectrograma de la señal resultante utilizando una frecuencia de muestreo de 1024 Hz
        f, t, Sxx = signal.spectrogram(results, fs=1024, nperseg=50)
        # Se recalcula el espectrograma sin especificar nperseg para obtener la configuración por defecto
        # f, t, Sxx = signal.spectrogram(results, fs=1024)
        # Se imprime la cantidad de frecuencias, tiempos y la forma del espectrograma
        print(len(f), len(t), Sxx.shape)
        # Se grafica el espectrograma en dB utilizando el mapa de colores 'seismic'
        plt.pcolormesh(f, t, 10 * np.log10(Sxx.T), cmap='seismic')
        plt.ylabel('Time [sec]')
        plt.xlabel('Frequency [Hz]')
        plt.title('Spectrogram')
        plt.colorbar(label='Intensity [dB]')
        plt.show()
        
        # Se calcula la FFT de la suma de la señal original y la filtrada
        fft_results = np.fft.fft(results + filtered_results)
        # Se obtiene el vector de frecuencias correspondiente a la FFT
        fft_freq = np.fft.fftfreq(len(results), d=1/1024)
        # Se seleccionan solo las frecuencias positivas (hasta la frecuencia de Nyquist)
        positive_freqs = fft_freq[:len(fft_freq)//2]
        positive_fft_results = np.abs(fft_results[:len(fft_results)//2])
        # Se grafica la magnitud de la FFT para las frecuencias positivas
        plt.plot(positive_freqs, positive_fft_results)

        # Se calcula la FFT exclusivamente de la señal resultante
        fft_results = np.fft.fft(results)
        fft_freq = np.fft.fftfreq(len(results), d=1/1024)
        positive_freqs = fft_freq[:len(fft_freq)//2]
        positive_fft_results = np.abs(fft_results[:len(fft_results)//2])
        # Se añade la gráfica de la FFT de la señal resultante a la misma figura
        plt.plot(positive_freqs, positive_fft_results)
        
        # Se configuran el título y las etiquetas de la gráfica de FFT y se muestra
        plt.title('FFT of Results (0 to Nyquist Frequency)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.show()
        
        break
    

def main() -> None:
    # Obtenemos la carpeta principal
    main_folder = FILE_PATH.parent.parent / 'TFM-LennartTollke' / 'LennartTollke-repository' / 'hippoSimUpdated' / 'sorted_output'
    
    # Obtenemos la dirección de la carpeta del TFM de Lennart
    tfm_folder = FILE_PATH.parent.parent / 'TFM-LennartTollke' / 'LennartTollke-repository'
    
    # Creamos un diccionario para guardar todos los archivos de texto
    txt_files = {}
    for txt_file in main_folder.rglob('*.txt'):
        txt_files[txt_file.name] = txt_file
    
    # Probamos algunas gráficas
    # test_plots(txt_files)
    
    # Seleccionamos si queremos realizar el bucle o no para leer los resultados de Lennart
    lennart_results = False
    
    # Realizamos las gráficas de Lennart
    if lennart_results:
        for key, file_path in txt_files.items():
            # Realizamos las gráficas e Lennart
            single_sim_analysis(str(file_path), True, False)
    
    # Seleccionamos el archivo de resultados propios
    folder = 'results_2025-03-11_12.39.10'
    own_results_path = FILE_PATH.parent.parent / folder / 'LFP.txt'
    single_sim_analysis(str(own_results_path), True, False)
    
        
        
    
    
    
       
if __name__ == '__main__':
    main()