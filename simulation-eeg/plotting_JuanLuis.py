import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from LennartToellke_files.Event_detection_Toellke import event_detection
from LennartToellke_files.plotting_Toellke import create_list_from_timeSeries
from pathlib import Path
from typing import *
import itertools

def data_to_dict(folder_path: str) -> Dict[str, List[Any]]:
    """
    Función para generar un diccionario con todas las características de las diferentes simulaciones
    que hay dentro de una carpeta. Se buscarán archivos LFP.txt.

    El diccionario será del tipo:

        {
            característica1: [simulacion1, simulacion2, ..., simulacionK],
            característica2: [simulacion1, simulacion2, ..., simulacionK],
            ...
            característicaN: [simulacion1, simulacion2, ..., simulacionK]
        }

    Args:
        folder_path (str): Ruta de la carpeta que contiene las carpetas de las simulaciones.

    Returns:
        Dict[str, List[Any]]: Diccionario con las características de las simulaciones.
    """

    # Creamos el diccionario que contendrá los datos de las simulaciones
    data_dict = {
        "event_list": [],
        "event_list_filtered": [],
        "event_peak_frequencies": [],
        "event_durations": [],
        "ripple_list": [],
        "ripple_peak_frequencies": [],
        "ripple_durations": [],
        "psd_theta": [],
        "psd_gamma": [],
        "psd_ripple": [],
    }

    # Recorremos todas las carpetas de la carpeta principal
    folder = Path(folder_path)
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            # Buscamos el archivo LFP.txt dentro de la carpeta
            lfp_file = subfolder / "LFP.txt"
            if lfp_file.exists():
                # Cargamos los datos del archivo LFP.txt en un array
                data = create_list_from_timeSeries(lfp_file)
                # Sacamos las características de la simulación
                event_characteristics, ripple_characteristics, psd_characteristics = event_detection(data)
                # Guardamos las características en el diccionario
                data_dict["event_list"].append(event_characteristics[0])
                data_dict["event_list_filtered"].append(event_characteristics[1])
                data_dict["event_peak_frequencies"].append(event_characteristics[2])
                data_dict["event_durations"].append(event_characteristics[3])
                data_dict["ripple_list"].append(ripple_characteristics[0])
                data_dict["ripple_peak_frequencies"].append(ripple_characteristics[1])
                data_dict["ripple_durations"].append(ripple_characteristics[2])
                data_dict["psd_theta"].append(psd_characteristics[0])
                data_dict["psd_gamma"].append(psd_characteristics[1])
                data_dict["psd_ripple"].append(psd_characteristics[2])
    
    return data_dict
                

def n_detects_single(data_path: str="", data_dict: Dict[str, Any]={}, figsize: tuple=(10,6)) -> None:
    """
    Función para generar los gráficos del número de eventos detectados en las simulaciones.

    Args:
        data_path (str): Ruta de la carpeta que contiene las simulaciones.
        data_dict (Dict[str, Any]): Diccionario con los datos de las simulaciones. Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    
    # Cargamos los datos del diccionario
    if data_path != "":
        data_dict = data_to_dict(data_path)
    elif data_dict == {}:
        raise ValueError("No se han proporcionado datos para generar los gráficos. Proporcione un diccionario con los datos o una ruta de carpeta válida.")
    
    # Calculamos los datos a graficar
    counts = [len(x) for x in data_dict["event_list"]]
    ripple_counts = [len(x) for x in data_dict["ripple_list"]]
    means = (
        np.mean(counts),
        np.mean(ripple_counts)
    )
    ste = (
        np.std(counts, ddof=0) / np.sqrt(len(counts)),
        np.std(ripple_counts, ddof=0) / np.sqrt(len(ripple_counts))
    )
    
    # Generamos el gráfico
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        ["Eventos", "SWRs"],
        means,
        yerr=ste,
        color=["blue", "green"],
        width=1.0,
        align="center",
        error_kw={'zorder': 2}, zorder=3
    )
    ax.set_ylabel("Número medio de eventos")
    ax.set_xlabel("Tipo de evento")
    ax.set_title("Número medio de eventos detectados en las simulaciones")
    ax.set_ylim(np.trunc(np.min(means + ste)/2), np.max(means + ste)+1)
    fig.tight_layout()
    fig.show()
    
def n_detects_multiple(
    data_path: List[str] = [], 
    data_dict: List[Dict[str, Any]] = [], 
    labels: List[str] = [],
    figsize: tuple = (10, 6)
    ) -> None:
    """
    Función para generar los gráficos del número de eventos detectados en varias simulaciones.

    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones. 
                                           Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    # Cargamos los datos (igual que en tu función single)
    if data_path != []:
        data_dict = [data_to_dict(path) for path in data_path]
    elif data_dict == []:
        raise ValueError(
            "No se han proporcionado datos para generar los gráficos. "
            "Proporcione una lista de diccionarios con los datos o una lista de rutas de carpetas válidas."
        )
    
    # Calculamos medias y errores para cada simulación
    medias = []
    estes = []
    for data in data_dict:
        counts = [len(x) for x in data["event_list"]]
        ripple_counts = [len(x) for x in data["ripple_list"]]

        # Media de "Eventos" y media de "SWRs"
        media_eventos = np.mean(counts)
        media_swrs = np.mean(ripple_counts)

        # STE de "Eventos" y STE de "SWRs"
        ste_eventos = np.std(counts, ddof=0) / np.sqrt(len(counts))
        ste_swrs = np.std(ripple_counts, ddof=0) / np.sqrt(len(ripple_counts))

        medias.append((media_eventos, media_swrs))
        estes.append((ste_eventos, ste_swrs))

    # Número de simulaciones
    n_sims = len(medias)

    # Convertimos a arrays de forma (n_sims, 2)
    medias_arr = np.array(medias)
    estes_arr = np.array(estes)

    # Etiquetas de las dos categorías
    categorias = ["Eventos", "SWRs"]
    n_cats = len(categorias)  # == 2

    # Anchura total disponible (80% del espacio) para las barras de cada grupo de categoría
    total_bar_width = 0.8
    bar_width = total_bar_width / n_sims

    # Posiciones base para las dos barras (0 y 1)
    x_base = np.arange(n_cats)  # array([0, 1])

    # Creamos figura y ejes
    fig, ax = plt.subplots(figsize=figsize)

    # Etiquetas de las simulaciones
    if labels == []:
        labels = [f"Sim {i+1}" for i in range(n_sims)]
        
    # Dibujamos las barras de cada simulación
    for i in range(n_sims):
        # Desplazamiento centrado:
        desplazamiento = (i - (n_sims - 1) / 2) * bar_width
        x_positions = x_base + desplazamiento

        alturas = medias_arr[i]   # [media_eventos, media_swrs]
        errores = estes_arr[i]    # [ste_eventos,   ste_swrs  ]

        ax.bar(
            x_positions,
            alturas,
            width=bar_width,
            yerr=errores,
            capsize=3,
            label=labels[i],
            error_kw={'zorder': 2}, zorder=3
        )

    # Ajustes de ejes y etiquetas
    ax.set_xticks(x_base)
    ax.set_xticklabels(categorias)
    ax.set_ylabel("Número medio de eventos")
    ax.set_xlabel("Tipo de evento")
    ax.set_title("Número medio de eventos detectados en las simulaciones")

    # Calculamos límites de y similares a tu single, pero usando todos los datos
    todas_medias = medias_arr.flatten()
    todos_ste = estes_arr.flatten()

    # Límite inferior: mitad de la mínima (media+ste) (truncado)
    y_min = np.trunc(np.min(todas_medias + todos_ste) / 2)
    # Si por alguna razón y_min da < 0, lo forzamos a 0
    if y_min < 0:
        y_min = 0

    # Límite superior: máximo de (media+ste) + 1
    y_max = np.max(todas_medias + todos_ste) + 1

    ax.set_ylim(y_min, y_max)

    # Mostramos leyenda (una entrada por cada simulación)
    ax.legend(title="Simulaciones")

    fig.tight_layout()
    fig.show()
    


def duration_single(data_path: str="", data_dict: Dict[str, Any]={}, figsize: tuple=(10,6)) -> None:
    """
    Función para generar los gráficos de las duraciones de los eventos detectados en las simulaciones.

    Args:
        data_path (str): Ruta de la carpeta que contiene las simulaciones.
        data_dict (Dict[str, Any]): Diccionario con los datos de las simulaciones. Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    
    # Cargamos los datos del diccionario
    if data_path != "":
        data_dict = data_to_dict(data_path)
    elif data_dict == {}:
        raise ValueError("No se han proporcionado datos para generar los gráficos. Proporcione un diccionario con los datos o una ruta de carpeta válida.")
    
    # Generamos los datos a graficar
    means = (
        np.mean([np.mean(x) for x in data_dict["event_durations"]]),
        np.mean([np.mean(x) for x in data_dict["ripple_durations"]])
    )
    ste = (
        np.std([np.mean(x) for x in data_dict["event_durations"]], ddof=0) / np.sqrt(len(data_dict["event_durations"])),
        np.std([np.mean(x) for x in data_dict["ripple_durations"]], ddof=0) / np.sqrt(len(data_dict["ripple_durations"]))
    )
    
    # Generamos el gráfico
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        1,
        means[0],
        yerr=ste[0],
        fmt='o',
        markersize=10,
        color='blue',
        label='Eventos'        
    )
    ax.errorbar(
        1,
        means[1],
        yerr=ste[1],
        fmt='o',
        markersize=10,
        color='green',
        label='SWRs'
    )
    ax.set_ylabel("Duración media (ms)")
    ax.set_xlabel("Tipo de evento")
    ax.set_title("Duración media de los eventos detectados en las simulaciones")
    ax.set_xticks([1])
    ax.set_xticklabels(["Eventos"])
    ax.legend()
    fig.show()


def duration_multiple(
    data_path: List[str] = [], 
    data_dict: List[Dict[str, Any]] = [], 
    labels: List[str] = [],
    figsize: tuple = (10, 6)
    ) -> None:
    """
    Función para generar los gráficos de las duraciones de los eventos detectados
    en varias simulaciones, conectando los puntos de 'Eventos' y 'SWRs' con dos líneas.

    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
                                           Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    # Cargamos los datos (igual que en la versión "single")
    if data_path != []:
        data_dict = [data_to_dict(path) for path in data_path]
    elif data_dict == []:
        raise ValueError(
            "No se han proporcionado datos para generar los gráficos. "
            "Proporcione una lista de diccionarios con los datos o una lista de rutas de carpetas válidas."
        )

    # Calculamos medias y errores para cada simulación
    medias = []  # aquí irá: [(media_eventos, media_swrs), …]
    estes = []   # aquí irá: [(ste_eventos, ste_swrs), …]
    for data in data_dict:
        # Listas de arrays para cada simulación
        evs = data["event_durations"]
        rips = data["ripple_durations"]

        # Media de medias (una media por simulación)
        media_eventos = np.mean([np.mean(x) if len(x) > 0 else 0.0 for x in evs])
        media_swrs    = np.mean([np.mean(x) if len(x) > 0 else 0.0 for x in rips])

        # Error estándar global: sqrt( Σ (σ_k/√n_k)² ) / N_sim
        # Si len(x) == 1, np.std(x, ddof=0) == 0, así que SEM_k = 0/1 = 0.
        # Para mayor robustez, comprobamos explícitamente len(x) > 1.
        ste_eventos = (
            np.sqrt(
                sum(
                    (np.std(x, ddof=0) / np.sqrt(len(x)))**2
                    if len(x) > 1 else 0.0
                    for x in evs
                )
            ) / len(evs)
        )
        ste_swrs = (
            np.sqrt(
                sum(
                    (np.std(x, ddof=0) / np.sqrt(len(x)))**2
                    if len(x) > 1 else 0.0
                    for x in rips
                )
            ) / len(rips)
        )
        medias.append((media_eventos, media_swrs))
        estes.append((ste_eventos, ste_swrs))

    n_sims = len(medias)
    # Convertimos a arrays de forma (n_sims, 2)
    medias_arr = np.array(medias)  # cada fila: [media_eventos, media_swrs]
    estes_arr = np.array(estes)    # cada fila: [ste_eventos,   ste_swrs ]

    # Definimos posiciones en el eje x para cada simulación
    x = np.arange(n_sims)  # [0, 1, 2, …, n_sims-1]

    # Separamos los valores para facilitar la llamada a errorbar
    y_eventos = medias_arr[:, 0]
    y_swrs = medias_arr[:, 1]
    err_eventos = estes_arr[:, 0]
    err_swrs = estes_arr[:, 1]

    # Etiquetas de las  simulacines
    if labels == []:
        labels = [f"Sim {i+1}" for i in range(n_sims)]
    
    # Creamos figura y ejes
    fig, ax = plt.subplots(figsize=figsize)

    # Dibujamos la línea de 'Eventos' con puntos y barras de error
    ax.errorbar(
        x,
        y_eventos,
        yerr=err_eventos,
        fmt='-o',
        markersize=8,
        color='blue',
        label='Eventos',
        capsize=3,
        zorder=3
    )

    # Dibujamos la línea de 'SWRs' con puntos y barras de error
    ax.errorbar(
        x,
        y_swrs,
        yerr=err_swrs,
        fmt='-o',
        markersize=8,
        color='green',
        label='SWRs',
        capsize=3,
        zorder=3
    )

    # Etiquetas del eje x: "Sim 1", "Sim 2", …
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Simulación")
    ax.set_ylabel("Duración media (ms)")
    ax.set_title("Duración media de los eventos detectados\n(en varias simulaciones)")
    ax.legend(title="Categoría")
    fig.tight_layout()
    fig.show()

    
def peak_freqs_single(data_path: str="", data_dict: Dict[str, Any]={}, figsize: tuple=(10,6)) -> None:
    """
    Función para generar los gráficos de las frecuencias pico de los eventos detectados en las simulaciones.

    Args:
        data_path (str): Ruta de la carpeta que contiene las simulaciones.
        data_dict (Dict[str, Any]): Diccionario con los datos de las simulaciones. Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    
    # Cargamos los datos del diccionario
    if data_path != "":
        data_dict = data_to_dict(data_path)
    elif data_dict == {}:
        raise ValueError("No se han proporcionado datos para generar los gráficos. Proporcione un diccionario con los datos o una ruta de carpeta válida.")
    
    # Generamos los datos a graficar
    means = (
        np.mean([np.mean(x) for x in data_dict["event_peak_frequencies"]]),
        np.mean([np.mean(x) for x in data_dict["ripple_peak_frequencies"]])
    )
    ste = (
        np.std([np.mean(x) for x in data_dict["event_peak_frequencies"]], ddof=0) / np.sqrt(len(data_dict["event_peak_frequencies"])),
        np.std([np.mean(x) for x in data_dict["ripple_peak_frequencies"]], ddof=0) / np.sqrt(len(data_dict["ripple_peak_frequencies"]))
    )
    
    # Generamos el gráfico
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        1,
        means[0],
        yerr=ste[0],
        fmt='o',
        markersize=10,
        color='blue',
        label='Eventos'        
    )
    ax.errorbar(
        1,
        means[1],
        yerr=ste[1],
        fmt='o',
        markersize=10,
        color='green',
        label='SWRs'
    )
    ax.set_ylabel("Frecuencia pico (Hz)")
    ax.set_xlabel("Tipo de evento")
    ax.set_title("Frecuencia pico media de los eventos detectados en las simulaciones")
    ax.set_xticks([1])
    ax.set_xticklabels(["Eventos"])
    ax.legend()
    fig.show()


def peak_freqs_multiple(
    data_path: List[str] = [],
    data_dict: List[Dict[str, Any]] = [],
    labels: List[str] = [],
    figsize: tuple = (10, 6)
) -> None:
    """
    Función para generar los gráficos de las frecuencias pico de los eventos detectados
    en varias simulaciones, conectando los puntos de 'Eventos' y 'SWRs' con dos líneas.

    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
                                           Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    # Cargamos los datos (igual que en la versión "single")
    if data_path != []:
        data_dict = [data_to_dict(path) for path in data_path]
    elif data_dict == []:
        raise ValueError(
            "No se han proporcionado datos para generar los gráficos. "
            "Proporcione una lista de diccionarios con los datos o una lista de rutas de carpetas válidas."
        )

    # Calculamos medias y errores para cada simulación
    medias = []  # [(media_eventos, media_swrs), …]
    estes = []   # [(ste_eventos, ste_swrs), …]

    for data in data_dict:
        # Listas de arrays para cada simulación
        evpf = data.get("event_peak_frequencies", [])
        rpf = data.get("ripple_peak_frequencies", [])

        # 2.1) Media de medias (una media por grabación)
        if len(evpf) > 0:
            media_eventos = np.mean([np.mean(x) if len(x) > 0 else 0.0 for x in evpf])
        else:
            media_eventos = 0.0

        if len(rpf) > 0:
            media_swrs = np.mean([np.mean(x) if len(x) > 0 else 0.0 for x in rpf])
        else:
            media_swrs = 0.0

        # Error estándar global: sqrt( Σ (σ_k/√n_k)² ) / N_grabaciones
        if len(evpf) > 0:
            ste_eventos = (
                np.sqrt(
                    sum(
                        (np.std(x, ddof=0) / np.sqrt(len(x)))**2
                        if len(x) > 1 else 0.0
                        for x in evpf
                    )
                ) / len(evpf)
            )
        else:
            ste_eventos = 0.0

        if len(rpf) > 0:
            ste_swrs = (
                np.sqrt(
                    sum(
                        (np.std(x, ddof=0) / np.sqrt(len(x)))**2
                        if len(x) > 1 else 0.0
                        for x in rpf
                    )
                ) / len(rpf)
            )
        else:
            ste_swrs = 0.0

        medias.append((media_eventos, media_swrs))
        estes.append((ste_eventos, ste_swrs))

    # Graficamos los resultados
    n_sims = len(medias)
    medias_arr = np.array(medias)  # cada fila: [media_eventos, media_swrs]
    estes_arr = np.array(estes)    # cada fila: [ste_eventos,   ste_swrs ]

    # Posiciones en el eje x para cada simulación
    x = np.arange(n_sims)  # [0, 1, 2, …, n_sims-1]

    # Valores y errores por categoría
    y_eventos = medias_arr[:, 0]
    y_swrs = medias_arr[:, 1]
    err_eventos = estes_arr[:, 0]
    err_swrs = estes_arr[:, 1]

    # Etiquetas de las simulaciones
    if labels == []:
        labels = [f"Sim {i+1}" for i in range(n_sims)]
    
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=figsize)

    # Línea de 'Eventos' con puntos y barras de error
    ax.errorbar(
        x,
        y_eventos,
        yerr=err_eventos,
        fmt='-o',
        markersize=8,
        color='blue',
        label='Eventos',
        capsize=3,
        zorder=3
    )

    # Línea de 'SWRs' con puntos y barras de error
    ax.errorbar(
        x,
        y_swrs,
        yerr=err_swrs,
        fmt='-o',
        markersize=8,
        color='green',
        label='SWRs',
        capsize=3,
        zorder=3
    )

    # Etiquetas del eje x: "Sim 1", "Sim 2", …
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Simulación")
    ax.set_ylabel("Frecuencia pico (Hz)")
    ax.set_title("Frecuencia pico media de los eventos detectados\n(en varias simulaciones)")

    # Cálculo de límites de y si se necesitara (opcional):
    all_sums = medias_arr.flatten() + estes_arr.flatten()

    ax.legend(title="Categoría")
    fig.tight_layout()
    fig.show()


def psd_plots_single(data_path: str="", data_dict: Dict[str, Any]={}, figsize: tuple=(10,6)) -> None:
    """
    Función para generar los gráficos de las densidades espectrales de potencia (PSD) de las simulaciones.

    Args:
        data_path (str): Ruta de la carpeta que contiene las simulaciones.
        data_dict (Dict[str, Any]): Diccionario con los datos de las simulaciones. Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    
    # Definimos los colores para cada tipo de PSD
    colors = {
        "theta": "orange",
        "gamma": "green",
        "ripple": "blue"
    }
    
    # Cargamos los datos del diccionario
    if data_path != "":
        data_dict = data_to_dict(data_path)
    elif data_dict == {}:
        raise ValueError("No se han proporcionado datos para generar los gráficos. Proporcione un diccionario con los datos o una ruta de carpeta válida.")
    
    # Generamos los datos a graficar
    means = (
        np.mean([np.mean(x) for x in data_dict["psd_theta"]]),
        np.mean([np.mean(x) for x in data_dict["psd_gamma"]]),
        np.mean([np.mean(x) for x in data_dict["psd_ripple"]])
    )
    ste = (
        np.sqrt(np.sum([(np.std(x)/np.sqrt(len(x)))**2 for x in data_dict["psd_theta"]])) / len(data_dict["psd_theta"]),
        np.sqrt(np.sum([(np.std(x)/np.sqrt(len(x)))**2 for x in data_dict["psd_gamma"]])) / len(data_dict["psd_gamma"]),
        np.sqrt(np.sum([(np.std(x)/np.sqrt(len(x)))**2 for x in data_dict["psd_ripple"]])) / len(data_dict["psd_ripple"])
    )
    
    # Generamos el gráfico
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        ["Theta", "Gamma", "Ripple"],
        means,
        yerr=ste,
        color=[colors["theta"], colors["gamma"], colors["ripple"]],
        width=1.0,
        align="center",
        error_kw={'zorder': 2}, zorder=3
    )
    ax.set_ylabel("Potencia media (uV^2/Hz)")
    ax.set_xlabel("Banda de frecuencia")
    ax.set_title("Densidad espectral de potencia media de las simulaciones")
    ax.set_yscale("log")
    ax.set_ylim(np.min(means)*1e-2, np.max(means + ste)*10)
    fig.tight_layout()
    fig.show()


def psd_plots_multiple(
    data_path: List[str]=[], 
    data_dict: List[Dict[str, Any]]=[], 
    labels: List[str]=[],
    figsize: tuple=(10,6)
    ) -> None:
    """
    Función para generar los gráficos de las densidades espectrales de potencia (PSD) de las simulaciones.

    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones. Se ignora si se proporciona `data_path`.
        figsize (tuple): Tamaño de la figura del gráfico.
    """
    
    # Definimos los colores para cada tipo de PSD
    colors = {
        "theta": "orange",
        "gamma": "green",
        "ripple": "blue"
    }
    
    if data_path != []:
        data_dict = [data_to_dict(path) for path in data_path]
    elif data_dict == []:
        raise ValueError("No se han proporcionado datos para generar los gráficos. Proporcione una lista de diccionarios con los datos o una lista de rutas de carpetas válidas.")
    
    # Generamos los datos a graficar
    means = []
    stes = []
    for data in data_dict:
        means.append((
            np.mean([np.mean(x) for x in data["psd_theta"]]),
            np.mean([np.mean(x) for x in data["psd_gamma"]]),
            np.mean([np.mean(x) for x in data["psd_ripple"]])
        ))
        stes.append((
            np.sqrt(np.sum([(np.std(x)/np.sqrt(len(x)))**2 for x in data["psd_theta"]])) / len(data["psd_theta"]),
            np.sqrt(np.sum([(np.std(x)/np.sqrt(len(x)))**2 for x in data["psd_gamma"]])) / len(data["psd_gamma"]),
            np.sqrt(np.sum([(np.std(x)/np.sqrt(len(x)))**2 for x in data["psd_ripple"]])) / len(data["psd_ripple"])
        ))
    
    # Generamos el gráfico de barras
    n_sims = len(means)
    means_arr = np.array(means)
    stes_arr  = np.array(stes) 
    
    # Nombre de las tres bandas
    bandas = ["Theta", "Gamma", "Ripple"]
    n_bandas = len(bandas)
    
    # Definimos el ancho de barra para cada simulación dentro de cada banda
    # Dejamos un 80% de espacio total para las barras, y el resto (20%) para separación
    total_bar_width = 0.8
    bar_width = total_bar_width / n_sims
    
    # Posiciones base para los tres grupos (una posición por banda)
    x_base = np.arange(n_bandas)
    
    # Etiquetas de las simulaciones
    if labels == []:
        labels = [f"Sim {i+1}" for i in range(n_sims)]
    
    # Creamos la figura y los ejes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Para cada simulación, dibujamos sus 3 barras (una por banda), desplazadas adecuadamente
    for i in range(n_sims):
        # Calculamos el desplazamiento de la i-ésima simulación dentro de cada grupo de banda
        # Centrado en el tick de x_base:
        desplazamiento = (i - (n_sims - 1) / 2) * bar_width
        x_positions = x_base + desplazamiento
        
        # Alturas (medias) y errores estándar de la simulación i
        alturas = means_arr[i]
        errores = stes_arr[i]
        
        # Dibujamos las barras: le damos un color distinto a cada simulación (ciclo por defecto de Matplotlib)
        ax.bar(
            x_positions,
            alturas,
            width=bar_width,
            yerr=errores,
            capsize=3,
            label=labels[i],
            error_kw={'zorder': 2}, zorder=3
        )
    
    # Ajustes de ejes y etiquetas
    ax.set_xticks(x_base)
    ax.set_xticklabels(bandas)
    ax.set_ylabel("Potencia media (uV^2/Hz)")
    ax.set_xlabel("Banda de frecuencia")
    ax.set_title("Densidad espectral de potencia media de las simulaciones")
    ax.set_yscale("log")
    
    # Definimos el límite inferior y superior usando todos los datos
    todas_medias = means_arr.flatten()
    todos_ste   = stes_arr.flatten()
    # Para el límite inferior: tomamos el valor mínimo de medias y lo multiplicamos por 1e-2
    y_min = np.min(todas_medias) * 1e-2
    # Para el límite superior: valor máximo de (media + ste) multiplicado por 10
    y_max = np.max(todas_medias + todos_ste) * 10
    ax.set_ylim(y_min, y_max)
    
    # Mostramos la leyenda (una entrada por cada simulación)
    ax.legend(title="Simulaciones")
    
    fig.tight_layout()
    fig.show()
    
    