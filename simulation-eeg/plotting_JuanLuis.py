import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from event_detection_JuanLuis import event_detection
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
        "theta_list": [],
        "theta_peak_frequencies": [],
        "theta_durations": [],
        "gamma_list": [],
        "gamma_peak_frequencies": [],
        "gamma_durations": [],
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
                event_characteristics, theta_characteristics, gamma_characteristics, ripple_characteristics, psd_characteristics = event_detection(data)
                # Guardamos las características en el diccionario
                data_dict["event_list"].append(event_characteristics[0])
                data_dict["event_list_filtered"].append(event_characteristics[1])
                data_dict["event_peak_frequencies"].append(event_characteristics[2])
                data_dict["event_durations"].append(event_characteristics[3])
                data_dict["theta_list"].append(theta_characteristics[0])
                data_dict["theta_peak_frequencies"].append(theta_characteristics[1])
                data_dict["theta_durations"].append(theta_characteristics[2])
                data_dict["gamma_list"].append(gamma_characteristics[0])
                data_dict["gamma_peak_frequencies"].append(gamma_characteristics[1])
                data_dict["gamma_durations"].append(gamma_characteristics[2])
                data_dict["ripple_list"].append(ripple_characteristics[0])
                data_dict["ripple_peak_frequencies"].append(ripple_characteristics[1])
                data_dict["ripple_durations"].append(ripple_characteristics[2])
                data_dict["psd_theta"].append(psd_characteristics[0])
                data_dict["psd_gamma"].append(psd_characteristics[1])
                data_dict["psd_ripple"].append(psd_characteristics[2])
    
    return data_dict
                

def n_detects_plot(
    data_path: List[str] = [], 
    data_dict: List[Dict[str, Any]] = [], 
    labels: List[str] = [],
    figsize: tuple = (10, 6),
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos del número de eventos detectados en varias simulaciones,
    opcionalmente calculando y retornando límites automáticos, y/o dibujando líneas horizontales en lims.

    Args:
        data_path (List[str]): Lista de rutas de carpetas con las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
        labels (List[str]): Etiquetas para cada simulación.
        figsize (tuple): Tamaño de la figura.
        lims (Optional[Tuple[List[float], List[float], List[float]]]): Límites manuales para cada banda:
            (
              [l_inf_theta, l_sup_theta],
              [l_inf_gamma, l_sup_gamma],
              [l_inf_ripple, l_sup_ripple]
            )
        return_lims (bool): Si True, calcula y devuelve límites automáticos.

    Returns:
        Si return_lims=True, devuelve:
            (
              [min_theta, max_theta],
              [min_gamma, max_gamma],
              [min_ripple, max_ripple]
            )
        En otro caso, devuelve None.
    """
    # --- Carga de datos ---
    if data_path:
        data_dict = [data_to_dict(path) for path in data_path]
    elif not data_dict:
        raise ValueError(
            "No se han proporcionado datos para generar los gráficos. "
            "Proporcione data_dict o data_path."
        )

    # --- Cálculo de medias y errores ---
    medias, estes = [], []
    for data in data_dict:
        theta_counts  = [len(x) for x in data["theta_list"]]
        gamma_counts  = [len(x) for x in data["gamma_list"]]
        ripple_counts = [len(x) for x in data["ripple_list"]]

        media_theta = np.mean(theta_counts)
        media_gamma = np.mean(gamma_counts)
        media_ripple = np.mean(ripple_counts)

        ste_theta  = np.std(theta_counts,  ddof=0) / np.sqrt(len(theta_counts))
        ste_gamma  = np.std(gamma_counts,  ddof=0) / np.sqrt(len(gamma_counts))
        ste_ripple = np.std(ripple_counts, ddof=0) / np.sqrt(len(ripple_counts))

        medias.append((media_theta, media_gamma, media_ripple))
        estes.append((ste_theta,   ste_gamma,   ste_ripple))

    means_arr = np.array(medias)  # shape (n_sims, 3)
    stes_arr  = np.array(estes)   # shape (n_sims, 3)
    n_sims    = len(medias)
    bandas    = ["Theta", "Gamma", "Ripple"]

    # --- Cálculo de lims automáticos si se pide ---
    auto_lims = None
    if return_lims:
        mins = means_arr - stes_arr
        maxs = means_arr + stes_arr
        auto_lims = (
            [float(np.min(mins[:,0])), float(np.max(maxs[:,0]))],  # theta
            [float(np.min(mins[:,1])), float(np.max(maxs[:,1]))],  # gamma
            [float(np.min(mins[:,2])), float(np.max(maxs[:,2]))],  # ripple
        )

    # --- Preparación para las barras ---
    total_bar_width = 0.8
    bar_width       = total_bar_width / n_sims
    x_base          = np.arange(len(bandas))

    if not labels:
        labels = [f"Sim {i+1}" for i in range(n_sims)]

    # --- Generación del gráfico de barras ---
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_sims):
        offset      = (i - (n_sims - 1) / 2) * bar_width
        x_positions = x_base + offset
        alturas     = means_arr[i]
        errores     = stes_arr[i]
        ax.bar(
            x_positions,
            alturas,
            width=bar_width,
            yerr=errores,
            capsize=3,
            label=labels[i],
            error_kw={'zorder': 3},
            zorder=2
        )

    # --- Etiquetas y título ---
    ax.set_xticks(x_base)
    ax.set_xticklabels(bandas)
    ax.set_ylabel("Número medio de eventos")
    ax.set_xlabel("Tipo de evento")
    ax.set_title("Número medio de eventos detectados en las simulaciones")

    # --- Límites de eje y existentes ---
    todas_medias = means_arr.flatten()
    todos_ste    = stes_arr.flatten()
    y_min = np.trunc(np.min(todas_medias + todos_ste) / 2)
    y_min = max(y_min, 0)
    y_max = np.max(todas_medias + todos_ste) + 1
    ax.set_ylim(y_min, y_max)

    # --- Dibujar líneas rojas discontinuas en lims ---
    line_colors = ["blue", "orange", "green"]
    if lims is not None:
        for idx, color in enumerate(line_colors):
            low, high = lims[idx]
            ax.axhline(low,  linestyle='--', color=color, label='_nolegend_')
            ax.axhline(high, linestyle='--', color=color, label='_nolegend_')

    ax.legend(title="Simulaciones")
    fig.tight_layout()
    plt.show()

    # --- Retorno opcional de lims ---
    if return_lims:
        return auto_lims

    return None
    


def durations_plot(
    data_path: List[str] = [], 
    data_dict: List[Dict[str, Any]] = [], 
    labels: List[str] = [],
    figsize: tuple = (10, 6),
    lims: Optional[Tuple[list, list, list]] = None,
    return_lims: bool = False,
    ) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos de las duraciones de los eventos detectados
    en varias simulaciones, y opcionalmente calcular límites automáticos.

    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
                                           Se ignora si se proporciona `data_path`.
        labels (List[str]): Etiquetas para cada simulación.
        figsize (tuple): Tamaño de la figura del gráfico.
        lims (Optional[Tuple[list, list, list]]): Límites manuales para cada categoría:
            (
              [l_inf_theta, l_sup_theta],
              [l_inf_gamma, l_sup_gamma],
              [l_inf_ripple, l_sup_ripple]
            )
        return_lims (bool): Si True, calcula y retorna los límites automáticos.

    Returns:
        Si return_lims=True, devuelve:
            (
              [min_theta, max_theta],
              [min_gamma, max_gamma],
              [min_ripple, max_ripple]
            )
        En otro caso, devuelve None.
    """
    # --- Carga de datos ---
    if data_path:
        data_dict = [data_to_dict(path) for path in data_path]
    elif not data_dict:
        raise ValueError(
            "No se han proporcionado datos para generar los gráficos. "
            "Proporcione data_dict o data_path."
        )

    # --- Cálculo de medias y errores ---
    medias, estes = [], []
    for data in data_dict:
        theta = data["theta_durations"]
        gamma = data["gamma_durations"]
        rips  = data["ripple_durations"]

        media_theta = np.mean([np.mean(x) if len(x) else 0.0 for x in theta])
        media_gamma = np.mean([np.mean(x) if len(x) else 0.0 for x in gamma])
        media_swrs  = np.mean([np.mean(x) if len(x) else 0.0 for x in rips])

        def comp_ste(arrays):
            sems_sq = [
                (np.std(x, ddof=0) / np.sqrt(len(x)))**2 if len(x) > 1 else 0.0
                for x in arrays
            ]
            return np.sqrt(sum(sems_sq)) / len(arrays)

        ste_theta = comp_ste(theta)
        ste_gamma = comp_ste(gamma)
        ste_swrs  = comp_ste(rips)

        medias.append((media_theta, media_gamma, media_swrs))
        estes.append((ste_theta,   ste_gamma,   ste_swrs))

    medias_arr = np.array(medias)  # shape (n_sims, 3)
    estes_arr  = np.array(estes)   # shape (n_sims, 3)
    n_sims     = len(medias)
    x          = np.arange(n_sims)

    # --- Cálculo de lims automáticos si se pide ---
    auto_lims = None
    if return_lims:
        mins = medias_arr - estes_arr   # matrix of (media - ste)
        maxs = medias_arr + estes_arr   # matrix of (media + ste)
        auto_lims = (
            [float(np.min(mins[:,0])), float(np.max(maxs[:,0]))],  # theta
            [float(np.min(mins[:,1])), float(np.max(maxs[:,1]))],  # gamma
            [float(np.min(mins[:,2])), float(np.max(maxs[:,2]))],  # ripple
        )

    # --- Preparación de categorías y colores ---
    categorias = {
        "Theta":  (medias_arr[:, 0], estes_arr[:, 0], "blue"),
        "Gamma":  (medias_arr[:, 1], estes_arr[:, 1], "orange"),
        "Ripple": (medias_arr[:, 2], estes_arr[:, 2], "green"),
    }
    limits_map = {}
    if lims is not None:
        limits_map = {
            "Theta":  lims[0],
            "Gamma":  lims[1],
            "Ripple": lims[2],
        }

    # --- Etiquetas X ---
    if not labels:
        labels = [f"Sim {i+1}" for i in range(n_sims)]

    # --- Dibujado de gráficas ---
    for nombre, (y_vals, y_errs, color) in categorias.items():
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(
            x, y_vals, yerr=y_errs,
            fmt='-o', markersize=8,
            color=color, label=nombre,
            capsize=3, zorder=3
        )

        # Líneas de límite si existen
        if nombre in limits_map:
            low, high = limits_map[nombre]
            ax.axhline(low,  linestyle='--', color='red',   label='_nolegend_')
            ax.axhline(high, linestyle='--', color='red',   label='_nolegend_')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Simulación")
        ax.set_ylabel("Duración media (ms)")
        ax.set_title(f"Duración media de {nombre}\n(en varias simulaciones)")
        ax.legend(title="Categoría")
        fig.tight_layout()
        plt.show()

    # --- Retorno opcional de lims ---
    if return_lims:
        return auto_lims

    return None


def peak_freqs_plot(
    data_path: List[str] = [],
    data_dict: List[Dict[str, Any]] = [],
    labels: List[str] = [],
    figsize: tuple = (10, 6),
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos de las frecuencias pico de los eventos detectados
    en varias simulaciones, y opcionalmente calcular límites automáticos.

    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
                                           Se ignora si se proporciona `data_path`.
        labels (List[str]): Etiquetas para cada simulación.
        figsize (tuple): Tamaño de la figura del gráfico.
        lims (Optional[Tuple[List[float], List[float], List[float]]]): Límites manuales para cada categoría:
            (
              [l_inf_theta, l_sup_theta],
              [l_inf_gamma, l_sup_gamma],
              [l_inf_swr,    l_sup_swr]
            )
        return_lims (bool): Si True, calcula y retorna los límites automáticos.

    Returns:
        Si return_lims=True, devuelve:
            (
              [min_theta, max_theta],
              [min_gamma, max_gamma],
              [min_swr,   max_swr]
            )
        En otro caso, devuelve None.
    """
    # --- Carga de datos ---
    if data_path:
        data_dict = [data_to_dict(path) for path in data_path]
    elif not data_dict:
        raise ValueError(
            "No se han proporcionado datos para generar los gráficos. "
            "Proporcione data_dict o data_path."
        )

    # --- Cálculo de medias y errores ---
    medias, estes = [], []
    for data in data_dict:
        tpf = data.get("theta_peak_frequencies", [])
        gpf = data.get("gamma_peak_frequencies", [])
        rpf = data.get("ripple_peak_frequencies", [])

        # medias de medias
        media_theta = np.mean([np.mean(x) if len(x) else 0.0 for x in tpf]) if tpf else 0.0
        media_gamma = np.mean([np.mean(x) if len(x) else 0.0 for x in gpf]) if gpf else 0.0
        media_swr   = np.mean([np.mean(x) if len(x) else 0.0 for x in rpf]) if rpf else 0.0

        # función auxiliar para STE compuesto
        def comp_ste(arrays):
            sems_sq = [
                (np.std(x, ddof=0) / np.sqrt(len(x)))**2 if len(x) > 1 else 0.0
                for x in arrays
            ]
            return np.sqrt(sum(sems_sq)) / len(arrays) if arrays else 0.0

        ste_theta = comp_ste(tpf)
        ste_gamma = comp_ste(gpf)
        ste_swr   = comp_ste(rpf)

        medias.append((media_theta, media_gamma, media_swr))
        estes.append((ste_theta,   ste_gamma,   ste_swr))

    medias_arr = np.array(medias)  # shape (n_sims, 3)
    estes_arr  = np.array(estes)   # shape (n_sims, 3)
    n_sims     = len(medias)
    x          = np.arange(n_sims)

    # --- Cálculo de lims automáticos si se pide ---
    auto_lims = None
    if return_lims:
        mins = medias_arr - estes_arr
        maxs = medias_arr + estes_arr
        auto_lims = (
            [float(np.min(mins[:,0])), float(np.max(maxs[:,0]))],  # theta
            [float(np.min(mins[:,1])), float(np.max(maxs[:,1]))],  # gamma
            [float(np.min(mins[:,2])), float(np.max(maxs[:,2]))],  # swr
        )

    # --- Preparación de categorías y colores ---
    categorias = {
        "Theta": (medias_arr[:, 0], estes_arr[:, 0], "blue"),
        "Gamma": (medias_arr[:, 1], estes_arr[:, 1], "orange"),
        "SWRs":  (medias_arr[:, 2], estes_arr[:, 2], "green"),
    }
    limits_map = {}
    if lims is not None:
        limits_map = {
            "Theta": lims[0],
            "Gamma": lims[1],
            "SWRs":  lims[2],
        }

    # --- Etiquetas X ---
    if not labels:
        labels = [f"Sim {i+1}" for i in range(n_sims)]

    # --- Dibujado de gráficas ---
    for nombre, (y_vals, y_errs, color) in categorias.items():
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(
            x, y_vals, yerr=y_errs,
            fmt='-o', markersize=8,
            color=color, label=nombre,
            capsize=3, zorder=3
        )

        # Líneas de límite si existen
        if nombre in limits_map:
            low, high = limits_map[nombre]
            ax.axhline(low,  linestyle='--', color='red',   label='_nolegend_')
            ax.axhline(high, linestyle='--', color='red',   label='_nolegend_')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Simulación")
        ax.set_ylabel("Frecuencia pico (Hz)")
        ax.set_title(f"Frecuencia pico media de {nombre}\n(en varias simulaciones)")
        ax.legend(title="Categoría")
        fig.tight_layout()
        plt.show()

    # --- Retorno opcional de lims ---
    if return_lims:
        return auto_lims

    return None


def psd_plot(
    data_path: List[str] = [], 
    data_dict: List[Dict[str, Any]] = [], 
    labels: List[str] = [],
    figsize: tuple = (10, 6),
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos de las densidades espectrales de potencia (PSD) de las simulaciones,
    opcionalmente calculando retornando límites automáticos y/o dibujando líneas horizontales en lims.

    Args:
        data_path (List[str]): Lista de rutas de carpetas con las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
        labels (List[str]): Etiquetas para cada simulación.
        figsize (tuple): Tamaño de la figura.
        lims (Optional[Tuple[List[float], List[float], List[float]]]): Límites manuales para cada banda:
            (
              [l_inf_theta, l_sup_theta],
              [l_inf_gamma, l_sup_gamma],
              [l_inf_ripple, l_sup_ripple]
            )
        return_lims (bool): Si True, calcula y retorna límites automáticos.

    Returns:
        Si return_lims=True, devuelve tupla de listas:
            (
              [min_theta, max_theta],
              [min_gamma, max_gamma],
              [min_ripple, max_ripple]
            )
        En otro caso, devuelve None.
    """
    # --- Carga de datos ---
    if data_path:
        data_dict = [data_to_dict(path) for path in data_path]
    elif not data_dict:
        raise ValueError(
            "No se han proporcionado datos para generar los gráficos. "
            "Proporcione data_dict o data_path."
        )

    # --- Cálculo de medias y errores ---
    means, stes = [], []
    for data in data_dict:
        theta_psd  = data.get("psd_theta", [])
        gamma_psd  = data.get("psd_gamma", [])
        ripple_psd = data.get("psd_ripple", [])

        mean_theta  = np.mean([np.mean(x) for x in theta_psd])  if theta_psd  else 0.0
        mean_gamma  = np.mean([np.mean(x) for x in gamma_psd])  if gamma_psd  else 0.0
        mean_ripple = np.mean([np.mean(x) for x in ripple_psd]) if ripple_psd else 0.0

        def comp_ste(arrays):
            sems_sq = [
                (np.std(x, ddof=0) / np.sqrt(len(x)))**2
                for x in arrays
                if len(x) > 0
            ]
            return np.sqrt(sum(sems_sq)) / len(arrays) if arrays else 0.0

        ste_theta  = comp_ste(theta_psd)
        ste_gamma  = comp_ste(gamma_psd)
        ste_ripple = comp_ste(ripple_psd)

        means.append((mean_theta, mean_gamma, mean_ripple))
        stes.append((ste_theta,  ste_gamma,  ste_ripple))

    means_arr = np.array(means)  # shape (n_sims, 3)
    stes_arr  = np.array(stes)   # shape (n_sims, 3)
    n_sims    = len(means)
    bandas    = ["Theta", "Gamma", "Ripple"]

    # --- Cálculo de lims automáticos si se pide ---
    auto_lims = None
    if return_lims:
        mins = means_arr - stes_arr
        maxs = means_arr + stes_arr
        auto_lims = (
            [float(np.min(mins[:,0])), float(np.max(maxs[:,0]))],
            [float(np.min(mins[:,1])), float(np.max(maxs[:,1]))],
            [float(np.min(mins[:,2])), float(np.max(maxs[:,2]))],
        )

    # --- Preparación de valores para barras ---
    total_bar_width = 0.8
    bar_width = total_bar_width / n_sims
    x_base = np.arange(len(bandas))

    if not labels:
        labels = [f"Sim {i+1}" for i in range(n_sims)]

    # --- Generación del gráfico de barras ---
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_sims):
        offset = (i - (n_sims - 1) / 2) * bar_width
        x_positions = x_base + offset
        alturas = means_arr[i]
        errores = stes_arr[i]
        ax.bar(
            x_positions,
            alturas,
            width=bar_width,
            yerr=errores,
            capsize=3,
            label=labels[i],
            error_kw={'zorder': 3},
            zorder=2
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(bandas)
    ax.set_ylabel("Potencia media (uV^2/Hz)")
    ax.set_xlabel("Banda de frecuencia")
    ax.set_title("Densidad espectral de potencia media de las simulaciones")
    ax.set_yscale("log")

    # --- Límites de eje y existentes ---
    todas_medias = means_arr.flatten()
    todos_ste   = stes_arr.flatten()
    y_min = np.min(todas_medias) * 1e-2
    y_max = np.max(todas_medias + todos_ste) * 10
    ax.set_ylim(y_min, y_max)

    # --- Dibujar líneas rojas discontinuas en lims ---
    colors = ['blue', 'orange', 'green'] 
    if lims is not None:
        for i, (band, color) in enumerate(zip(bandas, colors)):
            low, high = lims[i]
            ax.axhline(low,  linestyle='--', color=color)
            ax.axhline(high, linestyle='--', color=color)

    ax.legend(title="Simulaciones")
    fig.tight_layout()
    plt.show()

    if return_lims:
        return auto_lims

    return None
    