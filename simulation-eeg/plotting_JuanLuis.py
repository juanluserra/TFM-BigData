import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from event_detection_JuanLuis import event_detection
from LennartToellke_files.plotting_Toellke import create_list_from_timeSeries
from pathlib import Path
from typing import *
import seaborn as sns
import pandas as pd
import math

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
    xrotation: int = 0,
    ncol: int = 1,
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar un gráfico de barras con la media y el error estándar de las
    densidades espectrales de potencia de las simulaciones.
    Args:
        data_path (List[str]): Lista de rutas a las carpetas que contienen los datos de las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
        labels (List[str]): Lista de etiquetas para las simulaciones.
        figsize (tuple): Tamaño de la figura del gráfico.
        lims (Optional[Tuple[List[float], List[float], List[float]]]): Límites para rellenar en el gráfico.
        return_lims (bool): Si se deben devolver los límites calculados automáticamente.
    Returns:
        Optional[Tuple[List[float], List[float], List[float]]]: Si `return_lims` es True, devuelve los límites calculados automáticamente.
    """
    # --- Carga de datos ---
    if data_path:
        data_dict = [data_to_dict(path) for path in data_path]
    elif not data_dict:
        raise ValueError("No se han proporcionado datos para generar los gráficos.")

    # --- Cálculo de medias y errores ---
    medias, estes = [], []
    for data in data_dict:
        theta_counts  = [len(x) for x in data["theta_list"]]
        gamma_counts  = [len(x) for x in data["gamma_list"]]
        ripple_counts = [len(x) for x in data["ripple_list"]]

        medias.append((
            np.mean(theta_counts),
            np.mean(gamma_counts),
            np.mean(ripple_counts)
        ))
        estes.append((
            np.std(theta_counts, ddof=0)/np.sqrt(len(theta_counts)),
            np.std(gamma_counts, ddof=0)/np.sqrt(len(gamma_counts)),
            np.std(ripple_counts, ddof=0)/np.sqrt(len(ripple_counts))
        ))

    means_arr = np.array(medias)
    stes_arr  = np.array(estes)
    n_sims    = len(means_arr)
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

    # --- Preparar DataFrame para seaborn ---
    records = []
    for sim_idx in range(n_sims):
        sim_label = labels[sim_idx] if labels else f"Sim {sim_idx+1}"
        for band_idx, banda in enumerate(bandas):
            records.append({
                "Simulación": sim_label,
                "Banda": banda,
                "Media": means_arr[sim_idx, band_idx],
                "STE": stes_arr[sim_idx, band_idx]
            })
    df = pd.DataFrame.from_records(records)

    # --- Plot con seaborn ---
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x="Banda", y="Media", hue="Simulación",
        ci=None, dodge=True
    )

    # --- Añadir barras de error manualmente ---
    # calcular offsets
    total_bar_width = 0.8
    bar_width = total_bar_width / n_sims
    x_base = np.arange(len(bandas))
    offsets = (np.arange(n_sims) - (n_sims-1)/2) * bar_width

    for i, sim_label in enumerate(df["Simulación"].unique()):
        sub = df[df["Simulación"] == sim_label]
        errs = sub["STE"].values
        xpos = x_base + offsets[i]
        ax.errorbar(
            xpos, sub["Media"].values,
            yerr=errs,
            fmt='none', capsize=3, ecolor='black', zorder=2
        )

    # --- Límites de eje y existentes ---
    todas_medias = means_arr.flatten()
    todos_ste    = stes_arr.flatten()
    y_min = max(np.trunc(np.min(todas_medias + todos_ste) / 2), 0)
    y_max = np.max(todas_medias + todos_ste) + 1
    ax.set_ylim(y_min, y_max)

    # --- Relleno y líneas en lims restringidas a cada banda ---
    if lims is not None:
        center_offset = (n_sims / 2) * bar_width * 1.15
        for idx, (low, high) in enumerate(lims):
            xmin = idx - center_offset
            xmax = idx + center_offset
            ax.fill_betweenx(
                [low, high],
                xmin, xmax,
                color='red', alpha=0.25, zorder=1
            )
            ax.hlines(low,  xmin, xmax, linestyles='-', colors='red', label='_nolegend_')
            ax.hlines(high, xmin, xmax, linestyles='-', colors='red', label='_nolegend_')

    label_font = {'fontweight': 'bold', 'fontsize': 14}
    ax.set_xlabel("Frequency band", **label_font)
    ax.set_ylabel("Mean number of events", **label_font)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles, labels,
        ncol=ncol,
        title="Experiments",
        title_fontsize=13,
        prop={"size": 13},
        frameon=True,
        framealpha=0.9
    )
    leg.get_title().set_fontweight('bold')
    plt.tight_layout()
    plt.show()

    if return_lims:
        return auto_lims

    return None
    


def durations_plot_points(
    data_path: List[str] = [], 
    data_dict: List[Dict[str, Any]] = [], 
    labels: List[str] = [],
    figsize: tuple = (10, 6),
    xrotation: int = 0,
    ncol: int = 1,
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos de las duraciones de los eventos detectados
    en varias simulaciones, usando seaborn para estilizar, y opcionalmente
    calculando y retornando límites automáticos, y/o dibujando rellenos y líneas en lims.
    """
    # --- Carga de datos ---
    if data_path:
        data_dict = [data_to_dict(path) for path in data_path]
    elif not data_dict:
        raise ValueError("No se han proporcionado datos para generar los gráficos.")

    # --- Cálculo de medias y errores ---
    medias, estes = [], []
    for data in data_dict:
        theta = data["theta_durations"]
        gamma = data["gamma_durations"]
        rips  = data["ripple_durations"]

        medias.append((
            np.mean([np.mean(x) if len(x) else 0.0 for x in theta]),
            np.mean([np.mean(x) if len(x) else 0.0 for x in gamma]),
            np.mean([np.mean(x) if len(x) else 0.0 for x in rips])
        ))
        def comp_ste(arrays):
            sems_sq = [
                (np.std(x, ddof=0)/np.sqrt(len(x)))**2 if len(x)>1 else 0.0
                for x in arrays
            ]
            return np.sqrt(sum(sems_sq))/len(arrays)
        estes.append((
            comp_ste(theta),
            comp_ste(gamma),
            comp_ste(rips)
        ))

    medias_arr = np.array(medias)  # (n_sims,3)
    estes_arr  = np.array(estes)
    n_sims     = len(medias_arr)
    x          = np.arange(n_sims)
    categorias = {
        "Theta":  (medias_arr[:,0], estes_arr[:,0], "blue"),
        "Gamma":  (medias_arr[:,1], estes_arr[:,1], "orange"),
        "Ripple": (medias_arr[:,2], estes_arr[:,2], "green"),
    }

    # --- Cálculo de lims automáticos si se pide ---
    auto_lims = None
    if return_lims:
        mins = medias_arr - estes_arr
        maxs = medias_arr + estes_arr
        auto_lims = (
            [float(np.min(mins[:,0])), float(np.max(maxs[:,0]))],
            [float(np.min(mins[:,1])), float(np.max(maxs[:,1]))],
            [float(np.min(mins[:,2])), float(np.max(maxs[:,2]))],
        )

    # --- Etiquetas X ---
    if not labels:
        labels = [f"Sim {i+1}" for i in range(n_sims)]

    # --- Seaborn styling ---
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # --- Dibujado de gráficas ---
    for nombre, (y_vals, y_errs, color) in categorias.items():
        fig, ax = plt.subplots(figsize=figsize)
        # Plot manual de líneas con puntos y errores
        ax.errorbar(
            x, y_vals, yerr=y_errs,
            fmt='-o', markersize=8,
            color=color, label=nombre,
            capsize=3, zorder=3
        )

        # Si hay límites, rellenar y dibujar líneas
        if lims is not None:
            low, high = lims[ list(categorias).index(nombre) ]
            xmin, xmax = x[0] - 0.5, x[-1] + 0.5
            ax.fill_betweenx(
                [low, high],
                xmin, xmax,
                color='red', alpha=0.25, zorder=1
            )
            ax.hlines(low,  xmin, xmax, linestyles='-', colors='red', label='_nolegend_')
            ax.hlines(high, xmin, xmax, linestyles='-', colors='red', label='_nolegend_')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        label_font = {'fontweight': 'bold', 'fontsize': 14}
        ax.set_xlabel("Frequency band", **label_font)
        ax.set_ylabel("Mean duration of events", **label_font)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles, labels,
            ncol=ncol,
            title="Experiments",
            title_fontsize=13,
            prop={"size": 13},
            frameon=True,
            framealpha=0.9
        )
        leg.get_title().set_fontweight('bold')
        fig.tight_layout()
        plt.show()

    if return_lims:
        return auto_lims

    return None


def durations_plot(
    data_path: List[str] = [], 
    data_dict: List[Dict[str, Any]] = [], 
    labels: List[str] = [],
    figsize: tuple = (10, 6),
    xrotation: int = 0,
    ncol: int = 1,
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos de las duraciones de los eventos detectados
    en varias simulaciones, usando seaborn para estilizar, y opcionalmente
    calculando y retornando límites automáticos, y/o dibujando rellenos y líneas en lims.
    
    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
        labels (List[str]): Etiquetas para cada simulación.
        figsize (tuple): Tamaño de la figura del gráfico.
        lims (Optional[Tuple[List[float], List[float], List[float]]]): Límites para rellenar y dibujar líneas horizontales.
        return_lims (bool): Si es True, retorna los límites calculados automáticamente.
    Returns:
        Optional[Tuple[List[float], List[float], List[float]]]: Si return_lims es True, retorna los límites calculados automáticamente.
    """
    # --- Carga de datos ---
    if data_path:
        data_dict = [data_to_dict(path) for path in data_path]
    elif not data_dict:
        raise ValueError("No se han proporcionado datos para generar los gráficos.")

    # --- Cálculo de medias y errores ---
    records = []
    for sim_idx, data in enumerate(data_dict):
        sim_label = labels[sim_idx] if labels else f"Sim {sim_idx+1}"
        theta = data["theta_durations"]
        gamma = data["gamma_durations"]
        ripple = data["ripple_durations"]

        mean_vals = [
            np.mean([np.mean(x) if len(x) else 0.0 for x in theta]),
            np.mean([np.mean(x) if len(x) else 0.0 for x in gamma]),
            np.mean([np.mean(x) if len(x) else 0.0 for x in ripple])
        ]
        def comp_ste(arrays):
            sems_sq = [
                (np.std(x, ddof=0)/np.sqrt(len(x)))**2 if len(x)>1 else 0.0
                for x in arrays
            ]
            return np.sqrt(sum(sems_sq))/len(arrays)
        ste_vals = [
            comp_ste(theta),
            comp_ste(gamma),
            comp_ste(ripple)
        ]

        for banda, mean_v, ste_v in zip(["Theta","Gamma","Ripple"], mean_vals, ste_vals):
            records.append({
                "Simulación": sim_label,
                "Banda": banda,
                "Media (ms)": mean_v,
                "STE": ste_v
            })

    df = pd.DataFrame(records)
    n_sims = df["Simulación"].nunique()

    # --- Cálculo de lims automáticos ---
    auto_lims = None
    if return_lims:
        # para cada banda recoger medias±STE
        mins = df["Media (ms)"] - df["STE"]
        maxs = df["Media (ms)"] + df["STE"]
        auto_lims = tuple(
            [float(mins[df["Banda"]==b].min()), float(maxs[df["Banda"]==b].max())]
            for b in ["Theta","Gamma","Ripple"]
        )

    # --- Plot con seaborn ---
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df, x="Banda", y="Media (ms)", hue="Simulación",
        ci=None, dodge=True
    )

    # --- Añadir barras de error manualmente ---
    total_bar_width = 0.8
    bar_width = total_bar_width / n_sims
    x_base = np.arange(3)
    offsets = (np.arange(n_sims) - (n_sims-1)/2) * bar_width

    for i, sim_label in enumerate(df["Simulación"].unique()):
        sub = df[df["Simulación"] == sim_label]
        xpos = x_base + offsets[i]
        ax.errorbar(
            xpos, sub["Media (ms)"].values,
            yerr=sub["STE"].values,
            fmt='none', capsize=3, ecolor='black', zorder=2
        )

    label_font = {'fontweight': 'bold', 'fontsize': 14}
    ax.set_xlabel("Frequency band", **label_font)
    ax.set_ylabel("Mean duration of events $[ms]$", **label_font)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)

    # --- Relleno y líneas en lims ---
    if lims is not None:
        center_offset = (n_sims/2) * bar_width * 1.15
        for idx, (low, high) in enumerate(lims):
            xmin = idx - center_offset
            xmax = idx + center_offset
            ax.fill_betweenx([low, high], xmin, xmax, color='red', alpha=0.25, zorder=1)
            ax.hlines(low, xmin, xmax, linestyles='-', colors='red', label='_nolegend_')
            ax.hlines(high, xmin, xmax, linestyles='-', colors='red', label='_nolegend_')

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles, labels,
        ncol=ncol,
        title="Experiments",
        title_fontsize=13,
        prop={"size": 13},
        frameon=True,
        framealpha=0.9
    )
    leg.get_title().set_fontweight('bold')
    plt.tight_layout()
    plt.show()

    return auto_lims if return_lims else None


def peak_freqs_plot(
    data_path: List[str] = [],
    data_dict: List[Dict[str, Any]] = [],
    labels: List[str] = [],
    figsize: tuple = (10, 6),
    xrotation: int = 0,
    ncol: int = 1,
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos de las frecuencias pico de los eventos detectados
    en varias simulaciones, opcionalmente calculando y retornando límites automáticos,
    y/o dibujando rellenos y líneas horizontales en lims.

    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
        labels (List[str]): Etiquetas para cada simulación.
        figsize (tuple): Tamaño de la figura.
        lims (Optional[Tuple[List[float], List[float], List[float]]]): Límites manuales:
            (
              [l_inf_theta, l_sup_theta],
              [l_inf_gamma, l_sup_gamma],
              [l_inf_swr,    l_sup_swr]
            )
        return_lims (bool): Si True, calcula y retorna límites automáticos.

    Returns:
        Si return_lims=True, devuelve una tupla de listas de límites:
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
        raise ValueError("No se han proporcionado datos para generar los gráficos.")

    # --- Cálculo de medias y errores ---
    medias, estes = [], []
    for data in data_dict:
        tpf = data.get("theta_peak_frequencies", [])
        gpf = data.get("gamma_peak_frequencies", [])
        rpf = data.get("ripple_peak_frequencies", [])

        media_theta = np.mean([np.mean(x) if len(x) > 0 else 0.0 for x in tpf]) if tpf else 0.0
        media_gamma = np.mean([np.mean(x) if len(x) > 0 else 0.0 for x in gpf]) if gpf else 0.0
        media_swr   = np.mean([np.mean(x) if len(x) > 0 else 0.0 for x in rpf]) if rpf else 0.0

        def comp_ste(arrays):
            sems_sq = [
                (np.std(x, ddof=0) / np.sqrt(len(x)))**2
                for x in arrays if len(x) > 1
            ]
            return np.sqrt(sum(sems_sq)) / len(arrays) if arrays else 0.0

        ste_theta = comp_ste(tpf)
        ste_gamma = comp_ste(gpf)
        ste_swr   = comp_ste(rpf)

        medias.append((media_theta, media_gamma, media_swr))
        estes.append((ste_theta, ste_gamma, ste_swr))

    medias_arr = np.array(medias)
    estes_arr  = np.array(estes)
    n_sims     = len(medias_arr)
    x          = np.arange(n_sims)

    # --- Cálculo de lims automáticos si se pide ---
    auto_lims = None
    if return_lims:
        mins = medias_arr - estes_arr
        maxs = medias_arr + estes_arr
        auto_lims = (
            [float(np.min(mins[:,0])), float(np.max(maxs[:,0]))],
            [float(np.min(mins[:,1])), float(np.max(maxs[:,1]))],
            [float(np.min(mins[:,2])), float(np.max(maxs[:,2]))],
        )

    # --- Preparación de categorías ---
    colorblind = sns.color_palette("colorblind", 8)
    categorias = {
        "Theta": (medias_arr[:, 0], estes_arr[:, 0], colorblind[0]),
        "Gamma": (medias_arr[:, 1], estes_arr[:, 1], colorblind[1]),
        "SWRs":  (medias_arr[:, 2], estes_arr[:, 2], colorblind[2]),
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

        # Errorbar
        ax.errorbar(
            x, y_vals, yerr=y_errs,
            fmt='-o', markersize=8,
            color=color, label=nombre,
            capsize=3, zorder=3
        )

        # Relleno y líneas en lims para esta categoría
        if nombre in limits_map:
            low, high = limits_map[nombre]
            xmin, xmax = x[0] - 0.5, x[-1] + 0.5

            # Relleno semitransparente
            ax.fill_betweenx(
                [low, high],
                xmin, xmax,
                color='red',
                alpha=0.25,
                zorder=1
            )

            # Líneas horizontales
            ax.hlines(low,  xmin, xmax, linestyles='-', colors='red', label='_nolegend_')
            ax.hlines(high, xmin, xmax, linestyles='-', colors='red', label='_nolegend_')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        label_font = {'fontweight': 'bold', 'fontsize': 14}
        ax.set_xlabel("Experiments", **label_font)
        ax.set_ylabel("Mean peak frequency of events $[Hz]$", **label_font)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)
        handles, labs = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles, labs,
            ncol=ncol,
            title="Experiments",
            title_fontsize=13,
            prop={"size": 13},
            frameon=True,
            framealpha=0.9
        )
        leg.get_title().set_fontweight('bold')
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
    xrotation: int = 0,
    ncol: int = 1,
    y_lims: Optional[Tuple[float, float]] = None,
    lims: Optional[Tuple[List[float], List[float], List[float]]] = None,
    return_lims: bool = False
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    Función para generar los gráficos de la densidad espectral de potencia media
    de las simulaciones, usando seaborn para estilizar, y opcionalmente
    calculando y retornando límites automáticos, y/o dibujando rellenos y líneas horizontales en lims.
    Args:
        data_path (List[str]): Lista de rutas de carpetas que contienen las simulaciones.
        data_dict (List[Dict[str, Any]]): Lista de diccionarios con los datos de las simulaciones.
        labels (List[str]): Etiquetas para cada simulación.
        figsize (tuple): Tamaño de la figura del gráfico.
        lims (Optional[Tuple[List[float], List[float], List[float]]]): Límites manuales:
            (
              [l_inf_theta, l_sup_theta],
              [l_inf_gamma, l_sup_gamma],
              [l_inf_swr,    l_sup_swr]
            )
        return_lims (bool): Si True, calcula y retorna límites automáticos.
    Returns:
        Si return_lims=True, devuelve una tupla de listas de límites:
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
        raise ValueError("No se han proporcionado datos para generar los gráficos.")

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
                for x in arrays if len(x) > 0
            ]
            return np.sqrt(sum(sems_sq)) / len(arrays) if arrays else 0.0

        ste_theta  = comp_ste(theta_psd)
        ste_gamma  = comp_ste(gamma_psd)
        ste_ripple = comp_ste(ripple_psd)

        means.append((mean_theta, mean_gamma, mean_ripple))
        stes.append((ste_theta,  ste_gamma,  ste_ripple))

    means_arr = np.array(means)  # (n_sims, 3)
    stes_arr  = np.array(stes)
    n_sims    = len(means_arr)
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

    # --- Preparar DataFrame para seaborn ---
    records = []
    for sim_idx in range(n_sims):
        sim_label = labels[sim_idx] if labels else f"Sim {sim_idx+1}"
        for band_idx, banda in enumerate(bandas):
            records.append({
                "Simulación": sim_label,
                "Banda": banda,
                "Media": means_arr[sim_idx, band_idx],
                "STE": stes_arr[sim_idx, band_idx]
            })
    df = pd.DataFrame.from_records(records)

    # --- Plot con seaborn ---
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x="Banda", y="Media", hue="Simulación",
        ci=None, dodge=True
    )

    # --- Añadir barras de error manualmente ---
    total_bar_width = 0.8
    bar_width = total_bar_width / n_sims
    x_base = np.arange(len(bandas))
    offsets = (np.arange(n_sims) - (n_sims-1)/2) * bar_width

    for i, sim_label in enumerate(df["Simulación"].unique()):
        sub = df[df["Simulación"] == sim_label]
        errs = sub["STE"].values
        xpos = x_base + offsets[i]
        ax.errorbar(
            xpos, sub["Media"].values,
            yerr=errs,
            fmt='none', capsize=3, ecolor='black', zorder=2
        )

    # --- Límites de eje y existentes ---
    todas_medias = means_arr.flatten()
    todos_ste    = stes_arr.flatten()
    # y_min = np.min(todas_medias) * 0.1
    # y_max = np.max(todas_medias + todos_ste) * 10
    # ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")

    # --- Relleno y líneas en lims restringidas a cada banda ---
    if lims is not None:
        center_offset = (n_sims / 2) * bar_width * 1.15
        for idx, (low, high) in enumerate(lims):
            xmin = idx - center_offset
            xmax = idx + center_offset
            ax.fill_betweenx(
                [low, high],
                xmin, xmax,
                color='red', alpha=0.25, zorder=1
            )
            ax.hlines(low,  xmin, xmax, linestyles='-', colors='red', label='_nolegend_')
            ax.hlines(high, xmin, xmax, linestyles='-', colors='red', label='_nolegend_')

    label_font = {'fontweight': 'bold', 'fontsize': 14}
    ax.set_xlabel("Frequency band", **label_font)
    ax.set_ylabel("Mean PSD $[V²/Hz]$", **label_font)
    ax.tick_params(axis='both', labelsize=13)
    if y_lims is not None:
        ax.set_ylim(y_lims[0], y_lims[1])
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles, labels,
        ncol=ncol,
        title="Experiments",
        title_fontsize=13,
        prop={"size": 13},
        frameon=True,
        framealpha=0.9
    )
    leg.get_title().set_fontweight('bold')
    leg.get_title().set_fontweight('bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)
    plt.tight_layout()
    plt.show()
    
    if return_lims:
        return auto_lims

    return None
    