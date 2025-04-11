import os
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
FILE_FOLDER = Path(__file__).parent


def select_folders(directory: Path) -> List[Path]:
    """
    Selecciona carpetas en el directorio proporcionado que coincidan con el patrón "results_[n]",
    con `n` entero positivo (incluyendo 0).

    Args:
        directory (Path): El directorio donde se buscarán las carpetas.

    Returns:
        List[Path]: Una lista con las rutas de las carpetas seleccionadas.
    """
    return [folder for folder in directory.iterdir() if folder.is_dir() and folder.name.startswith("results_")]


def search_parameters_files(folder: Path) -> List[Path]:
    """
    Busca archivos de parámetros en la carpeta proporcionada.

    Args:
        folder (Path): La carpeta donde se buscarán los archivos.

    Returns:
        List[Path]: Una lista con las rutas de los archivos encontrados.
    """

    # Dividimos la carpeta en subcarpetas
    folders = select_folders(folder)

    # Buscamos una archivo de parámetros en cada subcarpeta
    parameters_files = {}
    for idx, subfolder in enumerate(folders):
        for file in subfolder.iterdir():
            if file.is_file() and file.name == "parameters.txt":
                file_name = file.name[:-4]  # Sin la extensión
                file_name = f"{file_name}_{idx}"
                parameters_files[file_name] = file

    return parameters_files


def read_txt_file(file_path: Path) -> Tuple[str, List[str]]:
    """
    Le un archivo de texto por líneas y devuelve una tupla con el nombre del archivo y su contenido.
    El contenido es una lista de cadenas, cada una representando una línea del archivo.
    Args:
        file_path (Path): La ruta del archivo a leer.
    Returns:
        Tuple[str, List[str]]: Una tupla con el nombre del archivo y su contenido.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    return file_path.name, [line.strip() for line in lines]


def get_values_from_lines(lines: List[str]) -> List[float]:
    """
    Extrae los valores numéricos de las líneas proporcionadas.
    Solo devuelve los valores de las líneas necesarias para el análisis:
        - "all_N"
        - "all_g_max_e"
        - "all_gains"
        - "gCAN"

    Args:
        lines (List[str]): Lista de líneas del archivo.

    Returns:
        List[float]: Lista de valores numéricos extraídos.
    """

    # Creamos un diccionario para almacenar los valores
    values = {}

    # Iteramos sobre las líneas y extraemos los valores
    for line in lines:
        # Solo nos interesan las líneas que comienzan con los nombres específicos
        if line.startswith("all_N") or line.startswith("all_g_max_e") or line.startswith("all_gains") or line.startswith("gCAN"):
            # Dividimos la línea en clave y valor
            key, value = line.split(":")
            # Eliminamos los corchetes "[" y "]" del valor
            value = value.strip().strip("[]")
            # Nos quedamos con los valores numéricos hasta llegar al primero carácter no numérico
            # (esto es para evitar errores si el valor tiene unidades)
            numeric_value = ""
            for char in value:
                if char.isdigit() or char == ".":
                    numeric_value += char
                else:
                    break
            # Almacenamos el valor en el diccionario
            values[key.strip()] = float(numeric_value)

    return values


def main() -> None:
    # Definimos la ruta de la carpeta que contiene los archivos de parámetros
    directory = FILE_FOLDER / "all_parameters"

    # Buscamos los archivos de parámetros en la carpeta
    files = search_parameters_files(directory)

    # Iteramos sobre los archivos encontrados
    for file_name in files.keys():
        # Leemos el archivo de texto
        file_content = read_txt_file(files[file_name])
        lines = file_content[1]

        # Extraemos los valores de las líneas
        values = get_values_from_lines(lines)

        # Renombramos el archivo con los valores extraídos
        all_N_string = f"all_N-{values['all_N']:.2f}"
        all_g_max_e_string = f"all_g_max_e-{values['all_g_max_e']:.2f}"
        all_gains_string = f"all_gains-{values['all_gains']:.2f}"
        gCAN_string = f"gCAN-{values['gCAN']:.2f}"
        new_folder_name = f"{all_N_string}_{all_g_max_e_string}_{all_gains_string}_{gCAN_string}"

        # Cambiamos el nombre de la carpeta al nuevo
        old_folder_path = files[file_name].parent
        new_folder_path = old_folder_path.parent / new_folder_name
        os.rename(old_folder_path, new_folder_path)
        print(f"Renamed {old_folder_path} to {new_folder_path}")


if __name__ == "__main__":
    main()
