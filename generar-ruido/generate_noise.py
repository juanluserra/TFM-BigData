import numpy as np


def white_noise(size: int, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """
    Genera ruido blanco.

    Parámetros
    ----------
    size : int
        El número de muestras a generar.
    mu : float, opcional
        La media del ruido, por defecto 0.
    sigma : float, opcional
        La desviación estándar del ruido, por defecto 1.

    Retorna
    -------
    np.ndarray
        El ruido blanco generado.
    """
    return np.random.normal(mu, sigma, size)


def gaussian_pulse(fs: float, t: np.ndarray, f_carrier: float, amp: float, center: float, sigma: float) -> np.ndarray:
    """
    Genera un pulso sinusoidal modulado por una envolvente gaussiana.

    Parámetros
    ----------
    fs : float
        Frecuencia de muestreo.
    t : np.ndarray
        Vector de tiempo.
    f_carrier : float
        Frecuencia portadora.
    amp : float
        Amplitud del pulso.
    center : float
        Centro de la envolvente gaussiana.
    sigma : float
        Desviación estándar de la envolvente gaussiana.

    Retorna
    -------
    np.ndarray
        El pulso sinusoidal modulado por la envolvente gaussiana generado.
    """
    # Creamos la envolvente gaussiana
    envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    # Generamos el pulso sinusoidal
    carrier = amp * np.sin(2 * np.pi * f_carrier * t)
    # Multiplicamos la envolvente por el pulso
    pulse = envelope * carrier

    return pulse


def multiple_guassian_pulses(fs: float, t: np.ndarray, f_carrier: float, amp: float, centers: list, sigmas: list) -> np.ndarray:
    """
    Genera múltiples pulsos gaussianos.

    Parámetros
    ----------
    fs : float
        Frecuencia de muestreo.
    t : np.ndarray
        Vector de tiempo.
    f_carrier : float
        Frecuencia portadora.
    amp : float
        Amplitud del pulso.
    centers : list
        Lista de centros de la envolvente gaussiana.
    sigmas : list
        Lista de desviaciones estándar de la envolvente gaussiana.

    Retorna
    -------
    np.ndarray
        Los pulsos gaussianos generados.
    """
    # Inicializamos el vector de pulsos
    pulses = np.zeros_like(t)

    # Recorremos los centros y sigmas para generar los pulsos y sumarlos
    for center, sigma in zip(centers, sigmas):
        pulses += gaussian_pulse(fs, t, f_carrier, amp, center, sigma)

    return pulses
