import numpy as np
import matplotlib.pyplot as plt
from typing import *


def white_noise(size: int, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """
    Genera ruido blanco gaussiano.

    Args:
        size (int): Número de muestras de ruido a generar.
        mu (float, optional): Media del ruido. Defaults to 0.
        sigma (float, optional): Desviación estándar del ruido. Defaults to 1.

    Returns:
        np.ndarray: Ruido blanco gaussiano generado.
    """
    return np.random.normal(mu, sigma, size)


def gaussian_pulse(fs: float, t: np.ndarray, f_carrier: float, amp: float, center: float, sigma: float) -> np.ndarray:
    """
    Genera un pulso gaussiano modulado en frecuencia.
    Este pulso es una envolvente gaussiana multiplicada por un pulso sinusoidal.

    Args:
        fs (float): Frecuencia de muestreo.
        t (np.ndarray): Vector de tiempo.
        f_carrier (float): Frecuencia portadora del pulso.
        amp (float): Amplitud del pulso.
        center (float): Centro de la envolvente gaussiana.
        sigma (float): Desviación estándar de la envolvente gaussiana.

    Returns:
        np.ndarray: Pulso gaussiano modulado en frecuencia generado.
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
    Genera múltiples pulsos gaussianos modulado en frecuencia.
    Este método combina varios pulsos gaussianos en una sola señal.

    Args:
        fs (float): Frecuencia de muestreo.
        t (np.ndarray): Vector de tiempo.
        f_carrier (float): Frecuencia portadora de los pulsos.
        amp (float): Amplitud de los pulsos.
        centers (list): Centros de las envolventes gaussianas para cada pulso.
        sigmas (list): Desviaciones estándar de las envolventes gaussianas para cada pulso.

    Returns:
        np.ndarray: Señal compuesta de múltiples pulsos gaussianos modulado en frecuencia generados.
    """
    # Inicializamos el vector de pulsos
    pulses = np.zeros_like(t)

    # Recorremos los centros y sigmas para generar los pulsos y sumarlos
    for center, sigma in zip(centers, sigmas):
        pulses += gaussian_pulse(fs, t, f_carrier, amp, center, sigma)

    return pulses


def flooding_jamming_attack(
    signal: np.ndarray,
    t: np.ndarray,
    factor: float,
    t_attack: Optional[Tuple[float, float]]=None,
) -> np.ndarray:
    """
    Realiza un ataque de inundación a una señal en una ventana de tiempo específica.
    Este ataque consiste en multiplicar la señal por un factor de potencia en una ventana de tiempo específica.

    Args:
        signal (np.ndarray): Señal a la que se le va a realizar el ataque.
        t (np.ndarray): Vector de tiempo correspondiente a la señal.
        factor (float): Factor de potencia por el que se multiplica la señal en la ventana de tiempo especificada.
        t_attack (Optional[Tuple[float, float]], optional): Ventana de tiempo en la que se realiza el ataque. Si es None, se usa todo el tiempo de la señal. Defaults to None.

    Returns:
        np.ndarray: Señal con el ataque aplicado.
    """
    
    # Calculamos la ventana de tiempo en la que se realiza el ataque
    if t_attack is None:
        t_attack = (t[0], t[-1])
        
    # Multiplicamos la señal por el factor de potencia en la ventana de tiempo especificada
    signal_attacked = signal.copy()
    signal_attacked[(t >= t_attack[0]) & (t <= t_attack[1])] *= factor
    
    return signal_attacked



def scanning_attack(
    signal: np.ndarray,
    fs: float,
    t: np.ndarray,
    t_attack: Optional[Tuple[float, float]]=None,
    dt: Optional[float]=None,
    gap: int=2,
    factor: float=2
) -> np.ndarray:
    """
    Realiza un ataque de inundación a una señal en una ventana de tiempo específica.
    Este ataque consiste en multiplicar la señal por un factor de potencia en una ventana de tiempo específica,
    dividiendo la ventana de tiempo en secciones y aplicando el ataque de forma periódica.
    El ataque se realiza cada 'gap' secciones de la señal, multiplicando por un factor de potencia especificado.

    Args:
        signal (np.ndarray): Señal a la que se le va a realizar el ataque.
        fs (float): Frecuencia de muestreo de la señal.
        t (np.ndarray): Vector de tiempo correspondiente a la señal.
        t_attack (Optional[Tuple[float, float]], optional): Ventana de tiempo en la que se realiza el ataque. Si es None, se usa todo el tiempo de la señal. Defaults to None.
        dt (Optional[float], optional): Intervalo de tiempo entre ataques. Si es None, se calcula como 1/fs. Defaults to None.
        gap (int, optional): Número de secciones entre ataques. Cada 'gap' secciones se aplica el ataque. Defaults to 2.
        factor (float, optional): Factor de potencia por el que se multiplica la señal en la ventana de tiempo especificada. Defaults to 2.

    Returns:
        np.ndarray: Señal con el ataque aplicado.
    """
    # Calculamos cada cuanto se realiza el ataque
    if dt is None:
        dt = 1/fs
    # Calculamos la ventana de tiempo en la que se realiza el ataque
    if t_attack is None:
        t_attack = (t[0], t[-1])
    
    # Dividimos el tiempo de ataque en tantos elementos del tamaño dt como sea posible
    n_dt = int((t_attack[1] - t_attack[0]) / dt)
    len_dt = len(t[t < dt])
    signal_to_attack = signal[(t >= t_attack[0]) & (t <= t_attack[1])][:len_dt * n_dt]
    signal_to_attack = signal_to_attack.reshape((n_dt, len_dt))
    
    # Realizamos el ataque de probabilidad en cada sección de la señal
    attack = np.ones(signal_to_attack.shape[0])
    attack[::gap] = factor  # Aplicamos el factor de ataque cada 'gap' secciones
    signal_attacked = signal_to_attack * attack[:, np.newaxis]
    
    # Volvemos plana la señal atacada
    signal_attacked = signal_attacked.flatten()
    # Sustituimos el tramo atacado en la señal original
    new_signal = signal.copy()
    # Localizamos los íncides a sustituir
    t_index = np.where((t >= t_attack[0]) & (t <= t_attack[1]))[:len_dt * n_dt]
    # Sustituimos el tramo atacado en la señal original
    new_signal[t_index] = signal_attacked
    
    return new_signal

    
def nonce_attack(
    signal: np.ndarray,
    fs: float, 
    t: np.ndarray, 
    dt: Optional[float]=None, 
    t_attack: Optional[Tuple[float, float]]=None,
    prob: Tuple[float, float, float]=(1/3, 1/3, 1/3),
    power: Tuple[float, float, float]=(2, 1, 0.5)
    ) -> np.ndarray:
    """
    Realiza un ataque de probabilidad (tipo 'nonce') a una señal en una ventana de tiempo específica.
    Este ataque consiste en multiplicar la señal por un factor de potencia aleatorio en una ventana de tiempo específica.
    La probabilidad de cada tipo de ataque se puede especificar, así como el factor de potencia para cada tipo de ataque.

    Args:
        signal (np.ndarray): Señal a la que se le va a realizar el ataque.
        fs (float): Frecuencia de muestreo de la señal.
        t (np.ndarray): Vector de tiempo correspondiente a la señal.
        dt (Optional[float], optional): Intervalo de tiempo entre ataques. Si es None, se calcula como 1/fs. Defaults to None.
        t_attack (Optional[Tuple[float, float]], optional): Ventana de tiempo en la que se realiza el ataque. Si es None, se usa todo el tiempo de la señal. Defaults to None.
        prob (Tuple[float, float, float], optional): Probabilidades de cada tipo de ataque. Deben sumar 1. Defaults to (1/3, 1/3, 1/3).
        power (Tuple[float, float, float], optional): Factores de potencia para cada tipo de ataque. Defaults to (2, 1, 0.5).

    Returns:
        np.ndarray: Señal con el ataque aplicado.
    """
    
    # Calculamos cada cuanto se realiza el ataque
    if dt is None:
        dt = 1/fs
    # Calculamos la ventana de tiempo en la que se realiza el ataque
    if t_attack is None:
        t_attack = (t[0], t[-1])
    
    # Dividimos el tiempo de ataque en tantos elementos del tamaño dt como sea posible
    n_dt = int((t_attack[1] - t_attack[0]) / dt)
    len_dt = len(t[t < dt])
    signal_to_attack = signal[(t >= t_attack[0]) & (t <= t_attack[1])][:len_dt * n_dt]
    signal_to_attack = signal_to_attack.reshape((n_dt, len_dt))
    
    # Realizamos el ataque de probabilidad en cada sección de la señal
    attack = np.random.choice(power, p=prob, size=signal_to_attack.shape[0])
    signal_attacked = signal_to_attack * attack[:, np.newaxis]
    
    # Volvemos plana la señal atacada
    signal_attacked = signal_attacked.flatten()
    # Sustituimos el tramo atacado en la señal original
    new_signal = signal.copy()
    # Localizamos los íncides a sustituir
    t_index = np.where((t >= t_attack[0]) & (t <= t_attack[1]))[:len_dt * n_dt]
    # Sustituimos el tramo atacado en la señal original
    new_signal[t_index] = signal_attacked
    
    return new_signal
