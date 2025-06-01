import numpy as np
import matplotlib.pyplot as plt
from typing import *


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
    
    
if __name__ == "__main__":
    # Ejemplo de uso de la funcion nonce_attack
    signal = np.sin(2 * np.pi * 5 * np.arange(0, 60, 1/1024))  # Señal de ejemplo (senoidal)
    fs = 1024  # Frecuencia de muestreo
    t = np.arange(0, 60, 1/fs) # Vector de tiempo de 60 segundos con frecuencia de muestreo fs
    dt = 1/fs*10  # Intervalo de tiempo entre ataques
    t_attack = (20, 40)  # Ventana de tiempo en la que se realiza el ataque
    p = (0.25, 0.5, 0.25)  # Probabilidades de cada tipo de ataque
    noise = scanning_attack(signal, fs, t, t_attack=t_attack, dt=dt, gap=3, factor=0.5)
    plt.plot(t, signal, label='Señal Original')
    plt.plot(t, noise, label='Señal con Ataque de Inundación', linestyle='--')
    plt.legend()
    plt.show()