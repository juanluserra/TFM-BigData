<!-- filepath: /home/juanluis/git-repositories/TFM-BigData/variables.md -->
# Explicación de Variables y Ecuaciones en el Modelo

Este archivo ofrece una descripción de las variables definidas en el script `global_vars_and_eqs.py`, el cual configura parámetros para una simulación neuronal con Brian2 utilizando una aproximación de Hodgkin-Huxley.

## Variables de Tiempo y Ruido

- **record_dt**  
  Define el intervalo en el que se recogen los datos (1/1024 segundo). Establece la resolución temporal de la simulación.

- **noise_amp** y **noise_amp_inh**  
  Representan la amplitud del ruido en las neuronas excitatorias e inhibitorias, respectivamente. Se utilizan para introducir fluctuaciones en las corrientes de la membrana.

- **r_noise** y **r_noise_inh**  
  Son factores escalares asociados al ruido (100 pamp para excitatorias y 10 pamp para inhibitorias) que, al multiplicarse por una función aleatoria (`randn()`), generan la fluctuación en la corriente.

## Variables de Umbral y Conductancias

- **V_th**  
  Define el potencial umbral (threshold) en -20 mV; es el valor a partir del cual se desencadena un potencial de acción.

- **gM**  
  Representa la conductancia para la corriente de la membrana (90 usiemens·cmetre⁻²); contribuye a la modulación de la excitabilidad y a la adaptación de la membrana.

## Parámetros de Ganancia y Variabilidad

- **var_coeff**  
  Es un coeficiente de variabilidad (valor 3). 

- **gain_stim**  
  Factor de ganancia (200) utilizado para escalar la intensidad de la estimulación aplicada a las neuronas.

## Variables Espaciales

- **sigma**  
  Parámetro (0.3 siemens/meter) 

- **scale** y **scale_str**  
  *scale* define una escala espacial (150 umetre) que puede relacionarse con dimensiones físicas de la red neuronal.  
  *scale_str* es la representación en cadena de la misma escala, útil para etiquetas o configuraciones textuales.

## Parámetros para Corrientes CAN

- **pCAN**  
  Indica la probabilidad o la activación de corrientes de tipo CAN (calcium-activated nonselective cation currents) en el modelo.


## Parámetros Relacionados con el Área de Sinapsis

- **taille_inh_normale** y **taille_inh_2**  
  - *taille_inh_normale* define el área normal de una sinapsis inhibitoria (14e3 umetre²).  
  - *taille_inh_2* es la mitad de ese valor, permitiendo investigar efectos de variaciones en el tamaño sináptico.

- **taille_exc_normale** y **taille_exc_2**  
  - *taille_exc_normale* establece el área típica de sinapsis excitatorias (29e3 umetre²).  
  - *taille_exc_2* es la mitad de dicho valor, lo que posibilita estudiar cómo la reducción en el área afecta la señal excitatoria.
