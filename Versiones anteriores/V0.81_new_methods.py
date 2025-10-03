import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import Menu
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
#sistema e interfaz
import sys
import os

#funciones de desarrollo
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# =============================================
# DEFINICIÓN DE TODOS LOS MODELOS MATEMÁTICOS
# =============================================

# 1. Modelos Sigmoidales
def sigmoidal_3params(x, a, b, x0):
    return a / (1 + np.exp(-(x - x0) / b))

def sigmoidal_4params(x, a, b, x0, y0):
    return y0 + (a / (1 + np.exp(-(x - x0) / b)))

def sigmoidal_5params(x, a, b, c, x0, y0):
    return y0 + (a / (1 + np.exp(-(x - x0) / b))) ** c

def logistic_3params(x, a, b, x0):
    condition1 = (x <= 0) & (b < 0)
    condition2 = (x <= 0) & (b >= 0)
    condition3 = (x > 0) & (b > 0)
    condition4 = (x > 0) & (b <= 0)
    result = np.zeros_like(x)
    result[condition1] = 0
    result[condition2] = a
    result[condition3] = a / (1 + np.abs(x[condition3] / x0) ** b)
    result[condition4] = a * np.abs(x[condition4] / x0) ** np.abs(b) / (1 + np.abs(x[condition4] / x0) ** np.abs(b))
    return result

def logistic_4params(x, a, b, x0, y0):
    condition1 = (x <= 0) & (b < 0)
    condition2 = (x <= 0) & (b >= 0)
    condition3 = (x > 0) & (b > 0)
    condition4 = (x > 0) & (b <= 0)
    result = np.zeros_like(x)
    result[condition1] = y0
    result[condition2] = y0 + a
    result[condition3] = y0 + (a / (1 + np.abs(x[condition3] / x0) ** b))
    result[condition4] = y0 + (a * np.abs(x[condition4] / x0) ** np.abs(b) / (1 + np.abs(x[condition4] / x0) ** np.abs(b)))
    return result

def weibull_4params(x, a, b, c, x0):
    threshold = x0 - b * (np.log(2) ** (1 / c))
    return np.where(x <= threshold, 0, a * (1 - np.exp(-((np.abs(x - x0 + b * (np.log(2) ** (1 / c))) / b) ** c))))

def weibull_5params(x, a, b, c, x0, y0):
    threshold = x0 - b * (np.log(2) ** (1 / c))
    return np.where(x <= threshold, y0, y0 + a * (1 - np.exp(-((np.abs(x - x0 + b * (np.log(2) ** (1 / c))) / b) ** c))))

def gompertz_3params(x, a, b, x0):
    return a * np.exp(-np.exp(-(x - x0) / b))

def gompertz_4params(x, a, b, x0, y0):
    return y0 + a * np.exp(-np.exp(-(x - x0) / b))

def hill_3params(x, a, b, c):
    return (a * x**b) / (c**b + x**b)

def hill_4params(x, a, b, c, y0):
    return y0 + (a * x**b) / (c**b + x**b)

def chapman_3params(x, a, b, c):
    return a * (1 - np.exp(-b * x)) ** c

def chapman_4params(x, a, b, c, y0):
    return y0 + a * (1 - np.exp(-b * x)) ** c

# 2. Modelos Polinomiales e Inversos
def linear_func(x, a, y0):
    return y0 + a * x

def quadratic_func(x, y0, a, b):
    return y0 + a * x + b * x**2

def cubic_func(x, y0, a, b, c):
    return y0 + a * x + b * x**2 + c * x**3

def inverse_first_order(x, y0, a):
    return y0 + a / x

def inverse_second_order(x, y0, a, b):
    return y0 + a / x + b / x**2

def inverse_third_order(x, y0, a, b, c):
    return y0 + a / x + b / x**2 + c / x**3

# 3. Modelos de Distribución Normal
def gaussian_3params(x, a, b, x0):
    return a * np.exp(-0.5 * ((x - x0)/b)**2)

def gaussian_4params(x, a, b, x0, y0):
    return y0 + a * np.exp(-0.5 * ((x - x0)/b)**2)

def weibull_5param(x, a, b, c, x0, y0):
    term = (c-1)/c
    threshold = x0 - b*(term)**(1/c)
    mask = x > threshold
    result = np.full_like(x, y0)
    
    x_transformed = (x[mask]-x0)/b + term**(1/c)
    result[mask] = y0 + a * term**((1-c)/c) * (np.abs(x_transformed)**(c-1)) * \
                  np.exp(-np.abs(x_transformed)**c + term)
    return result

def weibull_4param(x, a, b, c, x0):
    term = (c-1)/c
    threshold = x0 - b*(term)**(1/c)
    mask = x > threshold
    result = np.zeros_like(x)
    
    x_transformed = (x[mask]-x0)/b + term**(1/c)
    result[mask] = a * term**((1-c)/c) * (np.abs(x_transformed)**(c-1)) * \
                  np.exp(-np.abs(x_transformed)**c + term)
    return result

def pseudo_voigt_5param(x, a, b, c, x0, y0):
    lorentzian = 1 / (1 + ((x - x0)/b)**2)
    gaussian = np.exp(-0.5 * ((x - x0)/b)**2)
    return y0 + a * (c * lorentzian + (1 - c) * gaussian)

def pseudo_voigt_4param(x, a, b, c, x0):
    lorentzian = 1 / (1 + ((x - x0)/b)**2)
    gaussian = np.exp(-0.5 * ((x - x0)/b)**2)
    return a * (c * lorentzian + (1 - c) * gaussian)

def modified_gaussian_5param(x, a, b, c, x0, y0):
    return y0 + a * np.exp(-0.5 * np.abs((x - x0)/b)**c)

def modified_gaussian_4param(x, a, b, c, x0):
    return a * np.exp(-0.5 * np.abs((x - x0)/b)**c)

def lorentzian_4param(x, a, b, x0, y0):
    return y0 + a / (1 + ((x - x0)/b)**2)

def lorentzian_3param(x, a, b, x0):
    return a / (1 + ((x - x0)/b)**2)

def log_normal_4param(x, a, b, x0, y0):
    return np.where(x <= 0, y0, y0 + a * np.exp(-0.5 * (np.log(x/x0)/b)**2) / x)

def log_normal_3param(x, a, b, x0):
    return np.where(x <= 0, 0, a * np.exp(-0.5 * (np.log(x/x0)/b)**2 / x))

#4. MODELOS DE DECAIMIENTO EXPONENCIAL

def simple_exp_2param(x, a, b):
    return a * np.exp(-b * x)

def simple_exp_3param(x, y0, a, b):
    return y0 + a * np.exp(-b * x)

def double_exp_4param(x, a, b, c, d):
    return a * np.exp(-b * x) + c * np.exp(-d * x)

def double_exp_5param(x, y0, a, b, c, d):
    return y0 + a * np.exp(-b * x) + c * np.exp(-d * x)

def triple_exp_6param(x, a, b, c, d, g, h):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + g * np.exp(-h * x)

def triple_exp_7param(x, y0, a, b, c, d, g, h):
    return y0 + a * np.exp(-b * x) + c * np.exp(-d * x) + g * np.exp(-h * x)

def exp_linear_combination(x, y0, a, b, c):
    return y0 + a * np.exp(-b * x) + c * x

def modified_exp_3param(x, a, b, c):
    return a * np.exp(b / (x + c))

#5. modelos de crecimiento exponencial limitado
def exp_rise_simple_2param(x, a, b):
    return a * (1 - np.exp(-b * x))

def exp_rise_simple_3param(x, y0, a, b):
    return y0 + a * (1 - np.exp(-b * x))

def exp_rise_double_4param(x, a, b, c, d):
    return a * (1 - np.exp(-b * x)) + c * (1 - np.exp(-d * x))

def exp_rise_double_5param(x, y0, a, b, c, d):
    return y0 + a * (1 - np.exp(-b * x)) + c * (1 - np.exp(-d * x))

def exp_rise_simple_exp_2param(x, a, b):
    return a * (1 - b**x)

def exp_rise_simple_exp_3param(x, y0, a, b):
    return y0 + a * (1 - b**x)

#6. Modelos de potencia

def power_2param(x, a, b):
    return a * (x ** b)

def power_2param_modI(x, a, b):
    return a * (1 - x**(-b))

def power_2param_modII(x, a, b):
    return a * ((1 + x) ** b)

def power_3param(x, y0, a, b):
    return y0 + a * (x ** b)

def power_symmetric_3param(x, a, x0, b):
    return a * np.abs(x - x0) ** b

def power_symmetric_4param(x, a, x0, b, y0):
    return y0 + a * np.abs(x - x0) ** b

def power_pareto(x, a):
    return 1 - 1/(x**a)

def power_mod_pareto(x, a, b):
    return 1 - 1/((1 + a*x)**b)

# Modelos Beverton-Holt
def beverton_holt_A(x, r, K):
    return r / (1 + ((r-1)/K) * x)

def beverton_holt_B(x, r, K):
    return r * x / (1 + ((r-1)/K) * x)

#  von Bertalanffy
def von_bertalanffy(t, Linf, K, tzero):
    return Linf * (1.0 - np.exp(-K * (t - tzero)))

# Dose-Response models
def dose_response_A(x, a, b, c):
    return b + (a - b) / (1 + 10**(x - c))

def dose_response_B(x, a, b, c):
    return b + (a - b) / (1 + 10**(c - x))

def dose_response_C(x, a, b, c, d):
    return b + (a - b) / (1 + 10**(d * (x - c)))

def dose_response_D(x, a, b, c, d):
    return b + (a - b) / (1 + 10**(d * (c - x)))

def dose_response_E(x, a, b, c, d):
    return b + (a - b) / (1 + (x / c)**d)

# Plant Disease models
def plant_disease_exponential(t, y0, r):
    return y0 * np.exp(r * t)

def plant_disease_gompertz(t, y0, r):
    return np.exp(np.log(y0) * np.exp(-r * t))

def plant_disease_logistic(t, y0, r):
    return 1 / (1 + (1 - y0) / (y0 * np.exp(-r * t)))

def plant_disease_monomolecular(t, y0, r):
    return 1 - ((1 - y0) * np.exp(-r * t))

def plant_disease_weibull(t, a, b, c):
    return 1 - np.exp(-1.0 * ((t - a) / b)**c)

#  Standard Logistic
def standard_logistic_3p(x, a, c, d):
    return d + (a - d) / (1 + (x / c))

def standard_logistic_4p(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

def standard_logistic_5p(x, a, b, c, d, f):
    return d + (a - d) / (1 + (x / c)**b)**f

#  Affinity models
def high_low_affinity(x, a, b):
    return a * b * x / (1 + b * x)

def high_low_affinity_double(x, a, b, c, d):
    return a * b * x / (1 + b * x) + c * d * x / (1 + d * x)

#  Otros modelos
def membrane_transport(x, a, b, c, d):
    return a * (x - b) / (x**2 + c * x + d)

def aphid_population_growth(t, a, b, c):
    return a * np.exp(b * t) * (1 + c * np.exp(b * t))**(-2)

def generalized_negative_exponential(x, a, b, c):
    return a * (1.0 - np.exp(-b * x))**c

# Cinética enzimática
def michaelis_menten_basic(x, a, b):
    return a * x / (b + x)

def michaelis_menten_double(x, a, b, c, d):
    return a * x / (b + x) + c * x / (d + x)

# 7. MODELOS DE INGENIERÍA 2D
def dispersion_optical_2d(x, A1, A2, A3, A4):
    return A1 + A2*x**2 + A3/x**2 + A4/x**4

def dispersion_optical_sqrt_2d(x, A1, A2, A3, A4):
    return np.sqrt(A1 + A2*x**2 + A3/x**2 + A4/x**4)

def electron_beam_lithography_2d(x, a, b, c, d, f, g, h, i, j, k, l):
    return a*np.exp(-b*x) + c*np.exp(-(x-d)**2/f**2) + g*np.exp(-(x-h)**2/i**2) + j*np.exp(-(x-k)**2/l**2)

def extended_steinhart_hart_2d(R, A, B, C, D):
    return A + B*np.log(R) + C*(np.log(R))**2 + D*(np.log(R))**3

def graeme_paterson_motor_2d(t, A, b, omega, phi, A2, b2):
    return A*np.exp(-b*t)*np.cos(omega*t + phi) + A2*np.exp(-b2*t)

def klimpel_kinetics_flotation_A_2d(x, a, b):
    return a * (1 - (1 - np.exp(-b*x)) / (b*x))

def maxwell_wiechert_1_2d(X, a1, Tau1):
    return a1*np.exp(-X/Tau1)

def maxwell_wiechert_2_2d(X, a1, Tau1, a2, Tau2):
    return a1*np.exp(-X/Tau1) + a2*np.exp(-X/Tau2)

def maxwell_wiechert_3_2d(X, a1, Tau1, a2, Tau2, a3, Tau3):
    return a1*np.exp(-X/Tau1) + a2*np.exp(-X/Tau2) + a3*np.exp(-X/Tau3)

def maxwell_wiechert_4_2d(X, a1, Tau1, a2, Tau2, a3, Tau3, a4, Tau4):
    return a1*np.exp(-X/Tau1) + a2*np.exp(-X/Tau2) + a3*np.exp(-X/Tau3) + a4*np.exp(-X/Tau4)

def modified_arps_production_2d(x, qi_x, b_x, Di_x):
    return (qi_x/((1.0-b_x)*Di_x)) * (1.0-((1.0+b_x*Di_x*x)**(1.0-1.0/b_x)))

def ramberg_osgood_2d(Stress, Youngs_Modulus, K, n):
    return (Stress / Youngs_Modulus) + (Stress/K)**(1.0/n)

def reciprocal_extended_steinhart_hart_2d(R, A, B, C, D):
    return 1.0 / (A + B*np.log(R) + C*(np.log(R))**2 + D*(np.log(R))**3)

def reciprocal_steinhart_hart_2d(R, A, B, C):
    return 1.0 / (A + B*np.log(R) + C*(np.log(R))**3)

def sellmeier_optical_2d(x, B1, C1, B2, C2, B3, C3):
    return 1 + (B1*x**2)/(x**2-C1) + (B2*x**2)/(x**2-C2) + (B3*x**2)/(x**2-C3)

def sellmeier_optical_sqrt_2d(x, B1, C1, B2, C2, B3, C3):
    return np.sqrt(1 + (B1*x**2)/(x**2-C1) + (B2*x**2)/(x**2-C2) + (B3*x**2)/(x**2-C3))

def steinhart_hart_2d(R, A, B, C):
    return A + B*np.log(R) + C*(np.log(R))**3

def vandeemter_chromatography_2d(x, a, b, c):
    return a + b/x + c*x

# Modelos con offset
def electron_beam_lithography_offset_2d(x, a, b, c, d, f, g, h, i, j, k, l, Offset):
    return electron_beam_lithography_2d(x, a, b, c, d, f, g, h, i, j, k, l) + Offset

def graeme_paterson_motor_offset_2d(t, A, b, omega, phi, A2, b2, Offset):
    return graeme_paterson_motor_2d(t, A, b, omega, phi, A2, b2) + Offset

def klimpel_kinetics_flotation_A_offset_2d(x, a, b, Offset):
    return klimpel_kinetics_flotation_A_2d(x, a, b) + Offset

def maxwell_wiechert_1_offset_2d(X, a1, Tau1, Offset):
    return maxwell_wiechert_1_2d(X, a1, Tau1) + Offset

def maxwell_wiechert_2_offset_2d(X, a1, Tau1, a2, Tau2, Offset):
    return maxwell_wiechert_2_2d(X, a1, Tau1, a2, Tau2) + Offset

def maxwell_wiechert_3_offset_2d(X, a1, Tau1, a2, Tau2, a3, Tau3, Offset):
    return maxwell_wiechert_3_2d(X, a1, Tau1, a2, Tau2, a3, Tau3) + Offset

def maxwell_wiechert_4_offset_2d(X, a1, Tau1, a2, Tau2, a3, Tau3, a4, Tau4, Offset):
    return maxwell_wiechert_4_2d(X, a1, Tau1, a2, Tau2, a3, Tau3, a4, Tau4) + Offset

def modified_arps_production_offset_2d(x, qi_x, b_x, Di_x, Offset):
    return modified_arps_production_2d(x, qi_x, b_x, Di_x) + Offset

def ramberg_osgood_offset_2d(Stress, Youngs_Modulus, K, n, Offset):
    return ramberg_osgood_2d(Stress, Youngs_Modulus, K, n) + Offset

def reciprocal_extended_steinhart_hart_offset_2d(R, A, B, C, D, Offset):
    return reciprocal_extended_steinhart_hart_2d(R, A, B, C, D) + Offset

def reciprocal_steinhart_hart_offset_2d(R, A, B, C, Offset):
    return reciprocal_steinhart_hart_2d(R, A, B, C) + Offset

def sellmeier_optical_offset_2d(x, B1, C1, B2, C2, B3, C3, Offset):
    return sellmeier_optical_2d(x, B1, C1, B2, C2, B3, C3) + Offset

def sellmeier_optical_sqrt_offset_2d(x, B1, C1, B2, C2, B3, C3, Offset):
    return sellmeier_optical_sqrt_2d(x, B1, C1, B2, C2, B3, C3) + Offset

# Modelos con línea adicional
def klimpel_kinetics_flotation_A_plus_line_2d(x, a, b, c, d):
    return klimpel_kinetics_flotation_A_2d(x, a, b) + c*x + d

def maxwell_wiechert_1_plus_line_2d(X, a1, Tau1, c, d):
    return maxwell_wiechert_1_2d(X, a1, Tau1) + c*X + d

#NUEVOS MODELOS MATEMÁTICOS (NIST y otros)
# Hyperbolic Tangent (HT)
def hyperbolic_tangent_4p(T, LS, US, DBTT, C):
    return (LS + US)/2 + (US - LS)/2 * np.tanh((T - DBTT)/C)

def hyperbolic_tangent_5p(T, LS, US, DBTT, C, y0):
    return y0 + (LS + US)/2 + (US - LS)/2 * np.tanh((T - DBTT)/C)

# Asymmetric Hyperbolic Tangent (AHT)
def asymmetric_hyperbolic_tangent_5p(T, LS, US, DBTT, C, D):
    return (LS + US)/2 + (US - LS)/2 * np.tanh((T - DBTT)/(C + D*T))

def asymmetric_hyperbolic_tangent_6p(T, LS, US, DBTT, C, D, y0):
    return y0 + (LS + US)/2 + (US - LS)/2 * np.tanh((T - DBTT)/(C + D*T))

# Burr Model (BUR)
def burr_model_5p(T, LS, US, k, T0, m):
    return LS + (US - LS) * (1 + np.exp(-k*(T - T0)))**(-m)

def burr_model_6p(T, LS, US, k, T0, m, y0):
    return y0 + LS + (US - LS) * (1 + np.exp(-k*(T - T0)))**(-m)

# Arctangent Model (ACT)
def arctangent_model_4p(T, LS, US, DBTT, C):
    return (LS + US)/2 + (US - LS)/2 * np.arctan(np.pi/(2*C) * (T - DBTT))

def arctangent_model_5p(T, LS, US, DBTT, C, y0):
    return y0 + (LS + US)/2 + (US - LS)/2 * np.arctan(np.pi/(2*C) * (T - DBTT))

# Asymmetric Kohout Model (KHT)
def asymmetric_kohout_model_5p(T, LS, US, C, T0, p):
    result = np.zeros_like(T)
    mask_lower = T <= T0
    mask_upper = T > T0
    
    # Para T ≤ T0
    result[mask_lower] = LS + (US - LS)/(1 + p) * np.exp((1 + p)/(2*C) * (T[mask_lower] - T0))
    
    # Para T > T0
    result[mask_upper] = LS - p*(US - LS)/(1 + p) * np.exp(-(1 + p)/(2*C) * (T[mask_upper] - T0))
    
    return result

def asymmetric_kohout_model_6p(T, LS, US, C, T0, p, y0):
    result = np.zeros_like(T)
    mask_lower = T <= T0
    mask_upper = T > T0
    
    # Para T ≤ T0
    result[mask_lower] = y0 + LS + (US - LS)/(1 + p) * np.exp((1 + p)/(2*C) * (T[mask_lower] - T0))
    
    # Para T > T0
    result[mask_upper] = y0 + LS - p*(US - LS)/(1 + p) * np.exp(-(1 + p)/(2*C) * (T[mask_upper] - T0))
    
    return result

# Monotonic Four Parameter (MFP)
def monotonic_four_parameter(x, y0, x0, A, B):
    return y0 + A * (x - x0) * np.abs(x - x0)**(B - 1)

# Quadratic Four Parameter (QFP)
def quadratic_four_parameter(x, a, b, c, alpha):
    # Para evitar división por cero cuando alpha = 0
    if abs(alpha) < 1e-10:
        return a + b * np.log(x) + c * (np.log(x))**2
    else:
        return a + b * (x**alpha - 1)/alpha + c * ((x**alpha - 1)/alpha)**2

# SVD para 2 variables (implementación básica para 2 componentes)
def svd_2d_6p(x, y, theta1, u11, u12, v11, v12):
    """
    Implementación simplificada de SVD para 2 variables y 1 componente
    z_ij = θ₁ * u₁i * v₁j
    """
    # Para este ejemplo, asumimos que x e y son las coordenadas
    return theta1 * u11 * v11 * x + theta1 * u12 * v12 * y

def svd_2d_9p(x, y, theta1, theta2, u11, u12, u21, u22, v11, v12, v21, v22):
    """
    Implementación de SVD para 2 variables y 2 componentes
    z_ij = θ₁ * u₁i * v₁j + θ₂ * u₂i * v₂j
    """
    comp1 = theta1 * u11 * v11 * x + theta1 * u12 * v12 * y
    comp2 = theta2 * u21 * v21 * x + theta2 * u22 * v22 * y
    return comp1 + comp2

# =============================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# =============================================

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_menu()
        self.create_widgets()
        self.data = {'x': [], 'y': []}
        self.results = []
        self.suppress_alerts = False
        
    def setup_window(self):
        self.root.title("Regression Wizard - v0.81 the new models update update :3")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.root.iconbitmap(resource_path("icon_1.ico"))  
        
    def create_menu(self):
        menubar = Menu(self.root)  
        # Menú Archivo
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Cargar datos", command=self.load_txt_file)
        file_menu.add_command(label="Exportar resultados", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.destroy)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        
        # Menú Ayuda
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="Acerca de", command=self.show_about)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("Acerca de Regression Wizard")
        about_window.geometry("400x500")
        about_window.resizable(False, False)
        about_window.iconbitmap(resource_path("icon_1.ico"))
        
        # Frame principal
        main_frame = ttk.Frame(about_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Texto organizado
        about_text = """
        Regression Wizard - v0.75
        es una herramienta de ajuste de modelos matemáticos 
        diseñada para analizar datos experimentales mediante
        regresión no lineal. Permite ajustar múltiples
        modelos como sigmoidales, polinomiales, inversos,
        modelos de campana, Crecimiento exponencial limitada
        y Modelos de potencia y evaluar su calidad mediante
        el coeficiente de determinación (R^2).
        
        Características principales:
        - Más de 50 modelos matemáticos
        - Visualización gráfica interactiva
        - Exportación de resultados a TXT
        - Interfaz intuitiva
        
        Desarrollado con:
        - Python 3
        - Tkinter (GUI)
        - NumPy y SciPy (cálculos)
        - Matplotlib (gráficos)
        
        © 2025 - Todos los derechos reservados
        """
        
        ttk.Label(main_frame, 
                 text=about_text, 
                 justify=tk.LEFT,
                 font=('Arial', 10)).pack(fill=tk.X, pady=10)
        
        ttk.Button(main_frame, 
                 text="Cerrar", 
                 command=about_window.destroy).pack(pady=10)
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')  # Tema más actual
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TNotebook.Tab', padding=[10, 5], font=('Arial', 10, 'bold'))
        # Botones
        style.configure('Accent.TButton', foreground='white', background='#4a6baf', 
                      font=('Arial', 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#3a5b9f')])
        
        style.configure('Process.TButton', foreground='black', background='#e0e0e0', 
                      font=('Arial', 10, 'bold'))
        style.map('Process.TButton', background=[('active', '#d0d0d0')])
        
        style.configure('Custom.TButton', foreground='white', background='#2E8B57', 
                      font=('Arial', 10, 'bold'))
        style.map('Custom.TButton', background=[('active', '#3CB371')])
        
        # Treeview (tabla de resultados)
        style.configure('Treeview', rowheight=25, font=('Arial', 10))
        style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        
    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Datos
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="Datos")
        self.create_data_entry(data_tab)
        
        # Pestaña 2: Modelos
        models_tab = ttk.Frame(self.notebook)
        self.notebook.add(models_tab, text="Modelos")
        self.create_model_selection(models_tab)
        
        # Pestaña 3: Resultados
        results_tab = ttk.Frame(self.notebook)
        self.notebook.add(results_tab, text="Resultados")
        self.create_results_display(results_tab)
        
    def create_data_entry(self, parent):
        entry_frame = ttk.LabelFrame(parent, text="Carga de Datos", padding=10)
        entry_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        control_frame = ttk.Frame(entry_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        self.load_btn = ttk.Button(
            control_frame,
            text="Cargar Archivo TXT",
            style='Custom.TButton',
            command=self.load_txt_file
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            control_frame,
            text="Formato: 2 columnas (X y Y) separadas por espacio o tabulación",
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=10)
        
        text_container = ttk.Frame(entry_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        text_container.grid_rowconfigure(0, weight=1)
        text_container.grid_columnconfigure(0, weight=1)
        
        self.data_display = tk.Text(
            text_container,
            wrap=tk.NONE,
            state='disabled',
            font=('Consolas', 10),
            height=10,
            padx=5,
            pady=5
        )
        
        y_scroll = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=self.data_display.yview)
        x_scroll = ttk.Scrollbar(text_container, orient=tk.HORIZONTAL, command=self.data_display.xview)
        
        self.data_display.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )
        
        self.data_display.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        
        self.file_info_label = ttk.Label(
            entry_frame,
            text="No hay archivo cargado",
            foreground="gray60",
            font=('Arial', 9, 'italic')
        )
        self.file_info_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
    def create_model_selection(self, parent):
        model_frame = ttk.LabelFrame(parent, text="Selección de Modelos", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        dropdown_frame = ttk.Frame(model_frame)
        dropdown_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dropdown_frame, text="Categoría:").pack(side="left", padx=5)
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(dropdown_frame, 
                                            textvariable=self.category_var,
                                            values=["Sigmoidales", "Polinomiales", "Inversos polinomiales", "Modelos de campana", "Decaimiento Exponencial", "Crecimiento exponencial limitado", "Modelos de Potencia", "Ciencia-Bio", "Ingeniería 2D","Modelos NIST"],
                                            state="readonly",
                                            width=30)
        self.category_dropdown.pack(side="left", padx=5)
        self.category_dropdown.current(0)
        self.category_dropdown.bind("<<ComboboxSelected>>", self.update_model_list)
        
        ttk.Label(dropdown_frame, text="Modelo:").pack(side="left", padx=5)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(dropdown_frame, 
                                         textvariable=self.model_var,
                                         state="readonly",
                                         width=25)
        self.model_dropdown.pack(side="left", padx=5, expand=True, fill=tk.X)
        
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, 
                 text="PROCESAR MODELO SELECCIONADO", 
                 style='Process.TButton', 
                 command=self.process_selected_model).pack(side="left", padx=5)
        
        all_models_btn = ttk.Button(
            button_frame, 
            text="PROCESAR TODOS LOS MODELOS", 
            style='Process.TButton',
            command=self.ask_min_r2
        )
        all_models_btn.pack(side="right", padx=5)
        
        # Añadir tooltip informativo
        try:
            from tkinter import tix
            self.root.tk.call('package', 'require', 'tooltip')
            tooltip = tix.Balloon(self.root)
            tooltip.bind_widget(all_models_btn, 
                              balloonmsg="Nota: El bootstrapping no se aplica al procesar todos los modelos")
        except:
            pass  # Si no está disponible tix, continuar sin tooltip
        
        bootstrap_frame = ttk.Frame(model_frame)
        bootstrap_frame.pack(fill=tk.X, pady=5)
        
        self.bootstrap_var = tk.BooleanVar(value=False)
        self.bootstrap_check = ttk.Checkbutton(
            bootstrap_frame, 
            text="Usar Bootstrapping", 
            variable=self.bootstrap_var,
            command=self.toggle_bootstrap_settings
        )
        self.bootstrap_check.pack(side=tk.LEFT, padx=5)
        
        # Frame de configuración de bootstrapping (inicialmente oculto)
        self.bootstrap_settings_frame = ttk.Frame(bootstrap_frame)
        
        ttk.Label(self.bootstrap_settings_frame, text="Iteraciones:").pack(side=tk.LEFT)
        self.bootstrap_iter_var = tk.StringVar(value="200")
        self.bootstrap_iter_entry = ttk.Entry(
            self.bootstrap_settings_frame, 
            textvariable=self.bootstrap_iter_var, 
            width=5
        )
        self.bootstrap_iter_entry.pack(side=tk.LEFT, padx=5)
        
        # Inicializar el estado del frame de configuración
        if self.bootstrap_var.get():
            self.bootstrap_settings_frame.pack(side=tk.LEFT, padx=5)
        else:
            self.bootstrap_settings_frame.pack_forget()
        
        self.update_model_list()
        
    def toggle_bootstrap_settings(self):
        if self.bootstrap_var.get():
            self.bootstrap_settings_frame.pack(side=tk.LEFT, padx=5)
        else:
            self.bootstrap_settings_frame.pack_forget()
    
    def show_bootstrap_plot(self, result):
        """Muestra gráficos de distribución de parámetros del bootstrapping."""
        if not result.get("bootstrap_results"):
            return
        
        bootstrap_params = result["bootstrap_results"]["bootstrap_params"]
        n_samples = len(next(iter(bootstrap_params.values())))
        
        if n_samples == 0:
            return
        
        # Crear ventana para los gráficos de bootstrapping
        bootstrap_window = tk.Toplevel(self.root)
        bootstrap_window.title(f"Análisis Bootstrap - {result['name']}")
        bootstrap_window.geometry("1000x800")
        bootstrap_window.iconbitmap(resource_path("icon_1.ico"))
        
        # Crear figura con subplots para cada parámetro
        n_params = len(result["params"])
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params))
        if n_params == 1:
            axes = [axes]  # Para que funcione el bucle cuando solo hay un parámetro
        
        fig.suptitle(f"Distribución Bootstrap de Parámetros - {result['name']}", y=1.02)
        
        for i, (name, ax) in enumerate(zip(result["param_names"], axes)):
            param_values = bootstrap_params[name]
            
            # Histograma
            ax.hist(param_values, bins=30, alpha=0.7, color='blue', density=True)
            
            # Línea vertical para el valor estimado
            ax.axvline(result["params"][i], color='red', linestyle='--', linewidth=2)
            
            # Intervalo de confianza
            lower, upper = np.percentile(param_values, [2.5, 97.5])
            ax.axvline(lower, color='green', linestyle=':', linewidth=1.5)
            ax.axvline(upper, color='green', linestyle=':', linewidth=1.5)
            
            ax.set_title(f"Parámetro: {name}")
            ax.set_xlabel("Valor del parámetro")
            ax.set_ylabel("Densidad")
            ax.grid(True, alpha=0.3)
            
            # Añadir información estadística
            stats_text = (f"Estimado: {result['params'][i]:.4f}\n"
                         f"IC 95%: [{lower:.4f}, {upper:.4f}]\n"
                         f"Error estándar: {np.std(param_values):.4f}")
            ax.text(0.98, 0.85, stats_text, transform=ax.transAxes,
                   ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Mostrar en la ventana
        canvas = FigureCanvasTkAgg(fig, master=bootstrap_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_model_list(self, event=None):
        """Devuelve un diccionario con TODOS los modelos organizados por categoría."""
        all_models = {
        "Sigmoidales":[
                "Sigmoidal 3 parámetros",
                "Sigmoidal 4 parámetros",
                "Sigmoidal 5 parámetros",
                "Logística 3 parámetros",
                "Logística 4 parámetros",
                "Weibull 4 parámetros",
                "Weibull 5 parámetros",
                "Gompertz 3 parámetros",
                "Gompertz 4 parámetros",
                "Hill 3 parámetros",
                "Hill 4 parámetros",
                "Chapman 3 parámetros",
                "Chapman 4 parámetros"
            ],
        "Polinomiales":[
                "Lineal",
                "Cuadrático",
                "Cúbico"
            ],
        "Inversos polinomiales":[
                "Inverso 1º orden",
                "Inverso 2º orden",
                "Inverso 3º orden"
            ],
        "Modelos de campana":[
                "Gaussiana 3 parámetros",
                "Gaussiana 4 parámetros",
                "Weibull 4 parámetros (dist)",
                "Weibull 5 parámetros (dist)",
                "Pseudo-Voigt 4 parámetros",
                "Pseudo-Voigt 5 parámetros",
                "Gaussiana modificada 4 parámetros",
                "Gaussiana modificada 5 parámetros",
                "Lorentzian 3 parámetros",
                "Lorentzian 4 parámetros",
                "Log Normal 3 parámetros",
                "Log Normal 4 parámetros"
            ],
        
        "Decaimiento Exponencial":[
            "Exponencial simple 2p",
            "Exponencial con offset 3p",
            "Doble exponencial 4p",
            "Doble exponencial con offset 5p",
            "Triple exponencial 6p",
            "Triple exponencial con offset 7p",
            "Combinación lineal-exponencial",
            "Exponencial modificado 3p"
            ],
        
        "Crecimiento exponencial limitado":[
                "Crec. Exp. L. Simple 2p",
                "Crec. Exp. L. Simple 3p",
                "Crec. Exp. L. Doble 4p",
                "Crec. Exp. L. Doble 5p",
                "Crec. Exp. L. Simple Alt 2p",
                "Crec. Exp. L. Simple Alt 3p"
            ],
            
        "Modelos de Potencia":[
            "Potencia 2 parámetros",
            "Potencia 2p Modificado I",
            "Potencia 2p Modificado II",
            "Potencia 3 parámetros",
            "Potencia Simétrica 3p",
            "Potencia Simétrica 4p",
            "Función Pareto",
            "Función Pareto Modificada"
            ],
        
        "Ciencia-Bio": [
            "Beverton-Holt A (BH-A)",
            "Beverton-Holt B (BH-B)", 
            "von Bertalanffy Growth (VBG)",
            "Michaelis-Menten básico (MM)",
            "Michaelis-Menten doble (MMD)",
            "Dose-Response A (DR-A)",
            "Dose-Response B (DR-B)",
            "Dose-Response C (DR-C)",
            "Dose-Response D (DR-D)",
            "Dose-Response E (DR-E)",
            "Enfermedad vegetal Exponencial (PV-Exp)",
            "Enfermedad vegetal Gompertz (PV-Gomp)",
            "Enfermedad vegetal Logística (PV-Log)",
            "Enfermedad vegetal Monomolecular (PV-Mono)",
            "Enfermedad vegetal Weibull (PV-Weib)",
            "High-Low Affinity (HLA)",
            "High-Low Affinity doble (HLAD)",
            "Logística 3P estándar (Log3P)",
            "Logística 4P estándar (Log4P)", 
            "Logística 5P estándar (Log5P)",
            "Transporte de membrana (MT)",
            "Aphid Population Growth (APHID)",
            "Negative Exponential generalizada (NEG)"
        ],
        
        "Ingeniería 2D": [
                "Dispersión Óptica 2D",
                "Dispersión Óptica Raíz Cuadrada 2D",
                "Litografía por Haz de Electrones 2D",
                "Steinhart-Hart Extendido 2D",
                "Motor Eléctrico Graeme Paterson 2D",
                "Klimpel Cinética Flotación A 2D",
                "Maxwell-Wiechert 1 2D",
                "Maxwell-Wiechert 2 2D",
                "Maxwell-Wiechert 3 2D",
                "Maxwell-Wiechert 4 2D",
                "Producción de Pozo Arps Modificado 2D",
                "Ramberg-Osgood 2D",
                "Steinhart-Hart Extendido Recíproco 2D",
                "Steinhart-Hart Recíproco 2D",
                "Sellmeier Óptico 2D",
                "Sellmeier Óptico Raíz Cuadrada 2D",
                "Steinhart-Hart 2D",
                "VanDeemter Cromatografía 2D",
                "Litografía por Haz de Electrones con Offset 2D",
                "Motor Eléctrico Graeme Paterson con Offset 2D",
                "Klimpel Cinética Flotación A con Offset 2D",
                "Maxwell-Wiechert 1 con Offset 2D",
                "Maxwell-Wiechert 2 con Offset 2D",
                "Maxwell-Wiechert 3 con Offset 2D",
                "Maxwell-Wiechert 4 con Offset 2D",
                "Producción de Pozo Arps Modificado con Offset 2D",
                "Ramberg-Osgood con Offset 2D",
                "Steinhart-Hart Extendido Recíproco con Offset 2D",
                "Steinhart-Hart Recíproco con Offset 2D",
                "Sellmeier Óptico con Offset 2D",
                "Sellmeier Óptico Raíz Cuadrada con Offset 2D",
                "Klimpel Cinética Flotación A más Línea 2D",
                "Maxwell-Wiechert 1 más Línea 2D"
            ],
        
        "Modelos NIST": [
            "Hyperbolic Tangent 4p (HT)",
            "Hyperbolic Tangent 5p (HT)",
            "Asymmetric Hyperbolic Tangent 5p (AHT)",
            "Asymmetric Hyperbolic Tangent 6p (AHT)",
            "Burr Model 5p (BUR)",
            "Burr Model 6p (BUR)",
            "Arctangent Model 4p (ACT)",
            "Arctangent Model 5p (ACT)",
            "Asymmetric Kohout Model 5p (KHT)",
            "Asymmetric Kohout Model 6p (KHT)",
            "Monotonic Four Parameter (MFP)",
            "Quadratic Four Parameter (QFP)",
            "SVD 2D 6 parámetros",
            "SVD 2D 9 parámetros"
        ]
        
        }
               
        
        if event is not None:
            category = self.category_var.get()
            self.model_dropdown['values'] = all_models.get(category, [])
        
        return all_models
        
    def create_results_display(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Resultados", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Modelo", "Categoría", "R²", "Evaluación", "Parámetros")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        self.tree.column("Modelo", width=200, anchor="center")
        self.tree.column("Categoría", width=120, anchor="center")
        self.tree.column("R²", width=80, anchor="center")
        self.tree.column("Evaluación", width=100, anchor="center")
        self.tree.column("Parámetros", width=400, anchor="w")
        
        for col in columns:
            self.tree.heading(col, text=col)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        
        scrollbar_x = ttk.Scrollbar(results_frame, orient="horizontal", command=self.tree.xview)
        scrollbar_x.pack(side="bottom", fill="x")
        
        self.tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=scrollbar_x.set)
        
        action_frame = ttk.Frame(results_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, 
            text="Ver Gráfico", 
            command=self.show_selected_plot).pack(side="left", padx=5)
        
        ttk.Button(action_frame,
             text="Ver Estadísticas",
             command=self.show_selected_stats).pack(side="left", padx=5)
        
        ttk.Button(action_frame,
            text="Ver Residuos",
            command=self.show_selected_residuals).pack(side="left", padx=5)
        
        ttk.Button(action_frame,
            text="Exportar Resultados (TXT)",
            command=self.export_results).pack(side="left", padx=5)
        
        ttk.Button(action_frame,
            text="Limpiar Resultados",
            command=self.clear_results).pack(side="right", padx=5)
    
    def load_txt_file(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            
            filename = filepath.split('/')[-1]  # Para Linux/Mac
            filename = filename.split('\\')[-1]  # Para Windows
            self.file_info_label.config(text=f"Archivo cargado: {filename}", foreground="green")
                
            x_data = []
            y_data = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                x_val = float(parts[0])
                                y_val = float(parts[1])
                                x_data.append(x_val)
                                y_data.append(y_val)
                            except ValueError:
                                continue
            
            if len(x_data) < 2:
                messagebox.showerror("Error", "El archivo debe contener al menos 2 puntos de datos válidos")
                return
            
            self.data = {'x': np.array(x_data), 'y': np.array(y_data)}
            
            self.data_display.config(state='normal')
            self.data_display.delete(1.0, tk.END)
            
            header = "Datos cargados (X vs Y):\n" + "="*30 + "\n"
            self.data_display.insert(tk.END, header)
            
            for x, y in zip(x_data, y_data):
                self.data_display.insert(tk.END, f"{x:.4f}\t{y:.4f}\n")
            
            self.data_display.config(state='disabled')
            
            messagebox.showinfo("Éxito", f"Datos cargados correctamente: {len(x_data)} puntos")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{str(e)}")
    
    def evaluate_r2(self, r2):
        if r2 >= 0.9: return ("Excelente", "#4CAF50")
        elif r2 >= 0.7: return ("Bueno", "#8BC34A")
        elif r2 >= 0.5: return ("Moderado", "#FFC107")
        else: return ("Pobre", "#F44336")
    
    def get_model_function(self, model_name):
        model_functions = {
            # Sigmoidales
            "Sigmoidal 3 parámetros": (sigmoidal_3params, ["a", "b", "x0"]),
            "Sigmoidal 4 parámetros": (sigmoidal_4params, ["a", "b", "x0", "y0"]),
            "Sigmoidal 5 parámetros": (sigmoidal_5params, ["a", "b", "c", "x0", "y0"]),
            "Logística 3 parámetros": (logistic_3params, ["a", "b", "x0"]),
            "Logística 4 parámetros": (logistic_4params, ["a", "b", "x0", "y0"]),
            "Weibull 4 parámetros": (weibull_4params, ["a", "b", "c", "x0"]),
            "Weibull 5 parámetros": (weibull_5params, ["a", "b", "c", "x0", "y0"]),
            "Gompertz 3 parámetros": (gompertz_3params, ["a", "b", "x0"]),
            "Gompertz 4 parámetros": (gompertz_4params, ["a", "b", "x0", "y0"]),
            "Hill 3 parámetros": (hill_3params, ["a", "b", "c"]),
            "Hill 4 parámetros": (hill_4params, ["a", "b", "c", "y0"]),
            "Chapman 3 parámetros": (chapman_3params, ["a", "b", "c"]),
            "Chapman 4 parámetros": (chapman_4params, ["a", "b", "c", "y0"]),
            
            # Polinomiales
            "Lineal": (linear_func, ["a", "y0"]),
            "Cuadrático": (quadratic_func, ["y0", "a", "b"]),
            "Cúbico": (cubic_func, ["y0", "a", "b", "c"]),
            
            # Inversos
            "Inverso 1º orden": (inverse_first_order, ["y0", "a"]),
            "Inverso 2º orden": (inverse_second_order, ["y0", "a", "b"]),
            "Inverso 3º orden": (inverse_third_order, ["y0", "a", "b", "c"]),
            
            # Distribuciones
            "Gaussiana 3 parámetros": (gaussian_3params, ["a", "b", "x0"]),
            "Gaussiana 4 parámetros": (gaussian_4params, ["a", "b", "x0", "y0"]),
            "Weibull 4 parámetros (dist)": (weibull_4param, ["a", "b", "c", "x0"]),
            "Weibull 5 parámetros (dist)": (weibull_5param, ["a", "b", "c", "x0", "y0"]),
            "Pseudo-Voigt 4 parámetros": (pseudo_voigt_4param, ["a", "b", "c", "x0"]),
            "Pseudo-Voigt 5 parámetros": (pseudo_voigt_5param, ["a", "b", "c", "x0", "y0"]),
            "Gaussiana modificada 4 parámetros": (modified_gaussian_4param, ["a", "b", "c", "x0"]),
            "Gaussiana modificada 5 parámetros": (modified_gaussian_5param, ["a", "b", "c", "x0", "y0"]),
            "Lorentzian 3 parámetros": (lorentzian_3param, ["a", "b", "x0"]),
            "Lorentzian 4 parámetros": (lorentzian_4param, ["a", "b", "x0", "y0"]),
            "Log Normal 3 parámetros": (log_normal_3param, ["a", "b", "x0"]),
            "Log Normal 4 parámetros": (log_normal_4param, ["a", "b", "x0", "y0"]),
        
            # Decaimiento Exponencial
            "Exponencial simple 2p": (simple_exp_2param, ["a", "b"]),
            "Exponencial con offset 3p": (simple_exp_3param, ["y0", "a", "b"]),
            "Doble exponencial 4p": (double_exp_4param, ["a", "b", "c", "d"]),
            "Doble exponencial con offset 5p": (double_exp_5param, ["y0", "a", "b", "c", "d"]),
            "Triple exponencial 6p": (triple_exp_6param, ["a", "b", "c", "d", "g", "h"]),
            "Triple exponencial con offset 7p": (triple_exp_7param, ["y0", "a", "b", "c", "d", "g", "h"]),
            "Combinación lineal-exponencial": (exp_linear_combination, ["y0", "a", "b", "c"]),
            "Exponencial modificado 3p": (modified_exp_3param, ["a", "b", "c"]),
            
            # NUEVOS MODELOS DE CRECIMIENTO EXPONENCIAL
            "Crec. Exp. L. Simple 2p": (exp_rise_simple_2param, ["a", "b"]),
            "Crec. Exp. L. Simple 3p": (exp_rise_simple_3param, ["y0", "a", "b"]),
            "Crec. Exp. L. Doble 4p": (exp_rise_double_4param, ["a", "b", "c", "d"]),
            "Crec. Exp. L. Doble 5p": (exp_rise_double_5param, ["y0", "a", "b", "c", "d"]),
            "Crec. Exp. L. Simple Alt 2p": (exp_rise_simple_exp_2param, ["a", "b"]),
            "Crec. Exp. L. Simple Alt 3p": (exp_rise_simple_exp_3param, ["y0", "a", "b"]),
            
            # Modelos de Potencia
            "Potencia 2 parámetros": (power_2param, ["a", "b"]),
            "Potencia 2p Modificado I": (power_2param_modI, ["a", "b"]),
            "Potencia 2p Modificado II": (power_2param_modII, ["a", "b"]),
            "Potencia 3 parámetros": (power_3param, ["y0", "a", "b"]),
            "Potencia Simétrica 3p": (power_symmetric_3param, ["a", "x0", "b"]),
            "Potencia Simétrica 4p": (power_symmetric_4param, ["a", "x0", "b", "y0"]),
            "Función Pareto": (power_pareto, ["a"]),
            "Función Pareto Modificada": (power_mod_pareto, ["a", "b"]),
            
            # Modelos poblacionales
            "Beverton-Holt A (BH-A)": (beverton_holt_A, ["r", "K"]),
            "Beverton-Holt B (BH-B)": (beverton_holt_B, ["r", "K"]),
            "von Bertalanffy Growth (VBG)": (von_bertalanffy, ["Linf", "K", "tzero"]),
            "Aphid Population Growth (APHID)": (aphid_population_growth, ["a", "b", "c"]),
            
            # Cinética enzimática
            "Michaelis-Menten básico (MM)": (michaelis_menten_basic, ["a", "b"]),
            "Michaelis-Menten doble (MMD)": (michaelis_menten_double, ["a", "b", "c", "d"]),
            
            # Curvas dosis-respuesta
            "Dose-Response A (DR-A)": (dose_response_A, ["a", "b", "c"]),
            "Dose-Response B (DR-B)": (dose_response_B, ["a", "b", "c"]),
            "Dose-Response C (DR-C)": (dose_response_C, ["a", "b", "c", "d"]),
            "Dose-Response D (DR-D)": (dose_response_D, ["a", "b", "c", "d"]),
            "Dose-Response E (DR-E)": (dose_response_E, ["a", "b", "c", "d"]),
            
            # Enfermedades vegetales
            "Enfermedad vegetal Exponencial (PV-Exp)": (plant_disease_exponential, ["y0", "r"]),
            "Enfermedad vegetal Gompertz (PV-Gomp)": (plant_disease_gompertz, ["y0", "r"]),
            "Enfermedad vegetal Logística (PV-Log)": (plant_disease_logistic, ["y0", "r"]),
            "Enfermedad vegetal Monomolecular (PV-Mono)": (plant_disease_monomolecular, ["y0", "r"]),
            "Enfermedad vegetal Weibull (PV-Weib)": (plant_disease_weibull, ["a", "b", "c"]),
            
            # Afinidad y unión
            "High-Low Affinity (HLA)": (high_low_affinity, ["a", "b"]),
            "High-Low Affinity doble (HLAD)": (high_low_affinity_double, ["a", "b", "c", "d"]),
            
            # Logísticas estándar
            "Logística 3P estándar (Log3P)": (standard_logistic_3p, ["a", "c", "d"]),
            "Logística 4P estándar (Log4P)": (standard_logistic_4p, ["a", "b", "c", "d"]),
            "Logística 5P estándar (Log5P)": (standard_logistic_5p, ["a", "b", "c", "d", "f"]),
            
            # Otros modelos bio
            "Transporte de membrana (MT)": (membrane_transport, ["a", "b", "c", "d"]),
            "Negative Exponential generalizada (NEG)": (generalized_negative_exponential, ["a", "b", "c"]),
            
            # Ingeniería 2D
            "Dispersión Óptica 2D": (dispersion_optical_2d, ["A1", "A2", "A3", "A4"]),
            "Dispersión Óptica Raíz Cuadrada 2D": (dispersion_optical_sqrt_2d, ["A1", "A2", "A3", "A4"]),
            "Litografía por Haz de Electrones 2D": (electron_beam_lithography_2d, ["a", "b", "c", "d", "f", "g", "h", "i", "j", "k", "l"]),
            "Steinhart-Hart Extendido 2D": (extended_steinhart_hart_2d, ["A", "B", "C", "D"]),
            "Motor Eléctrico Graeme Paterson 2D": (graeme_paterson_motor_2d, ["A", "b", "omega", "phi", "A2", "b2"]),
            "Klimpel Cinética Flotación A 2D": (klimpel_kinetics_flotation_A_2d, ["a", "b"]),
            "Maxwell-Wiechert 1 2D": (maxwell_wiechert_1_2d, ["a1", "Tau1"]),
            "Maxwell-Wiechert 2 2D": (maxwell_wiechert_2_2d, ["a1", "Tau1", "a2", "Tau2"]),
            "Maxwell-Wiechert 3 2D": (maxwell_wiechert_3_2d, ["a1", "Tau1", "a2", "Tau2", "a3", "Tau3"]),
            "Maxwell-Wiechert 4 2D": (maxwell_wiechert_4_2d, ["a1", "Tau1", "a2", "Tau2", "a3", "Tau3", "a4", "Tau4"]),
            "Producción de Pozo Arps Modificado 2D": (modified_arps_production_2d, ["qi_x", "b_x", "Di_x"]),
            "Ramberg-Osgood 2D": (ramberg_osgood_2d, ["Youngs_Modulus", "K", "n"]),
            "Steinhart-Hart Extendido Recíproco 2D": (reciprocal_extended_steinhart_hart_2d, ["A", "B", "C", "D"]),
            "Steinhart-Hart Recíproco 2D": (reciprocal_steinhart_hart_2d, ["A", "B", "C"]),
            "Sellmeier Óptico 2D": (sellmeier_optical_2d, ["B1", "C1", "B2", "C2", "B3", "C3"]),
            "Sellmeier Óptico Raíz Cuadrada 2D": (sellmeier_optical_sqrt_2d, ["B1", "C1", "B2", "C2", "B3", "C3"]),
            "Steinhart-Hart 2D": (steinhart_hart_2d, ["A", "B", "C"]),
            "VanDeemter Cromatografía 2D": (vandeemter_chromatography_2d, ["a", "b", "c"]),
            "Litografía por Haz de Electrones con Offset 2D": (electron_beam_lithography_offset_2d, ["a", "b", "c", "d", "f", "g", "h", "i", "j", "k", "l", "Offset"]),
            "Motor Eléctrico Graeme Paterson con Offset 2D": (graeme_paterson_motor_offset_2d, ["A", "b", "omega", "phi", "A2", "b2", "Offset"]),
            "Klimpel Cinética Flotación A con Offset 2D": (klimpel_kinetics_flotation_A_offset_2d, ["a", "b", "Offset"]),
            "Maxwell-Wiechert 1 con Offset 2D": (maxwell_wiechert_1_offset_2d, ["a1", "Tau1", "Offset"]),
            "Maxwell-Wiechert 2 con Offset 2D": (maxwell_wiechert_2_offset_2d, ["a1", "Tau1", "a2", "Tau2", "Offset"]),
            "Maxwell-Wiechert 3 con Offset 2D": (maxwell_wiechert_3_offset_2d, ["a1", "Tau1", "a2", "Tau2", "a3", "Tau3", "Offset"]),
            "Maxwell-Wiechert 4 con Offset 2D": (maxwell_wiechert_4_offset_2d, ["a1", "Tau1", "a2", "Tau2", "a3", "Tau3", "a4", "Tau4", "Offset"]),
            "Producción de Pozo Arps Modificado con Offset 2D": (modified_arps_production_offset_2d, ["qi_x", "b_x", "Di_x", "Offset"]),
            "Ramberg-Osgood con Offset 2D": (ramberg_osgood_offset_2d, ["Youngs_Modulus", "K", "n", "Offset"]),
            "Steinhart-Hart Extendido Recíproco con Offset 2D": (reciprocal_extended_steinhart_hart_offset_2d, ["A", "B", "C", "D", "Offset"]),
            "Steinhart-Hart Recíproco con Offset 2D": (reciprocal_steinhart_hart_offset_2d, ["A", "B", "C", "Offset"]),
            "Sellmeier Óptico con Offset 2D": (sellmeier_optical_offset_2d, ["B1", "C1", "B2", "C2", "B3", "C3", "Offset"]),
            "Sellmeier Óptico Raíz Cuadrada con Offset 2D": (sellmeier_optical_sqrt_offset_2d, ["B1", "C1", "B2", "C2", "B3", "C3", "Offset"]),
            "Klimpel Cinética Flotación A más Línea 2D": (klimpel_kinetics_flotation_A_plus_line_2d, ["a", "b", "c", "d"]),
            "Maxwell-Wiechert 1 más Línea 2D": (maxwell_wiechert_1_plus_line_2d, ["a1", "Tau1", "c", "d"]),
            
            # Modelos NIST
            "Hyperbolic Tangent 4p (HT)": (hyperbolic_tangent_4p, ["LS", "US", "DBTT", "C"]),
            "Hyperbolic Tangent 5p (HT)": (hyperbolic_tangent_5p, ["LS", "US", "DBTT", "C", "y0"]),
            "Asymmetric Hyperbolic Tangent 5p (AHT)": (asymmetric_hyperbolic_tangent_5p, ["LS", "US", "DBTT", "C", "D"]),
            "Asymmetric Hyperbolic Tangent 6p (AHT)": (asymmetric_hyperbolic_tangent_6p, ["LS", "US", "DBTT", "C", "D", "y0"]),
            "Burr Model 5p (BUR)": (burr_model_5p, ["LS", "US", "k", "T0", "m"]),
            "Burr Model 6p (BUR)": (burr_model_6p, ["LS", "US", "k", "T0", "m", "y0"]),
            "Arctangent Model 4p (ACT)": (arctangent_model_4p, ["LS", "US", "DBTT", "C"]),
            "Arctangent Model 5p (ACT)": (arctangent_model_5p, ["LS", "US", "DBTT", "C", "y0"]),
            "Asymmetric Kohout Model 5p (KHT)": (asymmetric_kohout_model_5p, ["LS", "US", "C", "T0", "p"]),
            "Asymmetric Kohout Model 6p (KHT)": (asymmetric_kohout_model_6p, ["LS", "US", "C", "T0", "p", "y0"]),
            "Monotonic Four Parameter (MFP)": (monotonic_four_parameter, ["y0", "x0", "A", "B"]),
            "Quadratic Four Parameter (QFP)": (quadratic_four_parameter, ["a", "b", "c", "alpha"]),
            "SVD 2D 6 parámetros": (svd_2d_6p, ["theta1", "u11", "u12", "v11", "v12"]),
            "SVD 2D 9 parámetros": (svd_2d_9p, ["theta1", "theta2", "u11", "u12", "u21", "u22", "v11", "v12"])
                
        }
        
        return model_functions.get(model_name, (None, None))
    
    def perform_bootstrap(self, func, x, y, params, param_names, n_iterations=200):
        """Realiza el análisis de bootstrapping para estimar intervalos de confianza."""
        n = len(x)
        bootstrap_params = {name: [] for name in param_names}
        
        try:
            for _ in range(n_iterations):
                # Muestreo con reemplazo
                indices = np.random.choice(n, n, replace=True)
                x_sample = x[indices]
                y_sample = y[indices]
                
                try:
                    # Ajustar el modelo a la muestra bootstrap
                    popt, _ = curve_fit(func, x_sample, y_sample, p0=params, maxfev=5000)
                    
                    # Almacenar parámetros
                    for name, val in zip(param_names, popt):
                        bootstrap_params[name].append(val)
                        
                except RuntimeError:
                    continue  # Saltar iteraciones que no convergen
            
            # Calcular intervalos de confianza (95%)
            conf_intervals = {}
            for name in param_names:
                if bootstrap_params[name]:
                    lower = np.percentile(bootstrap_params[name], 2.5)
                    upper = np.percentile(bootstrap_params[name], 97.5)
                    conf_intervals[name] = (lower, upper)
                else:
                    conf_intervals[name] = (np.nan, np.nan)
                    
            return conf_intervals, bootstrap_params
            
        except Exception as e:
            print(f"Error en bootstrapping: {str(e)}")
            return None, None
    
    def show_bootstrap_plots(self, result):
        """Muestra gráficos de distribución de parámetros del bootstrapping, cada uno en su propia ventana."""
        if not result.get("bootstrap_results"):
            return
        
        bootstrap_params = result["bootstrap_results"]["bootstrap_params"]
        n_samples = len(next(iter(bootstrap_params.values())))
        
        if n_samples == 0:
            return
        
        # Crear una ventana independiente para cada parámetro
        for i, name in enumerate(result["param_names"]):
            param_values = bootstrap_params[name]
            
            # Crear ventana para este parámetro
            param_window = tk.Toplevel(self.root)
            param_window.title(f"Bootstrap: {result['name']} - {name}")
            param_window.geometry("600x500")
            param_window.iconbitmap(resource_path("icon_1.ico"))
            
            # Crear figura para este parámetro
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Histograma
            ax.hist(param_values, bins=30, alpha=0.7, color='blue', density=True)
            
            # Línea vertical para el valor estimado
            ax.axvline(result["params"][i], color='red', linestyle='--', linewidth=2, 
                      label=f"Estimado: {result['params'][i]:.4f}")
            
            # Intervalo de confianza
            lower, upper = np.percentile(param_values, [2.5, 97.5])
            ax.axvline(lower, color='green', linestyle=':', linewidth=1.5,
                      label=f"IC 95%: [{lower:.4f}, {upper:.4f}]")
            ax.axvline(upper, color='green', linestyle=':', linewidth=1.5)
            
            ax.set_title(f"Distribución Bootstrap\n{result['name']} - {name}")
            ax.set_xlabel("Valor del parámetro")
            ax.set_ylabel("Densidad")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Añadir información estadística en el gráfico
            stats_text = (f"Muestras: {n_samples}\n"
                         f"Media: {np.mean(param_values):.4f}\n"
                         f"Desv. Est.: {np.std(param_values):.4f}")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Mostrar en la ventana
            canvas = FigureCanvasTkAgg(fig, master=param_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def get_initial_params(self, model_name, x, y):
        if model_name == "Potencia 2 parámetros":
           return [0.0001, 1.75]
        elif model_name == "Potencia 2p Modificado I":
           return [43435, 0.0002]
        elif model_name == "Potencia 2p Modificado II":
           return [3.9456e-05, 1.8846]
        elif model_name == "Potencia 3 parámetros":
           return [-8046, 1.6963, 1.1139]
        elif model_name == "Potencia Simétrica 3p":
           return [16.7806, 2000.75, 0.6090]
        elif model_name == "Potencia Simétrica 4p":
           return [-0.5625, 2019.89, 1.7429, 99.9262]
        elif model_name == "Función Pareto":
           return [12315.96]
        elif model_name == "Función Pareto Modificada":
           return [0.0551, 80.16]
        elif model_name == "Crec. Exp. L. Simple 2p":
            return [np.max(y), 0.1]
        elif model_name == "Crec. Exp. L. Simple 3p":
            return [np.min(y), np.max(y)-np.min(y), 0.1]
        elif model_name == "Crec. Exp. L. Doble 4p":
            amp = np.max(y)/2
            return [amp, 0.1, amp, 0.1]
        elif model_name == "Crec. Exp. L. Doble 5p":
            y0 = np.min(y)
            amp = (np.max(y)-y0)/2
            return [y0, amp, 0.1, amp, 0.1]
        elif model_name == "Crec. Exp. L. Simple Alt 2p":
            return [np.max(y), 0.5]
        elif model_name == "Crec. Exp. L. Simple Alt 3p":
            return [np.min(y), (np.max(y)-np.min(y))/2, 0.5]
        elif model_name == "Exponencial simple 2p":
            return [np.max(y), 0.1]
        elif model_name == "Exponencial con offset 3p":
            return [np.min(y), np.max(y)-np.min(y), 0.1]
        elif model_name == "Doble exponencial 4p":
            amp = np.max(y)/2
            return [amp, 0.1, amp, 0.1]
        elif model_name == "Doble exponencial con offset 5p":
            y0 = np.min(y)
            amp = (np.max(y)-y0)/2
            return [y0, amp, 0.1, amp, 0.1]
        elif model_name == "Triple exponencial 6p":
            amp = np.max(y)/3
            return [amp, 0.1, amp, 0.1, amp, 0.1]
        elif model_name == "Triple exponencial con offset 7p":
            y0 = np.min(y)
            amp = (np.max(y)-y0)/3
            return [y0, amp, 0.1, amp, 0.1, amp, 0.1]
        elif model_name == "Combinación lineal-exponencial":
            return [-3.0, 100.0, 0.1, 0.06]
        elif model_name == "Exponencial modificado 3p":
            return [5e-5, 1867.0, 128.0]
        elif model_name == "Lineal":
            return np.polyfit(x, y, 1)[::-1]  # [a, y0]
        elif model_name == "Cuadrático":
            return np.polyfit(x, y, 2)[::-1]  # [y0, a, b]
        elif model_name == "Cúbico":
            return np.polyfit(x, y, 3)[::-1]  # [y0, a, b, c]
        elif model_name == "Inverso 1º orden":
            return [np.min(y), 1.0]  # y0, a
        elif model_name == "Inverso 2º orden":
            return [np.min(y), 1.0, 1.0]  # y0, a, b
        elif model_name == "Inverso 3º orden":
            return [np.min(y), 1.0, 1.0, 1.0]  # y0, a, b, c
        elif model_name == "Sigmoidal 3 parámetros":
            return [np.max(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))]]
        elif model_name == "Sigmoidal 4 parámetros":
           return [np.max(y)-np.min(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))], np.min(y)]
        elif model_name == "Sigmoidal 5 parámetros":
           return [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))], np.min(y)]
        elif model_name == "Logística 3 parámetros":
            return [np.max(y), 1.0, np.median(x)]
        elif model_name == "Logística 4 parámetros":
            return [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)]
        elif model_name == "Weibull 4 parámetros":
            return [np.max(y), (x[-1]-x[0])/10, 1.0, np.median(x)]
        elif model_name == "Weibull 5 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, np.median(x), np.min(y)]
        elif model_name == "Gompertz 3 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x)]
        elif model_name == "Gompertz 4 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x), np.min(y)]
        elif model_name == "Hill 3 parámetros":
            return [np.max(y), 1.0, np.median(x)]
        elif model_name == "Hill 4 parámetros":
            return [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)]
        elif model_name == "Chapman 3 parámetros":
            return [np.max(y), 2/np.median(x), 1.0]
        elif model_name == "Chapman 4 parámetros":
            return [np.max(y)-np.min(y), 2/np.median(x), 1.0, np.min(y)]
        elif model_name == "Gaussiana 3 parámetros":
            return [np.max(y), (x[-1]-x[0])/4, x[np.argmax(y)]]
        elif model_name == "Gaussiana 4 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/4, x[np.argmax(y)], np.min(y)]
        elif model_name == "Weibull 4 parámetros (dist)":
            return [np.max(y), (x[-1]-x[0])/4, 2.0, x[np.argmax(y)]]
        elif model_name == "Weibull 5 parámetros (dist)":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/4, 2.0, x[np.argmax(y)], np.min(y)]
        elif model_name == "Pseudo-Voigt 4 parámetros":
            return [np.max(y), (x[-1]-x[0])/4, 0.5, x[np.argmax(y)]]
        elif model_name == "Pseudo-Voigt 5 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/4, 0.5, x[np.argmax(y)], np.min(y)]
        elif model_name == "Gaussiana modificada 4 parámetros":
            return [np.max(y), (x[-1]-x[0])/4, 1.5, x[np.argmax(y)]]
        elif model_name == "Gaussiana modificada 5 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/4, 1.5, x[np.argmax(y)], np.min(y)]
        elif model_name == "Lorentzian 3 parámetros":
            return [np.max(y), (x[-1]-x[0])/4, x[np.argmax(y)]]
        elif model_name == "Lorentzian 4 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/4, x[np.argmax(y)], np.min(y)]
        elif model_name == "Log Normal 3 parámetros":
            return [np.max(y), 1.0, x[np.argmax(y)]]
        elif model_name == "Log Normal 4 parámetros":
            return [np.max(y)-np.min(y), 1.0, x[np.argmax(y)], np.min(y)]
        elif model_name == "Beverton-Holt A (BH-A)":
            # r ≈ y_max, K ≈ x donde y alcanza ~63% de r
            r_est = np.max(y)
            if r_est > 0:
                idx_63 = np.argmin(np.abs(y - 0.63 * r_est))
                K_est = x[idx_63] if idx_63 < len(x) else np.median(x)
                return [r_est, K_est]
            return [1.0, np.median(x)]
        elif model_name == "Beverton-Holt B (BH-B)":
            return [1.5, np.median(x)]
        elif model_name == "von Bertalanffy Growth (VBG)":
            # Linf ≈ y_max, K estimado de la pendiente, tzero ≈ x_min
            Linf_est = np.max(y)
            if len(x) > 2:
                # Estimación de K a partir de crecimiento inicial
                dy = np.diff(y) / np.diff(x)
                K_est = np.mean(dy[:min(3, len(dy))]) / Linf_est if Linf_est > 0 else 0.1
                return [Linf_est, max(0.01, K_est), np.min(x)]
            return [Linf_est, 0.1, np.min(x)]
        elif model_name == "Michaelis-Menten básico (MM)":
            # a ≈ y_max, b ≈ x donde y = a/2 (Km)
            a_est = np.max(y)
            if a_est > 0:
                idx_half = np.argmin(np.abs(y - a_est/2))
                b_est = x[idx_half] if idx_half < len(x) else np.median(x)
                return [a_est, b_est]
            return [1.0, np.median(x)]
        elif model_name == "Michaelis-Menten doble (MMD)":
            amp = np.max(y)/2
            return [amp, np.median(x)/2, amp, np.median(x)*2]
        elif model_name in ["Dose-Response A (DR-A)", "Dose-Response B (DR-B)"]:
            # a ≈ y_max, b ≈ y_min, c ≈ x donde y = (a+b)/2 (EC50)
            a_est = np.max(y)
            b_est = np.min(y)
            ec50_est = x[np.argmin(np.abs(y - (a_est + b_est)/2))]
            return [a_est, b_est, ec50_est] 
        elif model_name in ["Dose-Response C (DR-C)", "Dose-Response D (DR-D)"]:
            a_est = np.max(y)
            b_est = np.min(y)
            ec50_est = x[np.argmin(np.abs(y - (a_est + b_est)/2))]
            return [a_est, b_est, ec50_est, 1.0]  # Pendiente inicial = 1
        elif model_name == "Dose-Response E (DR-E)":
            a_est = np.max(y)
            b_est = np.min(y)
            ec50_est = x[np.argmin(np.abs(y - (a_est + b_est)/2))]
            return [a_est, b_est, ec50_est, 1.0]  # Pendiente Hill inicial = 1
        elif model_name in ["Enfermedad vegetal Exponencial (PV-Exp)", 
                       "Enfermedad vegetal Gompertz (PV-Gomp)",
                       "Enfermedad vegetal Logística (PV-Log)",
                       "Enfermedad vegetal Monomolecular (PV-Mono)"]:
            # y0 ≈ primer punto, r estimado de crecimiento inicial
            y0_est = y[0] if len(y) > 0 else 0.1
            if len(x) > 1 and len(y) > 1:
                r_est = (np.log(y[1]) - np.log(y0_est)) / (x[1] - x[0]) if y0_est > 0 else 0.1
                return [y0_est, max(0.01, r_est)]
            return [y0_est, 0.1]
        elif model_name == "Enfermedad vegetal Weibull (PV-Weib)":
            return[np.min(x), (np.max(x)-np.min(x))/4, 2.0]
        elif model_name == "High-Low Affinity (HLA)":
            # a ≈ y_max, b ≈ 1/x_medio
            a_est = np.max(y)
            b_est = 1.0 / np.median(x) if np.median(x) > 0 else 1.0
            return [a_est, b_est]
        elif model_name == "High-Low Affinity doble (HLAD)":
            # Dos sitios con diferentes afinidades
            a1_est = np.max(y)/2
            a2_est = np.max(y)/2
            b1_est = 1.0 / np.median(x) if np.median(x) > 0 else 1.0
            b2_est = b1_est / 10.0  # Segunda afinidad más baja
            return [a1_est, b1_est, a2_est, b2_est]
        elif model_name == "Logística 3P estándar (Log3P)":
            return [np.max(y), np.median(x), np.min(y)]
        elif model_name == "Logística 4P estándar (Log4P)":
            return [np.max(y), 1.0, np.median(x), np.min(y)]
        elif model_name == "Logística 5P estándar (Log5P)":
            return [np.max(y), 1.0, np.median(x), np.min(y), 1.0]
        elif model_name == "Transporte de membrana (MT)":
            return [np.max(y), np.min(x), 1.0, 1.0]
        elif model_name == "Aphid Population Growth (APHID)":
            return [np.max(y), 0.1, 0.1]
        elif model_name == "Negative Exponential generalizada (NEG)":
            return [np.max(y), 0.1, 1.0]
        # Modelos de Ingeniería 2D
        elif model_name == "Dispersión Óptica 2D":
            return [1.0, 0.001, 0.001, 0.001]
        elif model_name == "Dispersión Óptica Raíz Cuadrada 2D":
            return [1.0, 0.001, 0.001, 0.001]
        elif model_name == "Litografía por Haz de Electrones 2D":
            return [np.max(y), 0.1, np.max(y)/4, np.median(x), 1.0, 
                    np.max(y)/4, np.median(x), 1.0, np.max(y)/4, np.median(x), 1.0]
        elif model_name == "Steinhart-Hart Extendido 2D":
            return [0.001, 0.001, 0.001, 0.001]
        elif model_name == "Motor Eléctrico Graeme Paterson 2D":
            return [np.max(y), 0.1, 1.0, 0.0, np.max(y)/2, 0.05]
        elif model_name == "Klimpel Cinética Flotación A 2D":
            return [np.max(y), 0.1]
        elif model_name == "Maxwell-Wiechert 1 2D":
            return [np.max(y), np.median(x)]
        elif model_name == "Maxwell-Wiechert 2 2D":
            return [np.max(y)/2, np.median(x)/2, np.max(y)/2, np.median(x)*2]
        elif model_name == "Maxwell-Wiechert 3 2D":
            return [np.max(y)/3, np.median(x)/3, np.max(y)/3, np.median(x), np.max(y)/3, np.median(x)*3]
        elif model_name == "Maxwell-Wiechert 4 2D":
            return [np.max(y)/4, np.median(x)/4, np.max(y)/4, np.median(x)/2, 
                    np.max(y)/4, np.median(x), np.max(y)/4, np.median(x)*4]
        elif model_name == "Producción de Pozo Arps Modificado 2D":
            return [np.max(y), 0.5, 0.1]
        elif model_name == "Ramberg-Osgood 2D":
            return [1000.0, 100.0, 0.1]
        elif model_name == "Steinhart-Hart Extendido Recíproco 2D":
            return [0.001, 0.001, 0.001, 0.001]
        elif model_name == "Steinhart-Hart Recíproco 2D":
            return [0.001, 0.001, 0.001]
        elif model_name == "Sellmeier Óptico 2D":
            return [1.0, 100.0, 1.0, 1000.0, 1.0, 10000.0]
        elif model_name == "Sellmeier Óptico Raíz Cuadrada 2D":
            return [1.0, 100.0, 1.0, 1000.0, 1.0, 10000.0]
        elif model_name == "Steinhart-Hart 2D":
            return [0.001, 0.001, 0.001]
        elif model_name == "VanDeemter Cromatografía 2D":
            return [1.0, 1.0, 0.001]
        # Modelos con offset (añadir offset como último parámetro)
        elif "con Offset" in model_name:
            base_model = model_name.replace(" con Offset", "")
            base_params = self.get_initial_params(base_model, x, y)
            if base_params is not None:
                return base_params + [0.0]  # Añadir offset
        # Modelos con línea adicional
        elif model_name == "Klimpel Cinética Flotación A más Línea 2D":
            return [np.max(y), 0.1, 0.001, 0.0]
        elif model_name == "Maxwell-Wiechert 1 más Línea 2D":
            return [np.max(y), np.median(x), 0.001, 0.0]
        elif model_name == "Hyperbolic Tangent 4p (HT)":
           return [np.min(y), np.max(y), np.median(x), (np.max(x)-np.min(x))/10]
        elif model_name == "Hyperbolic Tangent 5p (HT)":
           return [np.min(y), np.max(y), np.median(x), (np.max(x)-np.min(x))/10, 0.0]
        elif model_name == "Asymmetric Hyperbolic Tangent 5p (AHT)":
           return [np.min(y), np.max(y), np.median(x), (np.max(x)-np.min(x))/10, 0.001]
        elif model_name == "Asymmetric Hyperbolic Tangent 6p (AHT)":
           return [np.min(y), np.max(y), np.median(x), (np.max(x)-np.min(x))/10, 0.001, 0.0]
        elif model_name == "Burr Model 5p (BUR)":
           return [np.min(y), np.max(y), 0.1, np.median(x), 1.0]
        elif model_name == "Burr Model 6p (BUR)":
           return [np.min(y), np.max(y), 0.1, np.median(x), 1.0, 0.0]
        elif model_name == "Arctangent Model 4p (ACT)":
           return [np.min(y), np.max(y), np.median(x), (np.max(x)-np.min(x))/10]
        elif model_name == "Arctangent Model 5p (ACT)":
           return [np.min(y), np.max(y), np.median(x), (np.max(x)-np.min(x))/10, 0.0]
        elif model_name == "Asymmetric Kohout Model 5p (KHT)":
           return [np.min(y), np.max(y), (np.max(x)-np.min(x))/10, np.median(x), 1.0]
        elif model_name == "Asymmetric Kohout Model 6p (KHT)":
           return [np.min(y), np.max(y), (np.max(x)-np.min(x))/10, np.median(x), 1.0, 0.0]
        elif model_name == "Monotonic Four Parameter (MFP)":
           return [np.min(y), np.median(x), 1.0, 1.0]
        elif model_name == "Quadratic Four Parameter (QFP)":
           return [np.min(y), 1.0, 1.0, 1.0]
        elif model_name == "SVD 2D 6 parámetros":
           return [1.0, 1.0, 1.0, 1.0, 1.0]
        elif model_name == "SVD 2D 9 parámetros":
           return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            return [1.0] * len(self.get_model_function(model_name)[1])
    
    def validate_bio_parameters(self, model_name, params, x, y):
        """Validación específica para parámetros de modelos biológicos"""
        
        # Modelos poblacionales
        if "Beverton-Holt" in model_name:
            r, K = params[0], params[1]
            if r <= 0 or K <= 0:
                return False, "Parámetros r y K deben ser > 0"
        
        elif "von Bertalanffy" in model_name:
            Linf, K, tzero = params[0], params[1], params[2]
            if Linf <= 0 or K <= 0:
                return False, "Linf y K deben ser > 0"
        
        # Cinética enzimática
        elif "Michaelis-Menten" in model_name:
            if "doble" in model_name.lower():
                a, b, c, d = params[0], params[1], params[2], params[3]
                if a <= 0 or b <= 0 or c <= 0 or d <= 0:
                    return False, "Todos los parámetros Vmax y Km deben ser > 0"
            else:
                a, b = params[0], params[1]
                if a <= 0 or b <= 0:
                    return False, "Vmax y Km deben ser > 0"
        
        # Curvas dosis-respuesta
        elif "Dose-Response" in model_name:
            a, b = params[0], params[1]
            if a <= b:  # a (respuesta máxima) debe ser > b (respuesta basal)
                return False, "Respuesta máxima debe ser mayor que respuesta basal"
            if len(params) > 2 and params[2] <= 0:  # EC50
                return False, "EC50 debe ser > 0"
            if len(params) > 3 and params[3] <= 0:  # Pendiente Hill
                return False, "Pendiente Hill debe ser > 0"
        
        # Enfermedades vegetales
        elif "Enfermedad vegetal" in model_name:
            if any(p <= 0 for p in params):
                return False, "Todos los parámetros deben ser > 0"
        
        # Modelos de afinidad
        elif "Affinity" in model_name:
            if "doble" in model_name.lower():
                a, b, c, d = params[0], params[1], params[2], params[3]
                if a <= 0 or b <= 0 or c <= 0 or d <= 0:
                    return False, "Todos los parámetros de afinidad deben ser > 0"
            else:
                a, b = params[0], params[1]
                if a <= 0 or b <= 0:
                    return False, "Parámetros de afinidad deben ser > 0"
        
        # Logísticas estándar
        elif "Logística" in model_name and "estándar" in model_name:
            a, d = params[0], params[-1]  # a (máximo), d (mínimo)
            if a <= d:
                return False, "El máximo debe ser mayor que el mínimo"
            if len(params) > 2 and params[1] <= 0:  # Pendiente
                return False, "Pendiente debe ser > 0"
            if len(params) > 3 and params[2] <= 0:  # EC50
                return False, "EC50 debe ser > 0"
        
        # Transporte de membrana
        elif "Transporte de membrana" in model_name:
            a, b, c, d = params[0], params[1], params[2], params[3]
            if c**2 < 4*d:  # Discriminante del denominador
                return False, "Parámetros no válidos para el modelo de transporte"
        
        # Crecimiento de áfidos
        elif "Aphid" in model_name:
            a, b, c = params[0], params[1], params[2]
            if a <= 0 or b <= 0 or c <= 0:
                return False, "Todos los parámetros deben ser > 0"
        
        # Exponencial negativa generalizada
        elif "Negative Exponential" in model_name:
            a, b, c = params[0], params[1], params[2]
            if a <= 0 or b <= 0 or c <= 0:
                return False, "Todos los parámetros deben ser > 0"
        
        # Validaciones para modelos NIST
        elif "Hyperbolic Tangent" in model_name:
            LS, US = params[0], params[1]
            if LS >= US:
                return False, "LS debe ser menor que US"
                
        elif "Burr Model" in model_name:
            LS, US, k, m = params[0], params[1], params[2], params[4]
            if LS >= US:
                return False, "LS debe ser menor que US"
            if k <= 0 or m <= 0:
                return False, "k y m deben ser > 0"
                
        elif "Asymmetric Kohout" in model_name:
            LS, US, C, p = params[0], params[1], params[2], params[4]
            if LS >= US:
                return False, "LS debe ser menor que US"
            if C <= 0:
                return False, "C debe ser > 0"
            if p <= 0:
                return False, "p debe ser > 0"
                
        elif "Monotonic Four Parameter" in model_name:
            A, B = params[2], params[3]
            if A <= 0:
                return False, "A debe ser > 0"
                
        elif "Quadratic Four Parameter" in model_name:
            # Validar que alpha no cause problemas numéricos
            alpha = params[3]
            if np.any(x <= 0) and alpha < 0:
                return False, "Para alpha < 0, x debe ser > 0"
        
        # Validaciones adicionales de dominio biológico
        if any(np.isnan(p) or np.isinf(p) for p in params):
            return False, "Parámetros contienen valores NaN o infinitos"
        
        # Validar que los parámetros produzcan valores reales en el rango de x
        try:
            func, _ = self.get_model_function(model_name)
            test_y = func(x, *params)
            if np.any(np.isnan(test_y)) or np.any(np.isinf(test_y)):
                return False, "Parámetros producen valores no válidos en el rango de datos"
        except:
            return False, "Error al validar parámetros con los datos"
        
        return True, "Parámetros válidos"
    
    def process_selected_model(self, suppress_alerts=False):
        if not self.data or len(self.data['x']) == 0:
            messagebox.showwarning("Advertencia", "Primero cargue un archivo con datos")
            return
            
        model_name = self.model_var.get()
        category = self.category_var.get()
        
        if not model_name:
            messagebox.showwarning("Advertencia", "Seleccione un modelo para procesar")
            return
            
        func, param_names = self.get_model_function(model_name)
        
        if not func:
            messagebox.showerror("Error", f"Modelo {model_name} no encontrado")
            return
            
        x = self.data['x']
        y = self.data['y']
        
        if "Inverso" in model_name and np.any(x == 0):
            messagebox.showerror("Error", "Los modelos inversos no son válidos para valores de x = 0")
            return
            
        min_points_required = {
            "Lineal": 2,
            "Cuadrático": 3,
            "Cúbico": 4,
            "Inverso 1º orden": 2,
            "Inverso 2º orden": 3,
            "Inverso 3º orden": 4,
            "Gaussiana 3 parámetros": 3,
            "Gaussiana 4 parámetros": 4,
            "Weibull 4 parámetros (dist)": 4,
            "Weibull 5 parámetros (dist)": 5,
            "Pseudo-Voigt 4 parámetros": 4,
            "Pseudo-Voigt 5 parámetros": 5,
            "Gaussiana modificada 4 parámetros": 4,
            "Gaussiana modificada 5 parámetros": 5,
            "Lorentzian 3 parámetros": 3,
            "Lorentzian 4 parámetros": 4,
            "Log Normal 3 parámetros": 3,
            "Log Normal 4 parámetros": 4
        }.get(model_name, 2)
        
        if len(x) < min_points_required:
            messagebox.showerror("Error", f"El modelo {model_name} requiere al menos {min_points_required} puntos de datos")
            return
            
        try:
            p0 = self.get_initial_params(model_name, x, y)
            
            if p0 is None or not hasattr(p0, '__len__'):
                messagebox.showerror("Error", f"No se pudo estimar parámetros iniciales para {model_name}")
                return
            
            func, param_names = self.get_model_function(model_name)
            
            if len(p0) != len(param_names):
                messagebox.showerror("Error", f"Número de parámetros iniciales incorrecto para {model_name}")
                return
            
            #Ajustar el modelo y obtener la matriz de covarianza
            params, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
            if isinstance(params, (set, np.ndarray)):
                params = list(params)
            #validacion para ciencia bio
            if category == "Ciencia-Bio":
                is_valid, error_msg = self.validate_bio_parameters(model_name, params, x, y)
                if not is_valid:
                    messagebox.showerror("Error de validación", 
                                       f"Parámetros biológicamente inválidos para {model_name}:\n{error_msg}")
                    return
            
            y_pred = func(x, *params)
            
            # Bootstrapping si está activado
            bootstrap_results = None
            if self.bootstrap_var.get():
                try:
                    n_iter = int(self.bootstrap_iter_var.get())
                    conf_intervals, bootstrap_params = self.perform_bootstrap(
                        func, x, y, params, param_names, n_iterations=n_iter
                    )         
                    # Asegurar que los parámetros bootstrap también sean listas
                    for name in bootstrap_params:
                        if isinstance(bootstrap_params[name], set):
                            bootstrap_params[name] = list(bootstrap_params[name])
                    # Formatear parámetros con intervalos de confianza
                    param_text = ", ".join([
                        f"{name}={val:.4f} [{conf_intervals[name][0]:.4f}, {conf_intervals[name][1]:.4f}]" 
                        for name, val in zip(param_names, params)
                    ])
                    
                    bootstrap_results = {
                        'conf_intervals': conf_intervals,
                        'bootstrap_params': bootstrap_params
                    }
                    
                except Exception as e:
                    if not suppress_alerts:
                        messagebox.showwarning(
                            "Bootstrapping", 
                            f"Bootstrapping falló: {str(e)}\nMostrando resultados sin intervalos de confianza."
                        )
                    param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(param_names, params)])
            else:
                param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(param_names, params)])
                
            #calcular estadisticas de calidad de ajuste
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)) #r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            eval_text, eval_color = self.evaluate_r2(r2)
            param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(param_names, params)])
            
            #calcular errores estandar de los parametros
            perr = np.sqrt(np.diag(pcov)) if np.any(pcov) else np.zeros_like(params)
            
            #calcular estadisticas de error
            abs_errors = np.abs(residuals)
            rel_errors = np.abs(residuals/y) * 100 if np.all(y != 0) else np.zeros_like(residuals)
            
            if isinstance(params, set):
                params = list(params)  # Convierte set a lista
            
            # Crear reporte estadístico
            stats_report = f"""
Estadísticas de Ajuste para el Modelo: {model_name}
{'='*60}
Grados de libertad (error): {len(x) - len(params)}
Grados de libertad (regresión): {len(params)}
Coeficiente de determinación (R²): {r2:.6f}
R² ajustado: {1 - (1 - r2) * (len(x) - 1)/(len(x) - len(params) - 1):.6f}
Error Cuadrático Medio (RMSE): {np.sqrt(np.mean(residuals**2)):.6f}

Estimación de Parámetros:
{'='*60}"""
        
            for i, (name, param, err) in enumerate(zip(param_names, params, perr)):
                # Intervalo de confianza del 95%
                ci_low = param - 1.96 * err
                ci_high = param + 1.96 * err
            
            stats_report += f"""
{name} = {param:.6E}
Error estándar: {err:.6E}
Intervalo de confianza 95%: [{ci_low:.6E}, {ci_high:.6E}]"""
        
            stats_report += f"""

Estadísticas de Error:
{'='*60}
Error Absoluto:
Mínimo: {np.min(abs_errors):.6E}
Máximo: {np.max(abs_errors):.6E}
Media: {np.mean(abs_errors):.6E}
Desviación estándar: {np.std(abs_errors):.6E}

Error Relativo (%):
Mínimo: {np.min(rel_errors):.6E}
Máximo: {np.max(rel_errors):.6E}
Media: {np.mean(rel_errors):.6E}
Desviación estándar: {np.std(rel_errors):.6E}

{'='*60}
Nota: Los errores estándar e intervalos de confianza son aproximados
y asumen una distribución normal de los errores.
"""
            
            params_list = list(params) if isinstance(params, (set, np.ndarray)) else params

            self.results.append({
                "name": model_name,
                "category": category,
                "params": params_list,  # ← Ahora siempre será una lista
                "pcov": pcov,
                "r2": r2,
                "func": func,
                "x": x,
                "y": y,
                "y_pred": y_pred,
                "param_names": param_names,
                "stats_report": stats_report,
                "bootstrap_results": bootstrap_results
            })
            
            if not suppress_alerts:
                messagebox.showinfo("Éxito", f"Modelo {model_name} ajustado\nR² = {r2:.4f}")
            
            item = self.tree.insert("", "end", values=(model_name, category, f"{r2:.4f}", eval_text, param_text))
            self.tree.tag_configure(eval_text, background=eval_color)
            
        except RuntimeError as e:
            messagebox.showerror("Error", f"No se pudo ajustar el modelo {model_name}:\nEl algoritmo no convergió")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo ajustar el modelo {model_name}:\n{str(e)}")
    
    def get_model_equation(self, model_name, params, param_names):
        """Devuelve la representación matemática de la ecuación del modelo"""
        """Devuelve la representación matemática de la ecuación del modelo"""
        def format_param(val):
            return f"{val:.4g}"
        
        if model_name == "Sigmoidal 3 parámetros":
            return f"y = {params[0]:.4f} / (1 + exp(-(x - {params[2]:.4f})/{params[1]:.4f}))"
        elif model_name == "Sigmoidal 4 parámetros":
            return f"y = {params[3]:.4f} + {params[0]:.4f} / (1 + exp(-(x - {params[2]:.4f})/{params[1]:.4f}))"
        elif model_name == "Sigmoidal 5 parámetros":
            return f"y = {params[4]:.4f} + ({params[0]:.4f} / (1 + exp(-(x - {params[3]:.4f})/{params[1]:.4f})))^{params[2]:.4f}"
        elif model_name == "Logística 3 parámetros":
            return f"y = {params[0]:.4f} / (1 + |x/{params[2]:.4f}|^{params[1]:.4f})"
        elif model_name == "Logística 4 parámetros":
            return f"y = {params[3]:.4f} + {params[0]:.4f} / (1 + |x/{params[2]:.4f}|^{params[1]:.4f})"
        elif model_name == "Weibull 4 parámetros":
            return f"y = {params[0]:.4f} * (1 - exp(-((x - {params[3]:.4f} + {params[1]:.4f}*(ln(2)^{{1/{params[2]:.4f}}}))/{params[1]:.4f})^{params[2]:.4f}))"
        elif model_name == "Weibull 5 parámetros":
            return f"y = {params[4]:.4f} + {params[0]:.4f} * (1 - exp(-((x - {params[3]:.4f} + {params[1]:.4f}*(ln(2)^{{1/{params[2]:.4f}}}))/{params[1]:.4f})^{params[2]:.4f}))"
        elif model_name == "Gompertz 3 parámetros":
            return f"y = {params[0]:.4f} * exp(-exp(-(x - {params[2]:.4f})/{params[1]:.4f}))"
        elif model_name == "Gompertz 4 parámetros":
            return f"y = {params[3]:.4f} + {params[0]:.4f} * exp(-exp(-(x - {params[2]:.4f})/{params[1]:.4f}))"
        elif model_name == "Hill 3 parámetros":
            return f"y = ({params[0]:.4f} * x^{params[1]:.4f}) / ({params[2]:.4f}^{params[1]:.4f} + x^{params[1]:.4f})"
        elif model_name == "Hill 4 parámetros":
            return f"y = {params[3]:.4f} + ({params[0]:.4f} * x^{params[1]:.4f}) / ({params[2]:.4f}^{params[1]:.4f} + x^{params[1]:.4f})"
        elif model_name == "Chapman 3 parámetros":
            return f"y = {params[0]:.4f} * (1 - exp(-{params[1]:.4f} * x))^{params[2]:.4f}"
        elif model_name == "Chapman 4 parámetros":
            return f"y = {params[3]:.4f} + {params[0]:.4f} * (1 - exp(-{params[1]:.4f} * x))^{params[2]:.4f}"
        elif model_name == "Lineal":
            return f"y = {params[1]:.4f} + {params[0]:.4f} * x"
        elif model_name == "Cuadrático":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * x + {params[2]:.4f} * x²"
        elif model_name == "Cúbico":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * x + {params[2]:.4f} * x² + {params[3]:.4f} * x³"
        elif model_name == "Inverso 1º orden":
            return f"y = {params[0]:.4f} + {params[1]:.4f} / x"
        elif model_name == "Inverso 2º orden":
            return f"y = {params[0]:.4f} + {params[1]:.4f} / x + {params[2]:.4f} / x²"
        elif model_name == "Inverso 3º orden":
            return f"y = {params[0]:.4f} + {params[1]:.4f} / x + {params[2]:.4f} / x² + {params[3]:.4f} / x³"
        elif model_name == "Gaussiana 3 parámetros":
            return f"y = {params[0]:.4f} * exp(-0.5 * ((x - {params[2]:.4f})/{params[1]:.4f})²)"
        elif model_name == "Gaussiana 4 parámetros":
            return f"y = {params[3]:.4f} + {params[0]:.4f} * exp(-0.5 * ((x - {params[2]:.4f})/{params[1]:.4f})²)"
        elif model_name == "Weibull 4 parámetros (dist)":
            return f"y = {params[0]:.4f} * (({params[2]-1})/{params[2]})^{(1-{params[2]})/{params[2]}} * |(x-{params[3]})/{params[1]} + (({params[2]-1})/{params[2]})^{1/{params[2]}}|^{{params[2]-1}} * exp(-|(x-{params[3]})/{params[1]} + (({params[2]-1})/{params[2]})^{1/{params[2]}}|^{params[2]} + ({params[2]-1})/{params[2]})"
        elif model_name == "Weibull 5 parámetros (dist)":
            return f"y = {params[4]:.4f} + {params[0]:.4f} * (({params[2]-1})/{params[2]})^{(1-{params[2]})/{params[2]}} * |(x-{params[3]})/{params[1]} + (({params[2]-1})/{params[2]})^{1/{params[2]}}|^{{params[2]-1}} * exp(-|(x-{params[3]})/{params[1]} + (({params[2]-1})/{params[2]})^{1/{params[2]}}|^{params[2]} + ({params[2]-1})/{params[2]})"
        elif model_name == "Pseudo-Voigt 4 parámetros":
            return f"y = {params[0]:.4f} * ({params[2]:.4f} * (1/(1+((x-{params[3]:.4f})/{params[1]:.4f})²)) + (1-{params[2]:.4f}) * exp(-0.5*((x-{params[3]:.4f})/{params[1]:.4f})²))"
        elif model_name == "Pseudo-Voigt 5 parámetros":
            return f"y = {params[4]:.4f} + {params[0]:.4f} * ({params[2]:.4f} * (1/(1+((x-{params[3]:.4f})/{params[1]:.4f})²)) + (1-{params[2]:.4f}) * exp(-0.5*((x-{params[3]:.4f})/{params[1]:.4f})²))"
        elif model_name == "Gaussiana modificada 4 parámetros":
            return f"y = {params[0]:.4f} * exp(-0.5 * |(x-{params[3]:.4f})/{params[1]:.4f}|^{params[2]:.4f})"
        elif model_name == "Gaussiana modificada 5 parámetros":
            return f"y = {params[4]:.4f} + {params[0]:.4f} * exp(-0.5 * |(x-{params[3]:.4f})/{params[1]:.4f}|^{params[2]:.4f})"
        elif model_name == "Lorentzian 3 parámetros":
            return f"y = {params[0]:.4f} / (1 + ((x-{params[2]:.4f})/{params[1]:.4f})²)"
        elif model_name == "Lorentzian 4 parámetros":
            return f"y = {params[3]:.4f} + {params[0]:.4f} / (1 + ((x-{params[2]:.4f})/{params[1]:.4f})²)"
        elif model_name == "Log Normal 3 parámetros":
            return f"y = {params[0]:.4f} * exp(-0.5 * (ln(x/{params[2]:.4f})/{params[1]:.4f})²) / x"
        elif model_name == "Log Normal 4 parámetros":
            return f"y = {params[3]:.4f} + {params[0]:.4f} * exp(-0.5 * (ln(x/{params[2]:.4f})/{params[1]:.4f})²) / x"
        elif model_name == "Exponencial simple 2p":
            return f"y = {params[0]:.4f} * exp(-{params[1]:.4f} * x)"
        elif model_name == "Exponencial con offset 3p":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * exp(-{params[2]:.4f} * x)"
        elif model_name == "Doble exponencial 4p":
            return f"y = {params[0]:.4f} * exp(-{params[1]:.4f} * x) + {params[2]:.4f} * exp(-{params[3]:.4f} * x)"
        elif model_name == "Doble exponencial con offset 5p":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * exp(-{params[2]:.4f} * x) + {params[3]:.4f} * exp(-{params[4]:.4f} * x)"
        elif model_name == "Triple exponencial 6p":
            return f"y = {params[0]:.4f} * exp(-{params[1]:.4f} * x) + {params[2]:.4f} * exp(-{params[3]:.4f} * x) + {params[4]:.4f} * exp(-{params[5]:.4f} * x)"
        elif model_name == "Triple exponencial con offset 7p":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * exp(-{params[2]:.4f} * x) + {params[3]:.4f} * exp(-{params[4]:.4f} * x) + {params[5]:.4f} * exp(-{params[6]:.4f} * x)"
        elif model_name == "Combinación lineal-exponencial":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * exp(-{params[2]:.4f} * x) + {params[3]:.4f} * x"
        elif model_name == "Exponencial modificado 3p":
            return f"y = {params[0]:.4f} * exp({params[1]:.4f} / (x + {params[2]:.4f}))"
        elif model_name == "Crec. Exp. L. Simple 2p":
            return f"y = {params[0]:.4f} * (1 - exp(-{params[1]:.4f} * x))"
        elif model_name == "Crec. Exp. L. Simple 3p":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * (1 - exp(-{params[2]:.4f} * x))"
        elif model_name == "Crec. Exp. L. Doble 4p":
            return f"y = {params[0]:.4f} * (1 - exp(-{params[1]:.4f} * x)) + {params[2]:.4f} * (1 - exp(-{params[3]:.4f} * x))"
        elif model_name == "Crec. Exp. L. Doble 5p":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * (1 - exp(-{params[2]:.4f} * x)) + {params[3]:.4f} * (1 - exp(-{params[4]:.4f} * x))"
        elif model_name == "Crec. Exp. L. Simple Alt 2p":
            return f"y = {params[0]:.4f} * (1 - {params[1]:.4f}^x)"
        elif model_name == "Crec. Exp. L. Simple Alt 3p":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * (1 - {params[2]:.4f}^x)"
        elif model_name == "Potencia 2 parámetros":
            return f"y = {params[0]:.4f} * x^{params[1]:.4f}"
        elif model_name == "Potencia 2p Modificado I":
            return f"y = {params[0]:.4f} * (1 - x^{-params[1]:.4f})"
        elif model_name == "Potencia 2p Modificado II":
            return f"y = {params[0]:.4f} * (1 + x)^{params[1]:.4f}"
        elif model_name == "Potencia 3 parámetros":
            return f"y = {params[0]:.4f} + {params[1]:.4f} * x^{params[2]:.4f}"
        elif model_name == "Potencia Simétrica 3p":
            return f"y = {params[0]:.4f} * |x - {params[1]:.4f}|^{params[2]:.4f}"
        elif model_name == "Potencia Simétrica 4p":
            return f"y = {params[3]:.4f} + {params[0]:.4f} * |x - {params[1]:.4f}|^{params[2]:.4f}"
        elif model_name == "Función Pareto":
            return f"y = 1 - 1/x^{params[0]:.4f}"
        elif model_name == "Función Pareto Modificada":
            return f"y = 1 - 1/(1 + {params[0]:.4f} * x)^{params[1]:.4f}"
        elif model_name == "Beverton-Holt A (BH-A)":
            return f"y = {params[0]:.4f} / (1 + (({params[0]:.4f}-1)/{params[1]:.4f}) * x)"
        elif model_name == "Beverton-Holt B (BH-B)":
            return f"y = {params[0]:.4f} * x / (1 + (({params[0]:.4f}-1)/{params[1]:.4f}) * x)"
        elif model_name == "von Bertalanffy Growth (VBG)":
            return f"y = {params[0]:.4f} * (1 - exp(-{params[1]:.4f} * (x - {params[2]:.4f})))"
        elif model_name == "Michaelis-Menten básico (MM)":
            return f"y = {params[0]:.4f} * x / ({params[1]:.4f} + x)"
        elif model_name == "Michaelis-Menten doble (MMD)":
            return f"y = {params[0]:.4f} * x / ({params[1]:.4f} + x) + {params[2]:.4f} * x / ({params[3]:.4f} + x)"
        elif model_name == "Dose-Response A (DR-A)":
            return f"y = {params[1]:.4f} + ({params[0]:.4f} - {params[1]:.4f}) / (1 + 10^(x - {params[2]:.4f}))"
        elif model_name == "Dose-Response B (DR-B)":
            return f"y = {params[1]:.4f} + ({params[0]:.4f} - {params[1]:.4f}) / (1 + 10^({params[2]:.4f} - x))"
        elif model_name == "Dose-Response C (DR-C)":
            return f"y = {params[1]:.4f} + ({params[0]:.4f} - {params[1]:.4f}) / (1 + 10^({params[3]:.4f} * (x - {params[2]:.4f})))"
        elif model_name == "Dose-Response D (DR-D)":
            return f"y = {params[1]:.4f} + ({params[0]:.4f} - {params[1]:.4f}) / (1 + 10^({params[3]:.4f} * ({params[2]:.4f} - x)))"
        elif model_name == "Dose-Response E (DR-E)":
            return f"y = {params[1]:.4f} + ({params[0]:.4f} - {params[1]:.4f}) / (1 + (x/{params[2]:.4f})^{params[3]:.4f})"
        elif model_name == "Enfermedad vegetal Exponencial (PV-Exp)":
            return f"y = {params[0]:.4f} * exp({params[1]:.4f} * x)"
        elif model_name == "Enfermedad vegetal Gompertz (PV-Gomp)":
            return f"y = exp(ln({params[0]:.4f}) * exp(-{params[1]:.4f} * x))"
        elif model_name == "Enfermedad vegetal Logística (PV-Log)":
            return f"y = 1 / (1 + (1 - {params[0]:.4f})/({params[0]:.4f} * exp(-{params[1]:.4f} * x)))"
        elif model_name == "Enfermedad vegetal Monomolecular (PV-Mono)":
            return f"y = 1 - ((1 - {params[0]:.4f}) * exp(-{params[1]:.4f} * x))"
        elif model_name == "Enfermedad vegetal Weibull (PV-Weib)":
            return f"y = 1 - exp(-1.0 * ((x - {params[0]:.4f})/{params[1]:.4f})^{params[2]:.4f})"
        elif model_name == "High-Low Affinity (HLA)":
            return f"y = {params[0]:.4f} * {params[1]:.4f} * x / (1 + {params[1]:.4f} * x)"
        elif model_name == "High-Low Affinity doble (HLAD)":
            return f"y = {params[0]:.4f} * {params[1]:.4f} * x / (1 + {params[1]:.4f} * x) + {params[2]:.4f} * {params[3]:.4f} * x / (1 + {params[3]:.4f} * x)"
        elif model_name == "Logística 3P estándar (Log3P)":
            return f"y = {params[2]:.4f} + ({params[0]:.4f} - {params[2]:.4f}) / (1 + x/{params[1]:.4f})"
        elif model_name == "Logística 4P estándar (Log4P)":
            return f"y = {params[3]:.4f} + ({params[0]:.4f} - {params[3]:.4f}) / (1 + (x/{params[2]:.4f})^{params[1]:.4f})"
        elif model_name == "Logística 5P estándar (Log5P)":
            return f"y = {params[3]:.4f} + ({params[0]:.4f} - {params[3]:.4f}) / (1 + (x/{params[2]:.4f})^{params[1]:.4f})^{params[4]:.4f}"
        elif model_name == "Transporte de membrana (MT)":
            return f"y = {params[0]:.4f} * (x - {params[1]:.4f}) / (x² + {params[2]:.4f} * x + {params[3]:.4f})"
        elif model_name == "Aphid Population Growth (APHID)":
            return f"y = {params[0]:.4f} * exp({params[1]:.4f} * x) * (1 + {params[2]:.4f} * exp({params[1]:.4f} * x))^(-2)"
        elif model_name == "Negative Exponential generalizada (NEG)":
            return f"y = {params[0]:.4f} * (1.0 - exp(-{params[1]:.4f} * x))^{params[2]:.4f}"
        elif model_name == "Hyperbolic Tangent 4p (HT)":
            return f"y = ({params[0]:.4f} + {params[1]:.4f})/2 + ({params[1]:.4f} - {params[0]:.4f})/2 * tanh((x - {params[2]:.4f})/{params[3]:.4f})"
        elif model_name == "Hyperbolic Tangent 5p (HT)":
            return f"y = {params[4]:.4f} + ({params[0]:.4f} + {params[1]:.4f})/2 + ({params[1]:.4f} - {params[0]:.4f})/2 * tanh((x - {params[2]:.4f})/{params[3]:.4f})"
        elif model_name == "Asymmetric Hyperbolic Tangent 5p (AHT)":
            return f"y = ({params[0]:.4f} + {params[1]:.4f})/2 + ({params[1]:.4f} - {params[0]:.4f})/2 * tanh((x - {params[2]:.4f})/({params[3]:.4f} + {params[4]:.4f}*x))"
        elif model_name == "Asymmetric Hyperbolic Tangent 6p (AHT)":
            return f"y = {params[5]:.4f} + ({params[0]:.4f} + {params[1]:.4f})/2 + ({params[1]:.4f} - {params[0]:.4f})/2 * tanh((x - {params[2]:.4f})/({params[3]:.4f} + {params[4]:.4f}*x))"
        elif model_name == "Burr Model 5p (BUR)":
            return f"y = {params[0]:.4f} + ({params[1]:.4f} - {params[0]:.4f}) * [1 + exp(-{params[2]:.4f}*(x - {params[3]:.4f}))]^(-{params[4]:.4f})"
        elif model_name == "Burr Model 6p (BUR)":
            return f"y = {params[5]:.4f} + {params[0]:.4f} + ({params[1]:.4f} - {params[0]:.4f}) * [1 + exp(-{params[2]:.4f}*(x - {params[3]:.4f}))]^(-{params[4]:.4f})"
        elif model_name == "Arctangent Model 4p (ACT)":
            return f"y = ({params[0]:.4f} + {params[1]:.4f})/2 + ({params[1]:.4f} - {params[0]:.4f})/2 * arctan(π/(2*{params[3]:.4f}) * (x - {params[2]:.4f}))"
        elif model_name == "Arctangent Model 5p (ACT)":
            return f"y = {params[4]:.4f} + ({params[0]:.4f} + {params[1]:.4f})/2 + ({params[1]:.4f} - {params[0]:.4f})/2 * arctan(π/(2*{params[3]:.4f}) * (x - {params[2]:.4f}))"
        elif model_name == "Asymmetric Kohout Model 5p (KHT)":
            return f"Para x ≤ {params[3]:.4f}: y = {params[0]:.4f} + ({params[1]:.4f} - {params[0]:.4f})/(1 + {params[4]:.4f}) * exp((1 + {params[4]:.4f})/(2*{params[2]:.4f}) * (x - {params[3]:.4f}))\nPara x > {params[3]:.4f}: y = {params[0]:.4f} - {params[4]:.4f}*({params[1]:.4f} - {params[0]:.4f})/(1 + {params[4]:.4f}) * exp(-(1 + {params[4]:.4f})/(2*{params[2]:.4f}) * (x - {params[3]:.4f}))"
        elif model_name == "Asymmetric Kohout Model 6p (KHT)":
            return f"Para x ≤ {params[3]:.4f}: y = {params[5]:.4f} + {params[0]:.4f} + ({params[1]:.4f} - {params[0]:.4f})/(1 + {params[4]:.4f}) * exp((1 + {params[4]:.4f})/(2*{params[2]:.4f}) * (x - {params[3]:.4f}))\nPara x > {params[3]:.4f}: y = {params[5]:.4f} + {params[0]:.4f} - {params[4]:.4f}*({params[1]:.4f} - {params[0]:.4f})/(1 + {params[4]:.4f}) * exp(-(1 + {params[4]:.4f})/(2*{params[2]:.4f}) * (x - {params[3]:.4f}))"
        elif model_name == "Monotonic Four Parameter (MFP)":
            return f"y = {params[0]:.4f} + {params[2]:.4f} * (x - {params[1]:.4f}) * |x - {params[1]:.4f}|^({params[3]:.4f} - 1)"
        elif model_name == "Quadratic Four Parameter (QFP)":
            if abs(params[3]) < 1e-10:
                return f"y = {params[0]:.4f} + {params[1]:.4f} * ln(x) + {params[2]:.4f} * (ln(x))^2"
            else:
                return f"y = {params[0]:.4f} + {params[1]:.4f} * (x^{params[3]:.4f} - 1)/{params[3]:.4f} + {params[2]:.4f} * [(x^{params[3]:.4f} - 1)/{params[3]:.4f}]^2"
        elif model_name == "SVD 2D 6 parámetros":
            return f"z = {params[0]:.4f} * {params[1]:.4f} * {params[3]:.4f} * x + {params[0]:.4f} * {params[2]:.4f} * {params[4]:.4f} * y"
        elif model_name == "SVD 2D 9 parámetros":
            return f"z = {params[0]:.4f} * {params[2]:.4f} * {params[6]:.4f} * x + {params[0]:.4f} * {params[3]:.4f} * {params[7]:.4f} * y + {params[1]:.4f} * {params[4]:.4f} * {params[6]:.4f} * x + {params[1]:.4f} * {params[5]:.4f} * {params[7]:.4f} * y"
        elif model_name == "Dispersión Óptica 2D":
            return f"y = {params[0]:.4f} + {params[1]:.4f}*x² + {params[2]:.4f}/x² + {params[3]:.4f}/x⁴"
        elif model_name == "Dispersión Óptica Raíz Cuadrada 2D":
            return f"y = √({params[0]:.4f} + {params[1]:.4f}*x² + {params[2]:.4f}/x² + {params[3]:.4f}/x⁴)"
        elif model_name == "Litografía por Haz de Electrones 2D":
            return f"y = {params[0]:.4f}*exp(-{params[1]:.4f}*x) + {params[2]:.4f}*exp(-((x-{params[3]:.4f})/{params[4]:.4f})²) + {params[5]:.4f}*exp(-((x-{params[6]:.4f})/{params[7]:.4f})²) + {params[8]:.4f}*exp(-((x-{params[9]:.4f})/{params[10]:.4f})²)"
        elif model_name == "Steinhart-Hart Extendido 2D":
            return f"y = {params[0]:.4f} + {params[1]:.4f}*ln(R) + {params[2]:.4f}*(ln(R))² + {params[3]:.4f}*(ln(R))³"
        elif model_name == "Motor Eléctrico Graeme Paterson 2D":
            return f"y = {params[0]:.4f}*exp(-{params[1]:.4f}*t)*cos({params[2]:.4f}*t + {params[3]:.4f}) + {params[4]:.4f}*exp(-{params[5]:.4f}*t)"
        elif model_name == "Klimpel Cinética Flotación A 2D":
            return f"y = {params[0]:.4f} * [1 - (1 - exp(-{params[1]:.4f}*x)) / ({params[1]:.4f}*x)]"
        elif model_name == "Maxwell-Wiechert 1 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f})"
        elif model_name == "Maxwell-Wiechert 2 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}*exp(-X/{params[3]:.4f})"
        elif model_name == "Maxwell-Wiechert 3 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}*exp(-X/{params[3]:.4f}) + {params[4]:.4f}*exp(-X/{params[5]:.4f})"
        elif model_name == "Maxwell-Wiechert 4 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}*exp(-X/{params[3]:.4f}) + {params[4]:.4f}*exp(-X/{params[5]:.4f}) + {params[6]:.4f}*exp(-X/{params[7]:.4f})"
        elif model_name == "Producción de Pozo Arps Modificado 2D":
            return f"y = ({params[0]:.4f}/((1.0-{params[1]:.4f})*{params[2]:.4f})) * (1.0-((1.0+{params[1]:.4f}*{params[2]:.4f}*x)^(1.0-1.0/{params[1]:.4f})))"
        elif model_name == "Ramberg-Osgood 2D":
            return f"y = (Stress/{params[0]:.4f}) + (Stress/{params[1]:.4f})^(1.0/{params[2]:.4f})"
        elif model_name == "Steinhart-Hart Extendido Recíproco 2D":
            return f"y = 1.0 / ({params[0]:.4f} + {params[1]:.4f}*ln(R) + {params[2]:.4f}*(ln(R))² + {params[3]:.4f}*(ln(R))³)"
        elif model_name == "Steinhart-Hart Recíproco 2D":
            return f"y = 1.0 / ({params[0]:.4f} + {params[1]:.4f}*ln(R) + {params[2]:.4f}*(ln(R))³)"
        elif model_name == "Sellmeier Óptico 2D":
            return f"y = 1 + ({params[0]:.4f}*x²)/(x²-{params[1]:.4f}) + ({params[2]:.4f}*x²)/(x²-{params[3]:.4f}) + ({params[4]:.4f}*x²)/(x²-{params[5]:.4f})"
        elif model_name == "Sellmeier Óptico Raíz Cuadrada 2D":
            return f"y = √[1 + ({params[0]:.4f}*x²)/(x²-{params[1]:.4f}) + ({params[2]:.4f}*x²)/(x²-{params[3]:.4f}) + ({params[4]:.4f}*x²)/(x²-{params[5]:.4f})]"
        elif model_name == "Steinhart-Hart 2D":
            return f"y = {params[0]:.4f} + {params[1]:.4f}*ln(R) + {params[2]:.4f}*(ln(R))³"
        elif model_name == "VanDeemter Cromatografía 2D":
            return f"y = {params[0]:.4f} + {params[1]:.4f}/x + {params[2]:.4f}*x"
        elif model_name == "Litografía por Haz de Electrones con Offset 2D":
            return f"y = {params[0]:.4f}*exp(-{params[1]:.4f}*x) + {params[2]:.4f}*exp(-((x-{params[3]:.4f})/{params[4]:.4f})²) + {params[5]:.4f}*exp(-((x-{params[6]:.4f})/{params[7]:.4f})²) + {params[8]:.4f}*exp(-((x-{params[9]:.4f})/{params[10]:.4f})²) + {params[11]:.4f}"
        elif model_name == "Motor Eléctrico Graeme Paterson con Offset 2D":
            return f"y = {params[0]:.4f}*exp(-{params[1]:.4f}*t)*cos({params[2]:.4f}*t + {params[3]:.4f}) + {params[4]:.4f}*exp(-{params[5]:.4f}*t) + {params[6]:.4f}"
        elif model_name == "Klimpel Cinética Flotación A con Offset 2D":
            return f"y = {params[0]:.4f} * [1 - (1 - exp(-{params[1]:.4f}*x)) / ({params[1]:.4f}*x)] + {params[2]:.4f}"
        elif model_name == "Maxwell-Wiechert 1 con Offset 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}"
        elif model_name == "Maxwell-Wiechert 2 con Offset 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}*exp(-X/{params[3]:.4f}) + {params[4]:.4f}"
        elif model_name == "Maxwell-Wiechert 3 con Offset 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}*exp(-X/{params[3]:.4f}) + {params[4]:.4f}*exp(-X/{params[5]:.4f}) + {params[6]:.4f}"
        elif model_name == "Maxwell-Wiechert 4 con Offset 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}*exp(-X/{params[3]:.4f}) + {params[4]:.4f}*exp(-X/{params[5]:.4f}) + {params[6]:.4f}*exp(-X/{params[7]:.4f}) + {params[8]:.4f}"
        elif model_name == "Producción de Pozo Arps Modificado con Offset 2D":
            return f"y = ({params[0]:.4f}/((1.0-{params[1]:.4f})*{params[2]:.4f})) * (1.0-((1.0+{params[1]:.4f}*{params[2]:.4f}*x)^(1.0-1.0/{params[1]:.4f}))) + {params[3]:.4f}"
        elif model_name == "Ramberg-Osgood con Offset 2D":
            return f"y = (Stress/{params[0]:.4f}) + (Stress/{params[1]:.4f})^(1.0/{params[2]:.4f}) + {params[3]:.4f}"
        elif model_name == "Steinhart-Hart Extendido Recíproco con Offset 2D":
            return f"y = 1.0 / ({params[0]:.4f} + {params[1]:.4f}*ln(R) + {params[2]:.4f}*(ln(R))² + {params[3]:.4f}*(ln(R))³) + {params[4]:.4f}"
        elif model_name == "Steinhart-Hart Recíproco con Offset 2D":
            return f"y = 1.0 / ({params[0]:.4f} + {params[1]:.4f}*ln(R) + {params[2]:.4f}*(ln(R))³) + {params[3]:.4f}"
        elif model_name == "Sellmeier Óptico con Offset 2D":
            return f"y = 1 + ({params[0]:.4f}*x²)/(x²-{params[1]:.4f}) + ({params[2]:.4f}*x²)/(x²-{params[3]:.4f}) + ({params[4]:.4f}*x²)/(x²-{params[5]:.4f}) + {params[6]:.4f}"
        elif model_name == "Sellmeier Óptico Raíz Cuadrada con Offset 2D":
            return f"y = √[1 + ({params[0]:.4f}*x²)/(x²-{params[1]:.4f}) + ({params[2]:.4f}*x²)/(x²-{params[3]:.4f}) + ({params[4]:.4f}*x²)/(x²-{params[5]:.4f})] + {params[6]:.4f}"
        elif model_name == "Klimpel Cinética Flotación A más Línea 2D":
            return f"y = {params[0]:.4f} * [1 - (1 - exp(-{params[1]:.4f}*x)) / ({params[1]:.4f}*x)] + {params[2]:.4f}*x + {params[3]:.4f}"
        elif model_name == "Maxwell-Wiechert 1 más Línea 2D":
            return f"y = {params[0]:.4f}*exp(-X/{params[1]:.4f}) + {params[2]:.4f}*X + {params[3]:.4f}"
        else:
            # Para modelos no definidos específicamente, usar formato genérico
            equation = f"y = {self.get_model_function(model_name)[0].__name__}("
            equation += ", ".join([f"{name}={format_param(val)}" for name, val in zip(param_names, params)])
            equation += ")"
            return equation

    def show_selected_stats(self):
        selected = self.tree.focus()
        if not selected:
            messagebox.showwarning("Advertencia", "Seleccione un modelo de la tabla")
            return
            
        item = self.tree.item(selected)
        model_name = item['values'][0]
        
        result = next((r for r in self.results if r["name"] == model_name), None)
        if not result:
            messagebox.showerror("Error", "No hay datos estadísticos para este modelo")
            return
        
        equation = self.get_model_equation(result["name"], result["params"], result["param_names"])
        
        # Calcular errores estándar si no están en los resultados
        if 'pcov' not in result:
            try:
                _, pcov = curve_fit(result["func"], result["x"], result["y"], 
                                  p0=result["params"], maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
            except:
                perr = np.zeros_like(result["params"])
        else:
            perr = np.sqrt(np.diag(result['pcov']))
        
        # Crear reporte base
        stats_report = f"""
    {'='*80}
    ESTADÍSTICAS DE AJUSTE - {model_name.upper()}
    {'='*80}
    
    ► Datos del Modelo:
    {'─'*40}
    Categoría: {result["category"]}
    Ecuación: {equation}
    Función: {result["func"].__name__}
    Puntos de datos: {len(result["x"])}
    Parámetros: {len(result["params"])}
    
    ► Estadísticas de Calidad de Ajuste:
    {'─'*40}
    Coeficiente de determinación (R²): {result["r2"]:.6f}
    R² ajustado: {1 - (1 - result["r2"]) * (len(result["x"]) - 1)/(len(result["x"]) - len(result["params"]) - 1):.6f}
    Error Cuadrático Medio (RMSE): {np.sqrt(np.mean((result["y"] - result["y_pred"])**2)):.6f}
    
    ► Estimación de Parámetros:
    {'─'*40}"""
        
        # Añadir información de cada parámetro
        for name, param, err in zip(result["param_names"], result["params"], perr):
            ci_low = param - 1.96 * err
            ci_high = param + 1.96 * err
            
            stats_report += f"""
    {name + ':':<15} {param:.6E}
    {'Error estándar:':<15} {err:.6E}
    {'IC 95%:':<15} [{ci_low:.6E}, {ci_high:.6E}]
    {'─'*40}"""
        
        # Añadir estadísticas de error
        residuals = result["y"] - result["y_pred"]
        abs_errors = np.abs(residuals)
        rel_errors = np.abs(residuals/result["y"]) * 100 if np.all(result["y"] != 0) else np.zeros_like(residuals)
        
        stats_report += f"""
    
    ► Estadísticas de Error:
    {'─'*40}
    Error Absoluto:
        Mínimo: {np.min(abs_errors):.6E}
        Máximo: {np.max(abs_errors):.6E}
        Media: {np.mean(abs_errors):.6E}
        Desviación estándar: {np.std(abs_errors):.6E}
    
    Error Relativo (%):
        Mínimo: {np.min(rel_errors):.6E}%
        Máximo: {np.max(rel_errors):.6E}%
        Media: {np.mean(rel_errors):.6E}%
        Desviación estándar: {np.std(rel_errors):.6E}%
    """
        
        # Añadir bootstrapping si existe
        if result.get("bootstrap_results"):
            conf_intervals = result["bootstrap_results"]["conf_intervals"]
            bootstrap_params = result["bootstrap_results"]["bootstrap_params"]
            n_samples = len(next(iter(bootstrap_params.values())))
            
            stats_report += f"""
    
    ► Resultados de Bootstrapping:
    {'─'*40}
    Muestras bootstrap exitosas: {n_samples}
    
    """
            
            for name in result["param_names"]:
                if name in conf_intervals:
                    param_values = bootstrap_params[name]
                    stats_report += f"""
    {name + ':':<15} {result['params'][result['param_names'].index(name)]:.6E}
    {'IC 95% bootstrap:':<15} [{conf_intervals[name][0]:.6E}, {conf_intervals[name][1]:.6E}]
    {'Error estándar:':<15} {np.std(param_values):.6E}
    {'─'*40}"""
        
        stats_report += f"""
    {'='*80}
    Nota: Los intervalos de confianza asumen una distribución normal de los errores.
    Los resultados de bootstrapping pueden ser más robustos para muestras pequeñas.
    {'='*80}"""
        
        # Crear ventana para mostrar el reporte
        stats_window = tk.Toplevel(self.root)
        stats_window.title(f"Estadísticas de Ajuste - {model_name}")
        stats_window.geometry("800x700")
        stats_window.iconbitmap(resource_path("icon_1.ico"))
        
        # Frame principal con scrollbars
        main_frame = ttk.Frame(stats_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget para mostrar el reporte
        text_widget = tk.Text(
            main_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            padx=10,
            pady=10,
            tabs=('0.5i', '1i', '1.5i')  # Configuración de tabulaciones
        )
        text_widget.insert(tk.END, stats_report)
        text_widget.config(state=tk.DISABLED)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=y_scroll.set)
        
        # Layout
        text_widget.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Botón de cierre
        ttk.Button(
            stats_window,
            text="Cerrar",
            command=stats_window.destroy
        ).pack(pady=10)
    
    def ask_min_r2(self):
        """Ventana emergente para preguntar el R² mínimo deseado."""
        
        if self.bootstrap_var.get():
            messagebox.showinfo(
                "Información", 
                "El bootstrapping no se aplicará cuando se procesan todos los modelos.\n"
                "Para usar bootstrapping, procese los modelos individualmente."
            )
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Filtrar por calidad de ajuste")
        dialog.geometry("400x200")
        dialog.resizable(False, False)
        dialog.grab_set()  # Bloquea la ventana principal hasta cerrar esta
    
        # Texto explicativo
        label = ttk.Label(
            dialog, 
            text="Ingrese el R² mínimo aceptable (0.0 a 1.0):\n\n" +
                 "Ejemplo: 0.7 para mostrar solo modelos con R² ≥ 70%",
            justify=tk.CENTER
        )
        label.pack(pady=10)
    
        # Campo de entrada
        entry = ttk.Entry(dialog, font=('Arial', 12), width=10)
        entry.pack(pady=10)
        entry.insert(0, "0.0")  # Valor por defecto
    
        # Función al hacer clic en "Aceptar"
        def on_accept():
            try:
                min_r2 = float(entry.get())
                if 0.0 <= min_r2 <= 1.0:
                    self.clear_results()  # Limpiar resultados anteriores
                    self.process_all_models(min_r2)  # Procesar con el filtro
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "¡R² debe estar entre 0.0 y 1.0!")
            except ValueError:
                messagebox.showerror("Error", "Ingrese un número válido (ej: 0.7)")
    
        # Botones
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
    
        ttk.Button(
            btn_frame, 
            text="Aceptar", 
            style='Accent.TButton',
            command=on_accept
        ).pack(side=tk.LEFT, padx=10)
    
        ttk.Button(
            btn_frame, 
            text="Cancelar", 
            command=dialog.destroy
        ).pack(side=tk.RIGHT, padx=10)        
    
    def show_selected_residuals(self):
        """Muestra los gráficos de residuos y residuos normalizados en pestañas"""
        selected = self.tree.focus()
        if not selected:
            messagebox.showwarning("Advertencia", "Seleccione un modelo de la tabla")
            return
            
        item = self.tree.item(selected)
        model_name = item['values'][0]
        
        result = next((r for r in self.results if r["name"] == model_name), None)
        if not result:
            messagebox.showerror("Error", "No hay datos para este modelo")
            return
        
        # Crear ventana para los gráficos de residuos
        residuals_window = tk.Toplevel(self.root)
        residuals_window.title(f"Análisis de Residuos - {result['name']}")
        residuals_window.geometry("1000x800")
        residuals_window.iconbitmap(resource_path("icon_1.ico"))
        
        # Crear notebook con pestañas
        notebook = ttk.Notebook(residuals_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Residuos vs X
        tab_residuals = ttk.Frame(notebook)
        notebook.add(tab_residuals, text="Residuos vs Variable Independiente")
        
        # Pestaña 2: Residuos Normalizados
        tab_normalized = ttk.Frame(notebook)
        notebook.add(tab_normalized, text="Residuos Normalizados")
        
        # Calcular residuos
        residuals = result["y"] - result["y_pred"]
        
        # Calcular residuos normalizados (estandarizados)
        if len(residuals) > 1:
            residuals_std = np.std(residuals)
            if residuals_std > 0:
                normalized_residuals = residuals / residuals_std
            else:
                normalized_residuals = np.zeros_like(residuals)
        else:
            normalized_residuals = np.zeros_like(residuals)
        
        # Gráfico 1: Residuos vs X
        fig1 = plt.Figure(figsize=(10, 6), dpi=100)
        ax1 = fig1.add_subplot(111)
        
        ax1.scatter(result["x"], residuals, alpha=0.6, color='blue', s=50)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Variable Independiente (X)')
        ax1.set_ylabel('Residuos')
        ax1.set_title(f'Residuos vs X - {result["name"]}')
        ax1.grid(True, alpha=0.3)
        
        # Añadir información estadística
        stats_text = (f"Media residuos: {np.mean(residuals):.4f}\n"
                     f"Desv. estándar: {np.std(residuals):.4f}\n"
                     f"R²: {result['r2']:.4f}")
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        canvas1 = FigureCanvasTkAgg(fig1, master=tab_residuals)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gráfico 2: Residuos Normalizados
        fig2 = plt.Figure(figsize=(10, 6), dpi=100)
        ax2 = fig2.add_subplot(111)
        
        ax2.scatter(result["x"], normalized_residuals, alpha=0.6, color='green', s=50)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        # Líneas para ±2 desviaciones estándar (aproximadamente 95% de los datos)
        ax2.axhline(y=2, color='orange', linestyle=':', linewidth=1.5)
        ax2.axhline(y=-2, color='orange', linestyle=':', linewidth=1.5)
        
        ax2.set_xlabel('Variable Independiente (X)')
        ax2.set_ylabel('Residuos Normalizados')
        ax2.set_title(f'Residuos Normalizados vs X - {result["name"]}')
        ax2.grid(True, alpha=0.3)
        
        # Añadir información estadística
        norm_stats_text = (f"Media residuos norm.: {np.mean(normalized_residuals):.4f}\n"
                          f"Desv. estándar: {np.std(normalized_residuals):.4f}\n"
                          f"Fuera de ±2σ: {np.sum(np.abs(normalized_residuals) > 2)} puntos")
        ax2.text(0.02, 0.98, norm_stats_text, transform=ax2.transAxes,
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        canvas2 = FigureCanvasTkAgg(fig2, master=tab_normalized)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Botón de cierre
        ttk.Button(residuals_window, 
                 text="Cerrar", 
                 command=residuals_window.destroy).pack(pady=10)
        
        # Configurar cierre correcto de las figuras
        def on_close():
            plt.close(fig1)
            plt.close(fig2)
            residuals_window.destroy()
        
        residuals_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def process_all_models(self, min_r2=0.0):
        """Procesa todos los modelos de la categoría actual y filtra por R²."""
        if not self.data or len(self.data['x']) == 0:
            messagebox.showwarning("Advertencia", "Primero cargue un archivo con datos")
            return
    
        # Desactivar bootstrapping temporalmente si está activado
        was_bootstrap_enabled = self.bootstrap_var.get()
        if was_bootstrap_enabled:
            self.bootstrap_var.set(False)
            messagebox.showinfo(
                "Información", 
                "El bootstrapping ha sido desactivado temporalmente.\n"
                "No se aplicará al procesar todos los modelos."
            )
    
        all_models = self.update_model_list()  # Obtener todos los modelos
        total_models = sum(len(models) for models in all_models.values())
        success_models = 0
        filtered_models = 0  # Modelos que cumplen R² ≥ min_r2
    
        # Procesar cada modelo
        for category, models in all_models.items():
            for model in models:
                try:
                    self.model_var.set(model)
                    self.category_var.set(category)  # Establecer categoría actual
                    self.process_selected_model(suppress_alerts=True)
                    success_models += 1
    
                    # Filtrar por R²
                    last_result = self.results[-1]
                    if last_result["r2"] >= min_r2:
                        filtered_models += 1
                    else:
                        for item in self.tree.get_children():
                            if self.tree.item(item)['values'][0] == model:
                                self.tree.delete(item)
                                break
                except Exception as e:
                    print(f"Error en {model}: {str(e)}")

        if was_bootstrap_enabled:
            self.bootstrap_var.set(True)

        # Mostrar resumen
        messagebox.showinfo(
            "Resumen",
            f"► Modelos procesados: {total_models}\n"
            f"► Ajustes exitosos: {success_models}\n"
            f"► Modelos con R² ≥ {min_r2:.2f}: {filtered_models}"
        )
         
    def clear_results(self):
        self.results = []
        for item in self.tree.get_children():
            self.tree.delete(item)
            
    def show_selected_plot(self):
        """
        Muestra los gráficos del modelo seleccionado.
        - Siempre muestra el gráfico de la función ajustada
        - Si bootstrap está activado y hay datos, muestra además los gráficos de distribución de parámetros
        """
        # Verificar si hay un modelo seleccionado en la tabla
        selected = self.tree.focus()
        if not selected:
            messagebox.showwarning("Advertencia", "Seleccione un modelo de la tabla")
            return
        
        # Obtener información del modelo seleccionado
        item = self.tree.item(selected)
        model_name = item['values'][0]
        
        # Buscar los resultados del modelo
        result = next((r for r in self.results if r["name"] == model_name), None)
        if not result:
            messagebox.showerror("Error", "No hay datos de gráfico para este modelo")
            return
        
        # 1. Mostrar siempre el gráfico principal de ajuste (con bandas de confianza si hay bootstrap)
        self._show_main_fit_plot(result)
        
        # 2. Si bootstrap está activado y hay resultados, mostrar gráficos de distribución de parámetros
        if self.bootstrap_var.get() and result.get("bootstrap_results"):
            self._show_bootstrap_distributions(result)
    
    def _show_main_fit_plot(self, result):
        """Muestra el gráfico principal con los datos y la curva ajustada"""
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Ajuste: {result['name']}")
        plot_window.geometry("900x700")
        plot_window.iconbitmap(resource_path("icon_1.ico"))
        
        fig = plt.Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Configuración del gráfico
        ax.set_title(f"{result['name']}\n({result['category']})", fontsize=14, pad=20)
        ax.set_xlabel('Variable independiente (X)', fontsize=12)
        ax.set_ylabel('Variable dependiente (Y)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 1. Graficar datos originales
        ax.scatter(result["x"], result["y"], 
                  color='royalblue', label='Datos observados', 
                  s=60, alpha=0.7, edgecolors='w')
        
        # 2. Graficar curva ajustada
        x_vals = np.linspace(min(result["x"]), max(result["x"]), 500)
        y_vals = result["func"](x_vals, *result["params"])
        ax.plot(x_vals, y_vals, 'r-', linewidth=2.5, 
               label='Modelo ajustado', alpha=0.9)
        
        # 3. Si hay bootstrap, agregar bandas de confianza
        if result.get("bootstrap_results"):
            bootstrap_params = result["bootstrap_results"]["bootstrap_params"]
            n_samples = len(next(iter(bootstrap_params.values())))
            
            if n_samples > 0:
                # Calcular predicciones para todas las muestras bootstrap
                y_samples = np.zeros((n_samples, len(x_vals)))
                for i in range(n_samples):
                    params_sample = [bootstrap_params[name][i] for name in result["param_names"]]
                    y_samples[i] = result["func"](x_vals, *params_sample)
                
                # Calcular intervalos de confianza
                lower = np.percentile(y_samples, 2.5, axis=0)
                upper = np.percentile(y_samples, 97.5, axis=0)
                
                # Graficar bandas de confianza
                ax.fill_between(x_vals, lower, upper, 
                              color='coral', alpha=0.2, 
                              label='IC 95% (bootstrap)')
        
        # Añadir leyenda y métricas
        ax.legend(fontsize=10, loc='best')
        stats_text = f"R² = {result['r2']:.4f}\n"
        stats_text += f"Parámetros: {', '.join([f'{n}={v:.3g}' for n,v in zip(result['param_names'], result['params'])])}"
        
        ax.annotate(stats_text, 
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=11, bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                   ha='left', va='top')
        
        # Mostrar en la ventana
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configurar cierre correcto
        def on_close():
            plt.close(fig)
            plot_window.destroy()
        plot_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def _show_bootstrap_distributions(self, result):
        """Muestra gráficos de distribución de parámetros del bootstrap"""
        bootstrap_params = result["bootstrap_results"]["bootstrap_params"]
        n_samples = len(next(iter(bootstrap_params.values())))
        
        if n_samples == 0:
            messagebox.showinfo("Información", "No hay muestras bootstrap disponibles")
            return
        
        # Crear una ventana con pestañas para cada parámetro
        dist_window = tk.Toplevel(self.root)
        dist_window.title(f"Distribuciones Bootstrap - {result['name']}")
        dist_window.geometry("1000x800")
        dist_window.iconbitmap(resource_path("icon_1.ico"))
        
        notebook = ttk.Notebook(dist_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for param_name in result["param_names"]:
            if param_name not in bootstrap_params:
                continue
                
            param_values = bootstrap_params[param_name]
            param_estimate = result["params"][result["param_names"].index(param_name)]
            
            # Crear pestaña para este parámetro
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=param_name)
            
            # Crear figura
            fig = plt.Figure(figsize=(9, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Histograma de distribución
            ax.hist(param_values, bins=30, color='steelblue', 
                   alpha=0.7, density=True)
            
            # Líneas de referencia
            ax.axvline(param_estimate, color='red', linestyle='--', 
                      linewidth=2, label='Estimado')
            
            # Calcular IC 95%
            lower, upper = np.percentile(param_values, [2.5, 97.5])
            ax.axvline(lower, color='green', linestyle=':', 
                      linewidth=1.5, label='IC 95%')
            ax.axvline(upper, color='green', linestyle=':', 
                      linewidth=1.5)
            
            # Configuración del gráfico
            ax.set_title(f"Distribución Bootstrap - {param_name}", fontsize=14)
            ax.set_xlabel("Valor del parámetro", fontsize=12)
            ax.set_ylabel("Densidad", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Mostrar estadísticas
            stats_text = (f"Estimado: {param_estimate:.4f}\n"
                         f"IC 95%: [{lower:.4f}, {upper:.4f}]\n"
                         f"Media: {np.mean(param_values):.4f}\n"
                         f"Desv. Est.: {np.std(param_values):.4f}")
            
            ax.text(0.98, 0.85, stats_text, transform=ax.transAxes,
                   ha='right', va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Mostrar en la pestaña
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Configurar cierre correcto
            def on_close(fig=fig):
                plt.close(fig)
            tab.bind("<Destroy>", lambda e: on_close())
        
        # Botón de cierre
        ttk.Button(dist_window, text="Cerrar", 
                   command=dist_window.destroy).pack(pady=10)
        
    def export_results(self):
        if not self.results:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt")],
            title="Guardar resultados como"
        )
        
        if not filepath:
            return
            
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Encabezado (existente)
                f.write("="*80 + "\n")
                f.write("RESULTADOS DE REGRESIÓN - REGRESSION WIZARD\n".center(80) + "\n")
                f.write("="*80 + "\n\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de puntos de datos: {len(self.data['x'])}\n")
                f.write(f"Total de modelos procesados: {len(self.results)}\n")
                f.write("="*80 + "\n\n")
                
                # Datos originales (existente)
                f.write("DATOS ORIGINALES:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'X':<20}{'Y':<20}\n")
                f.write("-"*40 + "\n")
                for x, y in zip(self.data['x'], self.data['y']):
                    f.write(f"{x:<20.6f}{y:<20.6f}\n")
                f.write("\n" + "="*80 + "\n\n")
                
                # Organizar por categoría (existente)
                categories = set(r['category'] for r in self.results)
                
                for cat in sorted(categories):
                    f.write(f"CATEGORÍA: {cat.upper()}\n")
                    f.write("="*80 + "\n")
                    
                    cat_results = [r for r in self.results if r['category'] == cat]
                    cat_results.sort(key=lambda x: x['r2'], reverse=True)
                    
                    for result in cat_results:
                        eval_text, _ = self.evaluate_r2(result["r2"])
                        
                        # Obtener la ecuación del modelo
                        equation = self.get_model_equation(result["name"], result["params"], result["param_names"])
                        
                        # Encabezado del modelo - AÑADIR LA ECUACIÓN
                        f.write(f"\nMODELO: {result['name']}\n")
                        f.write("-"*80 + "\n")
                        f.write(f"R²: {result['r2']:.6f} ({eval_text})\n")
                        f.write(f"ECUACIÓN: {equation}\n")  # <-- AÑADIR ESTA LÍNEA
                        
                        # Parámetros con bootstrap si está disponible
                        f.write("\nPARÁMETROS:\n")
                        f.write("-"*40 + "\n")
                        
                        # Mostrar tanto los errores estándar como los intervalos de bootstrap
                        perr = np.sqrt(np.diag(np.atleast_2d(result.get('pcov', np.zeros((len(result['params']), len(result['params'])))))) if 'pcov' in result else np.zeros_like(result['params']))
                        
                        for i, (name, param, err) in enumerate(zip(result["param_names"], result["params"], perr)):
                            ci_low = param - 1.96 * err
                            ci_high = param + 1.96 * err
                            
                            f.write(f"{name:<10} = {param:>12.6E} ± {err:.3E}\n")
                            f.write(f"{'IC 95% (normal)':<15} [{ci_low:.6E}, {ci_high:.6E}]\n")
                            
                            # Añadir información de bootstrap si existe
                            if result.get("bootstrap_results") and name in result["bootstrap_results"]["conf_intervals"]:
                                bs_low, bs_high = result["bootstrap_results"]["conf_intervals"][name]
                                f.write(f"{'IC 95% (bootstrap)':<15} [{bs_low:.6E}, {bs_high:.6E}]\n")
                                
                                # Calcular error estándar bootstrap
                                bs_values = result["bootstrap_results"]["bootstrap_params"][name]
                                bs_std = np.std(bs_values) if bs_values else np.nan
                                f.write(f"{'Error std (bs)':<15} {bs_std:.6E}\n")
                            
                            f.write("\n")
                        
                        # Estadísticas de error (existente)
                        if "stats_report" in result:
                            f.write("\nESTADÍSTICAS DE ERROR:\n")
                            f.write("-"*40 + "\n")
                            f.write(result["stats_report"].split("Estadísticas de Error:")[-1])
                        
                        # Sección específica de bootstrap si existe
                        if result.get("bootstrap_results"):
                            f.write("\nRESULTADOS DE BOOTSTRAPPING:\n")
                            f.write("-"*40 + "\n")
                            bs_params = result["bootstrap_results"]["bootstrap_params"]
                            n_samples = len(next(iter(bs_params.values())))
                            
                            f.write(f"Muestras bootstrap exitosas: {n_samples}\n")
                            f.write("-"*40 + "\n")
                            
                            for name in result["param_names"]:
                                if name in bs_params:
                                    values = bs_params[name]
                                    if values:
                                        f.write(f"{name}:\n")
                                        f.write(f"  Media bootstrap: {np.mean(values):.6E}\n")
                                        f.write(f"  Mediana bootstrap: {np.median(values):.6E}\n")
                                        f.write(f"  Desviación estándar: {np.std(values):.6E}\n")
                                        f.write(f"  Rango: [{np.min(values):.6E}, {np.max(values):.6E}]\n")
                                        f.write("-"*40 + "\n")
                        
                        f.write("\n" + "="*80 + "\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("FIN DEL REPORTE\n".center(80) + "\n")
                f.write("="*80 + "\n")
            
            messagebox.showinfo("Éxito", f"Resultados exportados correctamente a:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar los resultados:\n{str(e)}")
        
    def _format_scientific(self, num):
        """Formatea números en notación científica legible"""
        if abs(num) < 1e-3 or abs(num) > 1e3:
            return f"{num:.4E}"
        return f"{num:.6f}" 

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()