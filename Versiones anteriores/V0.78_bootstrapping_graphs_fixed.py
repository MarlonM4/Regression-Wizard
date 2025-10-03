import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import Menu
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        self.root.title("Regression Wizard - v0.78 bootstraping update :3")
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
                                            values=["Sigmoidales", "Polinomiales", "Inversos polinomiales", "Modelos de campana", "Decaimiento Exponencial", "Crecimiento exponencial limitado", "Modelos de Potencia"],
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
            "Función Pareto Modificada": (power_mod_pareto, ["a", "b"])
            
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
        else:
            return None
    
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
            
            if p0 is None:
                messagebox.showerror("Error", f"No se pudo estimar parámetros iniciales para {model_name}")
                return
            
            #Ajustar el modelo y obtener la matriz de covarianza
            params, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
            y_pred = func(x, *params)
            
            # Bootstrapping si está activado
            bootstrap_results = None
            if self.bootstrap_var.get():
                try:
                    n_iter = int(self.bootstrap_iter_var.get())
                    conf_intervals, bootstrap_params = self.perform_bootstrap(
                        func, x, y, params, param_names, n_iterations=n_iter
                    )
                    
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
            
            self.results.append({
                "name": model_name,
                "category": category,
                "params": params,
                "pcov":pcov,
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
                f.write(f"Fecha: {np.datetime64('today')}\n")
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
                        
                        # Encabezado del modelo
                        f.write(f"\nMODELO: {result['name']}\n")
                        f.write("-"*80 + "\n")
                        f.write(f"R²: {result['r2']:.6f} ({eval_text})\n")
                        f.write(f"Función: y = {self._get_model_equation(result)}\n")
                        
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

    def _get_model_equation(self, result):
        """Devuelve una representación textual de la ecuación del modelo"""
        # Esta es una implementación básica - puedes mejorarla para cada modelo específico
        equation = f"{result['func'].__name__}("
        equation += ", ".join([f"{name}={val:.3g}" for name, val in zip(result['param_names'], result['params'])])
        equation += ")"
        return equation
    
    def _format_scientific(self, num):
        """Formatea números en notación científica legible"""
        if abs(num) < 1e-3 or abs(num) > 1e3:
            return f"{num:.4E}"
        return f"{num:.6f}" 

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()