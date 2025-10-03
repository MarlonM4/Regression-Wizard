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
        self.root.title("Regression Wizard - v0.77")
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
        
        ttk.Button(
            button_frame, 
            text="PROCESAR TODOS LOS MODELOS", 
            style='Process.TButton',
            command=self.ask_min_r2  # Ahora abre la ventana de filtro
        ).pack(side="right", padx=5)
        
        self.update_model_list()
    
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
                "r2": r2,
                "func": func,
                "x": x,
                "y": y,
                "y_pred": y_pred,
                "param_names": param_names,
                "stats_report": stats_report
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
        if not result or "stats_report" not in result:
            messagebox.showerror("Error", "No hay datos estadísticos para este modelo")
            return
        
        # Crear ventana para mostrar el reporte
        stats_window = tk.Toplevel(self.root)
        stats_window.title(f"Estadísticas de Ajuste - {model_name}")
        stats_window.geometry("500x600")
        
        # Frame principal con scrollbars
        main_frame = ttk.Frame(stats_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget para mostrar el reporte
        text_widget = tk.Text(
            main_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            padx=10,
            pady=10
        )
        text_widget.insert(tk.END, result["stats_report"])
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
        selected = self.tree.focus()
        if not selected:
            messagebox.showwarning("Advertencia", "Seleccione un modelo de la tabla")
            return
            
        item = self.tree.item(selected)
        model_name = item['values'][0]
        
        result = next((r for r in self.results if r["name"] == model_name), None)
        if not result:
            messagebox.showerror("Error", "No hay datos de gráfico para este modelo")
            return
            
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Gráfico: {model_name}")
        plot_window.geometry("800x800")
        plot_window.iconbitmap(resource_path("icon_1.ico"))
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.scatter(result["x"], result["y"], color='blue', label='Datos reales', s=50)
        
        x_vals = np.linspace(min(result["x"]), max(result["x"]), 200)
        y_vals = result["func"](x_vals, *result["params"])
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Modelo ajustado')
        
        ax.set_title(f"{model_name}\n({result['category']})", fontsize=12, pad=20)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        ax.annotate(f"R² = {result['r2']:.4f}", 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, bbox=dict(boxstyle="round", fc="w"))
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
          
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
                f.write("RESULTADOS DE REGRESIÓN - REGRESSION WIZARD\n")
                f.write("="*60 + "\n\n")
                f.write(f"Fecha: {np.datetime64('today')}\n")
                f.write(f"Total modelos procesados: {len(self.results)}\n")
                f.write("="*60 + "\n\n")
                
                f.write("DATOS ORIGINALES:\n")
                f.write("X\tY\n")
                for x, y in zip(self.data['x'], self.data['y']):
                    f.write(f"{x:.4f}\t{y:.4f}\n")
                f.write("\n" + "="*60 + "\n\n")
                
                categories = set(r['category'] for r in self.results)
                
                for cat in sorted(categories):
                    f.write(f"CATEGORÍA: {cat.upper()}\n")
                    f.write("-"*60 + "\n")
                    
                    cat_results = [r for r in self.results if r['category'] == cat]
                    cat_results.sort(key=lambda x: x['r2'], reverse=True)
                    
                    for result in cat_results:
                        eval_text, _ = self.evaluate_r2(result["r2"])
                        param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(result["param_names"], result["params"])])
                        
                        f.write(f"Modelo: {result['name']}\n")
                        f.write(f"R²: {result['r2']:.4f} ({eval_text})\n")
                        f.write(f"Parámetros: {param_text}\n")
                        f.write("-"*40 + "\n")
                    
                    if "stats_report" in result:
                        f.write("\n" + result["stats_report"] + "\n")
                    
                    f.write("-"*60 + "\n")
                    
                    f.write("\n")
                
                f.write("="*60 + "\n")
                f.write("FIN DEL REPORTE\n")
            
            messagebox.showinfo("Éxito", f"Resultados exportados correctamente a:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar los resultados:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()