import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

# =============================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# =============================================

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Wizard - Versión Completa")
        self.root.geometry("1600x800")
        self.root.resizable(False, False)
        self.data = {'x': [], 'y': []}
        self.results = []
        
        self.setup_styles()
        self.create_widgets()
    
    def setup_styles(self):
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Accent.TButton', foreground='white', background='#4a6baf', font=('Arial', 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#3a5b9f')])
        style.configure('Process.TButton', foreground='black', background='#cccccc', font=('Arial', 10, 'bold'))
        style.map('Process.TButton', background=[('active', '#bbbbbb')])
        style.configure('TCombobox', padding=5)
        style.configure('Custom.TButton', foreground='black', background='#2E8B57', font=('Arial', 10, 'bold'))
        style.map('Custom.TButton', background=[('active', '#3CB371')])
    
    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_data_entry(self.main_frame)
        self.create_model_selection(self.main_frame)
        self.create_results_display(self.main_frame)
    
    def create_data_entry(self, parent):
        entry_frame = ttk.LabelFrame(parent, text="Carga de Datos", padding=10)
        entry_frame.pack(fill=tk.X, pady=5)
        
        control_frame = ttk.Frame(entry_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, 
         text="Cargar Archivo TXT", 
         style='Custom.TButton',  # Nuevo estilo
         command=self.load_txt_file).pack(side="left", padx=5)
        
        ttk.Label(control_frame, 
                 text="Formato: 2 columnas (X y Y) separadas por espacio o tabulación").pack(side="left", padx=10)
        
        # Área de texto para mostrar datos
        self.data_display = tk.Text(entry_frame, 
                                  height=6, 
                                  wrap=tk.WORD,
                                  state='disabled')
        scroll_y = ttk.Scrollbar(entry_frame, 
                               orient="vertical", 
                               command=self.data_display.yview)
        self.data_display.configure(yscrollcommand=scroll_y.set)
        
        self.data_display.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
    
    def create_model_selection(self, parent):
        model_frame = ttk.LabelFrame(parent, text="Selección de Modelos", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # Frame para los dropdowns
        dropdown_frame = ttk.Frame(model_frame)
        dropdown_frame.pack(fill=tk.X, pady=5)
        
        # Dropdown para categorías
        ttk.Label(dropdown_frame, text="Categoría:").pack(side="left", padx=5)
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(dropdown_frame, 
                                            textvariable=self.category_var,
                                            values=["Sigmoidales", "Polinomiales", "Inversos polinomiales"],
                                            state="readonly",
                                            width=15)
        self.category_dropdown.pack(side="left", padx=5)
        self.category_dropdown.current(0)
        self.category_dropdown.bind("<<ComboboxSelected>>", self.update_model_list)
        
        # Dropdown para modelos específicos
        ttk.Label(dropdown_frame, text="Modelo:").pack(side="left", padx=5)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(dropdown_frame, 
                                         textvariable=self.model_var,
                                         state="readonly",
                                         width=25)
        self.model_dropdown.pack(side="left", padx=5, expand=True, fill=tk.X)
        
        # Frame para botones
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Botón de procesamiento
        ttk.Button(button_frame, 
                 text="PROCESAR MODELO SELECCIONADO", 
                 style='Process.TButton', 
                 command=self.process_selected_model).pack(side="left", padx=5)
        
        ttk.Button(button_frame,
                 text="PROCESAR TODOS LOS MODELOS",
                 style='Process.TButton',
                 command=self.process_all_models).pack(side="right", padx=5)
        
        # Actualizar lista inicial de modelos
        self.update_model_list()
    
    def update_model_list(self, event=None):
        category = self.category_var.get()
        
        if category == "Sigmoidales":
            models = [
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
            ]
        elif category == "Polinomiales":
            models = [
                "Lineal",
                "Cuadrático",
                "Cúbico"
            ]
        else:  # Inversos polinomiales
            models = [
                "Inverso 1º orden",
                "Inverso 2º orden",
                "Inverso 3º orden"
            ]
        
        self.model_dropdown['values'] = models
        if models:
            self.model_dropdown.current(0)
    
    def create_results_display(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Resultados", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Modelo", "Categoría", "R²", "Evaluación", "Parámetros")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        # Configurar columnas
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
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Frame para botones de acción
        action_frame = ttk.Frame(results_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, 
                 text="Ver Gráfico", 
                 command=self.show_selected_plot).pack(side="left", padx=5)
        
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
            
            # Mostrar datos en el área de texto
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
            "Inverso 3º orden": (inverse_third_order, ["y0", "a", "b", "c"])
        }
        
        return model_functions.get(model_name, (None, None))
    
    def get_initial_params(self, model_name, x, y):
        if model_name == "Lineal":
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
            return [np.max(y)-np.min(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2)), np.min(y)]]
        elif model_name == "Sigmoidal 5 parámetros":
            return [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2)), np.min(y)]]
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
        else:
            return None
    
    def process_selected_model(self):
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
        
        # Validaciones específicas
        if "Inverso" in model_name and np.any(x == 0):
            messagebox.showerror("Error", "Los modelos inversos no son válidos para valores de x = 0")
            return
            
        min_points_required = {
            "Lineal": 2,
            "Cuadrático": 3,
            "Cúbico": 4,
            "Inverso 1º orden": 2,
            "Inverso 2º orden": 3,
            "Inverso 3º orden": 4
        }.get(model_name, 2)
        
        if len(x) < min_points_required:
            messagebox.showerror("Error", f"El modelo {model_name} requiere al menos {min_points_required} puntos de datos")
            return
            
        try:
            # Obtener parámetros iniciales
            p0 = self.get_initial_params(model_name, x, y)
            
            if p0 is None:
                messagebox.showerror("Error", f"No se pudo estimar parámetros iniciales para {model_name}")
                return
            
            # Ajustar el modelo
            params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
            y_pred = func(x, *params)
            
            # Cálculo de R²
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            eval_text, eval_color = self.evaluate_r2(r2)
            param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(param_names, params)])
            
            # Agregar a resultados
            self.results.append({
                "name": model_name,
                "category": category,
                "params": params,
                "r2": r2,
                "func": func,
                "x": x,
                "y": y,
                "y_pred": y_pred,
                "param_names": param_names
            })
            
            # Actualizar árbol
            item = self.tree.insert("", "end", values=(model_name, category, f"{r2:.4f}", eval_text, param_text))
            self.tree.tag_configure(eval_text, background=eval_color)
            
            messagebox.showinfo("Éxito", f"Modelo {model_name} ajustado correctamente\nR² = {r2:.4f}")
            
        except RuntimeError as e:
            messagebox.showerror("Error", f"No se pudo ajustar el modelo {model_name}:\nEl algoritmo no convergió")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo ajustar el modelo {model_name}:\n{str(e)}")
    
    def process_all_models(self):
        if not self.data or len(self.data['x']) == 0:
            messagebox.showwarning("Advertencia", "Primero cargue un archivo con datos")
            return
            
        original_category = self.category_var.get()
        original_model = self.model_var.get()
        
        # Procesar todos los modelos por categoría
        categories = ["Sigmoidales", "Polinomiales", "Inversos polinomiales"]
        total_models = 0
        success_models = 0
        
        for cat in categories:
            self.category_var.set(cat)
            self.update_model_list()
            
            for model in self.model_dropdown['values']:
                total_models += 1
                self.model_var.set(model)
                try:
                    self.process_selected_model()
                    success_models += 1
                except:
                    continue
        
        # Restaurar selección original
        self.category_var.set(original_category)
        self.update_model_list()
        self.model_var.set(original_model)
        
        messagebox.showinfo("Proceso completado", 
                          f"Se procesaron {total_models} modelos\n"
                          f"{success_models} ajustes exitosos\n"
                          f"{total_models - success_models} fallidos")
    
    def clear_results(self):
        self.results = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        messagebox.showinfo("Limpiar", "Todos los resultados han sido eliminados")
    
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
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Datos originales
        ax.scatter(result["x"], result["y"], color='blue', label='Datos reales', s=50)
        
        # Curva ajustada
        x_vals = np.linspace(min(result["x"]), max(result["x"]), 200)
        y_vals = result["func"](x_vals, *result["params"])
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Modelo ajustado')
        
        # Configuración del gráfico
        ax.set_title(f"{model_name}\n({result['category']})", fontsize=12, pad=20)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Mostrar R²
        ax.annotate(f"R² = {result['r2']:.4f}", 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, bbox=dict(boxstyle="round", fc="w"))
        
        # Integrar gráfico en la ventana
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
                # Encabezado
                f.write("RESULTADOS DE REGRESIÓN - REGRESSION WIZARD\n")
                f.write("="*60 + "\n\n")
                f.write(f"Fecha: {np.datetime64('today')}\n")
                f.write(f"Total modelos procesados: {len(self.results)}\n")
                f.write("="*60 + "\n\n")
                
                # Datos originales
                f.write("DATOS ORIGINALES:\n")
                f.write("X\tY\n")
                for x, y in zip(self.data['x'], self.data['y']):
                    f.write(f"{x:.4f}\t{y:.4f}\n")
                f.write("\n" + "="*60 + "\n\n")
                
                # Resultados por categoría
                categories = set(r['category'] for r in self.results)
                
                for cat in sorted(categories):
                    f.write(f"CATEGORÍA: {cat.upper()}\n")
                    f.write("-"*60 + "\n")
                    
                    # Filtrar resultados por categoría
                    cat_results = [r for r in self.results if r['category'] == cat]
                    
                    # Ordenar por R² descendente
                    cat_results.sort(key=lambda x: x['r2'], reverse=True)
                    
                    for result in cat_results:
                        eval_text, _ = self.evaluate_r2(result["r2"])
                        param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(result["param_names"], result["params"])])
                        
                        f.write(f"Modelo: {result['name']}\n")
                        f.write(f"R²: {result['r2']:.4f} ({eval_text})\n")
                        f.write(f"Parámetros: {param_text}\n")
                        f.write("-"*40 + "\n")
                    
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