import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 1. DEFINICIÓN DE LOS 13 MODELOS (versión corregida)
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

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ajuste de Modelos Científicos")
        self.root.geometry("1000x700")
        self.data = {}
        self.results = []
        
        # Configurar estilo
        self.setup_styles()
        self.create_widgets()
    
    def setup_styles(self):
        """Configura los estilos visuales"""
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Accent.TButton', foreground='white', background='#4a6baf', font=('Arial', 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#3a5b9f')])
    
    def create_widgets(self):
        """Crea todos los componentes de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. Sección de entrada de datos
        self.create_data_entry(main_frame)
        
        # 2. Sección de resultados
        self.create_results_display(main_frame)
    
    def create_data_entry(self, parent):
        """Crea el área para ingresar datos por año"""
        entry_frame = ttk.LabelFrame(parent, text="Ingreso de Datos por Año", padding=10)
        entry_frame.pack(fill=tk.X, pady=5)
        
        # Controles para rango de años
        control_frame = ttk.Frame(entry_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Año inicial:").grid(row=0, column=0, padx=5)
        self.start_year = ttk.Spinbox(control_frame, from_=1900, to=2100, width=8)
        self.start_year.grid(row=0, column=1, padx=5)
        self.start_year.set(2000)
        
        ttk.Label(control_frame, text="Año final:").grid(row=0, column=2, padx=5)
        self.end_year = ttk.Spinbox(control_frame, from_=1900, to=2100, width=8)
        self.end_year.grid(row=0, column=3, padx=5)
        self.end_year.set(2023)
        
        ttk.Button(control_frame, text="Generar Campos", 
                  command=self.generate_fields).grid(row=0, column=4, padx=10)
        
        # Área para campos de datos con scroll
        self.data_canvas = tk.Canvas(entry_frame, height=150)
        self.scrollbar = ttk.Scrollbar(entry_frame, orient="vertical", command=self.data_canvas.yview)
        self.data_frame = ttk.Frame(self.data_canvas)
        
        self.data_frame.bind("<Configure>", lambda e: self.data_canvas.configure(scrollregion=self.data_canvas.bbox("all")))
        self.data_canvas.create_window((0, 0), window=self.data_frame, anchor="nw")
        self.data_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.data_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Botón para procesar
        ttk.Button(parent, text="PROCESAR MODELOS", style='Accent.TButton',
                  command=self.process_models).pack(pady=10)
    
    def create_results_display(self, parent):
        """Crea el área para mostrar resultados"""
        results_frame = ttk.LabelFrame(parent, text="Resultados de Modelos", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview para mostrar resultados
        columns = ("Modelo", "R²", "Evaluación", "Parámetros")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        # Configurar columnas
        self.tree.heading("Modelo", text="Modelo")
        self.tree.heading("R²", text="R²")
        self.tree.heading("Evaluación", text="Evaluación")
        self.tree.heading("Parámetros", text="Parámetros")
        
        self.tree.column("Modelo", width=200)
        self.tree.column("R²", width=100, anchor="center")
        self.tree.column("Evaluación", width=150, anchor="center")
        self.tree.column("Parámetros", width=400)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Barra de desplazamiento
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Botones de acción
        btn_frame = ttk.Frame(results_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Ver Gráfico", 
                  command=self.show_selected_plot).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Exportar CSV", 
                  command=self.export_results).pack(side="left", padx=5)
    
    def generate_fields(self):
        """Genera campos de entrada para cada año"""
        for widget in self.data_frame.winfo_children():
            widget.destroy()
            
        try:
            start = int(self.start_year.get())
            end = int(self.end_year.get())
            
            if start > end:
                messagebox.showerror("Error", "El año inicial debe ser menor al año final")
                return
                
            self.data = {}
            for i, year in enumerate(range(start, end+1)):
                row = i % 5
                col = i // 5
                
                frame = ttk.Frame(self.data_frame)
                frame.grid(row=row, column=col, padx=5, pady=2, sticky="w")
                
                ttk.Label(frame, text=f"{year}:").pack(side="left")
                var = tk.DoubleVar()
                entry = ttk.Entry(frame, textvariable=var, width=10)
                entry.pack(side="left")
                self.data[year] = var
                
        except ValueError:
            messagebox.showerror("Error", "Ingrese años válidos")
    
    def evaluate_r2(self, r2):
        """Evalúa la calidad del ajuste basado en R²"""
        if r2 >= 0.9: return ("Excelente", "#4CAF50")  # Verde
        elif r2 >= 0.7: return ("Bueno", "#8BC34A")    # Verde claro
        elif r2 >= 0.5: return ("Moderado", "#FFC107") # Amarillo
        else: return ("Pobre", "#F44336")              # Rojo
    
    def process_models(self):
        """Procesa todos los modelos con los datos ingresados"""
        if not self.data:
            messagebox.showwarning("Advertencia", "Primero genere los campos de datos")
            return
            
        # Obtener y validar datos
        years = sorted(self.data.keys())
        y_values = [self.data[year].get() for year in years]
        
        if all(v == 0 for v in y_values):
            messagebox.showerror("Error", "Ingrese valores distintos de cero")
            return
            
        x = np.array(years)
        y = np.array(y_values)
        
        # Limpiar resultados anteriores
        self.results = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Definición de modelos con parámetros iniciales (CORREGIDOS)
        modelos = [
            ("Sigmoidal 3 parámetros", sigmoidal_3params, 
             [np.max(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))]]),
            ("Sigmoidal 4 parámetros", sigmoidal_4params,
             [np.max(y)-np.min(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))], np.min(y)]),
            ("Sigmoidal 5 parámetros", sigmoidal_5params,
             [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))], np.min(y)]),
            ("Logística 3 parámetros", logistic_3params,
             [np.max(y), 1.0, np.median(x)]),
            ("Logística 4 parámetros", logistic_4params,
             [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)]),
            ("Weibull 4 parámetros", weibull_4params,
             [np.max(y), (x[-1]-x[0])/10, 1.0, np.median(x)]),
            ("Weibull 5 parámetros", weibull_5params,
             [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, np.median(x), np.min(y)]),
            ("Gompertz 3 parámetros", gompertz_3params,
             [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x)]),
            ("Gompertz 4 parámetros", gompertz_4params,
             [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x), np.min(y)]),
            ("Hill 3 parámetros", hill_3params,
             [np.max(y), 1.0, np.median(x)]),
            ("Hill 4 parámetros", hill_4params,
             [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)]),
            ("Chapman 3 parámetros", chapman_3params,
             [np.max(y), 2/np.median(x), 1.0]),
            ("Chapman 4 parámetros", chapman_4params,
             [np.max(y)-np.min(y), 2/np.median(x), 1.0, np.min(y)])
        ]
        
        # Ajustar cada modelo
        for name, func, p0 in modelos:
            try:
                params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
                y_pred = func(x, *params)
                
                # Calcular R²
                residuals = y - y_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Evaluar calidad
                eval_text, eval_color = self.evaluate_r2(r2)
                
                # Formatear parámetros
                param_text = ", ".join([f"{p:.4f}" for p in params])
                
                # Guardar resultados
                self.results.append({
                    "name": name,
                    "params": params,
                    "r2": r2,
                    "func": func,
                    "x": x,
                    "y": y,
                    "y_pred": y_pred
                })
                
                # Insertar en treeview
                item = self.tree.insert("", "end", values=(name, f"{r2:.4f}", eval_text, param_text))
                self.tree.tag_configure(eval_text, background=eval_color)
                
            except Exception as e:
                self.tree.insert("", "end", values=(name, "Error", str(e), ""))
                self.results.append({"name": name, "error": str(e)})
    
    def show_selected_plot(self):
        """Muestra el gráfico del modelo seleccionado"""
        selected = self.tree.focus()
        if not selected:
            messagebox.showwarning("Advertencia", "Seleccione un modelo de la tabla")
            return
            
        item = self.tree.item(selected)
        model_name = item['values'][0]
        
        # Buscar resultado correspondiente
        result = next((r for r in self.results if r["name"] == model_name), None)
        if not result or "error" in result:
            messagebox.showerror("Error", "No hay datos de gráfico para este modelo")
            return
            
        # Crear ventana de gráfico
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Gráfico: {model_name}")
        plot_window.geometry("800x600")
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Datos originales
        ax.scatter(result["x"], result["y"], color='blue', label='Datos reales', s=50)
        
        # Curva ajustada
        x_vals = np.linspace(min(result["x"]), max(result["x"]), 100)
        y_vals = result["func"](x_vals, *result["params"])
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Modelo ajustado')
        
        # Configuración del gráfico
        ax.set_title(model_name, fontsize=14, pad=20)
        ax.set_xlabel('Año', fontsize=12)
        ax.set_ylabel('Valor', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Añadir R² al gráfico
        ax.annotate(f"R² = {result['r2']:.4f}", 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, bbox=dict(boxstyle="round", fc="w"))
        
        # Integrar gráfico en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Botón para cerrar
        ttk.Button(plot_window, text="Cerrar", 
                  command=plot_window.destroy).pack(pady=5)
    
    def export_results(self):
        """Exporta los resultados a CSV"""
        if not self.results:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
            
        from tkinter import filedialog
        import csv
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv")],
            title="Guardar resultados como"
        )
        
        if not filepath:
            return
            
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Modelo", "R²", "Evaluación", "Parámetros"])
                
                for result in self.results:
                    if "error" in result:
                        writer.writerow([result["name"], "Error", result["error"], ""])
                    else:
                        eval_text, _ = self.evaluate_r2(result["r2"])
                        param_text = ", ".join([f"{p:.4f}" for p in result["params"]])
                        writer.writerow([result["name"], f"{result['r2']:.4f}", eval_text, param_text])
                
            messagebox.showinfo("Éxito", f"Resultados exportados a:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()