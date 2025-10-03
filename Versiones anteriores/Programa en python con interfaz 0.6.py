import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 1. DEFINICIÓN DE LOS 13 MODELOS MATEMÁTICOS
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
        self.root.title("Regression Wizard de Marlon M")
        self.root.geometry("1000x800")
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
    
    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_data_entry(self.main_frame)
        self.create_results_display(self.main_frame)
    
    def create_data_entry(self, parent):
        entry_frame = ttk.LabelFrame(parent, text="Carga de Datos", padding=10)
        entry_frame.pack(fill=tk.X, pady=5)
        
        control_frame = ttk.Frame(entry_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, 
                  text="Cargar Archivo TSV", 
                  command=self.load_tsv_file).pack(side="left", padx=5)
        
        ttk.Label(control_frame, text="Formato esperado: Columna 0 (X) | Columna 1 (Y)").pack(side="left", padx=10)
        
        # Frame para mostrar datos cargados
        self.data_display = tk.Text(entry_frame, 
                                  height=8, 
                                  wrap=tk.NONE, 
                                  state='disabled')
        scroll_x = ttk.Scrollbar(entry_frame, 
                               orient="horizontal", 
                               command=self.data_display.xview)
        scroll_y = ttk.Scrollbar(entry_frame, 
                               orient="vertical", 
                               command=self.data_display.yview)
        self.data_display.configure(xscrollcommand=scroll_x.set,
                                  yscrollcommand=scroll_y.set)
        
        self.data_display.pack(side="top", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        
        ttk.Button(parent, 
                  text="PROCESAR MODELOS", 
                  style='Process.TButton', 
                  command=self.process_models).pack(pady=10)
    
    def create_results_display(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Resultados de Modelos", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Modelo", "R²", "Evaluación", "Parámetros")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        self.tree.heading("Modelo", text="Modelo")
        self.tree.heading("R²", text="R²")
        self.tree.heading("Evaluación", text="Evaluación")
        self.tree.heading("Parámetros", text="Parámetros")
        
        self.tree.column("Modelo", width=200)
        self.tree.column("R²", width=100, anchor="center")
        self.tree.column("Evaluación", width=150, anchor="center")
        self.tree.column("Parámetros", width=400)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        btn_frame = ttk.Frame(results_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Ver Gráfico", command=self.show_selected_plot).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Exportar CSV", command=self.export_results).pack(side="left", padx=5)
    
    def load_tsv_file(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Archivos TSV", "*.tsv"), ("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            x_data = []
            y_data = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Ignorar líneas vacías y comentarios
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            try:
                                x_val = float(parts[0])
                                y_val = float(parts[1])
                                x_data.append(x_val)
                                y_data.append(y_val)
                            except ValueError:
                                continue
            
            if len(x_data) < 3:
                messagebox.showerror("Error", "El archivo debe contener al menos 3 puntos de datos válidos")
                return
            
            self.data = {'x': np.array(x_data), 'y': np.array(y_data)}
            
            # Mostrar datos en el área de texto
            self.data_display.config(state='normal')
            self.data_display.delete(1.0, tk.END)
            
            header = "X\tY\n" + "-"*30 + "\n"
            self.data_display.insert(tk.END, header)
            
            for x, y in zip(x_data, y_data):
                self.data_display.insert(tk.END, f"{x:.2f}\t{y:.2f}\n")
            
            self.data_display.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{str(e)}")
    
    def evaluate_r2(self, r2):
        if r2 >= 0.9: return ("Excelente", "#4CAF50")
        elif r2 >= 0.7: return ("Bueno", "#8BC34A")
        elif r2 >= 0.5: return ("Moderado", "#FFC107")
        else: return ("Pobre", "#F44336")
    
    def process_models(self):
        if not self.data or len(self.data['x']) == 0:
            messagebox.showwarning("Advertencia", "Primero cargue un archivo con datos")
            return
            
        x = self.data['x']
        y = self.data['y']
        
        if all(v == 0 for v in y):
            messagebox.showerror("Error", "Ingrese valores distintos de cero")
            return
            
        self.results = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Lista completa de los 13 modelos con sus parámetros y nombres
        modelos = [
            {
                "name": "Sigmoidal 3 parámetros",
                "func": sigmoidal_3params,
                "p0": [np.max(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))]],
                "param_names": ["a", "b", "x0"]
            },
            {
                "name": "Sigmoidal 4 parámetros",
                "func": sigmoidal_4params,
                "p0": [np.max(y)-np.min(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))], np.min(y)],
                "param_names": ["a", "b", "x0", "y0"]
            },
            {
                "name": "Sigmoidal 5 parámetros",
                "func": sigmoidal_5params,
                "p0": [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, x[np.argmin(np.abs(y-(np.max(y)+np.min(y))/2))], np.min(y)],
                "param_names": ["a", "b", "c", "x0", "y0"]
            },
            {
                "name": "Logística 3 parámetros",
                "func": logistic_3params,
                "p0": [np.max(y), 1.0, np.median(x)],
                "param_names": ["a", "b", "x0"]
            },
            {
                "name": "Logística 4 parámetros",
                "func": logistic_4params,
                "p0": [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)],
                "param_names": ["a", "b", "x0", "y0"]
            },
            {
                "name": "Weibull 4 parámetros",
                "func": weibull_4params,
                "p0": [np.max(y), (x[-1]-x[0])/10, 1.0, np.median(x)],
                "param_names": ["a", "b", "c", "x0"]
            },
            {
                "name": "Weibull 5 parámetros",
                "func": weibull_5params,
                "p0": [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, np.median(x), np.min(y)],
                "param_names": ["a", "b", "c", "x0", "y0"]
            },
            {
                "name": "Gompertz 3 parámetros",
                "func": gompertz_3params,
                "p0": [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x)],
                "param_names": ["a", "b", "x0"]
            },
            {
                "name": "Gompertz 4 parámetros",
                "func": gompertz_4params,
                "p0": [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x), np.min(y)],
                "param_names": ["a", "b", "x0", "y0"]
            },
            {
                "name": "Hill 3 parámetros",
                "func": hill_3params,
                "p0": [np.max(y), 1.0, np.median(x)],
                "param_names": ["a", "b", "c"]
            },
            {
                "name": "Hill 4 parámetros",
                "func": hill_4params,
                "p0": [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)],
                "param_names": ["a", "b", "c", "y0"]
            },
            {
                "name": "Chapman 3 parámetros",
                "func": chapman_3params,
                "p0": [np.max(y), 2/np.median(x), 1.0],
                "param_names": ["a", "b", "c"]
            },
            {
                "name": "Chapman 4 parámetros",
                "func": chapman_4params,
                "p0": [np.max(y)-np.min(y), 2/np.median(x), 1.0, np.min(y)],
                "param_names": ["a", "b", "c", "y0"]
            }
        ]
        
        for model in modelos:
            try:
                params, _ = curve_fit(model["func"], x, y, p0=model["p0"], maxfev=10000)
                y_pred = model["func"](x, *params)
                
                residuals = y - y_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                eval_text, eval_color = self.evaluate_r2(r2)
                
                param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(model["param_names"], params)])
                
                self.results.append({
                    "name": model["name"],
                    "params": params,
                    "r2": r2,
                    "func": model["func"],
                    "x": x,
                    "y": y,
                    "y_pred": y_pred,
                    "param_names": model["param_names"]
                })
                
                item = self.tree.insert("", "end", values=(model["name"], f"{r2:.4f}", eval_text, param_text))
                self.tree.tag_configure(eval_text, background=eval_color)
                
            except Exception as e:
                self.tree.insert("", "end", values=(model["name"], "Error", str(e), ""))
                self.results.append({"name": model["name"], "error": str(e)})
    
    def show_selected_plot(self):
        selected = self.tree.focus()
        if not selected:
            messagebox.showwarning("Advertencia", "Seleccione un modelo de la tabla")
            return
            
        item = self.tree.item(selected)
        model_name = item['values'][0]
        
        result = next((r for r in self.results if r["name"] == model_name), None)
        if not result or "error" in result:
            messagebox.showerror("Error", "No hay datos de gráfico para este modelo")
            return
            
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Gráfico: {model_name}")
        plot_window.geometry("800x600")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.scatter(result["x"], result["y"], color='blue', label='Datos reales', s=50)
        
        x_vals = np.linspace(min(result["x"]), max(result["x"]), 100)
        y_vals = result["func"](x_vals, *result["params"])
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Modelo ajustado')
        
        ax.set_title(model_name, fontsize=14, pad=20)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        ax.annotate(f"R² = {result['r2']:.4f}", 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, bbox=dict(boxstyle="round", fc="w"))
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(plot_window, text="Cerrar", command=plot_window.destroy).pack(pady=5)
    
    def export_results(self):
        if not self.results:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
            
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
                        param_text = ", ".join([f"{name}={val:.4f}" for name, val in zip(result["param_names"], result["params"])])
                        writer.writerow([result["name"], f"{result['r2']:.4f}", eval_text, param_text])
                
            messagebox.showinfo("Éxito", f"Resultados exportados a:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()