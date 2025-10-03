import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
import csv

# Definiciones de modelos (13)
def sigmoidal_3params(x, a, b, x0):
    return a / (1 + np.exp(-(x - x0) / b))

def sigmoidal_4params(x, a, b, x0, y0):
    return y0 + (a / (1 + np.exp(-(x - x0) / b)))

def sigmoidal_5params(x, a, b, c, x0, y0):
    return y0 + (a / (1 + np.exp(-(x - x0) / b)) ** c)

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

model_defs = [
    ("Sigmoidal 3 parámetros", sigmoidal_3params, lambda x, y: [np.max(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y - (np.max(y) + np.min(y))/2))]]),
    ("Sigmoidal 4 parámetros", sigmoidal_4params, lambda x, y: [np.max(y)-np.min(y), (x[-1]-x[0])/10, x[np.argmin(np.abs(y - (np.max(y) + np.min(y))/2))], np.min(y)]),
    ("Sigmoidal 5 parámetros", sigmoidal_5params, lambda x, y: [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, x[np.argmin(np.abs(y - (np.max(y) + np.min(y))/2))], np.min(y)]),
    ("Logística 3 parámetros", logistic_3params, lambda x, y: [np.max(y), 1.0, np.median(x)]),
    ("Logística 4 parámetros", logistic_4params, lambda x, y: [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)]),
    ("Weibull 4 parámetros", weibull_4params, lambda x, y: [np.max(y), (x[-1]-x[0])/10, 1.0, np.median(x)]),
    ("Weibull 5 parámetros", weibull_5params, lambda x, y: [np.max(y)-np.min(y), (x[-1]-x[0])/10, 1.0, np.median(x), np.min(y)]),
    ("Gompertz 3 parámetros", gompertz_3params, lambda x, y: [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x)]),
    ("Gompertz 4 parámetros", gompertz_4params, lambda x, y: [np.max(y)-np.min(y), (x[-1]-x[0])/10, np.median(x), np.min(y)]),
    ("Hill 3 parámetros", hill_3params, lambda x, y: [np.max(y), 1.0, np.median(x)]),
    ("Hill 4 parámetros", hill_4params, lambda x, y: [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)]),
    ("Chapman 3 parámetros", chapman_3params, lambda x, y: [np.max(y), 2/np.median(x), 1.0]),
    ("Chapman 4 parámetros", chapman_4params, lambda x, y: [np.max(y)-np.min(y), 2/np.median(x), 1.0, np.min(y)])
]

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ajuste de Modelos")
        self.root.geometry("900x600")

        self.data = {}
        self.resultados = []

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.X)

        self.data_frame = ttk.Frame(self.root)
        self.data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.start_var = tk.IntVar(value=2000)
        self.end_var = tk.IntVar(value=2023)

        ttk.Label(frame, text="Año inicial:").grid(row=0, column=0)
        ttk.Spinbox(frame, from_=1900, to=2100, textvariable=self.start_var, width=8).grid(row=0, column=1)

        ttk.Label(frame, text="Año final:").grid(row=0, column=2, padx=10)
        ttk.Spinbox(frame, from_=1900, to=2100, textvariable=self.end_var, width=8).grid(row=0, column=3)

        ttk.Button(frame, text="Generar Campos", command=self.generar_campos).grid(row=0, column=4, padx=10)
        ttk.Button(frame, text="Procesar Datos", command=self.procesar_modelos).grid(row=0, column=5, padx=10)
        ttk.Button(frame, text="Guardar CSV", command=self.exportar_csv).grid(row=0, column=6, padx=10)

        self.result_frame = ttk.Frame(self.root)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def generar_campos(self):
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        self.data.clear()

        for i, year in enumerate(range(self.start_var.get(), self.end_var.get()+1)):
            ttk.Label(self.data_frame, text=f"{year}:").grid(row=i, column=0)
            var = tk.DoubleVar()
            entry = ttk.Entry(self.data_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1)
            self.data[year] = var

    def evaluar(self, r2):
        if r2 >= 0.9: return "Excelente ajuste (R² ≥ 0.9)"
        elif r2 >= 0.7: return "Buen ajuste"
        elif r2 >= 0.5: return "Ajuste moderado"
        else: return "Ajuste pobre"

    def procesar_modelos(self):
        self.resultados.clear()
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        x_vals = np.array(sorted(self.data.keys()))
        y_vals = np.array([self.data[year].get() for year in x_vals])

        for name, func, init_func in model_defs:
            try:
                p0 = init_func(x_vals, y_vals)
                params, _ = curve_fit(func, x_vals, y_vals, p0=p0, maxfev=10000)
                y_pred = func(x_vals, *params)
                residuals = y_vals - y_pred
                r2 = 1 - (np.sum(residuals**2) / np.sum((y_vals - np.mean(y_vals))**2))
                eval_ = self.evaluar(r2)

                frame = ttk.LabelFrame(self.result_frame, text=name)
                frame.pack(fill=tk.X, padx=5, pady=5)
                ttk.Label(frame, text=f"R²: {r2:.4f} - {eval_}").pack(anchor='w')
                ttk.Label(frame, text="Parámetros:").pack(anchor='w')
                for p in params:
                    ttk.Label(frame, text=f"• {p:.4f}").pack(anchor='w')

                self.resultados.append((name, r2, eval_, [round(p, 4) for p in params]))

            except Exception as e:
                frame = ttk.LabelFrame(self.result_frame, text=name)
                frame.pack(fill=tk.X, padx=5, pady=5)
                ttk.Label(frame, text=f"Error: {e}", foreground="red").pack()

    def exportar_csv(self):
        if not self.resultados:
            messagebox.showwarning("Sin resultados", "Primero procesa los datos")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filepath:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Modelo", "R²", "Evaluación", "Parámetros"])
                for model, r2, eval_, params in self.resultados:
                    writer.writerow([model, f"{r2:.4f}", eval_, ", ".join(map(str, params))])
            messagebox.showinfo("Guardado", f"Archivo guardado en:\n{filepath}")

# Ejecutar
if __name__ == '__main__':
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()
