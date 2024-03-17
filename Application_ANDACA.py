# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 23:35:12 2023

@author: Jonathan Calvopina Merchan
"""

import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as mt
from scipy.stats import norm
#from scipy.cluster import hierarchy
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
#from numpy.matrixlib.defmatrix import matrix

from numpy.ma import count
import math
from PIL import Image

contenedor_nodos_m = []
contenedor_vector = []
contenedor_nodos = []
contenedor_x = []
xx = []
yy = []
contenedor_dxf = []
contenedor_df_concatenado = []
contenedor_valor = []
resultados = []
contendor_diccionario = []
contenedor_f = []
contenedor_sumaI = []
contenedor_rk = []
contenedor__sk = []
contenedor_skk = []
contenedor_rkk = []
fi = []
Ii = []
con_suma = []
contenedor_v = []
contenedor_indice_similaridad = []
contenedor_matriz = []
contenedor_lista_matriz = []
contenedor_de_la_matriz = []
contenedor_matrix_aa = []
contenedor_lista = []
contenedor_matriz = []
text_resultados = [] 
contenedor_valores = []
df = pd.DataFrame()
variables_seleccionadas = []
con_al =[]

def cargar_excel():
    global df, variables_seleccionadas
    archivo = filedialog.askopenfilename(filetypes=[("Archivos de Excel", "*.xlsx;*.xls")])
    if archivo:
        df = pd.read_excel(archivo)
        columnas = df.columns.tolist()
        listbox.delete(0, tk.END)
        for columna in columnas:
            listbox.insert(tk.END, columna)
            
def mostrar_resultados():
    if df is None:
        messagebox.showinfo('Información', 'No se ha cargado ningún archivo.')
        return

    # Calcular el ancho máximo de cada columna para encabezados y datos
    anchos_encabezados = [max(len(str(col)), max(len(str(valor)) for valor in df[col])) for col in df.columns]
    anchos_datos = [max(len(str(valor)) for valor in df[col]) for col in df.columns]

    # Calcular el ancho total de cada columna
    anchos_totales = [max(anchos_encabezados[i], anchos_datos[i]) for i in range(len(anchos_encabezados))]

    # Crear una nueva ventana
    ventana_resultados = tk.Toplevel(root)
    ventana_resultados.title("Resultados")

    # Crear un widget Text para mostrar los resultados
    text_resultados = tk.Text(ventana_resultados)
    text_resultados.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    # Insertar los encabezados en la primera fila
    encabezados = [f"{col: <{anchos_totales[i]}}" for i, col in enumerate(df.columns)]  # Alinear y fijar el ancho máximo
    linea_encabezados = "   ".join(encabezados) + "\n"
    text_resultados.insert(tk.END, linea_encabezados)

    # Iterar sobre las filas del DataFrame para mostrar los datos alineados en una tabla
    for _, row in df.iterrows():
        linea = ""
        for i, valor in enumerate(row):
            # Formatear el valor para que ocupe el ancho máximo de la columna
            linea += f"{valor: <{anchos_totales[i]}}   "
        linea += "\n"
        text_resultados.insert(tk.END, linea)

def exportar_variables():
    global variables_seleccionadas
    variables_seleccionadas = [listbox.get(idx) for idx in listbox.curselection()]
    actualizar_cuadro_seleccionadas()

def actualizar_cuadro_seleccionadas():
    cuadro_seleccionadas.delete(1.0, tk.END)  # Limpiar el cuadro de texto
    for variable in variables_seleccionadas:
        cuadro_seleccionadas.insert(tk.END, variable + '\n')
        
def limpiar_data():
    
    global df, variables_seleccionadas
    df.drop(df.index, inplace=True)
    df.drop(df.columns, axis=1, inplace=True)
    actualizar_listbox()

def actualizar_listbox():
    listbox.delete(0, tk.END)
    cuadro_seleccionadas.delete(1.0, tk.END)
    columnas = df.columns.tolist()
    variables_seleccionadas = df.columns.tolist()
    for columna in columnas:
        listbox.insert(tk.END, columna)   
    for variable in variables_seleccionadas:
        cuadro_seleccionadas.insert(tk.END, variable + '\n')
        
def convertir_a_categoricas():
    global variables_seleccionadas, df
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo("Variables convertidas a categorías", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    for variable in variables_seleccionadas:
        df[variable] = df[variable].astype('category')
        categorias = df[variable].unique()
        print(f"Categoría de: '{variable}': {categorias}")
    messagebox.showinfo("Convertir a Categóricas", "Variables convertidas a categorías.")
    
def tabla_de_contigencia():
    global variables_seleccionadas, df
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    subset_df = df[variables_seleccionadas]
    tabla_contingencia = pd.crosstab(index=subset_df[variables_seleccionadas[0]], columns=subset_df[variables_seleccionadas[1]])
    suma_total = tabla_contingencia.values.sum()
    filas = tabla_contingencia.sum(axis=1)
    columnas = tabla_contingencia.sum()
    ventana_tabla = tk.Toplevel(root)
    ventana_tabla.title("Tabla de Contingencia")
    etiqueta_tabla = tk.Label(ventana_tabla, text=f"Tabla de Contingencia:\n{tabla_contingencia}\n\n Suma Total: \n{suma_total}\n\n Suma Total fila: \n{filas}\n\n Suma Total columna: \n{columnas}")
    etiqueta_tabla.pack(padx=20, pady=20) 
    
def mostrar_graficos_de_contingencia():
    global filas, tabla_contingencia
    global variables_seleccionadas, df
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]])
    suma_total = tabla_contingencia.values.sum()
    porcentaje = (tabla_contingencia/suma_total)
    porcentaje_transpuesta = np.transpose(porcentaje)
    porcentaje_perfiles_columna = tabla_contingencia.div(np.transpose(tabla_contingencia.sum(axis=1)), axis=0)
    ventana_tabla1 = tk.Toplevel(root)
    ventana_tabla1.title("Distribución de frecuencias")
    fig, ax = plt.subplots()
    porcentaje.plot(kind='bar', ax=ax)
    ax.set_ylabel('Frecuencia')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=360, horizontalalignment='right')
    canvas = FigureCanvasTkAgg(fig, master=ventana_tabla1)
    canvas.get_tk_widget().pack()
    ventana_tabla2 = tk.Toplevel(root)
    ventana_tabla2.title("Distribución de frecuencias")
    fig, ax = plt.subplots()
    porcentaje_transpuesta.plot(kind='bar', ax=ax)
    ax.set_ylabel('Frecuencia')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=360, horizontalalignment='right')
    canvas = FigureCanvasTkAgg(fig, master=ventana_tabla2)
    canvas.get_tk_widget().pack()
    ventana_tabla3 = tk.Toplevel(root)
    ventana_tabla3.title("Distribución de frecuencias perfiles")
    fig, ax = plt.subplots()
    porcentaje_perfiles_columna.plot(kind='bar', ax=ax)
    ax.set_ylabel('Frecuencia')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=360, horizontalalignment='right')
    canvas = FigureCanvasTkAgg(fig, master=ventana_tabla3)
    canvas.get_tk_widget().pack()    

def mostrar_frecuencias_de_contingencia():
    global filas, tabla_contingencia
    global variables_seleccionadas, df
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]])
    suma_total = tabla_contingencia.values.sum()
    porcentaje = round(((tabla_contingencia/suma_total)*100),2)
    porcentaje_columna = round((tabla_contingencia/tabla_contingencia.sum()*100),2) 
    porcentaje_fila = round(((tabla_contingencia.div(np.transpose(tabla_contingencia.sum(axis=1)), axis=0))*100),2)
    ventana_tabla = tk.Toplevel(root)
    ventana_tabla.title("Distribución de frecuencias")
    etiqueta_tabla = tk.Label(ventana_tabla, text=f"Porcentajes:\n{porcentaje}\n\n Porc.Fila : \n{porcentaje_fila}\n\n Porc.Columna : \n{porcentaje_columna}\n")
    etiqueta_tabla.pack(padx=20, pady=20) 

def chi_cuadrado():
    global filas, tabla_contingencia
    global variables_seleccionadas, df 
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    suma_total = tabla_contingencia.values.sum()
    # Frecuencias observadas
    fo = np.array(tabla_contingencia); fo
    x = tabla_contingencia.sum()
    y = np.transpose(tabla_contingencia).sum()
    # Matriz fila y columna
    x_0 = np.array(x)
    y_0 =np.array(y).reshape(-1,1)
    # Frecuencias esperadas
    fe = (x_0*y_0)/suma_total; fe
    fw =((fo-fe)**2)/fe
    chi_calculado = np.sum(fw)
    num_filas = tabla_contingencia.shape[0]
    num_columnas = tabla_contingencia.shape[1]
    gl = (num_filas+num_columnas) -1
    p_value = 0.95  # Nivel de confianza del 95%
    chi_critico = chi2.ppf(p_value, gl)
    Nivel_significancia = 0.05
    print("Regla de desicion:")
    if chi_calculado > chi_critico:
        desicion = "chi calculado > chi critico, se rechaza la Ho: No son independientes (están asociadas)."
    else:
        desicion = "chi calculado < chi critico, se acepta la Ho: Son independientes (no están asociadas)."
    ventana_chi_cuadrado = tk.Toplevel(root)
    ventana_chi_cuadrado.title("Chi^2")
    etiqueta_chi_cuadrado = tk.Label(ventana_chi_cuadrado, text=f"Nivel de significancia: \n{Nivel_significancia}\n\n Prueba chi calculado: \n{round(chi_calculado,3)}\n\n Prueba chi critico: \n{round(chi_critico,3)}\n\n Regla de decisión: \n{desicion} ")
    etiqueta_chi_cuadrado.pack(padx=20, pady=10)

def razon_verosimilitud():
    global filas, tabla_contingencia
    global variables_seleccionadas, df 
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return    
    data = np.array(tabla_contingencia)
    statistic, p_value, _, _ = chi2_contingency(data, lambda_="log-likelihood")
    num_filas = tabla_contingencia.shape[0]
    num_columnas = tabla_contingencia.shape[1]
    gl = (num_filas+num_columnas) -1
    p_value = 0.95  # Nivel de confianza del 95%
    chi_critico = chi2.ppf(p_value, gl)
    Nivel_significancia = 0.05
    print("Regla de desicion:")
    if statistic > chi_critico:
        desicion = "G^2 > chi critico, se concluye que estos datos  de las 2 variables se muestran son dependientes. "
    else:
        desicion = "G^2 < chi critico, se concluye que estos datos de las 2 variables se muestran son independientes."
    ventana_g2 = tk.Toplevel(root)
    ventana_g2.title("Contraste mediante la razón de verosimilitudes  (G^2)")
    etiqueta_g2 = tk.Label(ventana_g2, text=f"Nivel de significancia: \n{Nivel_significancia}\n\n G^2: \n{round(statistic,3)}\n\n Prueba chi critico: \n{round(chi_critico,3)}\n\n Regla de decisión: \n{desicion} ")
    etiqueta_g2.pack(padx=20, pady=10)

def coeficiente_contingencia():
    global tabla_contingencia
    global variables_seleccionadas, df  
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    suma_total = tabla_contingencia.values.sum()
    fo = np.array(tabla_contingencia); fo
    x = tabla_contingencia.sum()
    y = np.transpose(tabla_contingencia).sum()
    x_0 = np.array(x)
    y_0 =np.array(y).reshape(-1,1)
    fe = (x_0*y_0)/suma_total; fe
    fw =((fo-fe)**2)/fe
    chi_calculado = np.sum(fw)
    c = mt.sqrt(chi_calculado/(chi_calculado+suma_total))
    Nivel_significancia = 0.05
    desicion1 = "Si el coeficiente de contingencia es cercano a 1 es alta asociación. "
    desicion2 = "Si el coeficiente de contingencia es cercano a 0 es baja asociación. "
    desicion3 = "Si el coeficiente de contingencia es 0 es nula asociación. "
    ventana_c = tk.Toplevel(root)
    ventana_c.title("Coeficiente de contingencia")
    etiqueta_c = tk.Label(ventana_c, text=f"Nivel de significancia: \n{Nivel_significancia}\n\n Coeficiente de contingencia (C): \n{round(c,3)}\n\n Condición: \n{desicion1} \n{desicion2} \n{desicion3}")
    etiqueta_c.pack(padx=20, pady=10)

def coeficiente_phi():
    global tabla_contingencia
    global variables_seleccionadas, df     
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    suma_total = tabla_contingencia.values.sum()
    fo = np.array(tabla_contingencia); fo
    x = tabla_contingencia.sum()
    y = np.transpose(tabla_contingencia).sum()
    x_0 = np.array(x)
    y_0 =np.array(y).reshape(-1,1)
    fe = (x_0*y_0)/suma_total; fe
    fw =((fo-fe)**2)/fe
    chi_calculado = np.sum(fw)
    phi = mt.sqrt(chi_calculado/(suma_total))
    Nivel_significancia = 0.05
    print("Regla de decisión:")
    if (phi == 0):
        desicion = ("phi == 0, puede afirmarse con alta certidumbre que las variables son independientes.")
    elif(phi > 0):
        desicion =  ("phi > 0, la interpretación no es concluyente, corroborando que las variables son dependientes.")
    else:
        desicion =  ("Error confirme los valores")
    ventana_phi = tk.Toplevel(root)
    ventana_phi.title("Coeficiente phi")
    etiqueta_phi = tk.Label(ventana_phi, text=f"Nivel de significancia: \n{Nivel_significancia}\n\n Coeficiente phi: \n{round(phi,3)}\n\n Desición: \n{desicion} \n")
        
    etiqueta_phi.pack(padx=20, pady=10)
    
def coeficiente_cramer():
    global tabla_contingencia
    global variables_seleccionadas, df    
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    suma_total = tabla_contingencia.values.sum()
    fo = np.array(tabla_contingencia); fo
    x = tabla_contingencia.sum()
    y = np.transpose(tabla_contingencia).sum()
    x_0 = np.array(x)
    y_0 =np.array(y).reshape(-1,1)
    fe = (x_0*y_0)/suma_total; fe
    fw =((fo-fe)**2)/fe
    chi_calculado = np.sum(fw)
    k = min((tabla_contingencia.shape[0] - 1),(tabla_contingencia.shape[1] - 1))
    v = mt.sqrt(chi_calculado/(suma_total*k))
    Nivel_significancia = 0.05
    print("Regla de decisión:")
    if (v == 0):
        desicion = ("V == 0, las variables son independientes")
    elif(v > 0):
        desicion = ("V > 0, las variables son dependientes")
    else:
        desicion = ("Error confirme los valores")
    ventana_v = tk.Toplevel(root)
    ventana_v.title("Coeficiente Cramer(V)")
    etiqueta_v = tk.Label(ventana_v, text=f"Nivel de significancia: \n{Nivel_significancia}\n\n Coeficiente de cramer: \n{round(v,3)}\n\n Desición: \n{desicion} \n")   
    etiqueta_v.pack(padx=20, pady=10)
    
def calculo_rpe():
    global tabla_contingencia
    global variables_seleccionadas, df  
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    z0 = np.transpose(tabla_contingencia.sum())
    for i in z0:
        z = np.sum(z0)
    w=tabla_contingencia/z
    # Lambda y|x
    x = w.max()
    x1 = tabla_contingencia.sum() # Suma por columnas
    x_ = x1/z
    # Lambda x|y
    y = np.transpose(w).max()
    y1 = np.transpose(tabla_contingencia).sum()# Suma de las filas
    y_ = y1/z
    #lambda_yx
    maximo_y = np.max(y_)
    suma_x = np.sum(x)
    lb_x = (suma_x - maximo_y)/(1-maximo_y)
    # Lambda x|y
    maximo_x = np.max(x_)
    suma_y = np.sum(y)
    lb_y = (suma_y - maximo_x)/(1-maximo_x)
    # Lambda
    lbd = ((suma_x + suma_y - maximo_x - maximo_y))/(2- maximo_x - maximo_y)
    Nivel_significancia = 0.05
    desicion1 = "Valores cercano a 1, hay evidencia de una alta asociación entre las variables."
    desicion2 = "Valores cercano a 0, hay evidencia de una baja asociación entre las variables."
    desicion3 = "Valores igual a 0, hay evidencia de una asociación nula entre las variables."
    
    ventana_lbd = tk.Toplevel(root)
    ventana_lbd.title("Medidas basadas en la reducción proporcional del error(RPE)")
    etiqueta_lbd = tk.Label(ventana_lbd, text=f"Nivel de significancia: \n{Nivel_significancia}\n\n Lambda asimétrica y|x: {round(lb_x,3)} \n\n Lambda asimétrica x|y: {round(lb_y,3)} \n\n Lambda simétrica: {round(lbd,3)} \n\n Desición: \n{desicion1} \n{desicion2} \n{desicion3} \n") 
    etiqueta_lbd.pack(padx=20, pady=10)
    
def cyd():
    global tabla_contingencia
    global variables_seleccionadas, df     
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    df1 = pd.DataFrame(tabla_contingencia)
    num_filas, num_columnas = df1.shape
    fil = len(tabla_contingencia)
    col = len(np.transpose(tabla_contingencia))
    def concordancia(k,l):
        concordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j > l :
                    concordancia = concordancia + df1.iloc[i,j]
        concordancia = concordancia*df1.iloc[k,l]
        return(concordancia)
    def discordancia(k,l):
        discordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j < l :
                    discordancia = discordancia + df1.iloc[i,j]
        discordancia = discordancia*df1.iloc[k,l]
        return(discordancia)
    c = 0
    d = 0
    print("(i,j) ", "Concordancias","   Discordancias")
    for p in range(num_filas):
        for q in range(num_columnas):
            val1 = concordancia(p,q)
            val2 = discordancia(p,q)
            #print("(i,j)", " Concordancias","Discordancias")
            print(p,q,":     ", val1, "        :",val2)
            c = c + concordancia(p,q)
            d = d + discordancia(p,q)
    print("Valores totales de la suma:")
    print("Concordancia:",c,", Discordancia:",d)
    Nivel_significancia = 0.05
    ventana_cyd = tk.Toplevel(root)
    ventana_cyd.title("Concordancias y Discordancias")
    etiqueta_cyd = tk.Label(ventana_cyd, text=f"Nivel de significancia: \n{Nivel_significancia}\n\n Suma de Concordancias: {round(c,3)} \n\n Suma de Discordancias: {round(d,3)} \n")  
    etiqueta_cyd.pack(padx=20, pady=10)
    
def estadistico_gamma():
    global tabla_contingencia
    global variables_seleccionadas, df  
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    df1 = pd.DataFrame(tabla_contingencia)
    num_filas, num_columnas = df1.shape
    fil = len(tabla_contingencia)
    col = len(np.transpose(tabla_contingencia))
    def concordancia(k,l):
        concordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j > l :
                    concordancia = concordancia + df1.iloc[i,j]
        concordancia = concordancia*df1.iloc[k,l]
        return(concordancia)
    def discordancia(k,l):
        discordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j < l :
                    discordancia = discordancia + df1.iloc[i,j]
        discordancia = discordancia*df1.iloc[k,l]
        return(discordancia)
    c = 0
    d = 0
    print("(i,j) ", "Concordancias","   Discordancias")
    for p in range(num_filas):
        for q in range(num_columnas):
            val1 = concordancia(p,q)
            val2 = discordancia(p,q)
            #print("(i,j)", " Concordancias","Discordancias")
            print(p,q,":     ", val1, "        :",val2)
            c = c + concordancia(p,q)
            d = d + discordancia(p,q)
    gamma = (c-d)/(c+d)
    condicion1 = ("Si gamma es cercano a +1, tiene tendencia lineal creciente fuerte.")
    condicion2 = ("Si gamma cercano a -1, tiene tendencia lineal creciente fuerte.")
    condicion3 = ("Si gamma es igual a cero, no tiene una tendencia lineal.")
    condicion4 = ("Si gamma es cercano a cero, tiene una tendencia lineal débil.")
    ventana_gamma = tk.Toplevel(root)
    ventana_gamma.title("Estadístico gamma")
    etiqueta_gamma = tk.Label(ventana_gamma, text=f"Estadístico gamma: \n {round(gamma,3)} \n\n Condición: \n {condicion1} \n {condicion2} \n {condicion3} \n {condicion4} \n")
    etiqueta_gamma.pack(padx=20, pady=10)

def estadistico_kendall():
    global tabla_contingencia
    global variables_seleccionadas, df   
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    df1 = pd.DataFrame(tabla_contingencia)
    num_filas, num_columnas = df1.shape
    fil = len(tabla_contingencia)
    col = len(np.transpose(tabla_contingencia))
    def concordancia(k,l):
        concordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j > l :
                    concordancia = concordancia + df1.iloc[i,j]
        concordancia = concordancia*df1.iloc[k,l]
        return(concordancia)
    def discordancia(k,l):
        discordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j < l :
                    discordancia = discordancia + df1.iloc[i,j]
        discordancia = discordancia*df1.iloc[k,l]
        return(discordancia)
    c = 0
    d = 0
    print("(i,j) ", "Concordancias","   Discordancias")
    for p in range(num_filas):
        for q in range(num_columnas):
            val1 = concordancia(p,q)
            val2 = discordancia(p,q)
            #print("(i,j)", " Concordancias","Discordancias")
            print(p,q,":     ", val1, "        :",val2)
            c = c + concordancia(p,q)
            d = d + discordancia(p,q)
    N2 = (tabla_contingencia.values.sum())**2
    n2i = np.sum(tabla_contingencia.sum(axis=1)**2)
    n2j = np.sum(tabla_contingencia.sum()**2)
    wf = (N2 - n2i)/2
    wc = (N2 - n2j)/2
    kendall=((c-d)/(mt.sqrt(wf*wc)))
    condicion1 = ("Si es cercano a +1, tiene tendencia lineal creciente fuerte.")
    condicion2 = ("Si es cercano a -1, tiene tendencia lineal decreciente fuerte.")
    condicion4 = ("Si es igual a cero, no tiene una tendencia lineal.")
    condicion3 = ("Si es cercano a cero, tiene una tendencia lineal débil.")
    ventana_gamma = tk.Toplevel(root)
    ventana_gamma.title("Estadística Taub de Kendall(τb)")
    etiqueta_gamma = tk.Label(ventana_gamma, text=f"Estadística Taub de Kendall(τb): {round(kendall,3)} \n\n Condición: \n {condicion1} \n {condicion2} \n {condicion3} \n {condicion4} \n")  
    etiqueta_gamma.pack(padx=20, pady=10)
    
def estadistico_stuart():
    global tabla_contingencia
    global variables_seleccionadas, df  
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    df1 = pd.DataFrame(tabla_contingencia)
    num_filas, num_columnas = df1.shape
    fil = len(tabla_contingencia)
    col = len(np.transpose(tabla_contingencia))  
    def concordancia(k,l):
        concordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j > l :
                    concordancia = concordancia + df1.iloc[i,j]
        concordancia = concordancia*df1.iloc[k,l]
        return(concordancia)
    def discordancia(k,l):
        discordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j < l :
                    discordancia = discordancia + df1.iloc[i,j]
        discordancia = discordancia*df1.iloc[k,l]
        return(discordancia)
    c = 0
    d = 0
    print("(i,j) ", "Concordancias","   Discordancias")
    for p in range(num_filas):
        for q in range(num_columnas):
            val1 = concordancia(p,q)
            val2 = discordancia(p,q)
            #print("(i,j)", " Concordancias","Discordancias")
            print(p,q,":     ", val1, "        :",val2)
            c = c + concordancia(p,q)
            d = d + discordancia(p,q)
    nt = len(df)
    col = len(tabla_contingencia)
    fil = len( np.transpose(tabla_contingencia))
    k = min(col,fil)
    stuart = (c-d)/(nt**2*((k-1)/2*k))
    condicio1 = "Si es igual a cero, falta de asociación entre las variables categóricas."
    condicio2 = "Si es mayor a cero, es una asociación positiva entre las dos variables categóricas."
    condicio3 = "Si es menor a cero, es una asociación negativa entre las dos variables categóricas."
    condicio4 = "Si se acerca a 1, es una asociación fuerte positiva entre las dos variables categóricas."
    condicio5 = "Si se acerca a -1, es una asociación fuerte negativa entre las dos variables categóricas."
    ventana_st = tk.Toplevel(root)
    ventana_st.title("Estadística Tauc de Stuart(τc)")
    etiqueta_st = tk.Label(ventana_st, text=f"Estadística Tauc de Stuart(τc): {round(stuart,3)} \n\n Condición: \n {condicio1} \n {condicio2} \n {condicio3} \n {condicio4} \n {condicio5} \n")   
    etiqueta_st.pack(padx=20, pady=10)

def estadistico_somers():
    global tabla_contingencia
    global variables_seleccionadas, df   
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    df1 = pd.DataFrame(tabla_contingencia)
    num_filas, num_columnas = df1.shape
    fil = len(tabla_contingencia)
    col = len(np.transpose(tabla_contingencia))
    def concordancia(k,l):
        concordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j > l :
                    concordancia = concordancia + df1.iloc[i,j]
        concordancia = concordancia*df1.iloc[k,l]
        return(concordancia)
    def discordancia(k,l):
        discordancia = 0
        for i in range(0,fil):
            for j in range(0,col):
                if i> k and j < l :
                    discordancia = discordancia + df1.iloc[i,j]
        discordancia = discordancia*df1.iloc[k,l]
        return(discordancia)
    c = 0
    d = 0
    print("(i,j) ", "Concordancias","   Discordancias")
    for p in range(num_filas):
        for q in range(num_columnas):
            val1 = concordancia(p,q)
            val2 = discordancia(p,q)
            #print("(i,j)", " Concordancias","Discordancias")
            print(p,q,":     ", val1, "        :",val2)
            c = c + concordancia(p,q)
            d = d + discordancia(p,q)
    
    N2 = (tabla_contingencia.values.sum())**2
    n2i = np.sum(tabla_contingencia.sum(axis=1)**2)
    n2j = np.sum(tabla_contingencia.sum()**2)
    wf = (N2 - n2i)/2
    wc = (N2 - n2j)/2
    dyx = (c-d)/wf
    dxy = (c-d)/wc
    
    print("Regla de desicion:")
    if(dyx < 0):
        desicionxy = ("D(Y|X) < 0, existe alguna asociación(INVERSA) entre las variables.")
    elif(dyx >0):
        desicionxy = ("D(Y|X) > 0, existe alguna asociación(DIRECTA) entre las variables.")
    else:
        desicionxy = ("D(Y|X) == 0, no existe asociación entre las variables.")
    
    print("Regla de desicion:")
    if(dxy < 0):
        desicionyx = ("D(X|Y) < 0, existe alguna asociación(INVERSA) entre las variables.")
    elif(dxy >0):
        desicionyx = ("D(X|Y) > 0, existe alguna asociación(DIRECTA)entre las variables.")
    else:
        desicionyx = ("D(X|Y) == 0, no existe asociación entre las variables.")
    ventana_somer = tk.Toplevel(root)
    ventana_somer.title("Estadística de Somers")
    etiqueta_somer = tk.Label(ventana_somer, text=f"D(Y|X): {round(dyx,3)} \n\n Regla de desición: \n {desicionxy} \n\n D(X|Y):: {round(dxy,3)} \n\n Regla de Desición: \n {desicionyx} \n \n")  
    etiqueta_somer.pack(padx=20, pady=10)
    
def coeficiente_pearson():
    
    global tabla_contingencia
    global variables_seleccionadas, df  
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    num_filas = tabla_contingencia.shape[0]
    num_columnas = tabla_contingencia.shape[1]
    sumas_filas = np.sum(tabla_contingencia, axis=1)
    sumas_columnas = np.sum(tabla_contingencia, axis=0)
    total = len(df)
    puntajes_filas = np.arange(1, num_filas + 1)
    puntajes_columnas = np.arange(1, num_columnas + 1)
    media_filas = np.sum(puntajes_filas*sumas_filas)/total
    media_columnas = np.sum(puntajes_columnas*sumas_columnas)/total
    #print("media de las filas:",media_filas,", media de las columnas:",media_columnas)
    desviaciones_filas = puntajes_filas - media_filas
    desviaciones_columnas = puntajes_columnas - media_columnas 
    numeradorfc = sum(np.sum(tabla_contingencia*desviaciones_filas.reshape(-1, 1)*desviaciones_columnas, axis=None))
    denominadorf = sum(np.sum(tabla_contingencia*desviaciones_filas.reshape(-1, 1)**2, axis=None))
    denominadorc = sum((np.sum(tabla_contingencia*desviaciones_columnas**2, axis=None)))
    coef_pearson=numeradorfc/np.sqrt((denominadorf*denominadorc))
    #print(coef_pearson)
    print("Regla de desicion:")
    resultado1 = ("Si el coefiente de Pearson es cercano a +1, tiene tendencia lineal creciente fuerte.")
    resultado2 = ("Si el coefiente de Pearson es cercano a -1, tiene tendencia lineal decreciente fuerte.")
    resultado4 = ("Coefiente de Pearson es igual a cero, no tiene una tendencia lineal.")
    resultado3 = ("Coefiente de Pearson es cercano a cero, tiene una tendencia lineal débil.")
    ventana_pearson = tk.Toplevel(root)
    ventana_pearson.title("Coefiente de Pearson")
    etiqueta_pearson = tk.Label(ventana_pearson, text=f"Coeficiente de Pearson: {round(coef_pearson,3)} \n\n Condición: \n {resultado1} \n {resultado2} \n {resultado3} \n {resultado4} \n")
    etiqueta_pearson.pack(padx=20, pady=10)         
    
def coeficiente_spearman():
    global tabla_contingencia
    global variables_seleccionadas, df 
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    coef_spearman, p_value = spearmanr(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]])
    nivel_significancia = 0.05    
    print("Regla de desicion:")
    if p_value < nivel_significancia:
        resultado1  = ("Se rechaza la hipótesis nula.")
        resultado2  = ("Hay evidencia de una correlación significativa entre las variables.")
    else:
        resultado1  =("No se rechaza la hipótesis nula.")
        resultado2  = ("No hay suficiente evidencia de una correlación significativa entre las variables.")
    ventana_pearson = tk.Toplevel(root)
    ventana_pearson.title("Coefiente de Spearman")
    etiqueta_pearson = tk.Label(ventana_pearson, text=f"Nivel de significancia:\n {nivel_significancia} \n\n Coeficiente de Spearman: \n {round(coef_spearman,3)} \n\n Desición: \n {resultado1} \n {resultado2} \n")
    etiqueta_pearson.pack(padx=20, pady=10)  
    
def coeficiente_incertidumbre(): 
    global tabla_contingencia
    global variables_seleccionadas, df   
    if not variables_seleccionadas or df.empty:
        messagebox.showinfo(" ", "No hay variables seleccionadas o el DataFrame está vacío.")
        return
    tabla_contingencia = pd.crosstab(df[variables_seleccionadas[0]], df[variables_seleccionadas[1]]) 
    n = tabla_contingencia.values.sum()
    ni = tabla_contingencia.sum(axis=1)
    nim = ni/n
    nl = nim*np.log(nim)
    hx= -np.sum(nl)
    nj = tabla_contingencia.sum()
    njm = nj/n
    nm = njm*np.log(njm)
    hy= -np.sum(nm)
    nij = tabla_contingencia
    nijm = nij/n
    nijlog = np.log(nijm)
    ijk = nijm*nijlog
    nmn = -np.sum(ijk)
    hxy = nmn.sum()
    coef_u = (hx+hy-hxy)/hy
    nivel_significancia = 0.05   
    print("Regla de desicion:")
    if(coef_u > 0.5):
        resultadosa = ("El valor es relativamente alto, explica la variable respuesta.")
    elif(coef_u <= 0.5):
        resultadosa = ("El valor es relativamente bajo, no explica la variable respuesta.") 
    else:
        resultadosa = ("El valor nulo, por lo cual es inexistente.")
    u = 2*((hx+hy-hxy)/(hx+hy))
    
    if(u > 0.5):
        resultadosb = ("El coefiente de incertidumbre u es relativamente alto, explica la variable respuesta.")
    elif(u <= 0.5):
        resultadosb = ("El coefiente de incertidumbre u es un valor relativamente bajo, no explica la variable respuesta.")
    else:
        resultadosb = ("El coefiente de incertidumbre u es un valor nulo, por ende es inexistente.")
    ventana_u = tk.Toplevel(root)
    ventana_u.title("Coeficiente de incertidumbre")
    etiqueta_u = tk.Label(ventana_u, text=f"Nivel de significancia:\n {nivel_significancia} \n\n U(Y|X): {round(coef_u,3)}\n Decisión: \n {resultadosa} \n\n U : {round(u,3)}\n Decisión: \n {resultadosb} \n")
    etiqueta_u.pack(padx=20, pady=10)
    
def analisis_similaridad():
    global df
    global contenedor_indice_similaridad, contenedor_lista_matriz, contenedor_matriz, contenedor_de_la_matriz
    contenedor_indice_similaridad.clear()
    contenedor_lista_matriz.clear()
    contenedor_matriz.clear()
    contenedor_de_la_matriz.clear()

    try:
        columnas = list(df.columns)
        x = len(columnas)

    except ValueError:
        messagebox.showinfo('Informacion', 'Formato incorrecto')
        return None

    except FileNotFoundError:
        messagebox.showinfo('Informacion', 'El archivo esta malogrado')
        return None

    matriz = np.zeros((x, x))


    for i in range(x):
        for j in range(i+1, x):
            A = df.columns[i]
            B = df.columns[j]
            ai = df[A]
            aj = df[B]
            card = np.count_nonzero((ai == 1) & (aj == 1))
            n = len(df)
            n_ai = np.count_nonzero(ai)
            n_aj = np.count_nonzero(aj)
            kc = (card - (n_ai * n_aj) / n) / np.sqrt((n_ai * n_aj) / n)
            sim = norm.cdf(kc)
            contenedor_indice_similaridad.append((A, B, card, kc, sim))
            ################ CONTENEDOR DE LOS INDICES DE SIMILARIDAD ################
            contenedor_lista_matriz.append((A, B, round(sim, 2)))
            ############### CREACION AUTOMATICA DE LA MATRIZ CUADRADA ################
            matriz[i, j] = sim
            matriz[j, i] = sim

    contenedor_matriz.append(matriz)
    column_labels = df.columns.tolist()  # encabezado automaticamente
    dfn = pd.DataFrame(matriz)
    encabezados = column_labels
    dfn.columns = encabezados  # encabezados de la columna
    dfn.index = encabezados   # encabezados de la fila

    ########################## INICIO DE CONTENEDORES ############################
    contenedor_matriz.append(matriz)
    column_labels = df.columns.tolist()  # encabezado automaticamente

    dfn = pd.DataFrame(matriz)
    encabezados = column_labels # Agregar encabezados a las columnas y filas de la matriz cero
    dfn.columns = encabezados  # encabezados de la columna
    dfn.index = encabezados   # encabezados de la fila
    contenedor_de_la_matriz.append(dfn)  # contenedor de la matriz data frame

    data_sim = pd.DataFrame(contenedor_indice_similaridad, columns=['var1', 'var2', 'card(ai ∩ aj)', 'kc', 's(ai,aj)'])
    
    resultado = "#####################################################################\n"
    resultado += "VALORES DE COPRESENCIAS ESTANDARIZADAS E ÍNDICES DE SIMILARIDAD\n"
    resultado += str(data_sim) + "\n\n"
    resultado += "#####################################################################\n"
    resultado += "MATRIZ NIVEL 0\n"
    resultado += str(dfn) + "\n\n"

    ventana_resultados = tk.Toplevel(root)
    ventana_resultados.title("Resultados de Análisis de Similaridad")
    text_resultados = tk.Text(ventana_resultados)
    text_resultados.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    text_resultados.insert(tk.END, resultado)
    text_resultados.tag_config("resaltado", foreground="red")

def ordenar_nodos():
    
    global df
    global contenedor_indice_similaridad, contenedor_lista_matriz, contenedor_valor, contendor_diccionario, xx, contenedor_sumaI, contenedor_f, contenedor_rkk, contenedor_skk, contenedor_v, contenedor_nodos_m, contenedor_valores, con_al, contenedor_vector
    global text_resultados
    
    contenedor_indice_similaridad.clear()
    contenedor_lista_matriz.clear()
    contenedor_valor.clear()
    contendor_diccionario.clear()
    xx.clear()
    contenedor_sumaI.clear()
    contenedor_f.clear()
    contenedor_rkk.clear()
    contenedor_skk.clear()
    contenedor_v.clear()
    contenedor_nodos_m.clear()
    contenedor_valores.clear()
    con_al.clear()
    contenedor_vector.clear()

    try:
        columnas = list(df.columns)
        df_fila = df.to_numpy().tolist()
        x = len(columnas)
        y = len(df_fila)

    except ValueError:
        messagebox.showinfo('Informacion', 'Formato incorrecto')
        return None

    except FileNotFoundError:
        messagebox.showinfo('Informacion', 'El archivo esta malogrado')
        return None

    matriz = np.zeros((x, x))

    for i in range(x):
        for j in range(i+1, x):
            A = df.columns[i]
            B = df.columns[j]
            ai = df[A]
            aj = df[B]
            card = np.count_nonzero((ai == 1) & (aj == 1))
            n = len(df)
            n_ai = np.count_nonzero(ai)
            n_aj = np.count_nonzero(aj)
            kc = (card - (n_ai * n_aj) / n) / np.sqrt((n_ai * n_aj) / n)
            sim = norm.cdf(kc)
            contenedor_indice_similaridad.append((A, B, card, kc, sim))
            ################ CONTENEDOR DE LOS INDICES DE SIMILARIDAD ################
            contenedor_lista_matriz.append((A, B, round(sim, 2)))
            ############### CREACION AUTOMATICA DE LA MATRIZ CUADRADA ################
            matriz[i, j] = sim
            matriz[j, i] = sim


    dfnuevo = pd.DataFrame(contenedor_lista_matriz) # Convertimos primero los datos en Data Frame
    v_1 = (dfnuevo[0] +","+ dfnuevo[1]) # Unimos las columnas de variables encabezado
    v_2 = (dfnuevo[1] +","+ dfnuevo[0]) # Unimos la variables de las columna valor
    a = np.array(v_1) # Transformacion en un array
    b = np.array(v_2) # Transformacion en un array
    c = np.array(dfnuevo[2]) # Transformacion en un array
    
    # Obtener el orden de 'c' de mayor a menor
    orden = np.argsort(c)[::-1]

    # Actualizar las filas 'a' y 'b' en función del orden de 'c'
    a_ordenada = a[orden]
    b_ordenada = b[orden]
    c_ordenada = c[orden]

    # Crear un nuevo DataFrame con las filas ordenadas
    df_ordenados = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada, 'Valor': c_ordenada})
    df_ordenadoc = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada})#, 'Valor': c_ordenada})
    df_ordenadov = pd.DataFrame({'Valor1': c_ordenada,'Valor2': c_ordenada})

    df_combined1 = pd.DataFrame(df_ordenadoc.values.reshape(-1), columns=['Variables'])
    df_combined2 = pd.DataFrame(df_ordenadov.values.reshape(-1), columns=['Valor'])
    df_concatenado = pd.concat([df_combined1, df_combined2], axis=1)

    contenedor_valor.append(df_combined2)
    c_ordenado = df_combined2['Valor']
    c_orden = pd.Series(c_ordenado)
    grupos = c_orden.groupby(c_orden).groups
    xx.append(grupos)

    ########### CONTENEDORES DE LOS GRUPOS Y TRANSFORMACION ######################
    diccionario = xx[0]
    
    contendor_diccionario.append(diccionario)
    ##############################################################################

    resultado = "######### GRUPOS SELECCIONADOS Y CODIFICADOS ###############################\n"
    
    dados = contendor_diccionario[0]
    w = 0
    for chave, valor in dados.items():
        w = w + 1
        #lista = valor
        indice = pd.Index(valor)
        valores = indice.values.tolist()
        contenedor_valores.append(valores)
        resultado += f"indice de similaridad: {chave} Grupo {w}: {valores}\n"

    variables = contenedor_valores
    contenedor = contenedor_valor[0]
    
    s_k = len(contenedor) 
    suma_total = 0
    mk = len(variables)
    for i in range(mk):
      contar = count(variables[i])
      resultados.append(contar)
      suma_total += contar
      f = (count(variables[i])-1)
      contenedor_sumaI.append(suma_total)
      contenedor_f.append(f)
            
       
    fila_invertida = contenedor_sumaI[::-1]
    columna_invertida = contenedor_f[::-1]
    s_k = (len((contenedor))-1)
    
    for i in range(len(contenedor_f)):
      fi.append(columna_invertida[i])
      Ii.append(fila_invertida[i])
      rk =  i+1
      sk = s_k - i
      contenedor_rkk.append(rk)
      contenedor_skk.append(sk)
      card = (sum(Ii) - rk*((rk+1)/2) - sum(fi))
      s_beta_k = round((card-(0.5*sk*rk))/math.sqrt((sk*rk*(sk+rk+1))/12),5)

      con_al.append(s_beta_k)

      ############# ***INICIO PROCESO ULTIMO VALOR*** ##########################
      valoresv = con_al
      
      for i in range(1, len(valoresv)):
        resta = valoresv[i] - valoresv[i-1]  # Resta entre el valor anterior y el actual
        contenedor_vector.append(resta)
      contenedor_vector.insert(0,valoresv[0])
      for j in range(len(contenedor_vector)):
        v = round(contenedor_vector[j],3)
      contenedor_v.append(v)

      
      contenedor_nodos_m.append((card, s_beta_k, v))


    data_nodos = pd.DataFrame(contenedor_nodos_m, columns=['Card()', 'S(Ω,k)', 'V(Ω,k)'])
    
    resultado += "###########################################################################\n"
    
    resultado += "VALORES DEL ÍNDICE CENTRADO Y DE LA FUNCIÓN V(Ω k)\n"
    resultado += str(data_nodos) + "\n\n"

    ventana_resultados = tk.Toplevel(root)
    ventana_resultados.title("Resultados del Análisis de Nodos Significativos")

    text_resultados = tk.Text(ventana_resultados)
    text_resultados.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    text_resultados.insert(tk.END, resultado)

    # Datos de la serie temporal
    n = len(contenedor_v)  # Número hasta el cual se generará la secuencia
    x = list(range(1, n+1))
    y = contenedor_v

    # Crear la gráfica
    plt.plot(x, y, 'bo-', label='Valores')  # Puntos marcados en azul

    # Agregar etiquetas con los valores de cada punto
    for i in range(len(x)):
        plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom', color='#006400')

    # Agregar una línea horizontal en cero
    plt.axhline(y=0, color='r', linestyle='--')

    # Etiquetas y título
    plt.xlabel('Grafica de los valores de v(Ω,k)')
    plt.title(r'V($\Omega$,k)')

    # Guardar la gráfica como una imagen
    plt.savefig('grafica_omega.png')

    # Mostrar la gráfica
    plt.show()

    # Abrir la imagen automáticamente
    Image.open('grafica_omega.png').show()
    
##########################FIN DE LAS FUNCIONES#################################
   
root = tk.Tk()
root.title("ANDACA")
toolbar = tk.Frame(root)
toolbar.pack(side=tk.TOP, fill=tk.X) 
nombres_cajones = ["Importar", "Contingencia", "Medidas"]
for i, nombre_cajon in enumerate(nombres_cajones):
    menubutton = tk.Menubutton(toolbar, text=nombre_cajon, relief=tk.RAISED)
    menubutton.pack(side=tk.LEFT)
    menu = tk.Menu(menubutton, tearoff=0)
    if i == 2:  # Crear 2 botones desplegables para "Medidas"
        submenu1 = tk.Menu(menu, tearoff=0)
        for j in range(2):
            if j == 0:
                submenu1.add_command(label=f'Chi cuadrado',command=chi_cuadrado)
            elif j == 1:
                submenu1.add_command(label=f'Contraste mediante la razón de verosimilitudes',command=razon_verosimilitud)
        submenu2 = tk.Menu(menu, tearoff=0)
        for j in range(3):
            if j == 0:
                submenu2.add_command(label=f'Coeficiente de Contingencia',command=coeficiente_contingencia)
            elif j == 1:
                submenu2.add_command(label=f'Coeficiente Phi',command=coeficiente_phi)
            elif j == 2:
                submenu2.add_command(label=f'Coeficiente de Cramer',command=coeficiente_cramer)            
        submenu3 = tk.Menu(menu, tearoff=0)
        for j in range(5):
            if j == 0:
                submenu3.add_command(label=f'Concordantes y Discordantes',command=cyd)
            elif j == 1:
                submenu3.add_command(label=f'Estadístico Gamma',command=estadistico_gamma)
            elif j == 2:
                submenu3.add_command(label=f'Estadística Taub de Kendall(τb)',command=estadistico_kendall)     
            elif j == 3:
                submenu3.add_command(label=f'Estadística de Tauc Stuart(τc)',command=estadistico_stuart)                 
            elif j == 4:
                submenu3.add_command(label=f'Estadística de Somers',command=estadistico_somers)
        submenu4 = tk.Menu(menu, tearoff=0)
        for j in range(3):
            if j == 0:
                submenu4.add_command(label=f'Coeficiente de Pearson',command=coeficiente_pearson)
            elif j == 1:
                submenu4.add_command(label=f'Coeficiente de Spearman',command=coeficiente_spearman)
            elif j == 2:
                submenu4.add_command(label=f'Coeficiente de Incertidumbre',command=coeficiente_incertidumbre)     
        submenu5 = tk.Menu(menu, tearoff=0)
        for j in range(3):
            if j == 0:
                submenu5.add_command(label=f'Análisis de similaridad',command=analisis_similaridad)
            elif j == 1:
                submenu5.add_command(label=f'Nodos significativos',command=ordenar_nodos)
            elif j == 2:
                submenu5.add_command(label=f'Coeficiente de Incertidumbre',command=coeficiente_incertidumbre)  
            #submenu5.add_command(label=f'Medida 5.{j+1}')  
           
        menu.add_cascade(label="Independencia", menu=submenu1)
        menu.add_cascade(label="Asociadas al chi^2", menu=submenu2)
        menu.add_cascade(label="Asociación ordinales", menu=submenu3)
        menu.add_cascade(label="Otras medidas de asociación", menu=submenu4)
        menu.add_cascade(label="Análisis estadístico implicativo", menu=submenu5)
        menu.add_cascade(label="Medidas basadas en la reducción proporcional del error(RPE)",command = calculo_rpe)
    else:
        for j in range(3):
            if j ==0 and i == 0:
                menu.add_command(label="Abrir", command=cargar_excel)
            elif j == 1 and i == 0:
                menu.add_command(label="Mostrar", command=mostrar_resultados)  
            elif j == 2 and i == 0:
                
                menu.add_command(label="Eliminar", command=limpiar_data) 
            elif j == 0 and i == 1:
                menu.add_command(label="Tabla de contingencia", command= tabla_de_contigencia)
            elif j == 1 and i == 1:
                menu.add_command(label="Tabla de frecuencia", command=mostrar_frecuencias_de_contingencia)
            elif j == 2 and i == 1:
                menu.add_command(label="Gráficas de frecuencias", command=mostrar_graficos_de_contingencia)        
            else:
                menu.add_command(label=f'Variable {i+1}.{j+1}')
    menubutton.config(menu=menu) 

listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
listbox.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
exportar_button = tk.Button(root, text="Exportar Variables", command=exportar_variables)
exportar_button.pack(pady=10)
cuadro_seleccionadas = tk.Text(root, height=10, width=30)
cuadro_seleccionadas.pack(pady=10)
exportar_button = tk.Button(root, text="Convertir a categóricas", command=convertir_a_categoricas)
exportar_button.pack(pady=10)
root.mainloop()

