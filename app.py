import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go
import os # Para verificar si existe la imagen

# --- CONFIGURACIÓN DE PÁGINA E ICONO ---
# Intentamos usar tu logo como favicon, si falla usamos un emoji
try:
    st.set_page_config(page_title="RAM Fitter Pro", layout="wide", page_icon="mi_logo.png")
except:
    st.set_page_config(page_title="RAM Fitter Pro", layout="wide", page_icon="📊")

# --- FUNCIONES DE CÁLCULO (Sin cambios) ---

def calculate_ad_stat(data, dist_name, params):
    n = len(data)
    sorted_data = np.sort(data)
    
    if dist_name == 'Normal':
        cdf = stats.norm.cdf(sorted_data, *params)
    elif dist_name == 'Lognormal':
        cdf = stats.lognorm.cdf(sorted_data, *params)
    elif dist_name == 'Weibull':
        cdf = stats.weibull_min.cdf(sorted_data, *params)
    elif dist_name == 'Gamma':
        cdf = stats.gamma.cdf(sorted_data, *params)
    elif dist_name == 'Exponencial':
        cdf = stats.expon.cdf(sorted_data, *params)
    
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    S = np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])))
    return -n - S / n

def monte_carlo_p_value(data, dist_name, params, n_sim=1000):
    n = len(data)
    ad_orig = calculate_ad_stat(data, dist_name, params)
    count = 0
    
    for _ in range(n_sim):
        if dist_name == 'Normal':
            sim_data = stats.norm.rvs(*params, size=n)
            new_params = stats.norm.fit(sim_data)
        elif dist_name == 'Lognormal':
            sim_data = stats.lognorm.rvs(*params, size=n)
            new_params = stats.lognorm.fit(sim_data, floc=0) 
        elif dist_name == 'Weibull':
            sim_data = stats.weibull_min.rvs(*params, size=n)
            new_params = stats.weibull_min.fit(sim_data, floc=0)
        elif dist_name == 'Gamma':
            sim_data = stats.gamma.rvs(*params, size=n)
            new_params = stats.gamma.fit(sim_data, floc=0)
        elif dist_name == 'Exponencial':
            sim_data = stats.expon.rvs(*params, size=n)
            new_params = stats.expon.fit(sim_data, floc=0)
            
        if calculate_ad_stat(sim_data, dist_name, new_params) >= ad_orig:
            count += 1
            
    return count / n_sim

# --- INTERFAZ DE USUARIO ---

# 1. ENCABEZADO CON LOGO PERSONALIZADO
col_logo, col_titulo = st.columns([1, 8])

with col_logo:
    # Verifica si la imagen existe para no romper la app si falta el archivo
    if os.path.exists("Logo_Prod_Risk_Solution.png"):
        st.image("Logo_Prod_Risk_Solution.png", width=100)
    else:
        st.warning("Sin Logo")

with col_titulo:
    st.title("Herramienta de Caracterización Probabilística (RAM)")

st.markdown("---")

# 2. BARRA LATERAL (Solo entradas de configuración)
st.sidebar.header("1. Configuración de Análisis")
input_text = st.sidebar.text_area("Datos de entrada (Pegar aquí):", height=200, value="105.5\n98.2\n134.1\n155.9\n78.4\n112.0\n143.8\n122.5\n95.0\n110.2\n130.5\n85.6\n145.2\n102.3\n118.7")
num_simulaciones = st.sidebar.slider("Simulaciones Monte Carlo", 100, 5000, 1000)

# Botón de ejecución PRINCIPAL
ejecutar = st.sidebar.button("🚀 Ejecutar Análisis Completo")

# 3. LÓGICA DE PROCESAMIENTO Y SESSION STATE

# Inicializar estado si no existe
if 'resultados' not in st.session_state:
    st.session_state['resultados'] = None
if 'datos_procesados' not in st.session_state:
    st.session_state['datos_procesados'] = None

# Si se presiona el botón, hacemos el cálculo pesado y GUARDAMOS en session_state
if ejecutar:
    if input_text:
        try:
            # Procesar datos
            raw_data = input_text.replace(',', '\n').split('\n')
            data = np.array([float(x.strip()) for x in raw_data if x.strip()])
            
            if len(data) < 5:
                st.error("Se necesitan al menos 5 datos.")
            else:
                st.session_state['datos_procesados'] = data # Guardar datos
                
                with st.spinner('Ajustando distribuciones y simulando... (Esto ocurre solo una vez)'):
                    results_list = []
                    dist_defs = [
                        ('Normal', stats.norm, {}),
                        ('Lognormal', stats.lognorm, {'floc': 0}),
                        ('Weibull', stats.weibull_min, {'floc': 0}),
                        ('Gamma', stats.gamma, {'floc': 0}),
                        ('Exponencial', stats.expon, {'floc': 0})
                    ]
                    
                    best_p = -1
                    best_name = ""
                    
                    prog_bar = st.progress(0)
                    for i, (name, func, const) in enumerate(dist_defs):
                        params = func.fit(data, **const)
                        p_val = monte_carlo_p_value(data, name, params, n_sim=num_simulaciones)
                        
                        results_list.append({
                            "Distribución": name,
                            "P-Value": p_val,
                            "Params": params,
                            "Obj": func # Guardamos el objeto función para usarlo luego
                        })
                        
                        if p_val > best_p:
                            best_p = p_val
                            best_name = name
                            
                        prog_bar.progress((i + 1) / len(dist_defs))
                    
                    # Ordenar resultados y guardar en memoria
                    df_res = pd.DataFrame(results_list).sort_values(by="P-Value", ascending=False)
                    st.session_state['resultados'] = df_res
                    st.session_state['mejor_ajuste'] = best_name
                    
        except ValueError:
            st.error("Error en formato de datos.")

# 4. VISUALIZACIÓN E INTERACTIVIDAD (Lee de session_state)

# Solo mostramos cosas si ya hay resultados en memoria
if st.session_state['resultados'] is not None:
    
    data = st.session_state['datos_procesados']
    df_results = st.session_state['resultados']
    best_dist = st.session_state['mejor_ajuste']
    
    # --- SECCIÓN DE RESULTADOS ---
    col_res, col_calc = st.columns([1.5, 1])
    
    with col_res:
        st.subheader("📊 Tabla de Resultados")
        # Mostramos tabla coloreando la mejor
        st.dataframe(
            df_results[['Distribución', 'P-Value']].style.apply(
                lambda x: ['background-color: #d4edda' if x['Distribución'] == best_dist else '' for i in x], 
                axis=1
            ), 
            use_container_width=True
        )
        st.caption(f"Mejor ajuste sugerido: **{best_dist}**")

    with col_calc:
        st.subheader("🧮 Calculadora Interactiva")
        st.markdown("Calcula percentiles **al instante** sin re-simular.")
        
        # Selector de distribución (por defecto la mejor)
        dist_options = df_results['Distribución'].tolist()
        # Encontrar índice de la mejor para ponerla por defecto
        default_idx = dist_options.index(best_dist)
        
        sel_dist = st.selectbox("Selecciona Distribución:", dist_options, index=default_idx)
        sel_percentil = st.number_input("Percentil deseado (Pxx):", 1.0, 99.9, 50.0, step=0.5)
        
        # --- CÁLCULO INSTANTÁNEO ---
        # Buscar la fila correspondiente en los resultados guardados
        row = df_results[df_results['Distribución'] == sel_dist].iloc[0]
        func_obj = row['Obj']
        params_obj = row['Params']
        
        # Calcular PPF (Percent Point Function)
        resultado_percentil = func_obj.ppf(sel_percentil / 100.0, *params_obj)
        
        st.success(f"**P{sel_percentil} ({sel_dist}) = {resultado_percentil:.4f}**")
        
        with st.expander("Ver parámetros técnicos"):
            st.write(f"Parámetros: {params_obj}")

    st.divider()

    # --- SECCIÓN DE GRÁFICOS ---
    st.subheader("📈 Curvas de Comportamiento")
    
    # Preparar datos Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(data)
    km_df = kmf.cumulative_density_
    km_surv = kmf.survival_function_
    
    x_vals = np.linspace(0, max(data)*1.2, 200)
    
    tab1, tab2, tab3 = st.tabs(["Densidad (PDF)", "Acumulada (CDF)", "Confiabilidad (R(t))"])
    
    # TAB 1: PDF
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Datos', opacity=0.3, marker_color='grey'))
        for _, row in df_results.iterrows():
            y = row['Obj'].pdf(x_vals, *row['Params'])
            width = 4 if row['Distribución'] == sel_dist else 1 # Resaltar la seleccionada en el dropdown
            opacity = 1 if row['Distribución'] == sel_dist else 0.3
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode='lines', name=row['Distribución'], line=dict(width=width), opacity=opacity))
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: CDF
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=km_df.index, y=km_df.iloc[:,0], mode='lines+markers', name='Empírico (K-M)', line=dict(dash='dash', color='black')))
        for _, row in df_results.iterrows():
            y = row['Obj'].cdf(x_vals, *row['Params'])
            width = 4 if row['Distribución'] == sel_dist else 1
            opacity = 1 if row['Distribución'] == sel_dist else 0.3
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode='lines', name=row['Distribución'], line=dict(width=width), opacity=opacity))
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Reliability
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=km_surv.index, y=km_surv.iloc[:,0], mode='lines+markers', name='Empírico (K-M)', line=dict(dash='dash', color='black')))
        for _, row in df_results.iterrows():
            y = row['Obj'].sf(x_vals, *row['Params'])
            width = 4 if row['Distribución'] == sel_dist else 1
            opacity = 1 if row['Distribución'] == sel_dist else 0.3
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode='lines', name=row['Distribución'], line=dict(width=width), opacity=opacity))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Pega tus datos en la barra lateral y presiona 'Ejecutar Análisis' para comenzar.")