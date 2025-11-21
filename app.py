import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go
import time

# Configuración de la página
st.set_page_config(page_title="Caracterización Probabilística & RAM", layout="wide")

# --- FUNCIONES DE CÁLCULO ---

def calculate_ad_stat(data, dist_name, params):
    """Calcula el estadístico Anderson-Darling para una distribución dada."""
    n = len(data)
    sorted_data = np.sort(data)
    
    # Obtener la CDF teórica según la distribución
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
    
    # Evitar log(0)
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    
    # Fórmula AD
    S = np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])))
    ad_stat = -n - S / n
    return ad_stat

def monte_carlo_p_value(data, dist_name, params, n_sim=1000):
    """
    Calcula el p-value usando Monte Carlo.
    NOTA: n_sim reducido a 1000 por defecto para velocidad en web, 
    aumentar para mayor precisión.
    """
    n = len(data)
    ad_orig = calculate_ad_stat(data, dist_name, params)
    
    count = 0
    # Simulaciones
    for _ in range(n_sim):
        # Generar muestra sintética
        if dist_name == 'Normal':
            sim_data = stats.norm.rvs(*params, size=n)
            new_params = stats.norm.fit(sim_data)
        elif dist_name == 'Lognormal':
            sim_data = stats.lognorm.rvs(*params, size=n)
            # Forzamos floc=0 si queremos lognormal de 2 parámetros estándar
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
            
        ad_sim = calculate_ad_stat(sim_data, dist_name, new_params)
        
        if ad_sim >= ad_orig:
            count += 1
            
    return count / n_sim

# --- INTERFAZ DE USUARIO ---

st.title("📊 Herramienta de Caracterización Probabilística (RAM)")
st.markdown("""
Esta herramienta ajusta datos a distribuciones (Normal, Lognormal, Weibull, Gamma, Exponencial),
calcula el **P-Value mediante Monte Carlo** y utiliza **Kaplan-Meier** como referencia empírica.
""")

# 1. Panel Lateral - Entrada de Datos
st.sidebar.header("1. Carga de Datos")
input_text = st.sidebar.text_area("Pega tus datos aquí (uno por línea o separados por coma):", height=200, value="105.5\n98.2\n134.1\n155.9\n78.4\n112.0\n143.8\n122.5\n95.0\n110.2\n130.5\n85.6\n145.2\n102.3\n118.7")

num_simulaciones = st.sidebar.slider("Simulaciones Monte Carlo (Precisión vs Velocidad)", 100, 5000, 1000)

percentil_req = st.sidebar.number_input("Calcular Percentil específico (Pxx):", min_value=1.0, max_value=99.0, value=50.0)

if input_text:
    # Procesar datos
    try:
        raw_data = input_text.replace(',', '\n').split('\n')
        data = [float(x.strip()) for x in raw_data if x.strip()]
        data = np.array(data)
        n_datos = len(data)
        
        st.sidebar.success(f"✅ {n_datos} datos cargados correctamente.")
        
        if n_datos < 5:
            st.error("Por favor ingresa al menos 5 datos para un análisis estadístico mínimo.")
            st.stop()
            
    except ValueError:
        st.sidebar.error("Error en el formato de datos. Asegúrate de usar números.")
        st.stop()

    # --- PROCESAMIENTO PRINCIPAL ---
    
    if st.button("Ejecutar Análisis"):
        
        with st.spinner('Ajustando distribuciones y ejecutando Monte Carlo...'):
            results = []
            
            # Definir distribuciones a probar
            dist_list = [
                ('Normal', stats.norm, {}),
                ('Lognormal', stats.lognorm, {'floc': 0}), # 2-parameter lognormal
                ('Weibull', stats.weibull_min, {'floc': 0}), # 2-parameter weibull
                ('Gamma', stats.gamma, {'floc': 0}),
                ('Exponencial', stats.expon, {'floc': 0})
            ]
            
            best_dist_name = ""
            best_p_value = -1
            best_params = None
            
            # Barra de progreso
            progress_bar = st.progress(0)
            
            for i, (name, func, constraints) in enumerate(dist_list):
                # 1. Ajuste (MLE)
                params = func.fit(data, **constraints)
                
                # 2. Calcular P-Value (Monte Carlo)
                p_val = monte_carlo_p_value(data, name, params, n_sim=num_simulaciones)
                
                # Guardar resultados
                param_str = ", ".join([f"{p:.4f}" for p in params])
                results.append({
                    "Distribución": name,
                    "P-Value (A-D)": p_val,
                    "Parámetros": param_str,
                    "Params_Obj": params, # Guardar objeto para gráficos
                    "Func_Obj": func
                })
                
                # Determinar el mejor ajuste (mayor p-value)
                if p_val > best_p_value:
                    best_p_value = p_val
                    best_dist_name = name
                    best_params = params
                
                progress_bar.progress((i + 1) / len(dist_list))
            
            # Crear DataFrame de resultados
            df_results = pd.DataFrame(results).drop(columns=["Params_Obj", "Func_Obj"])
            df_results = df_results.sort_values(by="P-Value (A-D)", ascending=False)
            
            # --- MOSTRAR RESULTADOS ---
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Resultados del Ajuste")
                st.dataframe(df_results.style.apply(lambda x: ['background-color: #d4edda' if x['Distribución'] == best_dist_name else '' for i in x], axis=1))
                
                st.info(f"🏆 Mejor ajuste: **{best_dist_name}** con p-value de {best_p_value:.4f}")
                
                # Cálculo de Percentil
                st.subheader(f"Cálculo de Percentiles ({best_dist_name})")
                
                # Recuperar función del mejor ajuste
                best_row = [r for r in results if r['Distribución'] == best_dist_name][0]
                best_func = best_row['Func_Obj']
                
                # Calcular valor del percentil solicitado
                val_percentil = best_func.ppf(percentil_req / 100.0, *best_params)
                st.metric(label=f"Valor para P{int(percentil_req)}", value=f"{val_percentil:.4f}")
                
                st.write("---")
                st.write("**Parámetros detallados:**")
                if best_dist_name == "Weibull":
                    st.write(f"Forma (k/beta): {best_params[0]:.4f}")
                    st.write(f"Escala (lambda/eta): {best_params[2]:.4f}")
                elif best_dist_name == "Normal":
                    st.write(f"Media: {best_params[0]:.4f}")
                    st.write(f"Desv. Estándar: {best_params[1]:.4f}")
                else:
                    st.write(f"Raw Params: {best_params}")

            with col2:
                st.subheader("Visualización Gráfica")
                
                # --- KAPLAN-MEIER (Empírico) ---
                kmf = KaplanMeierFitter()
                kmf.fit(data)
                km_df = kmf.cumulative_density_
                km_surv = kmf.survival_function_
                
                # Rango para gráficos (X axis)
                x_min, x_max = 0, max(data) * 1.2
                x_vals = np.linspace(x_min, x_max, 200)
                
                # Crear pestañas
                tab1, tab2, tab3 = st.tabs(["PDF (Densidad)", "CDF (Acumulada Directa)", "Supervivencia (Acumulada Indirecta)"])
                
                # --- TAB 1: PDF ---
                with tab1:
                    fig_pdf = go.Figure()
                    # Histograma de datos
                    fig_pdf.add_trace(go.Histogram(x=data, histnorm='probability density', name='Datos Reales', opacity=0.5, marker_color='gray'))
                    
                    # Curvas de las distribuciones seleccionadas
                    for res in results:
                        y_vals = res['Func_Obj'].pdf(x_vals, *res['Params_Obj'])
                        
                        # Resaltar la mejor
                        line_width = 4 if res['Distribución'] == best_dist_name else 1
                        opacity = 1 if res['Distribución'] == best_dist_name else 0.4
                        
                        fig_pdf.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=res['Distribución'], line=dict(width=line_width), opacity=opacity))
                    
                    fig_pdf.update_layout(title="Función de Densidad de Probabilidad (PDF)", xaxis_title="Valor", yaxis_title="Densidad")
                    st.plotly_chart(fig_pdf, use_container_width=True)

                # --- TAB 2: CDF (Comparación con Kaplan-Meier) ---
                with tab2:
                    fig_cdf = go.Figure()
                    
                    # Kaplan-Meier (Puntos empíricos) - Usamos steps para simular la escalera
                    fig_cdf.add_trace(go.Scatter(x=km_df.index, y=km_df.iloc[:, 0], mode='lines+markers', name='Empírico (Kaplan-Meier)', line=dict(shape='hv', color='black', dash='dash')))
                    
                    for res in results:
                        y_vals = res['Func_Obj'].cdf(x_vals, *res['Params_Obj'])
                        line_width = 4 if res['Distribución'] == best_dist_name else 1
                        
                        fig_cdf.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=res['Distribución'], line=dict(width=line_width)))
                        
                    fig_cdf.update_layout(title="Función de Distribución Acumulada (CDF)", xaxis_title="Valor", yaxis_title="Probabilidad Acumulada")
                    st.plotly_chart(fig_cdf, use_container_width=True)

                # --- TAB 3: RELIABILITY / SURVIVAL (1 - CDF) ---
                with tab3:
                    fig_rel = go.Figure()
                    
                    # Kaplan-Meier Supervivencia
                    fig_rel.add_trace(go.Scatter(x=km_surv.index, y=km_surv.iloc[:, 0], mode='lines+markers', name='Empírico (Kaplan-Meier)', line=dict(shape='hv', color='black', dash='dash')))
                    
                    for res in results:
                        # Survival = 1 - CDF (o usar sf si disponible)
                        y_vals = res['Func_Obj'].sf(x_vals, *res['Params_Obj'])
                        line_width = 4 if res['Distribución'] == best_dist_name else 1
                        
                        fig_rel.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=res['Distribución'], line=dict(width=line_width)))
                        
                    fig_rel.update_layout(title="Función de Confiabilidad/Supervivencia (R(t) = 1 - CDF)", xaxis_title="Valor", yaxis_title="Probabilidad de Supervivencia")
                    st.plotly_chart(fig_rel, use_container_width=True)

else:
    st.info("Esperando datos en el panel lateral para iniciar el cálculo...")