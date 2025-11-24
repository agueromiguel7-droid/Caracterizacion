import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go
import os

# --- CONFIGURACIÓN INICIAL ---
try:
    st.set_page_config(page_title="RAM Fitter Pro", layout="wide", page_icon="Logo_Prod_Risk_Solution.png")
except:
    st.set_page_config(page_title="RAM Fitter Pro", layout="wide", page_icon="📊")

# --- FUNCIONES DE CÁLCULO ---

def calculate_ad_stat(data, dist_name, params):
    """Calcula el estadístico A-D (A^2)."""
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
    
    # Clip para evitar log(0)
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    S = np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])))
    return -n - S / n

def monte_carlo_analysis(data, dist_name, params, n_sim=1000):
    """Devuelve una tupla: (p_value, ad_stat_original)"""
    n = len(data)
    # 1. Calcular estadístico original
    ad_orig = calculate_ad_stat(data, dist_name, params)
    
    count = 0
    # 2. Simulaciones
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
            
        # Calcular AD de la simulación
        ad_sim = calculate_ad_stat(sim_data, dist_name, new_params)
        
        if ad_sim >= ad_orig:
            count += 1
            
    return (count / n_sim), ad_orig

def format_params(dist_name, params):
    """Da formato legible y técnico a los parámetros calculados."""
    if dist_name == 'Normal':
        # params: (mu, sigma)
        return f"Media (μ): {params[0]:.4f}, Desv. (σ): {params[1]:.4f}"
    
    elif dist_name == 'Lognormal':
        # params scypy: (s, loc, scale) -> scale = exp(mu_log)
        # Queremos: mu_log y sigma_log
        sigma_log = params[0] # Shape parameter s es sigma_log
        scale = params[2]
        mu_log = np.log(scale)
        return f"Media Log (μ_log): {mu_log:.4f}, Desv. Log (σ_log): {sigma_log:.4f}"
    
    elif dist_name == 'Weibull':
        # params scipy: (c, loc, scale) -> c es forma, scale es escala
        forma = params[0]
        escala = params[2]
        return f"Forma (β): {forma:.4f}, Escala (η): {escala:.4f}"
    
    elif dist_name == 'Gamma':
        # params scipy: (a, loc, scale) -> a es forma, scale es theta
        forma = params[0]
        escala = params[2]
        return f"Forma (α): {forma:.4f}, Escala (θ): {escala:.4f}"
        
    elif dist_name == 'Exponencial':
        # params scipy: (loc, scale) -> scale = 1/lambda (Mean Time)
        mean_time = params[1]
        tasa = 1.0 / mean_time
        return f"Tasa (λ): {tasa:.5f}, MTBF (1/λ): {mean_time:.2f}"
    
    return str(params)

# --- INTERFAZ ---

col_logo, col_titulo = st.columns([1, 8])
with col_logo:
    if os.path.exists("Logo_Prod_Risk_Solution.png"):
        st.image("Logo_Prod_Risk_Solution.png", width=100)
    else:
        st.write("📊") # Emoji si no hay logo
with col_titulo:
    st.title("Herramienta de Caracterización Probabilística (R&DF)")

st.markdown("---")

# --- BARRA LATERAL ---
st.sidebar.header("1. Configuración")

# Botón de Reinicio (Punto 4)
if st.sidebar.button("🗑️ Borrar Todo / Reiniciar"):
    st.session_state.clear()
    st.rerun()

# Text Area vinculado a session_state para poder borrarlo si es necesario, 
# pero le damos un valor por defecto si está vacío.
if 'default_input' not in st.session_state:
    st.session_state['default_input'] = "105.5\n98.2\n134.1\n155.9\n78.4\n112.0\n143.8\n122.5\n95.0\n110.2\n130.5\n85.6\n145.2\n102.3\n118.7"

input_text = st.sidebar.text_area("Datos de entrada:", height=200, value=st.session_state['default_input'])
num_simulaciones = st.sidebar.slider("Simulaciones Monte Carlo", 100, 10000, 1000)
ejecutar = st.sidebar.button("🚀 Ejecutar Análisis")

# --- LÓGICA PRINCIPAL ---

if 'resultados' not in st.session_state:
    st.session_state['resultados'] = None

if ejecutar:
    if input_text:
        try:
            raw_data = input_text.replace(',', '\n').split('\n')
            data = np.array([float(x.strip()) for x in raw_data if x.strip()])
            
            if len(data) < 5:
                st.error("Se necesitan al menos 5 datos.")
            else:
                st.session_state['datos_procesados'] = data
                
                with st.spinner('Realizando ajustes y simulaciones...'):
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
                        # 1. Ajuste MLE
                        params = func.fit(data, **const)
                        
                        # 2. Monte Carlo (Obtenemos P-value y AD-Stat)
                        p_val, ad_stat = monte_carlo_analysis(data, name, params, n_sim=num_simulaciones)
                        
                        # 3. Formatear Parámetros (Puntos 1 y 2)
                        params_str = format_params(name, params)
                        
                        results_list.append({
                            "Distribución": name,
                            "Estadístico A-D": ad_stat, # (Punto 3)
                            "P-Value": p_val,
                            "Parámetros Técnicos": params_str,
                            "Params_Raw": params,
                            "Obj": func
                        })
                        
                        if p_val > best_p:
                            best_p = p_val
                            best_name = name
                            
                        prog_bar.progress((i + 1) / len(dist_defs))
                    
                    df_res = pd.DataFrame(results_list).sort_values(by="P-Value", ascending=False)
                    st.session_state['resultados'] = df_res
                    st.session_state['mejor_ajuste'] = best_name
                    
        except ValueError:
            st.error("Error en formato de datos.")

# --- VISUALIZACIÓN ---

if st.session_state['resultados'] is not None:
    data = st.session_state['datos_procesados']
    df_results = st.session_state['resultados']
    best_dist = st.session_state['mejor_ajuste']
    
    # 1. TABLA DE RESULTADOS
    st.subheader("📊 Tabla de Resultados de Ajuste")
    
    # Preparamos el dataframe para mostrar (ocultamos las columnas de objetos internos)
    display_df = df_results[['Distribución', 'Estadístico A-D', 'P-Value', 'Parámetros Técnicos']].copy()
    
    # Formato visual
    st.dataframe(
        display_df.style.format({
            "Estadístico A-D": "{:.4f}",
            "P-Value": "{:.4f}"
        }).apply(lambda x: ['background-color: #d4edda' if x['Distribución'] == best_dist else '' for i in x], axis=1),
        use_container_width=True
    )

    # 2. CALCULADORA Y GRÁFICOS
    col_izq, col_der = st.columns([2, 1])
    
    with col_der:
        st.info(f"🏆 Mejor Ajuste: **{best_dist}**")
        st.markdown("### 🧮 Calculadora de Percentiles")
        
        dist_options = df_results['Distribución'].tolist()
        default_idx = dist_options.index(best_dist)
        
        sel_dist = st.selectbox("Selecciona Distribución:", dist_options, index=default_idx)
        sel_percentil = st.number_input("Percentil (Pxx):", 0.1, 99.9, 50.0, step=0.5)
        
        # Cálculo on-the-fly
        row = df_results[df_results['Distribución'] == sel_dist].iloc[0]
        func_obj = row['Obj']
        params_raw = row['Params_Raw']
        
        val_res = func_obj.ppf(sel_percentil / 100.0, *params_raw)
        st.success(f"**P{sel_percentil} = {val_res:.4f}**")
        st.caption(f"Parámetros usados: {row['Parámetros Técnicos']}")

    with col_izq:
        st.markdown("### 📈 Visualización Gráfica")
        
        kmf = KaplanMeierFitter()
        kmf.fit(data)
        km_cdf = kmf.cumulative_density_
        
        x_vals = np.linspace(0, max(data)*1.2, 300)
        
        tab1, tab2 = st.tabs(["Densidad (PDF)", "Acumulada Dir (CDF)", "Acumulada Inv (R(t))"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Datos', opacity=0.3, marker_color='grey'))
            for _, r in df_results.iterrows():
                y = r['Obj'].pdf(x_vals, *r['Params_Raw'])
                is_selected = (r['Distribución'] == sel_dist)
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y, mode='lines', name=r['Distribución'],
                    line=dict(width=4 if is_selected else 1),
                    opacity=1 if is_selected else 0.3
                ))
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=km_cdf.index, y=km_cdf.iloc[:,0], mode='lines+markers', name='Kaplan-Meier', line=dict(dash='dash', color='black')))
            for _, r in df_results.iterrows():
                y = r['Obj'].cdf(x_vals, *r['Params_Raw'])
                is_selected = (r['Distribución'] == sel_dist)
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y, mode='lines', name=r['Distribución'],
                    line=dict(width=4 if is_selected else 1),
                    opacity=1 if is_selected else 0.3
                ))
            st.plotly_chart(fig, use_container_width=True)
	# TAB 3: Reliability
    
	with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=km_surv.index, y=km_surv.iloc[:,0], mode='lines+markers', name='Empírico (Kaplan-Meier)', line=dict(dash='dash', color='black')))
        for _, row in df_results.iterrows():
            y = row['Obj'].sf(x_vals, *row['Params_Raw'])
            width = 4 if row['Distribución'] == sel_dist else 1
            opacity = 1 if row['Distribución'] == sel_dist else 0.3
            fig.add_trace(go.Scatter(x=x_vals, y=y, mode='lines', name=row['Distribución'], line=dict(width=4 if is_selected else 1), opacity=1 if is_selected else 0.3))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Ingresa datos en el panel lateral y presiona 'Ejecutar'.")