# =====================================================
# MODELO SEMIESTRUTURAL DO BC COM CURVA IS
# (VARIÂNCIA DE MENSURAÇÃO COMUM A TODAS AS OBSERVÁVEIS)
# =====================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
import matplotlib.pyplot as plt

# =====================================================
# 1. CARREGAR DADOS
# =====================================================
df = pd.read_excel('Variáveis Hiato BC.xlsx', sheet_name='Dados')
df['Data'] = pd.to_datetime(df['Data'])
df = df.set_index('Data')

if 'EL_NINO' not in df.columns:
    df['EL_NINO'] = 0
if 'LA_NINA' not in df.columns:
    df['LA_NINA'] = 0

print("Dados carregados:")
print(df.head())
print(f"\nPeríodo: {df.index[0]} a {df.index[-1]}")
print(f"Observações: {len(df)}")


# =====================================================
# 2. DEFINIR MODELO STATE SPACE
#     — agora com UMA variância de mensuração comum
#       para todas as observáveis, e uma variância do estado
# =====================================================
class HiatoStateSpace(MLEModel):
    def __init__(self, endog, exog_obs, exog_trans):
        self.exog_obs = exog_obs
        self.exog_trans = exog_trans

        # Parâmetros estruturais fixos (calibrados)
        self.b1 = 0.85  # persistência do hiato (fixo)
        self.b2 = 0.44
        self.b4 = 0.054
        self.b5 = 0.84

        # Parâmetros da equação de preços livres (fixos)
        self.alpha_livres_lag = 0.24
        self.alpha_ipca_lag = 0.38
        self.alpha_focus = 1 - 0.38 - 0.24
        self.beta_brl = 0.011
        self.beta_ic = 0.023
        self.beta_sv1 = 0.12
        self.beta_el_nino = 0.0012
        self.beta_la_nina = 0.0007
        self.intercept_livres = 0.0

        # inicializar como antes
        super(HiatoStateSpace, self).__init__(endog, k_states=2, k_posdef=2, initialization='diffuse')

    @property
    def param_names(self):
        # agora estimamos apenas:
        #  - log_var_obs: log da variância de mensuração comum a todas as observáveis
        #  - log_var_state: log da variância do choque estrutural do(s) estado(s)
        return ['log_var_obs', 'log_var_state']

    @property
    def start_params(self):
        # chute inicial para os 2 parâmetros livres
        return np.array([0.0, 0.0])

    def transform_params(self, unconstrained):
        # não precisamos de transformações especiais aqui
        return unconstrained

    def untransform_params(self, constrained):
        return constrained

    def update(self, params, **kwargs):
        # chamada padrão para manter compatibilidade com statsmodels
        params = super(HiatoStateSpace, self).update(params, **kwargs)
        log_var_obs = params[0]
        log_var_state = params[1]

        # variâncias (positividade garantida exponenciando)
        var_obs = np.exp(log_var_obs)        # variância de mensuração comum (todas observáveis)
        var_state = np.exp(log_var_state)    # variância do choque do(s) estado(s)

        # --------------------------
        # MATRIZES DO MODELO (Z, T, R, Q, H)
        # --------------------------
        # design (Z): mapeia estados para observáveis
        self['design'] = np.array([
            [1.0, 0.0],            # PIB_CICLO <- h_t
            [1.0, 0.0],            # NUCI_CICLO <- h_t
            [0.0, 1.0],            # CAGED_CICLO <- s_t^h
            [self.beta_sv1, 0.0]   # LIVRES   <- beta_sv1 * h_t + exógenos
        ])

        # transition (T): dinâmica dos estados
        # [ h_t ]     [ b1   1 ] [ h_{t-1} ]
        # [ s_t ]  =  [ 0   b5 ] [ s_{t-1} ]
        self['transition'] = np.array([
            [self.b1, 1.0],
            [0.0, self.b5]
        ])

        # selection (R) e state_cov (Q): ruídos de estado
        # aqui assumimos o mesmo ruído estrutural para ambos os estados (pode ser alterado)
        self['selection'] = np.eye(2)
        self['state_cov'] = np.diag([var_state, var_state])

        # obs_cov (H): todas as observáveis têm a mesma variância de mensuração
        self['obs_cov'] = np.diag([var_obs, var_obs, var_obs, var_obs])

        # --------------------------
        # INTERCEPTOS DE OBSERVAÇÃO (d_t para LIVRES)
        # --------------------------
        livres_intercept = np.zeros(self.nobs)
        if self.exog_obs is not None:
            livres_lag = self.exog_obs[:, 0]
            focus = self.exog_obs[:, 1]
            ipca_lag = self.exog_obs[:, 2]
            brl = self.exog_obs[:, 3]
            ic_br = self.exog_obs[:, 4]
            el_nino_lag_sq = self.exog_obs[:, 5]
            la_nina_lag_sq = self.exog_obs[:, 6]
            livres_intercept = (
                self.alpha_livres_lag * livres_lag +
                self.alpha_focus * focus +
                self.alpha_ipca_lag * ipca_lag +
                self.beta_brl * brl +
                self.beta_ic * ic_br +
                self.beta_el_nino * el_nino_lag_sq +
                self.beta_la_nina * la_nina_lag_sq +
                self.intercept_livres
            )
        self['obs_intercept'] = np.zeros((self.k_endog, self.nobs))
        self['obs_intercept'][3, :] = livres_intercept

        # --------------------------
        # INTERCEPTOS DE TRANSIÇÃO (Curva IS): -b2*r_gap/4 + b4*h_star
        # --------------------------
        if self.exog_trans is not None:
            r_gap = self.exog_trans[:, 0]
            h_star = self.exog_trans[:, 1]
            trans_intercept = np.zeros((2, self.nobs))
            trans_intercept[0, :] = -self.b2 * (r_gap / 4) + self.b4 * h_star
            self['state_intercept'] = trans_intercept
        else:
            self['state_intercept'] = np.zeros((2, self.nobs))


# =====================================================
# 3. PREPARAR DADOS
# =====================================================
df['LIVRES_d11_lag1'] = df['LIVRES_d11'].shift(1)
df['IPCA_d11_lag1'] = df['IPCA_d11'].shift(1)
df['EL_NINO_lag1_sq'] = df['EL_NINO'].shift(1) ** 2
df['LA_NINA_lag1_sq'] = df['LA_NINA'].shift(1) ** 2
df_model = df.dropna()

endog = df_model[['PIB_CICLO', 'NUCI_CICLO', 'CAGED_CICLO', 'LIVRES_d11']].values
exog_obs = df_model[['LIVRES_d11_lag1', 'FOCUS', 'IPCA_d11_lag1',
                     'BRL', 'IC_BR', 'EL_NINO_lag1_sq', 'LA_NINA_lag1_sq']].values
exog_trans = df_model[['HIATO_JUROS', 'HIATO_MUNDIAL']].values

print(f"\nDados para estimação: {len(df_model)} observações")


# =====================================================
# 4. ESTIMAR MODELO (apenas VARIÂNCIAS livres: var_obs e var_state)
# =====================================================
print("\n" + "="*60)
print("ESTIMANDO MODELO STATE SPACE (variância comum às observáveis)")
print("="*60)

model = HiatoStateSpace(endog, exog_obs, exog_trans)
results = model.fit(method='lbfgs', maxiter=1000, disp=False)  # disp=False pra não poluir a saída

# Imprime apenas resumo curto (sem enfatizar log-verossimilhança)
print("\nResumo curto dos parâmetros estimados:")
for name, val in zip(results.param_names, results.params):
    print(f"  {name}: {val:.6f}")


# =====================================================
# 5. EXTRAIR ESTADOS SUAVIZADOS E IC 95%
# =====================================================
sv1_smooth = results.smoothed_state[0, :]
sv2_smooth = results.smoothed_state[1, :]
sv1_smooth_se = np.sqrt(results.smoothed_state_cov[0, 0, :])
sv2_smooth_se = np.sqrt(results.smoothed_state_cov[1, 1, :])

# IC 95%
sv1_lower = sv1_smooth - 1.96 * sv1_smooth_se
sv1_upper = sv1_smooth + 1.96 * sv1_smooth_se

# DataFrame de resultados
df_results = pd.DataFrame(index=df_model.index)
df_results['Hiato_suavizado'] = sv1_smooth
df_results['Hiato_lower_95'] = sv1_lower
df_results['Hiato_upper_95'] = sv1_upper
df_results['s_t_h'] = sv2_smooth

# =====================================================
# 6. SALVAR RESULTADOS (Excel)
# =====================================================
output_path = 'hiato_comis.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Hiato_Suavizado')
    pd.DataFrame({
        'Parametro': results.param_names,
        'Valor': results.params
    }).to_excel(writer, sheet_name='Parametros', index=False)

print(f"\nResultados salvos em: {output_path}")


# =====================================================
# 7. GRÁFICO DO HIATO
# =====================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_results.index, df_results['Hiato_suavizado'], label='Hiato (estimado)', linewidth=2, color='tab:blue')
ax.fill_between(df_results.index, df_results['Hiato_lower_95'], df_results['Hiato_upper_95'],
                alpha=0.25, color='tab:blue', label='IC 95%')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title('Hiato do Produto (Suavizado) – Variância de mensuração comum', fontsize=13)
ax.set_ylabel('Hiato (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# =====================================================
# 8. DIAGNÓSTICOS SIMPLES (resíduos padronizados)
# =====================================================
residuals = results.filter_results.standardized_forecasts_error
df_resid = pd.DataFrame(residuals.T, columns=['PIB', 'NUCI', 'CAGED', 'LIVRES'], index=df_model.index)
print("\nResumo dos resíduos (mean e std):")
print(df_resid.describe().loc[['mean', 'std']])

print("\nScript concluído com sucesso! ✅")
