# =====================================================
# MODELO SEMIESTRUTURAL DO BC COM CURVA IS (ATUALIZADO)
# (VARIÂNCIA DE MENSURAÇÃO COMUM A TODAS AS OBSERVÁVEIS)
# Estado: [h_t, s_t^h, h_{t-1}]
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
# 2. DEFINIR MODELO STATE SPACE (com s_t^h autorregressivo)
# =====================================================
class HiatoStateSpace(MLEModel):
    def __init__(self, endog, exog_obs, exog_trans):
        # exog_obs: matriz para intercepto de LIVRES (Z' gamma)
        # exog_trans: exógenos para intercepto de transição (Curva IS)
        self.exog_obs = exog_obs
        self.exog_trans = exog_trans

        # Parâmetros estruturais fixos (calibrados)
        self.b1 = 0.85   # persistência do hiato
        self.b2 = 0.44
        self.b3 = 0.003
        self.b4 = 0.054
        self.b5 = 0.84   # persistência do choque s_t^h

        # parâmetros para equação de LIVRES (Z' gamma)
        self.alpha_livres = 0.054
        self.alpha_livres_lag = 0.24
        self.alpha_ipca_lag = 0.38
        self.alpha_focus = 1 - 0.38 - 0.24
        self.beta_brl = 0.011
        self.beta_ic = 0.023
        self.beta_el_nino = 0.0012
        self.beta_la_nina = 0.0007
        self.intercept_livres = 0.0

        # agora o estado tem 3 componentes: [h_t, s_t^h, h_{t-1}]
        # k_posdef = 2 (dois choques: epsilon^h e epsilon^s)
        super(HiatoStateSpace, self).__init__(endog, k_states=3, k_posdef=2, initialization='diffuse')

    @property
    def param_names(self):
        # estimamos log das variâncias dos erros de mensuração e dos dois choques de estado
        return ['log_var_obs', 'log_var_h', 'log_var_s']

    @property
    def start_params(self):
        return np.array([0.0, 0.0, 0.0])

    def transform_params(self, unconstrained):
        return unconstrained

    def untransform_params(self, constrained):
        return constrained

    def update(self, params, **kwargs):
        params = super(HiatoStateSpace, self).update(params, **kwargs)
        log_var_obs = params[0]
        log_var_h = params[1]
        log_var_s = params[2]

        var_obs = np.exp(log_var_obs)     # variância de mensuração comum
        var_h = np.exp(log_var_h)         # variância do choque em h
        var_s = np.exp(log_var_s)         # variância do choque em s_t^h

        # --------------------------
        # MATRIZES DO MODELO
        # Estado: alpha_t = [h_t, s_t^h, h_{t-1}]'
        # Objetivo: h_t = b1*h_{t-1} + s_t^h + (exógenos) + eps^h
        #          s_t^h = b5*s_{t-1}^h + eps^s
        # companion para h_{t-1} na 3ª linha
        # --------------------------

        # Transition (T) dimension: (k_states x k_states) = 3x3
        # construído para que, ao resultar alpha_t = T * alpha_{t-1} + R * eta_t,
        # tenhamos h_t = b1*h_{t-1} + b5*s_{t-1} + ... + (eta_s + eta_h) equivalendo a b1*h_{t-1} + s_t^h + eta_h
        self['transition'] = np.array([
            [self.b1, self.b5, 0.0],   # h_t depende de h_{t-1} e b5*s_{t-1}
            [0.0,    self.b5, 0.0],    # s_t^h = b5 * s_{t-1}^h + eps_s
            [1.0,    0.0,   0.0]       # h_{t-1} <- h_{t-1} (companheiro)
        ])

        # Selection (R): quais choques contemporâneos entram em cada coordenada do estado
        # eta_t = [eps_h, eps_s]'
        # queremos:
        # - h_t recebe eps_h e também eps_s (porque s_t^h contém eps_s)
        # - s_t^h recebe eps_s
        # - h_{t-1} não recebe choque
        self['selection'] = np.array([
            [1.0, 1.0],   # h_t <- eps_h, eps_s
            [0.0, 1.0],   # s_t^h <- eps_s
            [0.0, 0.0]    # h_{t-1}
        ])

        # State covariance Q (dim k_posdef x k_posdef = 2x2)
        self['state_cov'] = np.diag([var_h, var_s])

        # Design (Z): mapeia estados [h_t, s_t^h, h_{t-1}] para observáveis
        # Observáveis: [PIB_CICLO, NUCI_CICLO, CAGED_CICLO, LIVRES_d11]
        # PIB_CICLO  <- h_t
        # NUCI_CICLO <- h_t
        # CAGED_CICLO<- h_{t-1}
        # LIVRES_d11 <- alpha_livres * h_t + Z_t' gamma + eps
        self['design'] = np.array([
            [1.0, 0.0, 0.0],               # PIB_CICLO
            [1.0, 0.0, 0.0],               # NUCI_CICLO
            [0.0, 0.0, 1.0],               # CAGED_CICLO (h_{t-1})
            [self.alpha_livres, 0.0, 0.0]   # LIVRES_d11 (alpha4 * h_t + Z'gamma)
        ])

        # obs_cov (H): variância comum para todas as observáveis
        self['obs_cov'] = np.diag([var_obs] * self.k_endog)

        # --------------------------
        # OBS_INTERCEPT: intercepto específico para LIVRES (Z' gamma)
        # --------------------------
        self['obs_intercept'] = np.zeros((self.k_endog, self.nobs))
        if self.exog_obs is not None:
            # exog_obs colunas: [LIVRES_d11_lag1, FOCUS, IPCA_d11_lag1, BRL, IC_BR, EL_NINO_lag1_sq, LA_NINA_lag1_sq]
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
            # atribui apenas à linha correspondente a LIVRES (index 3)
            self['obs_intercept'][3, :] = livres_intercept

        # --------------------------
        # STATE_INTERCEPT: interceptos na transição (ex.: Curva IS)
        # Trans_intercept aplica-se apenas na equação de h_t (primeira coordenada)
        # --------------------------
        self['state_intercept'] = np.zeros((self.k_states, self.nobs))
        if self.exog_trans is not None:
            # exog_trans colunas: [HIATO_JUROS, HIATO_MUNDIAL] (se quiser adicionar rp_t, inclua como coluna extra)
            r_gap = self.exog_trans[:, 0]
            h_star = self.exog_trans[:, 1]
            # b2 e b4 já definidos
            self['state_intercept'][0, :] = -self.b2 * (r_gap / 4) + self.b4 * h_star
        else:
            self['state_intercept'] = np.zeros((self.k_states, self.nobs))


# =====================================================
# 3. PREPARAR DADOS
# =====================================================
# gerar variáveis defasadas e drops
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
# 4. ESTIMAR MODELO (estimando variâncias dos choques e de mensuração)
# =====================================================
print("\n" + "="*60)
print("ESTIMANDO MODELO STATE SPACE (variâncias: mensuração e choques) - VERSÃO ATUALIZADA")
print("="*60)

model = HiatoStateSpace(endog, exog_obs, exog_trans)
results = model.fit(method='lbfgs', maxiter=1000, disp=False)

print("\nResumo curto dos parâmetros estimados:")
for name, val in zip(results.param_names, results.params):
    print(f"  {name}: {val:.6f}")


# =====================================================
# 5. EXTRAIR ESTADOS SUAVIZADOS E IC 95%
# =====================================================
# smoothed_state shape: (k_states, nobs) = (3, nobs)
h_t_smooth = results.smoothed_state[0, :]      # h_t
s_t_smooth = results.smoothed_state[1, :]      # s_t^h
h_lag_smooth = results.smoothed_state[2, :]    # h_{t-1}

# erros-padrão (desvio padrão das entradas da diagonal da cov suavizada)
h_t_se = np.sqrt(results.smoothed_state_cov[0, 0, :])
s_t_se = np.sqrt(results.smoothed_state_cov[1, 1, :])
h_lag_se = np.sqrt(results.smoothed_state_cov[2, 2, :])

h_lower = h_t_smooth - 1.96 * h_t_se
h_upper = h_t_smooth + 1.96 * h_t_se

df_results = pd.DataFrame(index=df_model.index)
df_results['Hiato_suavizado'] = h_t_smooth
df_results['Hiato_lower_95'] = h_lower
df_results['Hiato_upper_95'] = h_upper
df_results['s_t_h_suavizado'] = s_t_smooth
df_results['Hiato_lag'] = h_lag_smooth

# =====================================================
# 6. SALVAR RESULTADOS (Excel)
# =====================================================
output_path = 'hiato_comis_corrigido_sth.xlsx'
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
ax.plot(df_results.index, df_results['Hiato_suavizado'], label='Hiato (estimado)', linewidth=2)
ax.fill_between(df_results.index, df_results['Hiato_lower_95'], df_results['Hiato_upper_95'],
                alpha=0.25, label='IC 95%')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title('Hiato do Produto (Suavizado) – Variância de mensuração comum (corrigido, com s_t^h)', fontsize=13)
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
