import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
import matplotlib.pyplot as plt

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


class HiatoStateSpace(MLEModel):
    def __init__(self, endog, exog_obs, exog_trans):
        self.exog_obs = exog_obs
        self.exog_trans = exog_trans

        self.b1 = 0.85   # persistência do hiato
        self.b2 = 0.44
        self.b3 = 0.003
        self.b4 = 0.054
        self.b5 = 0.84   # persistência do choque s_t^h

        self.alpha_livres = 0.054
        self.alpha_livres_lag = 0.24
        self.alpha_ipca_lag = 0.38
        self.alpha_focus = 1 - 0.38 - 0.24
        self.beta_brl = 0.011
        self.beta_ic = 0.023
        self.beta_el_nino = 0.0012
        self.beta_la_nina = 0.0007
        self.intercept_livres = 0.0

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

        var_obs = np.exp(log_var_obs)     
        var_h = np.exp(log_var_h)  
        var_s = np.exp(log_var_s)         

        # matrizes do FK
        self['transition'] = np.array([
            [self.b1, self.b5, 0.0], 
            [0.0,    self.b5, 0.0],  
            [1.0,    0.0,   0.0]     ## <--- companion
        ])

        self['selection'] = np.array([
            [1.0, 1.0],  
            [0.0, 1.0],   
            [0.0, 0.0]    
        ])

        self['state_cov'] = np.diag([var_h, var_s])

        self['design'] = np.array([
            [1.0, 0.0, 0.0],               
            [1.0, 0.0, 0.0],              
            [0.0, 0.0, 1.0],              
            [self.alpha_livres, 0.0, 0.0]   
        ])


        self['obs_cov'] = np.diag([var_obs] * self.k_endog)

        self['obs_intercept'] = np.zeros((self.k_endog, self.nobs))
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
            self['obs_intercept'][3, :] = livres_intercept

        self['state_intercept'] = np.zeros((self.k_states, self.nobs))
        if self.exog_trans is not None:
            r_gap = self.exog_trans[:, 0]
            h_star = self.exog_trans[:, 1]
            self['state_intercept'][0, :] = -self.b2 * (r_gap / 4) + self.b4 * h_star
        else:
            self['state_intercept'] = np.zeros((self.k_states, self.nobs))


## rodando o modelo de fato
df['LIVRES_d11_lag1'] = df['LIVRES_d11'].shift(1)
df['IPCA_d11_lag1'] = df['IPCA_d11'].shift(1)
df['EL_NINO_lag1_sq'] = df['EL_NINO'].shift(1) ** 2
df['LA_NINA_lag1_sq'] = df['LA_NINA'].shift(1) ** 2
df_model = df.dropna()

endog = df_model[['PIB_CICLO', 'NUCI_CICLO', 'CAGED_CICLO', 'LIVRES_d11']].values
exog_obs = df_model[['LIVRES_d11_lag1', 'FOCUS', 'IPCA_d11_lag1',
                     'BRL', 'IC_BR', 'EL_NINO_lag1_sq', 'LA_NINA_lag1_sq']].values
exog_trans = df_model[['HIATO_JUROS', 'HIATO_MUNDIAL']].values



print("Estimando FK")

model = HiatoStateSpace(endog, exog_obs, exog_trans)
results = model.fit(method='lbfgs', maxiter=1000, disp=False)

print("\nResumo curto dos parâmetros estimados:")
for name, val in zip(results.param_names, results.params):
    print(f"  {name}: {val:.6f}")


# smoothining
h_t_smooth = results.smoothed_state[0, :]    
s_t_smooth = results.smoothed_state[1, :]      
h_lag_smooth = results.smoothed_state[2, :]  

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


output_path = 'hiato_mspp_bacen.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Hiato_Suavizado')
    pd.DataFrame({
        'Parametro': results.param_names,
        'Valor': results.params
    }).to_excel(writer, sheet_name='Parametros', index=False)

print("Resultados salvos!")

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

print("Rotina concluida :)")

