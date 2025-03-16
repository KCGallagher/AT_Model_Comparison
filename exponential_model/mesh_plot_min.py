import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append("../utils")
from myUtils import convert_ode_parameters
from odeModels import LotkaVolterraModel, ExponentialModel
from plotUtils import PlotSimulation

# Defaults for plotting
plt.style.use(["classic", "../utils/paper.mplstyle"])
plt.rcParams['font.size'] = '18'

colors = {'Sensitive': '#4c9150', 'Resistant': '#a80303', 'CT':'#c28d32' , 'AT':'#143159' ,'DRL_Monthly':'#66419e'}

expParamDic = {'rS': 0.00715, 'rR': 0.023, 'Ks': 1, 'Kr': 0.25, 'dDs': 2, 'dDr': 0, 
               'S0': 0.5, 'R0':2.5e-5, 'DMax':1, 'alpha':1, 'gamma':0.0021, 'N0': 0.5+2.5e-5}

def calc_ttp(params, n_crit, interval):
    arg = n_crit
    model = ExponentialModel(dt=1); model.SetParams(**params)
    model.Simulate_AT(atThreshold=arg, atMethod='Threshold',
                    intervalLength=interval, t_end=4000, refSize = params['N0'])
    model.resultsDf = model.resultsDf[model.resultsDf.TumourSize < 1.2 * params['N0']]
    model.resultsDf = model.resultsDf[model.resultsDf.R < 0.8 * params['Kr']]
    return model.resultsDf.Time.max()

def calc_critical_threshold_general(tau, Ks, N0, rS, alpha, **_args):
    inner_bracket = (Ks / (1.2 * N0))**alpha - 1
    outer_bracket = (inner_bracket * np.exp(alpha * rS * tau)) + 1
    return Ks * outer_bracket**(-1/alpha)

expParamDic_low = expParamDic.copy(); expParamDic_low['alpha'] = 0.5
expParamDic_high = expParamDic.copy(); expParamDic_high['alpha'] = 2

tau_values = np.linspace(0, 160, 100)
n_stars_high = [calc_critical_threshold_general(tau=t,**expParamDic_high)/expParamDic['N0'] for t in tau_values]
n_stars_low = [calc_critical_threshold_general(tau=t,**expParamDic_low)/expParamDic['N0'] for t in tau_values]


def calc_offset_ttp(params, n_crit, interval, offset):
    arg = n_crit
    model = ExponentialModel(dt=1); model.SetParams(**params)

    # Simulate offset
    initial_drug = n_crit < 1
    model.Simulate([[0, offset, initial_drug]])

    if model.resultsDf.TumourSize.iloc[-1] > 1.2 * params['N0']:
        model.resultsDf = model.resultsDf[model.resultsDf.TumourSize < 1.2 * params['N0']]
        return model.resultsDf.Time.max()
    else:
        new_params = params.copy()
        new_params['S0'] = model.resultsDf.S.iloc[-1]
        new_params['R0'] = model.resultsDf.R.iloc[-1]

        model2 = ExponentialModel(dt=1); model2.SetParams(**new_params)
        rescaled_prog = 1.2 * params['N0'] / model.resultsDf.TumourSize.iloc[-1]
        model2.Simulate_AT(atThreshold=arg, atMethod='Threshold',
                           intervalLength=interval, t_end=4000, 
                           refSize = params['N0'], tumourSizeWhenProgressed=rescaled_prog)
        model2.resultsDf = model2.resultsDf[model2.resultsDf.TumourSize < 1.2 * params['N0']]
        model2.resultsDf = model2.resultsDf[model2.resultsDf.R < 0.8 * params['Kr']]
        return model2.resultsDf.Time.max() + offset

def calc_ttp_repeats(params, n_crit, interval, output_func=np.mean):  
    N = 20  #CHANGE
    N_reps = min(N, interval)
    ttp_values = np.zeros(int(N_reps))

    if interval < N:
        offset_vals = np.arange(0, interval)
    else:
        offset_vals = np.linspace(0, interval, N)
    
    for i, offset in enumerate(offset_vals):
        ttp_values[i] = calc_offset_ttp(params, n_crit, interval, offset)
    return output_func(ttp_values)


# Once we are happy with the parameters above, lets plot some meshes

dx, dy = 1, 0.01
# dx, dy = 10, 0.1 #CHANGE

crit_n_grid, tau_grid = np.mgrid[slice(0, 1.2 + dy, dy),
                slice(1, 161 + dx, dx)]

calc_ttp_repeats_min = lambda params, n_crit, interval: calc_ttp_repeats(params, n_crit, interval, output_func=np.min)
vectorized_calc_ttp = np.vectorize(calc_ttp_repeats_min)

# Calculate TTP values for low alpha
ttp_values_low = vectorized_calc_ttp(
    params = expParamDic_low, n_crit = crit_n_grid,
    interval = tau_grid
)
ttp_values_low = ttp_values_low[:-1, :-1]

# Calculate TTP values for high alpha
ttp_values_high = vectorized_calc_ttp(
    params = expParamDic_high, n_crit = crit_n_grid,
    interval = tau_grid
)
ttp_values_high = ttp_values_high[:-1, :-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

im1 = ax1.pcolormesh(tau_grid, crit_n_grid, ttp_values_low, 
                   cmap=plt.get_cmap('viridis'), norm=None)
fig.colorbar(im1, ax=ax1)
ax1.plot(tau_values, n_stars_low, linewidth = 8, zorder=5, color='r', label = 'Optimal\nThreshold')
ax1.set_xlim(0, np.max(tau_grid))
ax1.set_xticks(ax1.get_xticks()[::2])
ax1.set_title(r'$\alpha = 0.5$')

im2 = ax2.pcolormesh(tau_grid, crit_n_grid, ttp_values_high, 
                   cmap=plt.get_cmap('viridis'), norm=None)
fig.colorbar(im2, ax=ax2)
ax2.plot(tau_values, n_stars_high, linewidth = 8, zorder=5, color='r', label = 'Optimal\nThreshold')
ax2.set_xlim(0, np.max(tau_grid))
ax2.set_xticks(ax2.get_xticks()[::2])
ax2.set_title(r'$\alpha = 2$')

fig.tight_layout()
plt.savefig("../figures/exponential_model_threshold_comparison_alpha_panel_min9.pdf")
plt.show()