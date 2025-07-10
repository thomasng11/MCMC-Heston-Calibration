import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.py import heston_mcmc
import multiprocessing
from tqdm import tqdm
import arviz as az

class HestonCalibrator:
    def __init__(self, market_prices_df, param_bounds=None, param_std=None, obs_std=None, S0=100.0, r=0.01, q=0.0):
        """ Set initial parameters, market prices dataframe, and market parameters """
       
        self.market_prices_df = market_prices_df
        self.param_bounds = [(0, 10), (0, 1), (0, 1), (-1, 1), (0, 1)] if param_bounds is None else param_bounds
        self.param_std = [0.01, 0.001, 0.005, 0.005, 0.001] if param_std is None else param_std
        self.obs_std = 0.1 if obs_std is None else obs_std
        self.S0, self.r, self.q = S0, r, q
        
        self.strikes = np.array(market_prices_df.index, dtype=float)
        self.maturities = np.array(market_prices_df.columns, dtype=float)
        self.market_prices = np.array(market_prices_df.values.flatten(), dtype=float)
        
        self.past_params = None
        self.past_accept_prob = None
        self.burn_in = None
        
    def fit(self, initial_params, n_iter, burn_in=0.3, use_multiprocessing=False):
        if use_multiprocessing:
            inputs = [(self.market_prices_df, self.param_bounds, self.param_std, self.obs_std, self.S0, self.r, self.q, params, n_iter) for params in initial_params]
            with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()/2) - 1) as pool:
                results = pool.starmap(self.fit_, tqdm(inputs, total=len(inputs), desc="MCMC Fitting"))
                past_params, past_accept_prob = zip(*results)
        else:
            past_params, past_accept_prob = self.fit_(self.market_prices_df, self.param_bounds, self.param_std, self.obs_std, self.S0, self.r, self.q, initial_params, n_iter)
        
        return HestonCalibrationResult(past_params, past_accept_prob, burn_in, n_iter)

    @staticmethod
    def call_price(S0, K, T, kappa, theta, sigma, rho, v0, r=0.01, q=0.0, alpha=2.0):
        return heston_mcmc.heston.call_price(S0, K, T, kappa, theta, sigma, rho, v0, r, q, alpha)

    @staticmethod
    def fit_(market_prices_df, param_bounds, param_std, obs_std, S0, r, q, initial_params, n_iter):
        strikes = np.array(market_prices_df.index, dtype=float)
        maturities = np.array(market_prices_df.columns, dtype=float)
        market_prices = np.array(market_prices_df.values.flatten(), dtype=float)
        mcmc = heston_mcmc.mcmc.MCMC(param_bounds, param_std, obs_std, market_prices, strikes, maturities, S0, r, q)
        mcmc.fit(initial_params, n_iter)
        return mcmc.get_past_params(), mcmc.get_past_accept_prob()

class HestonCalibrationResult:
    def __init__(self, past_params, past_accept_prob, burn_in, n_iter):
        self.past_params = past_params
        self.past_accept_prob = past_accept_prob
        self.burn_in = burn_in
        self.n_iter = n_iter
        self.param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0']

    def summary(self, include_burn_in=False):
        param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0']
        past_params_array = np.array(self.past_params)
        
        if past_params_array.ndim == 3:
            burn_idx = 0 if include_burn_in else int(past_params_array.shape[1] * self.burn_in)
            traces = past_params_array[:, burn_idx:, :].reshape(-1, 5)
            accept_probs = np.array(self.past_accept_prob)[:, burn_idx:].flatten()
        else:
            burn_idx = 0 if include_burn_in else int(len(self.past_accept_prob) * self.burn_in)
            traces = past_params_array[burn_idx:]
            accept_probs = np.array(self.past_accept_prob)[burn_idx:]
        
        avg_accept = np.mean(accept_probs)
        param_means = np.mean(traces, axis=0)
        param_stds = np.std(traces, axis=0)
        
        print(f"Average Acceptance Rate: {avg_accept:.3f}")
        for i, name in enumerate(param_names):
            print(f"{name}: mean = {param_means[i]:.4f}, std = {param_stds[i]:.4f}")
        
        if past_params_array.ndim == 3:
            arr = past_params_array[:, burn_idx:, :]
            data_dict = {name: arr[:, :, i] for i, name in enumerate(param_names)}
            dataset = az.from_dict(data_dict)
            rhat = az.rhat(dataset)
            print(f"\nR-hat (Gelman-Rubin):")
            for i, name in enumerate(param_names):
                print(f"{name}: {rhat[name].values:.4f}")

    def trace_plot(self, true_params=None, include_burn_in=False):
        param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0']
        arr = np.array(self.past_params)
        
        if arr.ndim == 3:
            burn_idx = 0 if include_burn_in else int(arr.shape[1] * self.burn_in)
            n_runs = arr.shape[0]
            
            for fig_idx in range((n_runs + 8) // 9):
                fig, ax = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
                ax = ax.flatten()
                
                start_idx = fig_idx * 9
                end_idx = min(start_idx + 9, n_runs)
                
                for i in range(start_idx, end_idx):
                    local_idx = i - start_idx
                    traces = arr[i][burn_idx:]
                    
                    for j, name in enumerate(param_names):
                        ax[local_idx].plot(traces[:, j], label=name, alpha=0.8)
                        if true_params is not None:
                            ax[local_idx].axhline(true_params[j], color=f'C{j}', linestyle='--', alpha=0.7, label=f'{name} true')
                    
                    ax[local_idx].set_title(f'Run {i+1}')
                    ax[local_idx].grid(True, alpha=0.3)
                    
                    if local_idx == 2:
                        ax[local_idx].legend(loc='upper right')
                
                for j in range(end_idx - start_idx, 9):
                    fig.delaxes(ax[j])
                
                plt.tight_layout()
                plt.show()
        else:
            burn_idx = 0 if include_burn_in else int(len(self.past_accept_prob) * self.burn_in)
            traces = arr[burn_idx:]
            
            for i, name in enumerate(param_names):
                plt.plot(traces[:, i], label=name, alpha=0.8)
                if true_params is not None:
                    plt.axhline(true_params[i], color=f'C{i}', linestyle='--', alpha=0.7, label=f'{name} true')
            
            plt.xlabel('Iteration')
            plt.ylabel('Parameter Value')
            plt.legend(loc='upper right')
            plt.title('MCMC Parameter Traces')
            plt.grid(True, alpha=0.3)
            plt.show()

    def density_plot(self, true_params=None, include_burn_in=False):
        param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0']
        arr = np.array(self.past_params)
        
        if arr.ndim == 3:
            burn_idx = 0 if include_burn_in else int(arr.shape[1] * self.burn_in)
            traces = arr[:, burn_idx:, :].reshape(-1, 5)
        else:
            burn_idx = 0 if include_burn_in else int(len(self.past_accept_prob) * self.burn_in)
            traces = arr[burn_idx:]
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, name in enumerate(param_names):
            axes[i].hist(traces[:, i], bins=60, density=True, alpha=0.7)
            if true_params is not None:
                axes[i].axvline(true_params[i], color='red', linestyle='--', alpha=0.8, label='True')    
            axes[i].set_title(name)
            axes[i].set_xlabel('Value')

        axes[0].set_ylabel('Density')
        if true_params is not None:
            axes[4].legend(loc='upper right')

        plt.tight_layout()
        plt.show()
