#include "mcmc.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace mcmc {

// Constants
const double PI = 3.14159265358979323846;

// MCMC implementation
MCMC::MCMC(const std::vector<std::pair<double, double>>& param_bounds, const std::vector<double>& param_std, double obs_std,
           const Eigen::VectorXd& market_prices, const Eigen::VectorXd& strikes, const Eigen::VectorXd& maturities,
           double S0, double r, double q)
    : param_bounds_(param_bounds), param_std_(param_std), obs_std_(obs_std)
    , market_prices_(market_prices), strikes_(strikes), maturities_(maturities)
    , S0_(S0), r_(r), q_(q)
    , normal_dist_(0.0, 1.0), uniform_dist_(0.0, 1.0) {
    rng_.seed(std::random_device{}());
}

double MCMC::likelihood(const std::vector<double>& params) {
    // Extract parameters
    double kappa = params[0], theta = params[1], sigma = params[2], rho = params[3], v0 = params[4];
    
    // Compute model prices using vectorized function
    Eigen::VectorXd model_prices = heston::call_price_vector(S0_, strikes_, maturities_, 
                                                            kappa, theta, sigma, rho, v0, r_, q_);
    
    // Compute log-likelihood using for loop
    double loglike = 0.0;
    for (int i = 0; i < model_prices.size(); ++i) {
        double diff = model_prices(i) - market_prices_(i);
        loglike += -0.5 * std::log(2 * PI * obs_std_ * obs_std_) - 0.5 * (diff / obs_std_) * (diff / obs_std_);
    }
    
    return loglike;
}

double MCMC::prior(const std::vector<double>& params) {
    // Uniform prior with bounds
    if (!param_bounds_.empty()) {
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i] < param_bounds_[i].first || params[i] > param_bounds_[i].second) {
                return -INFINITY;
            }
        }
    }
    return 0.0;
}

std::vector<double> MCMC::propose(const std::vector<double>& current_params) {
    std::vector<double> new_params = current_params;
    for (size_t i = 0; i < new_params.size(); ++i) {
        new_params[i] += param_std_[i] * normal_dist_(rng_);
    }
    return new_params;
}

void MCMC::accept_decision(std::vector<double>& current_params, const std::vector<double>& new_params) {
    double new_posterior = likelihood(new_params) + prior(new_params);
    double current_posterior = likelihood(current_params) + prior(current_params);
    
    double log_ratio = new_posterior - current_posterior;
    double accept_prob = std::min(1.0, std::exp(std::max(-700.0, std::min(700.0, log_ratio))));
    
    if (uniform_dist_(rng_) < accept_prob) {
        current_params = new_params;
    }
    
    past_params_.push_back(current_params);
    past_accept_prob_.push_back(accept_prob);
}

void MCMC::fit(const std::vector<double>& initial_params, int n_iter) {
    std::vector<double> current_params = initial_params;
    
    for (int i = 0; i < n_iter; ++i) {
        std::vector<double> new_params = propose(current_params);
        accept_decision(current_params, new_params);
    }
}

std::vector<std::vector<double>> MCMC::get_past_params() const {
    return past_params_;
}

std::vector<double> MCMC::get_past_accept_prob() const {
    return past_accept_prob_;
}

} // namespace mcmc 