#ifndef MCMC_HPP
#define MCMC_HPP

#include "heston.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>

namespace mcmc {

class MCMC {
private:
    // Data
    Eigen::VectorXd market_prices_;
    Eigen::VectorXd strikes_;
    Eigen::VectorXd maturities_;
    
    // Proposal stddev
    std::vector<double> param_std_;
    // Parameter bounds
    std::vector<std::pair<double, double>> param_bounds_;
    // Market parameters
    double S0_, r_, q_;
    
    // MCMC parameters
    double obs_std_;
    
    // History
    std::vector<double> past_accept_prob_;
    std::vector<std::vector<double>> past_params_;
    
    // Random number generation
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

public:
    MCMC(const std::vector<std::pair<double, double>>& param_bounds, const std::vector<double>& param_std, double obs_std,
         const Eigen::VectorXd& market_prices, const Eigen::VectorXd& strikes, const Eigen::VectorXd& maturities, 
         double S0, double r, double q);
    void fit(const std::vector<double>& initial_params, int n_iter);
    std::vector<std::vector<double>> get_past_params() const;
    std::vector<double> get_past_accept_prob() const;

private:
    double likelihood(const std::vector<double>& params);
    double prior(const std::vector<double>& params);
    std::vector<double> propose(const std::vector<double>& current_params);
    void accept_decision(std::vector<double>& current_params, const std::vector<double>& new_params);
};

} // namespace mcmc

#endif // MCMC_HPP 