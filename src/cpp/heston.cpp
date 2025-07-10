#include "heston.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <boost/math/quadrature/gauss_kronrod.hpp>

namespace heston {

// Constants
const double PI = 3.14159265358979323846;

Complex d(Complex u, double kappa, double sigma, double rho) {
    Complex i(0.0, 1.0);
    return std::sqrt(std::pow(rho * sigma * i * u - kappa, 2) + sigma * sigma * (i * u + u * u));
}

Complex g2(Complex u, double kappa, double sigma, double rho) {
    Complex i(0.0, 1.0);
    Complex d_val = d(u, kappa, sigma, rho);
    Complex numerator = kappa - rho * sigma * i * u - d_val;
    Complex denominator = kappa - rho * sigma * i * u + d_val;
    return numerator / denominator;
}

Complex phi2(Complex u, double t, double S0, 
            double kappa, double theta, double sigma, double rho, double v0,
            double r, double q) {
    Complex i(0.0, 1.0);
    Complex d_val = d(u, kappa, sigma, rho);
    Complex g2_val = g2(u, kappa, sigma, rho);
    
    Complex term1 = std::exp(i * u * (std::log(S0) + (r - q) * t));
    
    Complex temp = theta * kappa / (sigma * sigma) * 
                   ((kappa - rho * sigma * i * u - d_val) * t - 
                    2.0 * std::log((1.0 - g2_val * std::exp(-d_val * t)) / (1.0 - g2_val)));
    Complex term2 = std::exp(Complex(std::max(-600.0, std::min(600.0, temp.real())), temp.imag()));
    
    Complex term3 = std::exp(v0 * v0 / (sigma * sigma) * 
                            (kappa - rho * sigma * i * u - d_val) * 
                            (1.0 - std::exp(-d_val * t)) / 
                            (1.0 - g2_val * std::exp(-d_val * t)));
    
    return term1 * term2 * term3;
}

Complex rho2(double v, double T, double S0, 
            double kappa, double theta, double sigma, double rho, double v0,
            double r, double q, double alpha) {
    Complex i(0.0, 1.0);
    Complex numerator = std::exp(-r * T) * phi2(v - (alpha + 1.0) * i, T, S0, kappa, theta, sigma, rho, v0, r, q);
    Complex denominator = alpha * alpha + alpha - v * v + i * (2.0 * alpha + 1.0) * v;
    return numerator / denominator;
}

double call_price(double S0, double K, double T, 
                 double kappa, double theta, double sigma, double rho, double v0,
                 double r, double q, double alpha) {

    // Define the integrand function
    auto integrand = [&](double v) -> double {
        Complex integrand_val = std::exp(-Complex(0.0, 1.0) * v * std::log(K)) * 
                               rho2(v, T, S0, kappa, theta, sigma, rho, v0, r, q, alpha);
        return integrand_val.real();
    };
    
    // Use Boost's Gauss-Kronrod quadrature
    boost::math::quadrature::gauss_kronrod<double, 15> integrator;
    double integral = integrator.integrate(integrand, 0, 200);
    
    return (1.0 / PI) * std::exp(-alpha * std::log(K)) * integral;
}

Eigen::VectorXd call_price_vector(double S0, const Eigen::VectorXd& strikes, const Eigen::VectorXd& maturities,
                                 double kappa, double theta, double sigma, double rho, double v0,
                                 double r, double q, double alpha) {
    int n_strikes = static_cast<int>(strikes.size());
    int n_maturities = static_cast<int>(maturities.size());
    int total_size = n_strikes * n_maturities;
    
    Eigen::VectorXd prices(total_size);
    
    int idx = 0;
    for (int i = 0; i < n_strikes; ++i) {
        for (int j = 0; j < n_maturities; ++j) {
            prices(idx++) = call_price(S0, strikes(i), maturities(j), kappa, theta, sigma, rho, v0, r, q, alpha);
        }
    }
    
    return prices;
}

} // namespace heston 