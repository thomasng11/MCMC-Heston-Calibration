#ifndef HESTON_HPP
#define HESTON_HPP

#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/quadrature/gauss_kronrod.hpp>

namespace heston {

// Complex number type for calculations
using Complex = std::complex<double>;

// Main pricing function
double call_price(double S0, double K, double T, 
                  double kappa, double theta, double sigma, double rho, double v0, 
                  double r=0.01, double q=0.0, double alpha=2.0);

// Vectorized pricing function
Eigen::VectorXd call_price_vector(double S0, const Eigen::VectorXd& strikes, const Eigen::VectorXd& maturities,
                                 double kappa, double theta, double sigma, double rho, double v0, 
                                 double r=0.01, double q=0.0, double alpha=2.0);

// Characteristic function calculations
Complex phi2(Complex u, double t, double S0, 
            double kappa, double theta, double sigma, double rho, double v0,
            double r, double q);

Complex rho2(double v, double T, double S0, 
            double kappa, double theta, double sigma, double rho, double v0,
            double r, double q, double alpha);

// Helper functions for characteristic function
Complex d(Complex u, double kappa, double sigma, double rho);
Complex g2(Complex u, double kappa, double sigma, double rho);

} // namespace heston

#endif // HESTON_HPP 