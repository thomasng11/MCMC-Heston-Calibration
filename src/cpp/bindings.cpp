#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "heston.hpp"
#include "mcmc.hpp"

namespace py = pybind11;

PYBIND11_MODULE(heston_mcmc, m) {
    m.doc() = "Heston MCMC C++ implementation with Python bindings";
    
    // Heston namespace
    py::module_ heston_module = m.def_submodule("heston", "Heston model functions");
    
    // Heston pricing functions
    heston_module.def("call_price", &heston::call_price,
                      "Compute Heston call option price",
                      py::arg("S0"), py::arg("K"), py::arg("T"), 
                      py::arg("kappa"), py::arg("theta"), py::arg("sigma"), 
                      py::arg("rho"), py::arg("v0"), 
                      py::arg("alpha") = 1.5, py::arg("r") = 0.01, py::arg("q") = 0.0);
    
    heston_module.def("call_price_vector", &heston::call_price_vector,
                      "Compute Heston call prices as flattened vector",
                      py::arg("S0"), py::arg("strikes"), py::arg("maturities"), 
                      py::arg("kappa"), py::arg("theta"), py::arg("sigma"), 
                      py::arg("rho"), py::arg("v0"), 
                      py::arg("alpha") = 1.5, py::arg("r") = 0.01, py::arg("q") = 0.0);
    
    // Characteristic function functions
    heston_module.def("d", &heston::d,
                      "Compute d function for Heston characteristic function",
                      py::arg("u"), py::arg("kappa"), py::arg("sigma"), py::arg("rho"));
    
    heston_module.def("g2", &heston::g2,
                      "Compute g2 function for Heston characteristic function",
                      py::arg("u"), py::arg("kappa"), py::arg("sigma"), py::arg("rho"));
    
    heston_module.def("phi2", &heston::phi2,
                      "Compute phi2 characteristic function",
                      py::arg("u"), py::arg("t"), py::arg("S0"), py::arg("v0"),
                      py::arg("kappa"), py::arg("theta"), py::arg("sigma"), 
                      py::arg("rho"), py::arg("r"), py::arg("q") = 0.0);
    
    heston_module.def("rho2", &heston::rho2,
                      "Compute rho2 function for option pricing",
                      py::arg("v"), py::arg("T"), py::arg("S0"), py::arg("v0"),
                      py::arg("kappa"), py::arg("theta"), py::arg("sigma"), 
                      py::arg("rho"), py::arg("r"), 
                      py::arg("alpha") = 1.5, py::arg("q") = 0.0);
    
    // MCMC namespace
    py::module_ mcmc_module = m.def_submodule("mcmc", "MCMC engine functions");
    
    // MCMC class
    py::class_<mcmc::MCMC>(mcmc_module, "MCMC")
        .def(py::init<const std::vector<std::pair<double, double>>&, const std::vector<double>&, double,
                      const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      double, double, double>(),
             py::arg("param_bounds"), py::arg("param_std"), py::arg("obs_std"),
             py::arg("market_prices"), py::arg("strikes"), py::arg("maturities"),
             py::arg("S0"), py::arg("r"), py::arg("q"))
        
        .def("fit", &mcmc::MCMC::fit,
             "Run MCMC fitting for specified number of iterations",
             py::arg("initial_params"), py::arg("n_iter"))
        
        .def("get_past_params", &mcmc::MCMC::get_past_params,
             "Get parameter history")
        
        .def("get_past_accept_prob", &mcmc::MCMC::get_past_accept_prob,
             "Get acceptance probability history");
    
    // Convenience functions for Python
    m.def("call_price", [](double S0, double K, double T, 
                           double kappa, double theta, double sigma, double rho, double v0,
                           double alpha = 1.5, double r = 0.01, double q = 0.0) {
        return heston::call_price(S0, K, T, kappa, theta, sigma, rho, v0, alpha, r, q);
    }, "Convenience function for Heston call pricing with individual parameters",
    py::arg("S0"), py::arg("K"), py::arg("T"),
    py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"), py::arg("v0"),
    py::arg("alpha") = 1.5, py::arg("r") = 0.01, py::arg("q") = 0.0);
} 