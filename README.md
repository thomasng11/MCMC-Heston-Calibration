# MCMC Heston Calibration

Python/C++ code to calibrate the Heston stochastic volatility model to option prices using Markov Chain Monte-Carlo. 

Gradient-based samplers (e.g. HMC, NUTS) are not used because the likelihood is too complex and we don't have its gradients. Calibration is less accurate and efficient as a result.

See `Heston Calibration.ipynb` for usage example

## References

[1] Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.

[2] Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2006). The little Heston trap.