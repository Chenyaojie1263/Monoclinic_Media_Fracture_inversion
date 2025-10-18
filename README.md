**MATLAB Project for Monoclinic_Media_Fracture_inversion**

This MATLAB project implements a seismic anisotropic parameter inversion system based on the Bayesian Markov Chain Monte Carlo (MCMC) method. The system inverts seismic observation data to obtain anisotropic elastic parameters of subsurface media, including P-wave velocity, S-wave velocity, density, and anisotropic parameters.

**Main Features**

- **Forward Modeling**: Construction of forward operators based on anisotropic theory
- **Adaptive Sampling**: Automatic step size adjustment to improve sampling efficiency
- **Bayesian Inversion**: Parameter estimation using the MCMC method

**Required Data Variables**
- `S`: Seismic observation data matrix
- `L0_iso`: Initial model for isotropic parameters  
- `L0_ani`: Initial model for anisotropic parameters
- `wavelet`: Seismic wavelet information

**Important Notes**
- **Data Normalization**: Input data should undergo appropriate preprocessing and normalization
- **Parameter Tuning**: Adjust coefficients and noise parameters according to actual data characteristics
- **Convergence Diagnosis**: It is recommended to check the convergence of MCMC chains

