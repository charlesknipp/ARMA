# ARMA.jl

This submodule is meant as a proof of concept for possible extensions to the SSMProblems interface.

Specifically, `ARMA(p,q)` models describe the latent dynamics of a given series, and as such serves as an extension of the `LatentDynamics` struct from SSMProblems.jl. The construction of this model is as follows:

```julia
arma_model = ARMA(
    [1.3418, -0.7059],  # AR coefficients
    [-1.0543, 0.5188],  # MA coefficients
    0.9694              # standard error
)
```

Since this is built off of SSMProblems, let's evaluate the log-likelihood of the above model. We can do this one of two ways:

```julia
# 1. in the sequence space
logℓ = logdensity(arma_model, data)

# 2. in the state space with the Kalman filter
logℓ = begin
    ssm_model = StateSpaceModel(arma_model)
    sample(rng, ssm_model, data, KF())
end
```

For a working example of it's use, check `script.jl` for a replication of (Morely-Nelson-Zivot, 2004)