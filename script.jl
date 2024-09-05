include("arma.jl")
using CSV, DataFrames

## MORELY-NELSON-ZIVOT REPLICATION ############################################

# the Unobserved Components model defined in the appendix
function morely_nelson_zivot(φ::Vector{T}, ση::T, σε::T, ρ::T, μ::T) where {T<:Real}
    dyn = LinearGaussianLatentDynamics(
        T[1 0 0; 0 φ...; 0 1 0],
        T[ση^2 ρ 0; ρ σε^2 0; 0 0 0],
        T[μ, 0, 0]
    )

    obs = LinearGaussianObservationProcess(
        T[1 1 0],
        T[0;;]
    )

    return StateSpaceModel(dyn, obs)
end

# read the data
rng = MersenneTwister(1234)
mnz_data = begin
    raw_data = CSV.read("data/mnz_data.csv", DataFrame)
    100*raw_data.gdp
end

# since we're working with an I(1) series we must convert it
gdp_diff = diff(mnz_data)
demeaned_data = gdp_diff .- 0.8156

# create the ARMA model
mnz_arma = ARMA(
    [1.3418, -0.7059],  # AR coefficients
    [-1.0543, 0.5188],  # MA coefficients
    0.9694              # standard error
)

# convert it to a state space model
arma_model = StateSpaceModel(mnz_arma)

# use the UC-0 for comparison
mnz_model = morely_nelson_zivot(
    [1.5303, -0.6098], 0.6893, 0.6199, 0.0, 0.8119
)

# estimate with sequence space and Kalman filter
_, ssm_loglike = sample(rng, arma_model, demeaned_data, KF())
arma_loglike   = logdensity(mnz_arma, demeaned_data)
_, uc0_loglike = sample(rng, mnz_model, mnz_data, KF())