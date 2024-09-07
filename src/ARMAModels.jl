module ARMAModels

using FFTW
using ToeplitzMatrices
using Polynomials

using Random
using UnPack
using SSMProblems

using SSMProblems

using LinearAlgebra
using PDMats
using MatrixEquations

using GaussianDistributions
using Distributions

import AbstractMCMC: AbstractSampler
import SSMProblems: StateSpaceModel, logdensity
import PDMats: PDMat

export ARMA, AR, MA
export roots, spectral_density, autocovariance, empirical_autocovariance

include("linear-model.jl")
include("filters.jl")

#=
    NOTE: I could instead define ARIMA(p,r,q) which adds an integrated component,
    but that seems pretty overkill for now. I'm sure this will change in the near
    future when I work with more complicated models

    I could also construct an ARIMA by combining an I(r) LatendDynamic with an
    ARMA(p,q) to produce a different StateSpaceModel
=#

struct ARMA{T, p, q} <: LatentDynamics{T}
    """
        ARMA(p,q) models are defined for autoregressive polynomials φ, of order
        p, and moving-average polynomials θ, of order q. The model is defined
        by the following equation given the data y:
        (1-φ(L))*y[t] = (1+θ(L))*ε[t],  ε[t] ∼ N(0, σ²)
    """
    φ::Vector{T}
    θ::Vector{T}
    σ::T

    function ARMA(φ::Vector{T}, θ::Vector{T}, σ::T) where {T<:Real}
        # check for invertibility
        ma_poly = Polynomial([1; θ])
        ma_roots = inv.(roots(ma_poly))
        @assert all(abs2.(ma_roots) .< 1)

        # check for stationarity
        ar_poly = Polynomial([1; -φ])
        ar_roots = roots(ar_poly)
        @assert all(abs2.(ar_roots) .> 1)

        return new{T, length(φ), length(θ)}(φ, θ, σ)
    end
end

function polynomials(proc::ARMA{T, p, q}) where {T<:Real, p, q}
    return (
        φ = Polynomial([1; -proc.φ]),
        θ = Polynomial([1; proc.θ])
    )
end

function Polynomials.roots(proc::ARMA{T, p, q}) where {T<:Real, p, q}
    polys = polynomials(proc)
    return (
        φ = roots(polys.φ),
        θ = inv.(roots(polys.θ))
    )
end

# defined according to Hamilton's state space form (subject to change)
function SSMProblems.StateSpaceModel(proc::ARMA{T, p, q}) where {T<:Real, p, q}
    d = max(p, q+1)

    φ = cat(proc.φ, zeros(T, d-p), dims=1)
    θ = cat(proc.θ, zeros(T, d-1-q), dims=1)

    dyn = LinearGaussianLatentDynamics(
        vcat(φ[1:d]', diagm(d-1, d, ones(d-1))),
        diagm(d, d, T[proc.σ^2]),
        zeros(T, d)
    )

    obs = LinearGaussianObservationProcess(
        T[1 θ[1:d-1]'...],
        T[0.0;;]
    )

    # returns a linear Gaussian state space model
    return SSMProblems.StateSpaceModel(dyn, obs)
end

function Base.show(io::IO, proc::ARMA{T, p, q}) where {T<:Real, p, q}
    φ = round.(proc.φ, digits=3)
    θ = round.(proc.θ, digits=3)
    σ = round.(proc.σ, digits=3)
    print(io, "ARMA($p,$q){$T}:\n  φ: $φ\n  θ: $θ\n  σ: $σ")
end

# defined alias for autoregressive models AR(p) = ARMA(p,0)
const AR{T, p} = ARMA{T, p, 0}

function AR(φ::Vector{T}, σ) where {T<:Real}
    return ARMA(φ, T[], σ)
end

function Base.show(io::IO, proc::AR{T, p}) where {T<:Real, p}
    φ = round.(proc.φ, digits=3)
    σ = round.(proc.σ, digits=3)
    print(io, "AR($p){$T}:\n  φ: $φ\n  σ: $σ")
end

# define alias for moving-average models such that MA(q) = ARMA(0,q)
const MA{T, q} = ARMA{T, 0, q}

function MA(θ::Vector{T}, σ::T) where {T<:Real}
    return ARMA(T[], θ, σ)
end

function Base.show(io::IO, proc::MA{T, q}) where {T<:Real, q}
    θ = round.(proc.θ, digits=3)
    σ = round.(proc.σ, digits=3)
    print(io, "MA($q){$T}:\n  θ: $θ\n  σ: $σ")
end

function SSMProblems.logdensity(
        proc::ARMA{T},
        data::AbstractVector{T}
    ) where {T<:Real}
    # use the analytical autocovariance
    Γ = autocovariance(proc, length(data))

    # calculate and return the log likelihood
    logℓ = data'inv(Γ)*data
    logℓ += length(data)*log(2π) + logdet(Γ)
    return -0.5*logℓ
end

## SPECTRAL ANALYSIS ##########################################################

function frequency_response(
        proc::ARMA{<:Real, p, q}, z::T
    ) where {T<:Number, p, q}
    ma_poly = I + proc.θ'*[z^(k) for k in 1:q]
    ar_poly = I - proc.φ'*[z^(k) for k in 1:p]

    return abs2.(ma_poly / ar_poly)
end

frequency_response(proc::ARMA, freqs::AbstractVector{<:Real}) = begin
    frequency_response.(Ref(proc), exp.(im.*freqs))
end

function spectral_density(proc::ARMA; res::Int=257)
    ωs = range(0, stop=2π, length=res)
    hz = frequency_response(proc, ωs)
    spect = @. (proc.σ^2) * hz
    return ωs, spect
end

function empirical_autocovariance(
        proc::ARMA{MT}, T::Integer; order::Integer=16
    ) where {MT<:Real}
    Γ = zeros(MT, T, T)
    
    γ = begin
        _, spect = spectral_density(proc; res = max(1200, 2*order))
        acov = real(ifft(spect))
        acov[1:order]
    end

    for k in 1:T
        if k ≤ T-order
            Γ[k,k:(k+order-1)] = γ
        else
            Γ[k, k:end] = γ[1:(T-k+1)]
        end
    end

    return Matrix(Hermitian(UpperTriangular(Γ), :U))
end

## MATRIX REPRESENTATION FOR MLE ##############################################

# from Pollock chapter 17
function autocovariance(proc::ARMA{T, p, q}, n::Integer) where {T<:Real, p, q}
    polys = polynomials(proc)
    r = max(p, q)+1

    α = zeros(T, r)
    α[1:p+1] = coeffs(polys.φ)

    μ = zeros(T, r)
    μ[1:q+1] = coeffs(polys.θ)

    ψ = TriangularToeplitz(α, :L) \ ((proc.σ^2).*μ)

    # left hand side of eq 96, lower Toeplitz + other shit
    A = zeros(T, r, r)
    for i in 1:r
        for j in 1:p+1
            k = i-j+1
            if k<1; k = 2-k; end
            A[i,k] += α[j]
        end
    end

    # solve for the initial r autocovariances
    γ = zeros(T, n)
    γ[1:r] = begin
        H = Hankel([μ; zeros(T, r-1)], (r, r))
        A \ (H*ψ)
    end

    # find the succeeding autocovariances
    for i in (r+1):n, j in 1:p
        γ[i] += proc.φ[j]*γ[i-j]
    end

    return γ
end

end