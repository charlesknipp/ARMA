using FFTW

include("linear-model.jl")
include("filters.jl")

#=
    NOTE: I could instead define ARIMA(p,r,q) which adds an integrated component,
    but that seems pretty overkill for now. I'm sure this will change in the near
    future when I work with more complicated models

    I could also construct an ARIMA by combining an I(r) LatendDynamic with an
    ARMA(p,q) to produce a different StateSpaceModel
=#

struct ARMA{T<:Real, p, q} <: LatentDynamics
    """
        ARMA(p,q) models are defined for autoregressive polynomials φ, of order
        p, and moving-average polynomials θ, of order q. The model is defined
        by the following equation given the data y:
        φ(L)y[t] = θ(L)ε[t],  ε[t] ∼ N(0, σ²)
    """
    φ::Vector{T}
    θ::Vector{T}
    σ::T

    function ARMA(φ::Vector{T}, θ::Vector{T}, σ::T) where {T<:Real}
        return new{T, length(φ), length(θ)}(φ, θ, σ)
    end
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
    return StateSpaceModel(dyn, obs)
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

## SPECTRAL ANALYSIS ##########################################################

function frequency_response(proc::ARMA{<:Real, p, q}, z::T) where {T<:Number, p, q}
    ma_poly = I + proc.θ'*[z^(-k) for k in 1:q]
    ar_poly = I - proc.φ'*[z^(-k) for k in 1:p]

    return abs2.(ma_poly / ar_poly)
end

frequency_response(proc::ARMA, freqs::AbstractVector{<:Real}) = begin
    frequency_response.(Ref(proc), exp.(im.*freqs))
end

# this feels kinda wrong, but canonically it should work
function spectral_density(proc::ARMA; res=1200)
    ωs = range(0, stop=2π, length=res)
    hz = frequency_response(proc, ωs)
    spect = @. (proc.σ^2) * hz
    return ws, spect
end

function acov(proc::ARMA{MT}, T::Integer; order::Integer=16) where {MT<:Real}
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

function SSMProblems.logdensity(
        proc::ARMA{T},
        data::AbstractVector{T};
        kwargs...
    ) where {T<:Real}
    # get the approximate conditional covariance
    Γ = acov(proc, length(data); kwargs...)

    # calculate and return the log likelihood
    logℓ = data'inv(Γ)*data
    logℓ += length(data)*log(2π) + logdet(Γ)
    return -0.5*logℓ
end