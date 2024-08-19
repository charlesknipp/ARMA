using SSMProblems

using LinearAlgebra
using PDMats
using MatrixEquations

using GaussianDistributions
using Distributions

struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics
    """
        Latent dynamics for a linear Gaussian state space model.
        The model is defined by the following equation:
        x[t] = Ax[t-1] + μ + ε[t],  ε[t] ∼ N(0, Q)
    """
    A::Matrix{T}
    Q::PDMat{T, Matrix{T}}
    μ::Vector{T}
end

function LinearGaussianLatentDynamics(A::Matrix, Q::Matrix)
    μ = zeros(eltype(A), size(A, 1))
    return LinearGaussianLatentDynamics(A, PDMat(Q), μ)
end

function LinearGaussianLatentDynamics(A::Matrix, Q::Matrix, μ::Vector)
    return LinearGaussianLatentDynamics(A, PDMat(Q), μ)
end

struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess
    """
        Observation process for a linear Gaussian state space model.
        The model is defined by the following equation:
        y[t] = Hx[t] + η[t],        η[t] ∼ N(0, R)
    """
    H::Matrix{T}
    R::PDMat{T, Matrix{T}}
end

function LinearGaussianObservationProcess(H::Matrix, R::Matrix)
    return LinearGaussianObservationProcess(H, PDMat(R))
end

function SSMProblems.distribution(
        proc::LinearGaussianLatentDynamics{T},
        extra
    ) where {T<:Real}
        dx = size(proc.A, 1)
    return MvNormal(zeros(T, dx), init_cov(proc, convert(T, 1000)))
end

function SSMProblems.distribution(
        proc::LinearGaussianLatentDynamics{T},
        step::Int,
        state::AbstractVector{T},
        extra
    ) where {T<:Real}
    return MvNormal(proc.A*state+proc.μ, proc.Q)
end

function SSMProblems.distribution(
        proc::LinearGaussianObservationProcess{T},
        step::Int,
        state::AbstractVector{T},
        extra
    ) where {T<:Real}
    return MvNormal(proc.H*state, proc.R)
end

const LinearGaussianModel{T<:Real} = StateSpaceModel{
    LinearGaussianLatentDynamics{T},
    LinearGaussianObservationProcess{T}
}

Base.eltype(::LinearGaussianModel{T}) where T = T
Base.eltype(::StateSpaceModel) = error("model element type must be explicit")

## UTILITIES ##################################################################

PDMats.PDMat(mat::AbstractMatrix) = begin
    # this deals with rank definicient positive semi-definite matrices
    chol_mat = cholesky(mat, Val(true), check=false)
    Up = UpperTriangular(chol_mat.U[:, invperm(chol_mat.p)])
    PDMat(mat, Cholesky(Up))
end

function init_cov(dyn::LinearGaussianLatentDynamics{T}, σ::T) where {T<:Real}
    # compute the eigenvalues without permuting the matrix
    λ   = eigvals(dyn.A, permute=false, sortby=nothing)
    idx = isless.(norm.(λ, 2), 1.0)

    # fill all non-stationary component diagonals with σ
    Σ0 = diagm(convert(T, σ)*ones(size(dyn.A, 1)))
    Σ0[idx, idx] = lyapd(dyn.A[idx, idx], dyn.Q[idx, idx])

    return Σ0
end