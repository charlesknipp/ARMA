using Random
using UnPack
using SSMProblems

import AbstractMCMC: AbstractSampler

abstract type AbstractFilter <: AbstractSampler end

"""
    predict([rng,] states, model, alg, [step, extra])

propagate the filtered states forward in time.
"""
function predict end

"""
    update(states, model, data, alg, [step, extra])

update beliefs on the propagated states.
"""
function update end

"""
    prior([rng,] model, alg, [extra])

propose an initial state distribution.
"""
function prior end

function sample(
        rng::AbstractRNG,
        model::StateSpaceModel,
        observations::AbstractVector,
        filter::AbstractFilter
    )
    MT = eltype(model)
    
    filtered_states = prior(rng, model, filter, nothing)
    log_evidence = zero(MT)

    for t in eachindex(observations)
        proposed_states = predict(
            rng, filtered_states, model, filter, t, nothing
        )

        filtered_states, log_marginal = update(
            proposed_states, model, observations[t], filter, t, nothing
        )

        log_evidence += log_marginal
    end

    return filtered_states, log_evidence
end

## KALMAN FILTER ##############################################################

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function prior(
        rng::AbstractRNG,
        model::LinearGaussianModel,
        filter::KalmanFilter,
        extras
    )
    init_dist = SSMProblems.distribution(model.dyn, extras)
    return Gaussian(init_dist.μ, Matrix(init_dist.Σ))
end

function predict(
        rng::AbstractRNG,
        particles::Gaussian,
        model::LinearGaussianModel,
        filter::KalmanFilter,
        step::Integer,
        extra
    )
    @unpack A, Q, μ = model.dyn

    predicted_particles = let x = particles.μ, Σ = particles.Σ
        Gaussian(A*x + μ, A*Σ*A' + Q)
    end

    return predicted_particles
end

function update(
        proposed_particles::Gaussian,
        model::LinearGaussianModel{T},
        observation::AbstractVector,
        filter::KalmanFilter,
        step::Integer,
        extra
    ) where {T<:Real}
    @unpack H, R = model.obs

    particles, residual, S = GaussianDistributions.correct(
        proposed_particles,
        Gaussian(observation, R), H
    )

    log_marginal = if step > 1
        logpdf(
            Gaussian(zero(residual), Symmetric(S)),
            residual
        )
    else
        zero(T)
    end

    return particles, log_marginal
end

function update(
        proposed_particles::Gaussian,
        model::LinearGaussianModel,
        observartion::T,
        filter::KalmanFilter,
        step::Integer,
        extra
    ) where {T<:Real}
    return update(proposed_particles, model, [observartion], filter, step, extra)
end

## BOOTSTRAP FILTER ###########################################################

# TODO: adaptive resampling, set by default to always resmaple
struct BootstrapFilter <: AbstractFilter
    N::Int64
    threshold::Float64
end

BF(N::Integer) = BootstrapFilter(N, 1.0)

resample_threshold(filter::BootstrapFilter) = filter.threshold*filter.N

function prior(
        rng::AbstractRNG,
        model::StateSpaceModel,
        filter::BootstrapFilter,
        extra
    )
    init_dist = SSMProblems.distribution(model.dyn, extra)
    initial_states = map(
        x -> rand(rng, init_dist),
        1:filter.N
    )

    return ParticleContainer(initial_states, zeros(eltype(model), filter.N))
end

function predict(
        rng::AbstractRNG,
        particles::ParticleContainer,
        model::StateSpaceModel,
        filter::BootstrapFilter,
        step::Integer,
        extra
    )
    weights = softmax(particles.log_weights)
    idx = multinomial_resampling(rng, weights)

    proposed_states = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x, extra),
        particles[idx]
    )

    return ParticleContainer(proposed_states, particles.log_weights)
end

function update(
        particles::ParticleContainer,
        model::StateSpaceModel,
        observation,
        filter::BootstrapFilter,
        step::Integer,
        extra
    )
    log_marginals = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation, extra),
        collect(particles)
    )

    return (
        ParticleContainer(particles.vals, log_marginals),
        logsumexp(log_marginals) - convert(eltype(model), log(filter.N))
    )
end