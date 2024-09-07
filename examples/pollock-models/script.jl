using ARMAModels
using CairoMakie

## GRAPHING UTILITIES #########################################################

function plot_model(proc::ARMA, n)
    fig = Figure()

    # this is the full spectrum, we only want to see the first half
    ωs, S = spectral_density(proc; res=2048)

    # try to smooth it out
    ax1 = Axis(fig[1,1], title="Spectral Density")
    lines!(ax1, ωs[1:1024], S[1:1024])

    # plot autocovariance
    γs = autocovariance(proc, n)

    ax2 = Axis(fig[2,1], title="Autocovariance")
    stem!(ax2, γs)

    return fig
end

## AR(p) MODELS ###############################################################

ar2 = AR([0.273, -0.81], 1.0)
plot_model(ar2, 25)

ar4 = AR([1.061, -1.202, 0.679, -0.360], 1.0)
plot_model(ar4, 25)

## MA(q) MODELS ###############################################################

ma1 = MA([0.9], 1.0)
plot_model(ma1, 25)

ma2 = MA([1.25, 0.8], 1.0)
plot_model(ma2, 25)

## ARMA(p,q) MODELS ###########################################################

aex = ARMA([0.273, -0.81], [0.9], 1.0)
plot_model(aex, 25)

mnz = ARMA([1.3418, -0.7059], [-1.0543, 0.5188], 0.9694)
plot_model(mnz, 25)

qec = ARMA([0.5], [0, -0.8], 1.0)
plot_model(qec, 25)