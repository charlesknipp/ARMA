using ARMAModels
using SSMProblems
using Test

@testset "ARMAModels" begin

@testset "Model Constructors" begin
    @test isa(AR([0.9], 1.0), ARMA{Float64,1,0})
    @test isa(MA([0.7], 1.0), ARMA{Float64, 0, 1})
    @test isa(ARMA([1.6, -0.7], [0.6], 1.0), ARMA{Float64, 2, 1})
end

@testset "Autocovariance Generation" begin
    model = ARMA([1.6, -0.7], [0.6], 1.0)
    empΓ  = empirical_autocovariance(model, 8, order = 2048)
    trueΓ = autocovariance(model, 8)

    diffγ = empΓ-trueΓ
    @test diffγ'*diffγ < 5e-2
end

end