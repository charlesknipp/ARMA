using ARMAModels
using SSMProblems
using Test

@testset "ARMAModels" begin

@testset "Model Constructors" begin
    AR([0.9], 1.0)
    MA([0.7], 1.0)
    ARMA([1.6, -0.7], [0.6], 1.0)
end

@testset "Autocovariance Generation" begin
    model = ARMA([1.6, -0.7], [0.6], 1.0)
    empΓ  = empirical_autocovariance(model, 25)
    trueΓ = autocovariance(model, 25)

    @test similar_list(empΓ, trueΓ, 1e-2)
end

end