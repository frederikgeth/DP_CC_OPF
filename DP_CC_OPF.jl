#!/usr/bin/env julia
using Pkg
pkg"activate ."

using CSV
using DataStructures: SortedDict
using JuMP
using DataFrames
# using Mosek
# using MosekTools
using ECOS
using Ipopt
using Distributions
using LinearAlgebra
using ArgParse

# DATA MODEL set up
# - exactly one generator per bus
# - all generator buses are PV buses
# - generators are fixed power factor (tan phi)
# - exactly one load per bus
# - network is radial

optimizer = optimizer_with_attributes(
    ECOS.Optimizer, "maxit" => 300
    )
# parse arguments
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mechanism", "-m"
            help = "chose mechanism: D_OPF, CC_OPF, ToV_CC_OPF, TaV_CC_OPF or CVaR_CC_OPF"
            arg_type = String
            default = "CC_OPF"
    end
    return parse_args(s)
end
args = parse_commandline()
# mechanism = "D_OPF"
mechanism = "CC_OPF"
# mechanism = "ToV_CC_OPF"
# mechanism = "TaV_CC_OPF"
# mechanism = "CVaR_CC_OPF"
# mechanism = args["mechanism"]
outdir = "output/" * mechanism
mkpath(outdir)

# load scripts
include("scripts/data_manager.jl")
include("scripts/run_mechanism.jl")
include("scripts/run_mechanism_nlp.jl")

# load data
caseID = "feeder15"
(node,line,R,D_n,U_n,U_l,D_l,T)=load_data(caseID)

# privacy parameters
σ = zeros(length(line)); σ̂ = zeros(length(line)); Σ = zeros(length(line),length(line));
δ = 1/(length(node)-1); ε = 1; # DP parameters
# adjacency and noise parameters
for i in 1:length(line)
    β = 0.1*node[i].d_p # adjacency coefficients
    σ[i] = β*sqrt(2*log(1.25/δ))/ε
    σ̂[i] = σ[i] # target variance
    Σ[i,i] = σ[i]^2 # covariance matrix
end
if mechanism == "TaV_CC_OPF"
    σ = zeros(length(line)); Σ = zeros(length(line),length(line)) .+ 1e-10;
    for i in 1:length(line)
        β = 0.1*node[i].d_p # adjacency coefficients
        i ∈ [1 5 6 7 9 11 12 13] ? σ[i] = β*sqrt(2*log(1.25/δ))/ε : σ[i] = 10e-6 # selected network lines
        Σ[i,i] = σ[i]^2 .+ 1e-10 # covariance matrix
    end
end

# variance penalty
ψ = 1e6

# constraint violation probabilities
η_g = 0.01; η_u = 0.02; η_f = 0.10;

# CVaR parameters
ϱ = 0.1; θ = 1;

# run mechanism conic
(nodal_solution,exp_cost,CVaR,CPU_time, model)=run_mechanism(mechanism,node,line,σ,Σ,T,U_n,D_n,U_l,D_l,η_g,η_u,η_f,ψ,σ̂,θ,ϱ, optimizer=optimizer)

open("out.txt", "w") do file                      
    write(file, string(model));
end

# run mechanism nonlinear 
(nodal_solution_nlp,exp_cost_nlp,CVaR_nlp,CPU_time_nlp, model_nlp)=run_mechanism_nlp(mechanism,node,line,σ,Σ,T,U_n,D_n,U_l,D_l,η_g,η_u,η_f,ψ,σ̂,θ,ϱ)


# save results
CSV.write("$(outdir)/nodal_solution.csv",nodal_solution)
open("$(outdir)/summary.txt","a") do io
   println(io,"exp_cost = ", exp_cost)
   println(io,"CVaR = ", CVaR)
   println(io,"tot_variance = ", sum(nodal_solution[2:end,8]))
   println(io,"CPU_time = ", CPU_time)
end
