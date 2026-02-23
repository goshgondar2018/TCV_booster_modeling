using CSV, DifferentialEquations, DataFrames, StructArrays, Tables, Optim, StatsBase, Distributions, LinearAlgebra,LsqFit


params=CSV.read("params_latest_adapted_feb11.csv", DataFrame)
params.Bi.=params.Bi./12
params.u=params.u./12
params.pu=1 .- exp.(-1 .* params.u)
params.pu_in = [0; circshift(params.pu, 1)[2:end]];

# full typhoid model with demographic 
# note vaccination and disease compartments don't matter here because in pre-disease/pre-vaxx period
function typhoid_model_demo!(du, u, p, t) 
    ages=17
    
    S = @view u[1:ages]
    Is = @view u[ages+1:2*ages]
    Ia = @view u[2*ages+1:3*ages]
    C = @view u[3*ages+1:4*ages]
    R = @view u[4*ages+1:5*ages]
    V = @view u[5*ages+1:6*ages] 
    Vw = @view u[6*ages+1:7*ages]
    Iv = @view u[7*ages+1:8*ages]
    Sv = @view u[8*ages+1:9*ages]

    fill!(du, 0.0)
    dS = @view du[1:ages]
    dIs = @view du[ages+1:2*ages]
    dIa = @view du[2*ages+1:3*ages]
    dC = @view du[3*ages+1:4*ages]
    dR = @view du[4*ages+1:5*ages]
    dV = @view du[5*ages+1:6*ages] 
    dVw = @view du[6*ages+1:7*ages]
    dIv = @view du[7*ages+1:8*ages]
    dSv = @view du[8*ages+1:9*ages]

    N = sum(S + Is + Ia + C + R + V + Vw + Iv + Sv)

    #1. AGING, ALL-CAUSE MORTALITY, ROUTINE VACCINATION (these 3 don't "compete" with each other) - as you age you can die or be vx'd
    #treat 0-8 month-olds separately because they have births, no aging in, and no vx
    dS .= p.Bi_burn .* [N; zeros(ages-1)] .+ # births
        (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(S, 1)[2:end]] .- # aging in
        p.a_out .* (1 .- p.pu_burn) .* S .- p.u_burn .* S   #aging out and background mortality
    dIs .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Is, 1)[2:end]] .-  #aging in
         p.a_out .* (1 .- p.pu_burn) .* Is .- p.u_burn .* Is  #aging out and background mortality
    dIa .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Ia, 1)[2:end]] .- #aging in
        p.a_out .* (1 .- p.pu_burn) .* Ia .- p.u_burn .* Ia  #aging out and background mortality
    dC .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(C, 1)[2:end]] .- #aging in
        p.a_out .* (1 .- p.pu_burn) .* C .- p.u_burn .* C   #aging out and background mortality
    dR .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(R, 1)[2:end]] .- #aging in 
         p.a_out .* (1 .- p.pu_burn) .* R .- p.u_burn .* R   #aging out and background mortality
    dV .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(V, 1)[2:end]] .-  #aging in
        p.a_out .* (1 .- p.pu_burn) .* V .- p.u_burn .* V    #aging out and background mortality
    dVw .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Vw, 1)[2:end]] .-   #aging in
        p.a_out .* (1 .- p.pu_burn) .* Vw .- p.u_burn .* Vw  #aging out and background mortality
    dIv .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Iv, 1)[2:end]] .- #aging in
        p.a_out .* (1 .- p.pu_burn) .* Iv .- p.u_burn .* Iv   #aging out and background mortality
    dSv .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Sv, 1)[2:end]] .- #aging in
        p.a_out .* (1 .- p.pu_burn) .* Sv .- p.u_burn .* Sv   #aging out and background mortality

end

# process demographic aging output
function clean_output_full_demo!(sol) # (sol, t, ages)
    m=17
    time_points = sol.t
    S_values = [u[1:m] for u in sol.u]
    Is_values = [u[m+1:2*m] for u in sol.u]
    Ia_values = [u[2*m+1:3*m] for u in sol.u]
    C_values = [u[3*m+1:4*m] for u in sol.u]
    R_values = [u[4*m+1:5*m] for u in sol.u]
    V_values = [u[5*m+1:6*m] for u in sol.u]
    Vw_values = [u[6*m+1:7*m] for u in sol.u]
    Iv_values = [u[7*m+1:8*m] for u in sol.u]
    Sv_values = [u[8*m+1:9*m] for u in sol.u]

    # Create DataFrame
    model_df = DataFrame(
        time = time_points)

    for (idx, group) in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"])
        model_df[!, "S_$group"] = [S[idx] for S in S_values]
        model_df[!, "Is_$group"] = [Is[idx] for Is in Is_values]
        model_df[!, "Ia_$group"] = [Ia[idx] for Ia in Ia_values]
        model_df[!, "C_$group"] = [C[idx] for C in C_values]
        model_df[!, "R_$group"] = [R[idx] for R in R_values]
        model_df[!, "V_$group"] = [V[idx] for V in V_values]
        model_df[!, "Vw_$group"] = [Vw[idx] for Vw in Vw_values]
        model_df[!, "Iv_$group"] = [Iv[idx] for Iv in Iv_values]
        model_df[!, "Sv_$group"] = [Sv[idx] for Sv in Sv_values]
    end
    
    return(model_df)
end

# check population at equilibrium 
function check_demographic_equilibrium(sol, threshold=0.01) #0.001
    # last two years (24 months) of data
    if length(sol.t) < 25  
        return false, Inf
    end
    
    last_two_years = sol.u[(end-24):end]
    
    total_pops = [sum(u) for u in last_two_years]
    
    # annual growth rate (as percentage)
    annual_growth_rate = 100 * (total_pops[end] - total_pops[1]) / total_pops[1] / 2  # Divide by 2 years
    
    
    
    return annual_growth_rate
end

# check age specific population sizes at equilibrium 
function check_age_specific_equilibrium(sol, ages=17, threshold=0.005)
    # Get the last two years of data
    if length(sol.t) < 25
        return false, []
    end
    
    last_year_start = sol.u[end-12]
    last_year_end = sol.u[end]
    
    # Calculate age-specific growth rates
    age_growth_rates = []
    
    # For each age group in each compartment
    for i in 1:8  # 8 compartments
        compartment_start = last_year_start[(1+(i-1)*ages):(i*ages)]
        compartment_end = last_year_end[(1+(i-1)*ages):(i*ages)]
        
        for a in 1:ages
            if compartment_start[a] > 0
                growth_rate = 100 * (compartment_end[a] - compartment_start[a]) / compartment_start[a]
                push!(age_growth_rates, growth_rate)
            else
                push!(age_growth_rates, 0.0)
            end
        end
    end
    
    # Check if all age-specific growth rates are below threshold
    max_growth_rate = maximum(abs.(age_growth_rates))
    is_equilibrium = max_growth_rate < threshold
    
    return is_equilibrium, max_growth_rate
end


#GOF calculation function return objective value 
function demo_gof!(x, params) #y, params
    
    m=17
    ages=17
    T=12*100
    ts=range(0, stop=T, step=1)

    initial=[x[1:m];fill(0, 9*m)]
    params.u_burn = x[(m+1):(2*m)]
    params.pu_burn = 1 .- exp.(-params.u_burn)
    params.pu_in_burn = [0; circshift(params.pu_burn, 1)[2:ages]]
    params.Bi_burn = vcat(x[35],repeat([0],16))

    prob=ODEProblem(typhoid_model_demo!, initial, (0.0, T), params)
    sol=solve(prob, dense=false, save_everystep=false, tstops=ts, saveat=ts);

    annual_growth = check_demographic_equilibrium(sol)
    
    # Check birth-death balance
    total_pop = sum(sol.u[end])
    births = params.Bi_burn[1] * total_pop
    output_demo=clean_output_full_demo!(sol)
    final_row = size(output_demo,1)
    estimated_final_N_by_age=Array(select(output_demo,[:"S_1","S_2","S_3","S_4","S_5","S_6","S_7","S_8","S_9","S_10","S_11","S_12","S_13","S_14","S_15","S_16","S_17"])[final_row,:])
    deaths = sum(params.u_burn .* estimated_final_N_by_age)
    birth_death_ratio = births/deaths

    simulated_age_props = estimated_final_N_by_age ./ total_pop
    target_age_props = params.N_age ./ sum(params.N_age)
    
    sse_age_dist = sum(((simulated_age_props .- target_age_props) ./ target_age_props).^2)
    sse_u = sum(((params.u_burn .- params.u) ./params.u).^2) #./ params.u_burn) 

    # if birth/death balance, weight it heavily in objective value
    if abs(birth_death_ratio - 1.0) > 0.01
        gof = 1000 * (birth_death_ratio - 1.0)^2
    else
        # if not, focus on age distribution
        gof = sse_u + 10 * sse_age_dist
    end

    # calculate proportion in older age groups (25+) (initially a little off in these age groups without this step)
    simulated_older_prop = sum(simulated_age_props[(m÷2):end])
    target_older_prop = sum(target_age_props[(m÷2):end])

    # penalty for this mismatch
    older_age_penalty = 50 * ((simulated_older_prop - target_older_prop) / target_older_prop)^2 #100*
    gof += older_age_penalty
    
    neg_penalty = 0.0
    for i in (m+1):(2*m)  # For mortality rates
        if x[i] < 0
            neg_penalty += 1e6 * abs(x[i])  
        end
    end
    gof += neg_penalty
    
    return(gof)
end

T=12*100 #run calibration/burn-in for 100 years or 1200 months
ts=range(0, stop=T, step=1)
m=17

params_archetype1=params[(params.archetype.==1),:]
params_archetype1.inc_age=zeros(m)

start=[params_archetype1.N_age; params_archetype1.u; params_archetype1.Bi[1]]

# optimize
opt=optimize(x->demo_gof!(x, params_archetype1),
   start,NelderMead(),
    Optim.Options(g_tol=0.0005, f_tol=0.0005, x_tol=0.0005,
       iterations=15000))    

# obtain calibrated pop sizes, age-specific mortality rates, and birth rate
opt_demo=Optim.minimizer(opt)

# write out
demo_calib_output=hcat(start,opt_demo) 
sum((opt_demo .- start).^2/start.^2)
CSV.write("demo_calib_output.csv",Tables.table(demo_calib_output),writeheader=true)


