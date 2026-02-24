addprocs(15)

@everywhere using CSV, DifferentialEquations, DataFrames, StructArrays, Tables, Optim, StatsBase, Distributions, LinearAlgebra,LsqFit

# update with calibrated demographic parameters
@everywhere function update_params_demographics(params_start, calibrated_demographic_params,archetype)
    params=params_start[params_start.archetype.==archetype,:]
    params.N_age_burn.=calibrated_demographic_params[1:17,2]
    params.u_burn.=calibrated_demographic_params[18:34,2]
    params.pu_burn=1 .- exp.(-1 .* params.u_burn)
    params.pu_in_burn = [0; circshift(params.pu_burn, 1)[2:end]];
    params.Bi_burn.=vcat(calibrated_demographic_params[35,2],repeat([0],16));
    params = mapcols(x -> convert.(Float64, x), params)

    return(params)
end

# update with calibrated natural history parameters and incidence targets
@everywhere function update_params_psa(params_start, samples, sample)
    
    params = copy(params_start)
    p_sample = samples[sample,:]
    
    params.inc_age[1:3] .= p_sample.inc_age_cat1/100000
    params.inc_age[4] = p_sample.inc_age_cat2/100000
    params.inc_age[5] = p_sample.inc_age_cat3/100000
    params.inc_age[6] = p_sample.inc_age_cat4/100000
    params.inc_age[7:17] .= p_sample.inc_age_cat5/100000
    
    params.f[1:3] .= p_sample.f1
    params.f[4] = p_sample.f1
    params.f[5] = p_sample.f2
    params.f[6] = p_sample.f2
    params.f[7:17] .= p_sample.f3 
    
    params.alpha .= p_sample.alpha
    
    params.gamma .= p_sample.gamma
    params.r_a .= p_sample.r_a
    params.r_c .= p_sample.r_c
    params.theta_s .= p_sample.theta_s
    params.theta_a .= p_sample.theta_a
    params.phi .= p_sample.phi
    params.rho .= p_sample.rho
    
    params.alpha_v .= 0.0
    params.r_v .= 0.0
    params.l .= 0.0

 
    return(params)
end

# full typhoid model 
# note vaxx compartments don't matter because in a pre-vaccine period
@everywhere function typhoid_model!(du, u, p, t) 
    
    ages=17
    m = 17
    
    S = @view u[1:m]
    Is = @view u[m+1:2*m]
    Ia = @view u[2*m+1:3*m]
    C = @view u[3*m+1:4*m]
    R = @view u[4*m+1:5*m]
    V = @view u[5*m+1:6*m] 
    Vw = @view u[6*m+1:7*m]
    Iv = @view u[7*m+1:8*m]
    Sv = @view u[8*m+1:9*m]
    cases = @view u[9*m+1:10*m]
    
    fill!(du, 0.0)
    dS = @view du[1:m]
    dIs = @view du[m+1:2*m]
    dIa = @view du[2*m+1:3*m]
    dC = @view du[3*m+1:4*m]
    dR = @view du[4*m+1:5*m]
    dV = @view du[5*m+1:6*m] 
    dVw = @view du[6*m+1:7*m]
    dIv = @view du[7*m+1:8*m]
    dSv = @view du[8*m+1:9*m]
    dcases = @view du[9*m+1:10*m]

    N = sum(S + Is + Ia + C + R + V + Vw + Iv + Sv)
   
    #1. VACCINATION AND DEMOGRAPHICS
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

    #2. DISEASE TRANSMISSION

    Is_vec = fill(sum(Is[1:ages]), ages)/N #fill(sum(Is[1:ages])/N, ages)  
    Ia_vec = fill(sum(Ia[1:ages]), ages)/N#fill(sum(Ia[1:ages])/N, ages)
    Iv_vec = fill(sum(Iv[1:ages]), ages)/N#fill(sum(Iv[1:ages])/N, ages)
    C_vec = fill(sum(C[1:ages]), ages)/N#fill(sum(C[1:ages])/N, ages)
    
    dS .+= p.alpha.*R .+ p.gamma.*(1 .- p.rho).*(((1 .- p.theta_s).*Is) .+ ((1 .- p.theta_a).*Ia)) .- p.beta.*S.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec) 
    dIs .+= p.f.*(p.beta.*S.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .+ 
           p.f.*(p.beta.*Vw.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .+
           p.f.*(p.beta.*Sv.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .-
           p.gamma.*Is .- p.u_ic.*Is 
    dIa .+= (1 .- p.f).*(p.beta.*S.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .+
           (1 .- p.f).*(p.beta.*Sv.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .-
           p.gamma.*Ia 
    dC .+= p.rho.*p.gamma.*(Is .+ Ia .+ Iv) .- p.phi.*C 
    dR .+= p.phi.*C .+ (1 .- p.rho).* p.gamma.*(p.theta_s.*Is .+ p.theta_a.*Ia) .- p.alpha.*R 
    dV .+= (1 .- p.rho).*p.theta_a.*p.gamma.*Iv .- 
           (p.beta.*p.l.*V.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .- p.alpha_v.*V 
    dVw .+= p.alpha_v.*V  .+ (1 .- p.rho).*(1 .- p.theta_a).*p.gamma.*Iv .- (p.beta.*Vw.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) 
    dIv .+= (p.beta.*p.l.*V.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .+
           (1 .- p.f).*(p.beta.*Vw.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .- p.gamma.*Iv
    dSv .+= .- p.beta.*Sv.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec) 
    dcases .= p.f.*(p.beta.*S.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .+
              p.f.*(p.beta.*Vw.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) .+
              p.f.*(p.beta.*Sv.*(Is_vec .+ p.r_a.*Ia_vec .+ p.r_c.*C_vec .+ p.r_v.*Iv_vec)) 

end

# GOF calculation function - returns objective value
@everywhere function incidence_gof_calibration!(x, params_archetype, archetype,target_inc, type, calib)

    ages=17
    m=17
 
    params_archetype.beta[1:3]=fill(x[1], 3) # Archetype i
    params_archetype.beta[4]=x[2]
    params_archetype.beta[5]=x[3]
    params_archetype.beta[6]=x[4]
    params_archetype.beta[7:17]=fill(x[5], 11) 

    params_archetype.f[1:3]= fill(x[6],3) # Archetype 1
    params_archetype.f[4]= x[7]
    params_archetype.f[5]= x[8]
    params_archetype.f[6]= x[9]
    params_archetype.f[7:17]= fill(x[10],11)

    params_archetype.alpha .= x[11] # Shared

    alpha_extreme = (x[11] < 1/(25*12) || x[11] > 1/12)
    
    if (minimum(x) < 0 ||  alpha_extreme) # || h_extreme)  
        if calib==0
            return(NaN)
        end
        gof_sum = 100*10^9 
        return(gof_sum)
    end

    if archetype == 1
        T_max=100*12 
    else
        T_max=100*12
    end
    
    ts = range(0, stop=T_max, step=1)
    initial = gen_initial_burn_in!(params_archetype, ages)
    cal_prob = ODEProblem(typhoid_model!, initial, (0.0, T_max), params_archetype)
    cal_sol = solve(cal_prob, save_everystep=false, tstops=ts, saveat=ts) 
    cleaned_cal = clean_output_full_transmission_calib!(cal_sol)
    ## Extract incidence values
    cal_inc = calc_inc_burn(cleaned_cal)

    cal_inc[cal_inc .<= 0].=0.0000001 #to prevent NA errors in Poisson
    gof_poisson = [log(pdf(Poisson(cal_inc[i]), Int(round(target_inc[i])))) for i in 1:5]

    gof = gof_poisson .* (type .== "poisson") .* -10 
    gof_sum = sum(gof)

    ##if incidence is so much higher than targets that log pdf is -Inf, make it really big but scaled by incidence
    ##so that optim can find a direction to go in
    all_positive = all(>(0), @view cal_inc[1:5])
    
    if gof_sum == Inf && all_positive
        gof_inf_ind = findall(isinf, gof)
        inc_mean = mean(@view cal_inc[gof_inf_ind])
        target_mean = mean(@view target_inc[gof_inf_ind])
         if inc_mean > target_mean
            gof_sum = sum(1000000 .* @view cal_inc[gof_inf_ind]) # sum(1000000 .* @view cal_inc[gof_inf_ind]) 100
        elseif inc_mean < target_mean
            gof_sum = sum(100000000 ./ @view cal_inc[gof_inf_ind]) #sum(1000000 ./ @view cal_inc[gof_inf_ind]) 100000000
        end  
    end
    
    obj_value = gof_sum    

    if (calib==1)
        return(obj_value)
    else
        return(sum(gof_poisson)) 
    end
end

# initialize population for burn in 
@everywhere function gen_initial_burn_in!(params, ages)
    #initialize starting compartment sizes
    Is_0 = (params.inc_age .* params.N_age_burn)./12 #(params.inc_age .* params.N_age_burn).* (1 ./ params.gamma) #
    Ia_0 = (Is_0 ./ (params.f)) .- Is_0
    S_0 = params.N_age_burn[1:(ages)] - (Is_0 + Ia_0)
    C_0 = zeros(ages)
    R_0 = zeros(ages)
    V_0 = zeros(ages)
    Vw_0 = zeros(ages)
    Iv_0 = zeros(ages)
    Sv_0 = zeros(ages)
    cases_0 = zeros(ages)
    initial = vcat(S_0, Is_0, Ia_0, C_0, R_0, V_0, Vw_0, Iv_0, Sv_0,cases_0) 
    return(initial)
end

# process compartmental pop size output following transmission
@everywhere function clean_output_full_transmission_calib!(sol)
    ages=17
    m=17
    
    # Final two time steps
    final_index = length(sol.t)
    time_points = [sol.t[final_index-1], sol.t[final_index]]
    
    cases_values = [sol.u[final_index-1][9*m+1:10*m], sol.u[final_index][9*m+1:10*m]]
    
    model_df = DataFrame(time = time_points)
    
    for age in 1:ages
        age_label = string(age)
        model_df[!, "cases_$age_label"] = [cases_values[1][age], cases_values[2][age]]
    end
    
    for age in 1:ages
        age_label = string(age)
        for (comp_idx, comp_name) in enumerate(["S", "Is", "Ia", "C", "R", "V", "Vw", "Iv", "Sv"])
            idx_start = (comp_idx-1)*m + 1
            idx_end = comp_idx*m
            model_df[!, "$(comp_name)_$age_label"] = [missing, sol.u[final_index][idx_start:idx_end][age]]
        end
    end
    
    return model_df
end

# calculate burn-in incidence 
@everywhere function calc_inc_burn(sol_archetype)
    agecat1 = 1:3
    agecat2 = [4]
    agecat3 = [5]
    agecat4 = [6]
    agecat5 = 7:17
    
    final = size(sol_archetype, 1)
    
    # calculate incident cases 
    calculate_incidence = function(ages)
        cases_final = sum(sol_archetype[final, Symbol("cases_$i")] for i in ages)
        cases_prev = sum(sol_archetype[final-1, Symbol("cases_$i")] for i in ages)
        return cases_final - cases_prev
    end
    
    # calculate total population
    calculate_population = function(ages)
        return sum(
            sol_archetype[final, Symbol("$(comp)_$i")] 
            for i in ages 
            for comp in ["S", "Is", "Ia", "C", "R", "V", "Vw", "Iv", "Sv"]
            if !ismissing(sol_archetype[final, Symbol("$(comp)_$i")])
        )
    end
    
    inc_rates = Float64[]
    for ages in [agecat1, agecat2, agecat3, agecat4, agecat5]
        inc_cases = calculate_incidence(ages)
        total_pop = calculate_population(ages)
        push!(inc_rates, (inc_cases * 12 * 100000) / total_pop)
    end
    
    return inc_rates
end

# calculate calibrated incidence given input params
@everywhere function get_calibrated_incidence!(params, opt_burn)
    params.beta[1:3]=fill(opt_burn[1], 3) 
    params.beta[4]=opt_burn[2]
    params.beta[5]=opt_burn[3]
    params.beta[6]=opt_burn[4]
    params.beta[7:17]=fill(opt_burn[5], 11) 
    
    params.f[1:3]= fill(opt_burn[6],3) 
    params.f[4]= opt_burn[7]
    params.f[5]= opt_burn[8]
    params.f[6]= opt_burn[9]
    params.f[7:17]= fill(opt_burn[10],11)
    
    params.alpha .= opt_burn[11] 

    T_max=100*12
    ts = range(0, stop=T_max, step=1)
    ages=17
    initial = gen_initial_burn_in!(params, ages);
    prob = ODEProblem(typhoid_model!, initial, (0.0, T_max), params)
    sol = solve(prob, dense=false, save_everystep=false, tstops=ts, saveat=ts)
    
    cleaned_sol = clean_output_full_transmission_calib!(sol)
    steady_inc = calc_inc_burn(cleaned_sol)
end


# setup data and parameters on all workers
@everywhere begin
    params = CSV.read("params_latest_adapted_june14.csv", DataFrame)
    calibrated_demographic_params = CSV.read("demo_calib_output.csv", DataFrame)
    f_estimates = CSV.read("age_specific_symptomatic_fraction_estimates.csv", DataFrame)
    samples = CSV.read("params_psa.csv", DataFrame)
    samples_archetype1 = samples[samples.archetype.=="Medium",:]
    
    params_archetype1 = update_params_demographics(params, calibrated_demographic_params, 1)
end

# run distributed computation using pmap to collect results
results = pmap(1:250) do i # change to 251:500, 501:750, and 751:1000 to obtain all samples output
    println(string("starting ", i))
    
    # initialize parameters for this run
    local_params_archetype1 = update_params_psa(params_archetype1, samples_archetype1, i)

    T = 1200 # run calibration/burn-in for 100 years
    ts = range(0, stop=T, step=1)
    
    # incidence targets and initial parameter guesses
    target_inc_archetype1 = [samples_archetype1.inc_age_cat1[i], samples_archetype1.inc_age_cat2[i], 
    samples_archetype1.inc_age_cat3[i], samples_archetype1.inc_age_cat4[i], samples_archetype1.inc_age_cat5[i]]

    initial_betas_1 = [3.16, 3.16, 4.85, 5, mean([2.00, 0.514])]
    initial_f_archetype1 = [f_estimates.estimate[1],f_estimates.estimate[1],f_estimates.estimate[2],f_estimates.estimate[2],f_estimates.estimate[3]]
    initial_alpha = [1/(19.8*12)] 
    initial_params_archetype1 = [initial_betas_1; initial_f_archetype1; initial_alpha]
        
    # calibrate
    opt_archetype1 = optimize(
        x -> incidence_gof_calibration!(x, local_params_archetype1, 1, target_inc_archetype1, "poisson", 1), 
        initial_params_archetype1, NelderMead(),
        Optim.Options(show_trace=false, g_tol=0.01, f_tol=0.01, x_tol=0.01, iterations=5000)
    )
    opt_burn_archetype1 = Optim.minimizer(opt_archetype1)

    # compute likelihood and incidence
    opt_like_archetype1 = incidence_gof_calibration!(opt_burn_archetype1, local_params_archetype1, 1, target_inc_archetype1, "poisson", 0) 
    opt_inc_archetype1 = get_calibrated_incidence!(local_params_archetype1, opt_burn_archetype1)
 
    return (
        i = i,
        archetype1 = [opt_burn_archetype1; opt_inc_archetype1; opt_like_archetype1]
    )
end

# create result DataFrames
names_list = ["beta1", "beta2", "beta3", "beta4", "beta5",
             "f1", "f2", "f3", "f4", "f5", "alpha",
             "inc_age_cat1", "inc_age_cat2", "inc_age_cat3", "inc_age_cat4", "inc_age_cat5",
             "log_like"]

archetype1_results = DataFrame(Names = names_list)

for result in results
    i = result.i
    col_name = Symbol("run_$i")
    insertcols!(archetype1_results, col_name => result.archetype1)
end

# Write results to disk
CSV.write("archetype1_all_runs.csv", archetype1_results) 
