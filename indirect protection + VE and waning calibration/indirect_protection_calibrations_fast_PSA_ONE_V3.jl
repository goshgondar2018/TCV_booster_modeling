
using Distributed, CSV, DifferentialEquations, DataFrames, StructArrays, Tables, Optim, StatsBase, Distributions, LinearAlgebra, LsqFit, BlackBoxOptim, StatsFuns
addprocs(20) #addprocs(25) 

@everywhere using CSV, DifferentialEquations, DataFrames, StructArrays, Tables, Optim, StatsBase, Distributions, LinearAlgebra, LsqFit, BlackBoxOptim, StatsFuns

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

# update with calibrated natural history parameters 
@everywhere function update_params_psa_burn(params_start, psa_sample)
    params=copy(params_start)
    
    #calibrated parameters
    params.beta[1:3] .= fill(psa_sample.beta1, 3)
    params.beta[4] = psa_sample.beta2
    params.beta[5] = psa_sample.beta3
    params.beta[6] = psa_sample.beta4
    params.beta[7:17] .= fill(psa_sample.beta5, 11)

    params.f[1:3] .= fill(psa_sample.f1, 3)
    params.f[4] = psa_sample.f2
    params.f[5] = psa_sample.f3
    params.f[6] = psa_sample.f4
    params.f[7:17] .= fill(psa_sample.f5, 11)
    params.alpha .= fill(psa_sample.alpha, 17)

    # For these parameters, check if they're scalars or arrays
    params.gamma .= fill(psa_sample.gamma, 17) 
    params.r_a .= fill(psa_sample.r_a, 17)     
    params.r_c .= fill(psa_sample.r_c, 17)      
    params.theta_s .= fill(psa_sample.theta_s, 17)  
    params.theta_a .= fill(psa_sample.theta_a, 17)  
    params.phi .= fill(psa_sample.phi, 17)     
    params.rho .= fill(psa_sample.rho, 17)      

    params.VE[1:3] .= fill(psa_sample.VE_under2, 3)  #
    params.VE[4] = psa_sample.VE_2_4
    params.VE[5:17] .= fill(psa_sample.VE_5_plus, 13)  
    params.alpha_v[1:3] .= fill(2/psa_sample.duration_under2, 3)  
    params.alpha_v[4] = 2/psa_sample.duration_2_4
    params.alpha_v[5:17] .= fill(2/psa_sample.duration_5_plus, 13)  
    params.r_v .= 0.0
    params.l .= 0.0
    
    #incidence
    params.inc_age[1:3] .= psa_sample.inc_age_cat1/100000
    params.inc_age[4] = psa_sample.inc_age_cat2/100000
    params.inc_age[5] = psa_sample.inc_age_cat3/100000
    params.inc_age[6] = psa_sample.inc_age_cat4/100000
    params.inc_age[7:17] .= psa_sample.inc_age_cat5/100000
    
    return(params)
end

# full typhoid model with all vaccine-relevant compartments
# post-infection pathways vary depending on V, V2, Vw, or never vaccinated origin
@everywhere function typhoid_model_integrated!(du, u, p, t) 
    ages = 17
    m = 17
    
    # Main compartments (unchanged)
    S = @view u[1:m]                     # Susceptible (unvaccinated)
    V = @view u[m+1:2*m]                 # Vaccinated (partial protection)
    V2 = @view u[2*m+1:3*m]              # Vaccinated (partial protection 2)
    Vw = @view u[3*m+1:4*m]             # Vaccinated (waned protection)
    Is = @view u[4*m+1:5*m]              # Symptomatic infected
    Ia = @view u[5*m+1:6*m]               # Asymptomatic infected
    C = @view u[6*m+1:7*m]              # Carriers
    R = @view u[7*m+1:8*m]               # Recovered
    
    # Is compartment counts
    Is_never = @view u[8*m+1:9*m]        # Is who were never vaccinated (from S)
    Is_V = @view u[9*m+1:10*m]           # Is who came from V compartment
    Is_V2 = @view u[10*m+1:11*m]         # Is who came from V2 compartment  
    Is_Vw = @view u[11*m+1:12*m]         # Is who came from Vw compartment
    
    # Ia compartment counts
    Ia_never = @view u[12*m+1:13*m]      # Ia who were never vaccinated (from S)
    Ia_V = @view u[13*m+1:14*m]          # Ia who came from V compartment
    Ia_V2 = @view u[14*m+1:15*m]         # Ia who came from V2 compartment
    Ia_Vw = @view u[15*m+1:16*m]         # Ia who came from Vw compartment
    
    # C compartment counts  
    C_never = @view u[16*m+1:17*m]       # C who were never vaccinated
    C_V = @view u[17*m+1:18*m]           # C who came from V compartment
    C_V2 = @view u[18*m+1:19*m]          # C who came from V2 compartment
    C_Vw = @view u[19*m+1:20*m]          # C who came from Vw compartment
    
    # R compartment counts
    R_never = @view u[20*m+1:21*m]       # R who were never vaccinated
    R_V = @view u[21*m+1:22*m]           # R who came from V compartment  
    R_V2 = @view u[22*m+1:23*m]          # R who came from V2 compartment
    R_Vw = @view u[23*m+1:24*m]          # R who came from Vw compartment
    
    # cases
    cases_vaxx = @view u[24*m+1:25*m]    # Cases from previously vaccinated
    cases_control = @view u[25*m+1:26*m] # Cases from never vaccinated

    fill!(du, 0.0)
    
    dS = @view du[1:m]                    
    dV = @view du[m+1:2*m]                
    dV2 = @view du[2*m+1:3*m]             
    dVw = @view du[3*m+1:4*m]             
    dIs = @view du[4*m+1:5*m]             
    dIa = @view du[5*m+1:6*m]              
    dC = @view du[6*m+1:7*m]              
    dR = @view du[7*m+1:8*m]               
    
    dIs_never = @view du[8*m+1:9*m]
    dIs_V = @view du[9*m+1:10*m]
    dIs_V2 = @view du[10*m+1:11*m]
    dIs_Vw = @view du[11*m+1:12*m]
    
    dIa_never = @view du[12*m+1:13*m]
    dIa_V = @view du[13*m+1:14*m]
    dIa_V2 = @view du[14*m+1:15*m]
    dIa_Vw = @view du[15*m+1:16*m]
    
    dC_never = @view du[16*m+1:17*m]
    dC_V = @view du[17*m+1:18*m]
    dC_V2 = @view du[18*m+1:19*m]
    dC_Vw = @view du[19*m+1:20*m]
    
    dR_never = @view du[20*m+1:21*m]
    dR_V = @view du[21*m+1:22*m]
    dR_V2 = @view du[22*m+1:23*m]
    dR_Vw = @view du[23*m+1:24*m]
    
    dcases_vaxx = @view du[24*m+1:25*m]
    dcases_control = @view du[25*m+1:26*m]
    
    # 1. DEMOGRAPHICS - AGING & MORTALITY

    ## standard aging function for base compartments
    function apply_aging!(du_comp, u_comp, p, N, is_S=false)
        if is_S
            du_comp .+= p.Bi_burn .* [N; zeros(ages-1)]
        end
        du_comp .+= (1 .- p.pu_in_burn) .* p.a_in .* [0; circshift(u_comp, 1)[2:end]]
        du_comp .-= p.a_out .* (1 .- p.pu_burn) .* u_comp .+ p.u_burn .* u_comp
    end

    total_Is = sum(Is_never + Is_V + Is_V2 + Is_Vw)
    total_Ia = sum(Ia_never + Ia_V + Ia_V2 + Ia_Vw)
    total_C = sum(C_never + C_V + C_V2 + C_Vw)
    total_R =  sum(R_never + R_V + R_V2 + R_Vw)
    
    N = sum(S + V + V2 + Vw) + total_Is + total_Ia + total_C + total_R 
       
    apply_aging!(dS, S, p, N, true)  # S gets births
    apply_aging!(dV, V, p, N)
    apply_aging!(dV2, V2, p, N)
    apply_aging!(dVw, Vw, p, N)
    
    ## aging for population counts
    aging_factor_in = (1 .- p.pu_in_burn) .* p.a_in
    aging_factor_out = p.a_out .* (1 .- p.pu_burn) .+ p.u_burn
    
    dIs_never .+= aging_factor_in .* [0; circshift(Is_never, 1)[2:end]] .- aging_factor_out .* Is_never
    dIs_V .+= aging_factor_in .* [0; circshift(Is_V, 1)[2:end]] .- aging_factor_out .* Is_V
    dIs_V2 .+= aging_factor_in .* [0; circshift(Is_V2, 1)[2:end]] .- aging_factor_out .* Is_V2
    dIs_Vw .+= aging_factor_in .* [0; circshift(Is_Vw, 1)[2:end]] .- aging_factor_out .* Is_Vw
    
    dIa_never .+= aging_factor_in .* [0; circshift(Ia_never, 1)[2:end]] .- aging_factor_out .* Ia_never
    dIa_V .+= aging_factor_in .* [0; circshift(Ia_V, 1)[2:end]] .- aging_factor_out .* Ia_V
    dIa_V2 .+= aging_factor_in .* [0; circshift(Ia_V2, 1)[2:end]] .- aging_factor_out .* Ia_V2
    dIa_Vw .+= aging_factor_in .* [0; circshift(Ia_Vw, 1)[2:end]] .- aging_factor_out .* Ia_Vw
    
    dC_never .+= aging_factor_in .* [0; circshift(C_never, 1)[2:end]] .- aging_factor_out .* C_never
    dC_V .+= aging_factor_in .* [0; circshift(C_V, 1)[2:end]] .- aging_factor_out .* C_V
    dC_V2 .+= aging_factor_in .* [0; circshift(C_V2, 1)[2:end]] .- aging_factor_out .* C_V2
    dC_Vw .+= aging_factor_in .* [0; circshift(C_Vw, 1)[2:end]] .- aging_factor_out .* C_Vw
    
    dR_never .+= aging_factor_in .* [0; circshift(R_never, 1)[2:end]] .- aging_factor_out .* R_never
    dR_V .+= aging_factor_in .* [0; circshift(R_V, 1)[2:end]] .- aging_factor_out .* R_V
    dR_V2 .+= aging_factor_in .* [0; circshift(R_V2, 1)[2:end]] .- aging_factor_out .* R_V2
    dR_Vw .+= aging_factor_in .* [0; circshift(R_Vw, 1)[2:end]] .- aging_factor_out .* R_Vw

    # 2. DISEASE TRANSMISSION
    
    Is_vec = fill(total_Is, ages) / N
    C_vec = fill(total_C, ages) / N
    Ia_never_vec = fill(sum(Ia_never)) / N
    Ia_V_vec = fill(sum(Ia_V)) / N
    Ia_V2_vec = fill(sum(Ia_V2)) / N
    Ia_Vw_vec = fill(sum(Ia_Vw)) / N

    #effective_Ia = ((Ia_never) .+                           # Never vaccinated (normal infectiousness)
        #(Ia_V) .* p.r_v .+                     # V-origin (reduced infectiousness)
        ##(Ia_V2) .* p.r_v .+                    # V2-origin (reduced infectiousness)  
       # (Ia_Vw)) ./ N                               # Vw-origin (normal infectiousness)
    #println(effective_Ia)
    
    effective_Is = Is_vec  
    effective_C = C_vec    
    effective_Ia_never = Ia_never_vec
    effective_Ia_V = Ia_V_vec .* p.r_v
    effective_Ia_V2 = Ia_V2_vec .* p.r_v
    effective_Ia_Vw = Ia_Vw_vec 

    FOI = effective_Is .+ p.r_a .* effective_Ia_never .+ p.r_a .* effective_Ia_V .+ p.r_a .* effective_Ia_V2 .+ p.r_a .* effective_Ia_Vw .+ p.r_c .* effective_C 

    ## infection pathways
    new_Is_from_S = p.f .* p.beta .* S .* FOI
    new_Ia_from_S = (1 .- p.f) .* p.beta .* S .* FOI
    
    new_Is_from_V = zeros(ages)  # No symptomatic infections from V
    new_Ia_from_V = p.beta .* p.l .* V .* FOI

    new_Is_from_V2 = zeros(ages)  # No symptomatic infections from V2
    new_Ia_from_V2 = p.beta .* p.l .* V2 .* FOI 

    new_Is_from_Vw = p.f .* p.beta .* Vw .* FOI
    new_Ia_from_Vw = (1 .- p.f) .* p.beta .* Vw .* FOI
    
    dIs_never .+= new_Is_from_S
    dIs_V .+= new_Is_from_V  # zero
    dIs_V2 .+= new_Is_from_V2  # zero
    dIs_Vw .+= new_Is_from_Vw
    
    dIa_never .+= new_Ia_from_S
    dIa_V .+= new_Ia_from_V
    dIa_V2 .+= new_Ia_from_V2
    dIa_Vw .+= new_Ia_from_Vw
    
    #total_new_Is = new_Is_from_S + new_Is_from_Vw
    #total_new_Ia = new_Ia_from_S + new_Ia_from_V + new_Ia_from_V2 + new_Ia_from_Vw
    
    dS .-= new_Is_from_S .+ new_Ia_from_S
    dV .-= new_Ia_from_V
    dV2 .-= new_Ia_from_V2
    dVw .-= new_Is_from_Vw .+ new_Ia_from_Vw
     
    ### case tracking
    dcases_control .+= new_Is_from_S
    dcases_vaxx .+= new_Is_from_Vw
     
    ## recovery pathways
    gamma_factor = (1 .- p.rho) .* p.gamma

    ### never vaccinated Is either successfully recover to R or go back to S
    recover_Is_never_to_R = gamma_factor .* p.theta_s .* Is_never
    recover_Is_never_to_S = gamma_factor .* (1 .- p.theta_s) .* Is_never
    
    ### Vw-origin Is either successfully recover R or go back to Vw
    recover_Is_Vw_to_R = gamma_factor .* p.theta_s .* Is_Vw
    recover_Is_Vw_to_Vw = gamma_factor .* (1 .- p.theta_s) .* Is_Vw

    ### never vaccinated Ia either successfully recover to R or go back to S
    recover_Ia_never_to_R = gamma_factor .* p.theta_a .* Ia_never
    recover_Ia_never_to_S = gamma_factor .* (1 .- p.theta_a) .* Ia_never

    ### V-origin Ia either successfully recover to R or go back to V (OR go to Vw)
    recover_Ia_V_to_R = gamma_factor .* p.theta_a .* Ia_V
    recover_Ia_V_to_V = gamma_factor .* (1 .- p.theta_a) .* Ia_V
    #recover_Ia_V_to_Vw = gamma_factor .* (1 .- p.theta_a) .* Ia_V

    ### V2-origin Ia either successfully recover to R or go back to V2 (OR go to Vw)
    recover_Ia_V2_to_R = gamma_factor .* p.theta_a .* Ia_V2
    recover_Ia_V2_to_V2 = gamma_factor .* (1 .- p.theta_a) .* Ia_V2  
    #recover_Ia_V2_to_Vw = gamma_factor .* (1 .- p.theta_a) .* Ia_V2  

    ### Vw-origin Ia either successfully recover to R or go back to Vw
    recover_Ia_Vw_to_R = gamma_factor .* p.theta_a .* Ia_Vw
    recover_Ia_Vw_to_Vw = gamma_factor .* (1 .- p.theta_a) .* Ia_Vw
    
    ### carrier formation from Is
    become_carrier_from_Is_never = p.rho .* p.gamma .* Is_never
    become_carrier_from_Is_Vw = p.rho .* p.gamma .* Is_Vw

    ### carrier formation from Ia
    become_carrier_from_Ia_never = p.rho .* p.gamma .* Ia_never
    become_carrier_from_Ia_V = p.rho .* p.gamma .* Ia_V
    become_carrier_from_Ia_V2 = p.rho .* p.gamma .* Ia_V2
    become_carrier_from_Ia_Vw = p.rho .* p.gamma .* Ia_Vw
   
    ### apply Is recovery flows
    dIs_never .-= recover_Is_never_to_S .+ recover_Is_never_to_R .+ become_carrier_from_Is_never .+ p.u_ic .* Is_never
    dIs_Vw .-= recover_Is_Vw_to_Vw .+ recover_Is_Vw_to_R .+ become_carrier_from_Is_Vw .+ p.u_ic .* Is_Vw
    
    ### apply Ia recovery flows
    dIa_never .-= recover_Ia_never_to_S .+ recover_Ia_never_to_R .+ become_carrier_from_Ia_never
    dIa_V .-= recover_Ia_V_to_V .+ recover_Ia_V_to_R .+ become_carrier_from_Ia_V #recover_Ia_V_to_Vw .+ recover_Ia_V_to_R .+ become_carrier_from_Ia_V #
    dIa_V2 .-= recover_Ia_V2_to_V2 .+ recover_Ia_V2_to_R .+ become_carrier_from_Ia_V2 #recover_Ia_V2_to_Vw .+ recover_Ia_V2_to_R .+ become_carrier_from_Ia_V2
    dIa_Vw .-= recover_Ia_Vw_to_Vw .+ recover_Ia_Vw_to_R .+ become_carrier_from_Ia_Vw
    
    ### apply carrier flows
    dC_never .+= become_carrier_from_Is_never .+ become_carrier_from_Ia_never
    dC_V .+= become_carrier_from_Ia_V 
    dC_V2 .+= become_carrier_from_Ia_V2  
    dC_Vw .+= become_carrier_from_Is_Vw .+ become_carrier_from_Ia_Vw
    
    ### recovery from carriers
    recover_C_to_R_never = p.phi .* C_never
    recover_C_to_R_V = p.phi .* C_V
    recover_C_to_R_V2 = p.phi .* C_V2
    recover_C_to_R_Vw = p.phi .* C_Vw
    
    dC_never .-= recover_C_to_R_never
    dC_V .-= recover_C_to_R_V
    dC_V2 .-= recover_C_to_R_V2
    dC_Vw .-= recover_C_to_R_Vw
    
    dR_never .+= recover_Is_never_to_R .+ recover_Ia_never_to_R .+ recover_C_to_R_never
    dR_V .+= recover_Ia_V_to_R .+ recover_C_to_R_V
    dR_V2 .+= recover_Ia_V2_to_R .+ recover_C_to_R_V2
    dR_Vw .+= recover_Is_Vw_to_R .+ recover_Ia_Vw_to_R .+ recover_C_to_R_Vw

    ## waning immunity pathways

    ### waning from R to S or Vw (natural immunity waning)
    waning_R_never_to_S = p.alpha .* R_never
    waning_R_V_to_Vw = p.alpha .* R_V
    waning_R_V2_to_Vw = p.alpha .* R_V2  
    waning_R_Vw_to_Vw = p.alpha .* R_Vw
    
    dR_never .-= waning_R_never_to_S
    dR_V .-= waning_R_V_to_Vw
    dR_V2 .-= waning_R_V2_to_Vw
    dR_Vw .-= waning_R_Vw_to_Vw
    
    # waning from V to V2 to Vw (vaccine waning)
    waning_V_to_V2 = p.alpha_v .* V
    waning_V2_to_Vw = p.alpha_v .* V2
    
    # update main compartments with recovery flows
    dS .+= recover_Is_never_to_S .+ recover_Ia_never_to_S .+ waning_R_never_to_S
    dV .+=  recover_Ia_V_to_V.- waning_V_to_V2 #dV .-= waning_V_to_V2 
    dV2 .+= waning_V_to_V2 .+ recover_Ia_V2_to_V2 .- waning_V2_to_Vw #waning_V_to_V2 .- waning_V2_to_Vw #
    dVw .+= recover_Is_Vw_to_Vw .+ recover_Ia_Vw_to_Vw .+ waning_R_V_to_Vw .+ waning_R_V2_to_Vw .+ waning_R_Vw_to_Vw .+ waning_V2_to_Vw #recover_Is_Vw_to_Vw .+ recover_Ia_Vw_to_Vw .+ recover_Ia_V_to_Vw .+ recover_Ia_V2_to_Vw .+ waning_R_V_to_Vw .+ waning_R_V2_to_Vw .+ waning_R_Vw_to_Vw .+ waning_V2_to_Vw #
    dIs .= dIs_never .+ dIs_V .+ dIs_V2 .+ dIs_Vw
    dIa .= dIa_never .+ dIa_V .+ dIa_V2 .+ dIa_Vw  
    dC .= dC_never .+ dC_V .+ dC_V2 .+ dC_Vw
    dR .= dR_never .+ dR_V .+ dR_V2 .+ dR_Vw

end

# gen initial population sizes for full typhoid model 
@everywhere function gen_initial_integrated!(params, current_S, current_Is, current_Ia, current_C, current_R)
    vaxx_covg_trial = 155841/(155841+155448)
    ages = 17
    # Initialize susceptible compartments with proper VE distribution
    V_0 = current_S .* vaxx_covg_trial .* params.VE            # Actively protected vaccinated
    Vw_0 = current_S .* vaxx_covg_trial .* (1 .- params.VE)    # Partially protected vaccinated
    S_0 = current_S .- V_0 .- Vw_0                             # Truly unvaccinated
    
    # Initial disease compartments (same as before)
    V2_0 = zeros(ages)
    Is_0 = current_Is
    Ia_0 = current_Ia
    C_0 = current_C
    R_0 = current_R
    
    # Initial proportion variables - all start at 0 since we're initializing from pre-vaccine state
    Is_never = current_Is
    Is_V = zeros(ages)
    Is_V2 = zeros(ages)
    Is_Vw = zeros(ages)

    Ia_never = current_Ia
    Ia_V = zeros(ages)
    Ia_V2 = zeros(ages)
    Ia_Vw = zeros(ages)

    C_never = current_C
    C_V = zeros(ages)
    C_V2 = zeros(ages)
    C_Vw = zeros(ages)

    R_never = current_R
    R_V = zeros(ages)
    R_V2 = zeros(ages)
    R_Vw = zeros(ages)

    # Case tracking
    cases_vaxx_0 = zeros(ages)
    cases_control_0 = zeros(ages)

    # Combine all initial values
    initial = vcat(
        S_0, V_0, V2_0, Vw_0,                          # Susceptible compartments
        Is_0, Ia_0, C_0, R_0,                    # Disease compartments
        Is_never, Is_V, Is_V2, Is_Vw,  # Is proportions
        Ia_never, Ia_V, Ia_V2, Ia_Vw,  # Ia proportions
        C_never, C_V, C_V2, C_Vw,   # C proportions
        R_never, R_V, R_V2, R_Vw,                   # R proportion
        cases_vaxx_0, cases_control_0           # Case tracking
    )
    
    return initial
end

# process compartmental pop size output following transmission
@everywhere function clean_output_integrated!(sol)
    ages = 17
    m = 17
    
    time_points = sol.t

    S_values = [u[1:m] for u in sol.u]                   
    V_values = [u[m+1:2*m] for u in sol.u]             
    V2_values = [u[2*m+1:3*m] for u in sol.u]             
    Vw_values = [u[3*m+1:4*m] for u in sol.u]             
    Is_values = [u[4*m+1:5*m] for u in sol.u]              
    Ia_values = [u[5*m+1:6*m] for u in sol.u]               
    C_values = [u[6*m+1:7*m] for u in sol.u]          
    R_values = [u[7*m+1:8*m] for u in sol.u]
    
    Is_never_values = [u[8*m+1:9*m] for u in sol.u]
    Is_V_values = [u[9*m+1:10*m] for u in sol.u]
    Is_V2_values = [u[10*m+1:11*m] for u in sol.u]
    Is_Vw_values = [u[11*m+1:12*m] for u in sol.u]
    
    # Ia compartment counts
    Ia_never_values = [u[12*m+1:13*m] for u in sol.u]
    Ia_V_values = [u[13*m+1:14*m] for u in sol.u]
    Ia_V2_values = [u[14*m+1:15*m] for u in sol.u]
    Ia_Vw_values = [u[15*m+1:16*m] for u in sol.u]
    
    # C compartment counts  
    C_never_values = [u[16*m+1:17*m] for u in sol.u]
    C_V_values = [u[17*m+1:18*m] for u in sol.u]
    C_V2_values = [u[18*m+1:19*m] for u in sol.u]
    C_Vw_values = [u[19*m+1:20*m] for u in sol.u]
    
    # R compartment counts
    R_never_values = [u[20*m+1:21*m] for u in sol.u]
    R_V_values = [u[21*m+1:22*m] for u in sol.u]
    R_V2_values = [u[22*m+1:23*m] for u in sol.u]
    R_Vw_values = [u[23*m+1:24*m] for u in sol.u]
    
    # cases
    cases_vaxx_values = [u[24*m+1:25*m] for u in sol.u]
    cases_control_values = [u[25*m+1:26*m] for u in sol.u]

    model_df = DataFrame(time = time_points)
    
    for age in 1:ages
        age_label = string(age)
        
        # Add columns for each compartment and age
        model_df[!, "S_$age_label"] = [S[age] for S in S_values]
        model_df[!, "V_$age_label"] = [V[age] for V in V_values]
        model_df[!, "V2_$age_label"] = [V2[age] for V2 in V2_values]
        model_df[!, "Vw_$age_label"] = [Vw[age] for Vw in Vw_values]
        model_df[!, "Is_$age_label"] = [Is[age] for Is in Is_values]
        model_df[!, "Ia_$age_label"] = [Ia[age] for Ia in Ia_values]
        model_df[!, "C_$age_label"] = [C[age] for C in C_values]
        model_df[!, "R_$age_label"] = [R[age] for R in R_values]
        
        # We can also save the proportions for debugging or analysis
        model_df[!, "Is_never_$age_label"] = [Is_never[age] for Is_never in Is_never_values]
        model_df[!, "Is_V_$age_label"] = [Is_V[age] for Is_V in Is_V_values]
        model_df[!, "Is_V2_$age_label"] = [Is_V2[age] for Is_V2 in Is_V2_values]
        model_df[!, "Is_Vw_$age_label"] = [Is_Vw[age] for Is_Vw in Is_Vw_values]
    
        model_df[!, "Ia_never_$age_label"] = [Ia_never[age] for Ia_never in Ia_never_values]
        model_df[!, "Ia_V_$age_label"] = [Ia_V[age] for Ia_V in Ia_V_values]
        model_df[!, "Ia_V2_$age_label"] = [Ia_V2[age] for Ia_V2 in Ia_V2_values]
        model_df[!, "Ia_Vw_$age_label"] = [Ia_Vw[age] for Ia_Vw in Ia_Vw_values]
    
        model_df[!, "C_never_$age_label"] = [C_never[age] for C_never in C_never_values]
        model_df[!, "C_V_$age_label"] = [C_V[age] for C_V in C_V_values]
        model_df[!, "C_V2_$age_label"] = [C_V2[age] for C_V2 in C_V2_values]
        model_df[!, "C_Vw_$age_label"] = [C_Vw[age] for C_Vw in C_Vw_values]
    
        model_df[!, "R_never_$age_label"] = [R_never[age] for R_never in R_never_values]
        model_df[!, "R_V_$age_label"] = [R_V[age] for R_V in R_V_values]
        model_df[!, "R_V2_$age_label"] = [R_V2[age] for R_V2 in R_V2_values]
        model_df[!, "R_Vw_$age_label"] = [R_Vw[age] for R_Vw in R_Vw_values]
    
        model_df[!, "cases_vaxx_$age_label"] = [cases_vaxx[age] for cases_vaxx in cases_vaxx_values]
        model_df[!, "cases_control_$age_label"] = [cases_control[age] for cases_control in cases_control_values]

    end
    
    return model_df
end

# calculate incidence among controls 
@everywhere function calc_inc_control_integrated!(sol_archetype)
    all_cols = names(sol_archetype)
    
    # identify case columns for unvaccinated only 
    cases_control_cols = [col for col in all_cols if startswith(String(col), "cases_control_")]
    
    # age category definitions for case columns
    cases_agecat1_cols = [:cases_control_1, :cases_control_2, :cases_control_3]
    cases_agecat2_cols = [:cases_control_4]
    cases_agecat3_cols = [:cases_control_5]
    cases_agecat4_cols = [:cases_control_6]
    cases_agecat5_cols = [Symbol("cases_control_$i") for i in 7:17]
    
    # calculate new cases in unvaccinated (control group) by age category
    total_cases_archetype_agecat1 = sum(sol_archetype[end, col] - sol_archetype[end-1, col] for col in cases_agecat1_cols)
    total_cases_archetype_agecat2 = sum(sol_archetype[end, col] - sol_archetype[end-1, col] for col in cases_agecat2_cols)
    total_cases_archetype_agecat3 = sum(sol_archetype[end, col] - sol_archetype[end-1, col] for col in cases_agecat3_cols)
    total_cases_archetype_agecat4 = sum(sol_archetype[end, col] - sol_archetype[end-1, col] for col in cases_agecat4_cols)
    total_cases_archetype_agecat5 = sum(sol_archetype[end, col] - sol_archetype[end-1, col] for col in cases_agecat5_cols)
    
    # calculate unvaccinated population by age category
    total_pop_archetype_agecat1 = 0
    total_pop_archetype_agecat2 = 0
    total_pop_archetype_agecat3 = 0
    total_pop_archetype_agecat4 = 0
    total_pop_archetype_agecat5 = 0
    
    # age category 1 (1-3)
    for i in 1:3
        # S compartment
        if Symbol("S_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat1 += sol_archetype[end, Symbol("S_$i")]
        end
        
        # disease compartments (only unvaccinated portion)
        if Symbol("Is_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat1 += sol_archetype[end, Symbol("Is_never_$i")] 
        end
        
        if Symbol("Ia_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat1 += sol_archetype[end, Symbol("Ia_never_$i")] 
        end
        
        if Symbol("C_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat1 += sol_archetype[end, Symbol("C_never_$i")] 
        end
        
        if Symbol("R_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat1 += sol_archetype[end, Symbol("R_never_$i")]
        end
    end
    
    # age category 2 (4)
    i = 4
    # S compartment 
    if Symbol("S_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat2 += sol_archetype[end, Symbol("S_$i")] 
    end
    
    # disease compartments (only unvaccinated portion)
    if Symbol("Is_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat2 += sol_archetype[end, Symbol("Is_never_$i")] 
    end
    
    if Symbol("Ia_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat2 += sol_archetype[end, Symbol("Ia_never_$i")] 
    end
    
    if Symbol("C_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat2 += sol_archetype[end, Symbol("C_never_$i")] 
    end
    
    if Symbol("R_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat2 += sol_archetype[end, Symbol("R_never_$i")] 
    end
    
    # age category 3 (5)
    i = 5
    # S compartment 
    if Symbol("S_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat3 += sol_archetype[end, Symbol("S_$i")] 
    end
    
    # disease compartments (only unvaccinated portion)
    if Symbol("Is_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat3 += sol_archetype[end, Symbol("Is_never_$i")] 
    end
    
    if Symbol("Ia_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat3 += sol_archetype[end, Symbol("Ia_never_$i")]
    end
    
    if Symbol("C_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat3 += sol_archetype[end, Symbol("C_never_$i")] 
    end
    
    if Symbol("R_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat3 += sol_archetype[end, Symbol("R_never_$i")] 
    end
    
    # age category 4 (6)
    i = 6
    # S compartment 
    if Symbol("S_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat4 += sol_archetype[end, Symbol("S_$i")] 
    end    
    
    # disease compartments (only unvaccinated portion)
    if Symbol("Is_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat4 += sol_archetype[end, Symbol("Is_never_$i")] 
    end
    
    if Symbol("Ia_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat4 += sol_archetype[end, Symbol("Ia_never_$i")] 
    end
    
    if Symbol("C_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat4 += sol_archetype[end, Symbol("C_never_$i")]
    end
    
    if Symbol("R_never_$i") in propertynames(sol_archetype) 
        total_pop_archetype_agecat4 += sol_archetype[end, Symbol("R_never_$i")]
    end
    
    # age category 5 (7-17)
    for i in 7:17
        # S compartment (all unvaccinated)
        if Symbol("S_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat5 += sol_archetype[end, Symbol("S_$i")] 
        end    
        
        # Disease compartments (only unvaccinated portion)
        if Symbol("Is_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat5 += sol_archetype[end, Symbol("Is_never_$i")] 
        end
        
        if Symbol("Ia_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat5 += sol_archetype[end, Symbol("Ia_never_$i")] 
        end
        
        if Symbol("C_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat5 += sol_archetype[end, Symbol("C_never_$i")] 
        end
        
        if Symbol("R_never_$i") in propertynames(sol_archetype) 
            total_pop_archetype_agecat5 += sol_archetype[end, Symbol("R_never_$i")] 
        end
    end

    # calculate total incidence in control group
    total_cases_archetype = total_cases_archetype_agecat1 + total_cases_archetype_agecat2 + 
                           total_cases_archetype_agecat3 + total_cases_archetype_agecat4 + 
                           total_cases_archetype_agecat5
    
    total_N_archetype = total_pop_archetype_agecat1 + total_pop_archetype_agecat2 + 
                       total_pop_archetype_agecat3 + total_pop_archetype_agecat4 + 
                       total_pop_archetype_agecat5
    
    # calculate incidence per 100,000 unvaccinated population
    inc_archetype = (total_cases_archetype * 100000) / total_N_archetype
    
    return inc_archetype
end

# instead return incidence rates among controls for the two target age groups
@everywhere function calc_inc_control_integrated_VE!(sol_archetype,t_start,t_end)
    all_cols = names(sol_archetype)
    
    # identify case columns for unvaccinated only 
    cases_control_cols = [col for col in all_cols if startswith(String(col), "cases_control_")]
    
    # age category definitions for case columns
    cases_agecat1_cols = [:cases_control_1, :cases_control_2, :cases_control_3]
    cases_agecat2_cols = [:cases_control_4]
    cases_agecat3_cols = [Symbol("cases_control_$i") for i in 5:17]
    
    # calculate new cases in unvaccinated (control group) by age category
    total_cases_archetype_agecat1 = sum(sol_archetype[t_end, col] - sol_archetype[t_start, col] for col in cases_agecat1_cols) 
    total_cases_archetype_agecat2 = sum(sol_archetype[t_end, col] - sol_archetype[t_start, col] for col in cases_agecat2_cols)
    total_cases_archetype_agecat3 = sum(sol_archetype[t_end, col] - sol_archetype[t_start, col] for col in cases_agecat3_cols)
        
    # calculate unvaccinated population by age category
    total_person_time_archetype_agecat1 = 0
    total_person_time_archetype_agecat2 = 0
    total_person_time_archetype_agecat3 = 0

    # age category 1 (1-3)
    for i in 1:3
        # S compartment
        if Symbol("S_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("S_$i")][t] for t in t_start:t_end)
        end
        
        # disease compartments (only unvaccinated portion)
        if Symbol("Is_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("Is_never_$i")][t] for t in t_start:t_end)  #sol_archetype[end, Symbol("Is_never_$i")] 
        end
        
        if Symbol("Ia_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("Ia_never_$i")][t] for t in t_start:t_end)  #sol_archetype[end, Symbol("Ia_never_$i")] 
        end
        
        if Symbol("C_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("C_never_$i")][t] for t in t_start:t_end)
        end
        
        if Symbol("R_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("R_never_$i")][t] for t in t_start:t_end)
        end
    end
    

# age category 3 (7-17)
    for i in 5:17
        # S compartment (all unvaccinated)
        if Symbol("S_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("S_$i")][t] for t in t_start:t_end) 
        end    
        
        # disease compartments (only unvaccinated portion)
        if Symbol("Is_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("Is_never_$i")][t] for t in t_start:t_end)
        end
        
        if Symbol("Ia_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("Ia_never_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("C_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("C_never_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("R_never_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("R_never_$i")][t] for t in t_start:t_end)
        end
    end
    
    # calculate incidence per 100,000 unvaccinated population
    inc_archetype_agecat1 = (total_cases_archetype_agecat1)/(total_person_time_archetype_agecat1)
    inc_archetype_agecat3 = (total_cases_archetype_agecat3)/(total_person_time_archetype_agecat3)
    
    return inc_archetype_agecat1, inc_archetype_agecat3 

end

# instead return incidence rates among the vaccinated for the two target age groups
@everywhere function calc_inc_vaxx_integrated_VE!(sol_archetype,t_start,t_end)
    all_cols = names(sol_archetype)
    
    # identify case columns for vaccinated only 
    cases_vaxx_cols = [col for col in all_cols if startswith(String(col), "cases_vaxx_")]
    
    # age category definitions for case columns
    cases_agecat1_cols = [:cases_vaxx_1, :cases_vaxx_2, :cases_vaxx_3]
    cases_agecat2_cols = [:cases_vaxx_4]
    cases_agecat3_cols = [Symbol("cases_vaxx_$i") for i in 5:17]
    
    # calculate new cases in vaxx group by age category
    total_cases_archetype_agecat1 = sum(sol_archetype[t_end, col] - sol_archetype[t_start, col] for col in cases_agecat1_cols) 
    total_cases_archetype_agecat2 = sum(sol_archetype[t_end, col] - sol_archetype[t_start, col] for col in cases_agecat2_cols)
    total_cases_archetype_agecat3 = sum(sol_archetype[t_end, col] - sol_archetype[t_start, col] for col in cases_agecat3_cols)
        
    # calculate unvaccinated population by age category
    total_person_time_archetype_agecat1 = 0
    total_person_time_archetype_agecat2 = 0
    total_person_time_archetype_agecat3 = 0

    # age category 1 (1-3)
    for i in 1:3
        if Symbol("V_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("V_$i")][t] for t in t_start:t_end) 
        end

        if Symbol("V2_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("V2_$i")][t] for t in t_start:t_end)
        end
        
        if Symbol("Vw_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("Vw_$i")][t] for t in t_start:t_end) 
        end
        
        # disease compartments (only unvaccinated portion)
        if Symbol("Is_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("Is_Vw_$i")][t] for t in t_start:t_end) 
        end

        
        if Symbol("Ia_V_$i") in propertynames(sol_archetype) && Symbol("Ia_V2_$i") in propertynames(sol_archetype) && Symbol("Ia_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("Ia_V_$i")][t] + sol_archetype[:, Symbol("Ia_V2_$i")][t] + sol_archetype[:, Symbol("Ia_Vw_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("C_V_$i") in propertynames(sol_archetype) && Symbol("C_V2_$i") in propertynames(sol_archetype) && Symbol("C_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("C_V_$i")][t] + sol_archetype[:, Symbol("C_V2_$i")][t] + sol_archetype[:, Symbol("C_Vw_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("R_V_$i") in propertynames(sol_archetype) && Symbol("R_V2_$i") in propertynames(sol_archetype) && Symbol("R_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat1 += sum(sol_archetype[:, Symbol("R_V_$i")][t] + sol_archetype[:, Symbol("R_V2_$i")][t] + sol_archetype[:, Symbol("R_Vw_$i")][t] for t in t_start:t_end) 
        end
    end


# age category 3 (7-17)
    for i in 5:17
        if Symbol("V_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("V_$i")][t] for t in t_start:t_end) 
        end

        if Symbol("V2_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("V2_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("Vw_$i") in propertynames(sol_archetype) 
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("Vw_$i")][t] for t in t_start:t_end) 
        end
        
        # disease compartments (only unvaccinated portion)
        if Symbol("Is_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("Is_Vw_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("Ia_V_$i") in propertynames(sol_archetype) && Symbol("Ia_V2_$i") in propertynames(sol_archetype) && Symbol("Ia_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("Ia_V_$i")][t] + sol_archetype[:, Symbol("Ia_V2_$i")][t] + sol_archetype[:, Symbol("Ia_Vw_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("C_V_$i") in propertynames(sol_archetype) && Symbol("C_V2_$i") in propertynames(sol_archetype) && Symbol("C_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("C_V_$i")][t] + sol_archetype[:, Symbol("C_V2_$i")][t] + sol_archetype[:, Symbol("C_Vw_$i")][t] for t in t_start:t_end) 
        end
        
        if Symbol("R_V_$i") in propertynames(sol_archetype) && Symbol("R_V2_$i") in propertynames(sol_archetype) && Symbol("R_Vw_$i") in propertynames(sol_archetype)
            total_person_time_archetype_agecat3 += sum(sol_archetype[:, Symbol("R_V_$i")][t] + sol_archetype[:, Symbol("R_V2_$i")][t] + sol_archetype[:, Symbol("R_Vw_$i")][t] for t in t_start:t_end)
        end
    end

 # calculate incidence per 100,000 vaccinated population
   inc_archetype_agecat1 = (total_cases_archetype_agecat1)/(total_person_time_archetype_agecat1)
   inc_archetype_agecat3 = (total_cases_archetype_agecat3)/(total_person_time_archetype_agecat3)
            
   return inc_archetype_agecat1, inc_archetype_agecat3 
    
end

# GOF calculation - return objective value
@everywhere function incidence_gof_calibration!(x, params, target_inc_control_final, target_VE_final, target_VE_post, target_VE_post2,
        current_S, current_Is, current_Ia, current_C, current_R)
    ages=17
    m=17
 
    params.l.= x[1]
    params.r_v.= x[2]
    params.alpha_v[1:3] .= x[3]
    params.alpha_v[5:17] .= x[4]
    params.VE[1:3] .= x[5]
    params.VE[5:17] .= x[6]

    # parameter bounds check
    
     bounds_violated = (x[1] < 0.0 || x[1] > 1.0 ||
                      x[2] < 0.0 || x[2] > 1.0 ||
                      x[5] < 0.5 || x[5] > 1.0 ||
                      x[6] < 0.5 || x[6] > 1.0) 


    if bounds_violated
        gof_sum = 100.0*10^9
        return(gof_sum)
    end
    
    initial = gen_initial_integrated!(params, current_S, current_Is, current_Ia, current_C, current_R)
    
    # define time points for VE evaluation
    T_18month = 18  # For final VE and incidence
    ts_18month = range(0, stop=T_18month, step=1)

    T_6yr = 12*6  # For final VE and incidence
    ts_6yr = range(0, stop=T_6yr, step=1)

    # solve for 18-month period (final VE and incidence)
    cal_prob_18m = ODEProblem(typhoid_model_integrated!, initial, (0.0, T_18month), params)
    cal_sol_18m = solve(cal_prob_18m, Tsit5(), save_everystep=false, tstops=ts_18month, saveat=ts_18month)
    cleaned_cal_18m = clean_output_integrated!(cal_sol_18m)

    cal_prob_6y = ODEProblem(typhoid_model_integrated!, initial, (0.0, T_6yr), params)
    cal_sol_6y = solve(cal_prob_6y, Tsit5(), save_everystep=false, tstops=ts_6yr, saveat=ts_6yr)
    cleaned_cal_6y = clean_output_integrated!(cal_sol_6y)
   
    ## control group incidence rate
    cal_inc_control_1y = calc_inc_control_integrated_VE!(cleaned_cal_6y,1,13)
    cal_inc_control_4y = calc_inc_control_integrated_VE!(cleaned_cal_6y,14,50)
    cal_inc_control_6y = calc_inc_control_integrated_VE!(cleaned_cal_6y,51,71)

    ## vaccine group incidence rate
    cal_inc_vaxx_1y = calc_inc_vaxx_integrated_VE!(cleaned_cal_6y,1,13)
    cal_inc_vaxx_4y = calc_inc_vaxx_integrated_VE!(cleaned_cal_6y,14,50)
    cal_inc_vaxx_6y = calc_inc_vaxx_integrated_VE!(cleaned_cal_6y,51,71)

    ## control group incidence rate (aggregated)
    cal_inc_control_18m_aggregated = calc_inc_control_integrated!(cleaned_cal_18m)

    ## final VE
    cal_VE_final = (cal_inc_control_1y .- cal_inc_vaxx_1y) ./ cal_inc_control_1y
    cal_VE_post = (cal_inc_control_4y .- cal_inc_vaxx_4y) ./ cal_inc_control_4y
    cal_VE_post2 = (cal_inc_control_6y .- cal_inc_vaxx_6y) ./ cal_inc_control_6y

    cal_inc_control=cal_inc_control_18m_aggregated
    
    if cal_inc_control > 0 && target_inc_control_final > 0
        gof_incidence = -logpdf(Poisson(cal_inc_control), round(Int, target_inc_control_final))
    else
        gof_incidence = 1e3  # Large penalty for invalid incidence
    end
    
    # final VE fit (weighted squared errors for each VE component, with weights determined by number of ages captured in each age category)
    weights = [1/3, 1/13]
    
    if all(cal_VE_final.>0) && all(cal_VE_final.<1)
        gof_VE_final = sum(weights .* (logit.(cal_VE_final) .- logit.(target_VE_final)).^2)  #sum((logit.(cal_VE_final).- logit.(target_VE_final)).^2) 
    else
       gof_VE_final = 1e3
    end
    
    if all(cal_VE_post.>0) && all(cal_VE_post.<1)
        gof_VE_post = sum(weights .* (logit.(cal_VE_post) .- logit.(target_VE_post)).^2)
    else
       gof_VE_post = 1e3
    end
    
    if all(cal_VE_post2.>0) && all(cal_VE_post2.<1)
        gof_VE_post2 = sum(weights .* (logit.(cal_VE_post2) .- logit.(target_VE_post2)).^2)
    else
       gof_VE_post2 = 1e3
    end

    # both objectives (VE at each time point and final incidence among controls given indirect protection)
    w_incidence = 10.0 / target_inc_control_final
    
    gof_VE_terms = gof_VE_final + gof_VE_post + gof_VE_post2
    
    obj_value = w_incidence*gof_incidence + gof_VE_terms
    

    return Float64(obj_value)
    
end

# full typhoid model (pre-vaccination era)
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
@everywhere function clean_output_full_transmission_calib!(sol) # (sol, t, ages)
    ages=17
    m=17 #ages*3
    
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
    cases_values = [u[9*m+1:10*m] for u in sol.u]

    # Create DataFrame with time
    model_df = DataFrame(time = time_points)
    
    for age in 1:ages
        age_label = string(age)
        
        model_df[!, "S_$age_label"] = [S[age] for S in S_values]
        model_df[!, "Is_$age_label"] = [Is[age] for Is in Is_values]
        model_df[!, "Ia_$age_label"] = [Ia[age] for Ia in Ia_values]
        model_df[!, "C_$age_label"] = [C[age] for C in C_values]
        model_df[!, "R_$age_label"] = [R[age] for R in R_values]
        model_df[!, "V_$age_label"] = [V[age] for V in V_values]
        model_df[!, "Vw_$age_label"] = [Vw[age] for Vw in Vw_values]
        model_df[!, "Iv_$age_label"] = [Iv[age] for Iv in Iv_values]
        model_df[!, "Sv_$age_label"] = [Sv[age] for Sv in Sv_values]
        model_df[!, "cases_$age_label"] = [cases[age] for cases in cases_values]
    end
    
    return(model_df)
end

# setup data and parameters on all workers
@everywhere begin
    params=CSV.read("params_latest_adapted_june14.csv", DataFrame)
    
    calibrated_demographic_params=CSV.read("demo_calib_output.csv", DataFrame)
    #samples = CSV.read("params_psa.csv", DataFrame)
    #samples_archetype3=samples[samples.archetype.=="Very High",:];
    
    psa_samples = CSV.read("params_psa_full_fast.csv", DataFrame)

    params_archetype3=update_params_demographics(params, calibrated_demographic_params, 3)
end

# run distributed computation using pmap to collect results
results_fast = pmap(1:250) do i # change to 251:500, 501:750, and 751:1000 to obtain all samples 
    println("Starting run $i")
    
    sample_use = psa_samples[i, :]
    local_params = update_params_psa_burn(params_archetype3, sample_use)
    
    T = 1200
    initial = gen_initial_burn_in!(local_params, 17)
    prob = ODEProblem(typhoid_model!, initial, (0.0, T), local_params)
    
    # use explicit solver for speed 
    sol = solve(prob, Tsit5(), save_everystep=false, dense=false, saveat=[T-1, T])  # Only save last two points
    
    cleaned_sol = clean_output_full_transmission_calib!(sol)
    
    current_S = collect(select(cleaned_sol, 
        "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7", "S_8", "S_9", 
        "S_10", "S_11", "S_12", "S_13", "S_14", "S_15", "S_16", "S_17")[end,:])
    
    current_Is = collect(select(cleaned_sol,
        "Is_1", "Is_2", "Is_3", "Is_4", "Is_5", "Is_6", "Is_7", "Is_8", "Is_9",
        "Is_10", "Is_11", "Is_12", "Is_13", "Is_14", "Is_15", "Is_16", "Is_17")[end,:])
    
    current_Ia = collect(select(cleaned_sol,
        "Ia_1", "Ia_2", "Ia_3", "Ia_4", "Ia_5", "Ia_6", "Ia_7", "Ia_8", "Ia_9",
        "Ia_10", "Ia_11", "Ia_12", "Ia_13", "Ia_14", "Ia_15", "Ia_16", "Ia_17")[end,:])
    
    current_C = collect(select(cleaned_sol,
        "C_1", "C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "C_9",
        "C_10", "C_11", "C_12", "C_13", "C_14", "C_15", "C_16", "C_17")[end,:])
    
    current_R = collect(select(cleaned_sol,
        "R_1", "R_2", "R_3", "R_4", "R_5", "R_6", "R_7", "R_8", "R_9",
        "R_10", "R_11", "R_12", "R_13", "R_14", "R_15", "R_16", "R_17")[end,:])
    
    current_V = collect(select(cleaned_sol,
        "V_1", "V_2", "V_3", "V_4", "V_5", "V_6", "V_7", "V_8", "V_9",
        "V_10", "V_11", "V_12", "V_13", "V_14", "V_15", "V_16", "V_17")[end,:])

    current_Sv = collect(select(cleaned_sol,
        "Sv_1", "Sv_2", "Sv_3", "Sv_4", "Sv_5", "Sv_6", "Sv_7", "Sv_8", "Sv_9",
        "Sv_10", "Sv_11", "Sv_12", "Sv_13", "Sv_14", "Sv_15", "Sv_16", "Sv_17")[end,:])
   

    # calculate pre-vaccination metrics
    pre_vaxx_cases_all_ages = (collect(select(cleaned_sol,
        "cases_1", "cases_2", "cases_3", "cases_4", "cases_5", "cases_6",
        "cases_7", "cases_8", "cases_9", "cases_10", "cases_11", "cases_12",
        "cases_13", "cases_14", "cases_15", "cases_16", "cases_17")[end,:])) .-
    (collect(select(cleaned_sol,
        "cases_1", "cases_2", "cases_3", "cases_4", "cases_5", "cases_6",
        "cases_7", "cases_8", "cases_9", "cases_10", "cases_11", "cases_12",
        "cases_13", "cases_14", "cases_15", "cases_16", "cases_17")[end-1,:]))
    
    pre_vaxx_cases = [sum(pre_vaxx_cases_all_ages[1:3]);
                     pre_vaxx_cases_all_ages[4];
                     pre_vaxx_cases_all_ages[5];
                     pre_vaxx_cases_all_ages[6];
                     sum(pre_vaxx_cases_all_ages[7:17])]
    
    # calculate population sizes
    all_cols = names(cleaned_sol)
    case_cols = [col for col in all_cols if startswith(String(col), "cases_")]
    exclude_cols = vcat(case_cols, [:time])
    compartment_cols = filter(col -> !(col in exclude_cols), all_cols)
    
    pop_cols_agecat1 = filter(col -> any(endswith(String(col), "_$i") for i in 1:3), compartment_cols)
    pop_cols_agecat2 = filter(col -> endswith(String(col), "_4"), compartment_cols)
    pop_cols_agecat3 = filter(col -> endswith(String(col), "_5"), compartment_cols)
    pop_cols_agecat4 = filter(col -> endswith(String(col), "_6"), compartment_cols)
    pop_cols_agecat5 = filter(col -> any(endswith(String(col), "_$i") for i in 7:17), compartment_cols)
    
    pre_vaxx_pop = [sum(cleaned_sol[end, col] for col in pop_cols_agecat1),
                   sum(cleaned_sol[end, col] for col in pop_cols_agecat2),
                   sum(cleaned_sol[end, col] for col in pop_cols_agecat3),
                   sum(cleaned_sol[end, col] for col in pop_cols_agecat4),
                   sum(cleaned_sol[end, col] for col in pop_cols_agecat5)]
    
    pre_vaxx_inc = (sum(pre_vaxx_cases) * 100000) / sum(pre_vaxx_pop)
    
    # specify targets
    indirect_protection = 0.19
    target_inc = pre_vaxx_inc * (1 - indirect_protection)    
    target_VE_final=[0.81, 0.88] 
    target_VE_post=[0.24, 0.74] 
    target_VE_post2=[0.05, 0.32] 

    # optimized calibration with tighter bounds
    lower_bounds = [0.5, 0.5, 0.1, 0.01, 0.5, 0.5]
    upper_bounds = [1.0, 1.0, 0.5, 0.06, 1.0, 1.0]      
    
result_global = bboptimize(x -> incidence_gof_calibration!(x, local_params, target_inc,
                                                              target_VE_final, target_VE_post, 
target_VE_post2, current_S, current_Is, current_Ia, current_C, current_R);
        SearchRange = collect(zip(lower_bounds, upper_bounds)),
        Method = :de_rand_1_bin,  # Faster method
        MaxSteps = 5000,          # Reduced steps
        PopulationSize = 30,      # Smaller population
        randomSeed = 42)
    
    best_candidate = BlackBoxOptim.best_candidate(result_global)
    
    local_params.l .= best_candidate[1]
    local_params.r_v .= best_candidate[2]
    local_params.alpha_v[1:3] .= best_candidate[3]
    local_params.alpha_v[5:17] .= best_candidate[4]
    local_params.VE[1:3] .= best_candidate[5]
    local_params.VE[5:17] .= best_candidate[6]

    # calculate indirect protection
    T_max = 18
    ts = range(0, stop=T_max, step=1)
    
    initial_2 = gen_initial_integrated!(local_params, current_S, current_Is, current_Ia, current_C, current_R)
    prob_2 = ODEProblem(typhoid_model_integrated!, initial_2, (0.0, T_max), local_params)
    sol_2 = solve(prob_2, Tsit5(), save_everystep=false, dense=false, tstops=ts, saveat=ts)
    
    cleaned_sol_2 = clean_output_integrated!(sol_2)
    
    inc_2 = calc_inc_control_integrated!(cleaned_sol_2)
    indirect_protection_calibrated = 1.0 - (inc_2 / pre_vaxx_inc)
       
    # calculate VE at the different time intervals of collection
    T_max_final = 12*6
    ts_final = range(0, stop=T_max_final, step=1)
    
    prob_final = ODEProblem(typhoid_model_integrated!, initial_2, (0.0, T_max_final), local_params)
    sol_final = solve(prob_final, Tsit5(), save_everystep=false, dense=false, tstops=ts_final, saveat=ts_final)
    
    cleaned_sol_final = clean_output_integrated!(sol_final)
    
    inc_final_vaxx_1y = calc_inc_vaxx_integrated_VE!(cleaned_sol_final,1,13)
    inc_final_control_1y = calc_inc_control_integrated_VE!(cleaned_sol_final,1,13)
    VE_final_calibrated_1y = (inc_final_control_1y .- inc_final_vaxx_1y) ./ inc_final_control_1y
    
    inc_final_vaxx_4y = calc_inc_vaxx_integrated_VE!(cleaned_sol_final,14,50)
    inc_final_control_4y = calc_inc_control_integrated_VE!(cleaned_sol_final,14,50)
    VE_final_calibrated_4y = (inc_final_control_4y .- inc_final_vaxx_4y) ./ inc_final_control_4y

    inc_final_vaxx_6y = calc_inc_vaxx_integrated_VE!(cleaned_sol_final,51,71)
    inc_final_control_6y = calc_inc_control_integrated_VE!(cleaned_sol_final,51,71)
    VE_final_calibrated_6y = (inc_final_control_6y .- inc_final_vaxx_6y) ./ inc_final_control_6y

    return (
        i = i,
        archetype3 = [best_candidate; indirect_protection_calibrated; VE_final_calibrated_1y[1]; VE_final_calibrated_1y[2]; VE_final_calibrated_4y[1]; VE_final_calibrated_4y[2]; VE_final_calibrated_6y[1];
VE_final_calibrated_6y[2]]
    )
end

# create result DataFrames 
names_list_fast = ["l", "r_v", "alpha_v_agecat1","alpha_v_agecat3", "VE_agecat1", "VE_agecat3", "indirect_protection", "VE_final_agecat1", "VE_final_agecat3", "VE_post_agecat1", "VE_post_agecat3", "VE_post2_agecat1", "VE_post2_agecat3"]
indirect_protection_results_fast = DataFrame(Names = names_list_fast)

for result in results_fast
    i = result.i
    col_name = Symbol("run_$i")
    insertcols!(indirect_protection_results_fast, col_name => result.archetype3)
end

CSV.write("indirect_protection_results_fast_check.csv", indirect_protection_results_fast)
