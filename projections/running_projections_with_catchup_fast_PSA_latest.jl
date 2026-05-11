using Distributed, CSV, DifferentialEquations, DataFrames, StructArrays, Tables, Optim, StatsBase, Distributions, LinearAlgebra,LsqFit
addprocs(10)

@everywhere using CSV, DifferentialEquations, DataFrames, StructArrays, Tables, Optim, StatsBase, Distributions, LinearAlgebra,LsqFit


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

@everywhere function update_params_psa_burn_all(params_start, psa_sample, archetype, waning_scenario)
    params=copy(params_start)
    params=params[params.archetype.==archetype,:]
    
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

    #other nat'l history params
    params.gamma .= psa_sample.gamma
    params.r_a .= psa_sample.r_a
    params.r_c .= psa_sample.r_c
    params.theta_s .= psa_sample.theta_s
    params.theta_a .= psa_sample.theta_a
    params.phi .= psa_sample.phi
    params.rho .= psa_sample.rho
    
    if waning_scenario == "fast"
        params.VE[1:3] .= fill(psa_sample.VE_under2_fast, 3)  #
        params.VE[4] = psa_sample.VE_2_4_fast
        params.VE[5:17] .= fill(psa_sample.VE_5_plus_fast, 13)  
        params.alpha_v[1:3] .= fill(2/psa_sample.duration_under2_fast, 3)  
        params.alpha_v[4] = 2/psa_sample.duration_2_4_fast
        params.alpha_v[5:17] .= fill(2/psa_sample.duration_5_plus_fast, 13) 
    else 
        params.VE[1:4] .= fill(psa_sample.VE_under_5_fast, 4)  #
        params.VE[5:17] .= fill(psa_sample.VE_5_plus_fast, 13)  
        params.alpha_v[1:4] .= fill(1/psa_sample.duration_under_5_fast, 4)  
        params.alpha_v[5:17] .= fill(1/psa_sample.duration_5_plus_fast, 13) 
    end
    
    params.r_v .= psa_sample.r_v
    params.l .= psa_sample.l
    
    #incidence
    params.inc_age[1:3] .= psa_sample.inc_age_cat1/100000
    params.inc_age[4] = psa_sample.inc_age_cat2/100000
    params.inc_age[5] = psa_sample.inc_age_cat3/100000
    params.inc_age[6] = psa_sample.inc_age_cat4/100000
    params.inc_age[7:17] .= psa_sample.inc_age_cat5/100000
    
    #vaxx covg - to update later
    params.pvx_r_9mo.= fill(psa_sample.pvx_r_9mo,17)
    params.pvx_r_15mo.= fill(psa_sample.pvx_r_15mo,17)
    params.pvx_r_2yr.= fill(psa_sample.pvx_r_2yr,17)
    params.pvx_r_5yr.= fill(psa_sample.pvx_r_5yr,17)
    params.pvx_c.= fill(psa_sample.pvx_c,17)

    return(params)
end

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

    params.gamma .= fill(psa_sample.gamma, 17) 
    params.r_a .= fill(psa_sample.r_a, 17)     
    params.r_c .= fill(psa_sample.r_c, 17)      
    params.theta_s .= fill(psa_sample.theta_s, 17)  
    params.theta_a .= fill(psa_sample.theta_a, 17)  
    params.phi .= fill(psa_sample.phi, 17)     
    params.rho .= fill(psa_sample.rho, 17)      

    params.VE[1:3] .= fill(psa_sample.VE_under2_fast, 3)  #
    params.VE[4] = psa_sample.VE_2_4_fast
    params.VE[5:17] .= fill(psa_sample.VE_5_plus_fast, 13)  
    params.alpha_v[1:3] .= fill(2/psa_sample.duration_under2_fast, 3)  
    params.alpha_v[4] = 2/psa_sample.duration_2_4_fast
    params.alpha_v[5:17] .= fill(2/psa_sample.duration_5_plus_fast, 13)  
    params.r_v .= 0.0
    params.l .= 0.0
    
    params.inc_age[1:3] .= psa_sample.inc_age_cat1/100000
    params.inc_age[4] = psa_sample.inc_age_cat2/100000
    params.inc_age[5] = psa_sample.inc_age_cat3/100000
    params.inc_age[6] = psa_sample.inc_age_cat4/100000
    params.inc_age[7:17] .= psa_sample.inc_age_cat5/100000
    
    return(params)
end

@everywhere function typhoid_model_projections!(du, u, p, t) 
    
    ages = 17
    m = 17
    
    S = @view u[1:m]                     # Susceptible (unvaccinated)
    V = @view u[m+1:2*m]                 # Vaccinated (partial protection)
    V2 = @view u[2*m+1:3*m]              # Vaccinated (partial protection 2)
    Vw = @view u[3*m+1:4*m]             # Vaccinated (waned protection)
    Is = @view u[4*m+1:5*m]              # Symptomatic infected
    Ia = @view u[5*m+1:6*m]               # Asymptomatic infected
    C = @view u[6*m+1:7*m]              # Carriers
    R = @view u[7*m+1:8*m]               # Recovered
    
    Is_never = @view u[8*m+1:9*m]        # Is who were never vaccinated (from S)
    Is_V = @view u[9*m+1:10*m]           # Is who came from V compartment
    Is_V2 = @view u[10*m+1:11*m]         # Is who came from V2 compartment  
    Is_Vw = @view u[11*m+1:12*m]         # Is who came from Vw compartment
    
    Ia_never = @view u[12*m+1:13*m]      # Ia who were never vaccinated (from S)
    Ia_V = @view u[13*m+1:14*m]          # Ia who came from V compartment
    Ia_V2 = @view u[14*m+1:15*m]         # Ia who came from V2 compartment
    Ia_Vw = @view u[15*m+1:16*m]         # Ia who came from Vw compartment
    
    C_never = @view u[16*m+1:17*m]       # C who were never vaccinated
    C_V = @view u[17*m+1:18*m]           # C who came from V compartment
    C_V2 = @view u[18*m+1:19*m]          # C who came from V2 compartment
    C_Vw = @view u[19*m+1:20*m]          # C who came from Vw compartment
    
    # R compartment counts
    R_never = @view u[20*m+1:21*m]       # R who were never vaccinated
    R_V = @view u[21*m+1:22*m]           # R who came from V compartment  
    R_V2 = @view u[22*m+1:23*m]          # R who came from V2 compartment
    R_Vw = @view u[23*m+1:24*m]          # R who came from Vw compartment
    
    cases = @view u[24*m+1:25*m]    # Cases from previously vaccinated

    doses = @view u[25*m+1:26*m]
    doses_routine = @view u[26*m+1:27*m]
    doses_booster = @view u[27*m+1:28*m]
    doses_campaign = @view u[28*m+1:29*m]

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
    
    dcases = @view du[24*m+1:25*m]

    ddoses = @view du[25*m+1:26*m]
    ddoses_routine = @view du[26*m+1:27*m]
    ddoses_booster = @view du[27*m+1:28*m]
    ddoses_campaign = @view du[28*m+1:29*m]

    total_Is = sum(Is_never + Is_V + Is_V2 + Is_Vw)
    total_Ia = sum(Ia_never + Ia_V + Ia_V2 + Ia_Vw)
    total_C = sum(C_never + C_V + C_V2 + C_Vw)
    total_R =  sum(R_never + R_V + R_V2 + R_Vw)
    
    N = sum(S + V + V2 + Vw) + total_Is + total_Ia + total_C + total_R 
    
    # 1. VACCINATION FLOWS

    if t < 48
        p.vx_b_on.=repeat([0],17)
    elseif t >= 48  &&  t < 108
        p.vx_b_on.=[repeat([0],4);1;repeat([0],12)]
    else 
        p.vx_b_on.=[repeat([0],4);1;1;repeat([0],11)]
    end

    # calculate the total intended vaccination rate

    ## routine
    intended_total_r_vaxx = (p.pvx_r) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(S, 1)[2:ages]]
    intended_total_r_vaxx_ALL = (p.pvx_r) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(S, 1)[2:ages]] .+ 
                                (p.pvx_r) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Ia_never, 1)[2:ages]] .+
                                (p.pvx_r) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(C_never, 1)[2:ages]] .+
                                (p.pvx_r) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(R_never, 1)[2:ages]]
     
    ## boosters
    intended_total_b_vaxx_Vw = p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Vw, 1)[2:ages]]
    intended_total_b_vaxx_ALL = p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(V, 1)[2:ages]] .+ 
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(V2, 1)[2:ages]] .+ 
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Vw, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Ia_V, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Ia_V2, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Ia_Vw, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(C_V, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(C_V2, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(C_Vw, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(R_V, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(R_V2, 1)[2:ages]] .+
                                p.vx_b_on .* (p.pvx_b) .* (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(R_Vw, 1)[2:ages]] 
    

    # limit it to what's available
    actual_total_r_vaxx = min.(intended_total_r_vaxx, S)
    actual_total_b_vaxx_Vw = min.(intended_total_b_vaxx_Vw, Vw)

    actual_total_r_vaxx_ALL = min.(intended_total_r_vaxx_ALL, sum(S+Ia_never+C_never+R_never))
    actual_total_b_vaxx_ALL = min.(intended_total_b_vaxx_ALL, sum(V+V2+Vw+Ia_V+Ia_V2+Ia_Vw+C_V+C_V2+C_Vw+R_V+R_V2+R_Vw))

    # distribute between successful and unsuccessful vaccination
    successful_r_vaxx = p.VE .* actual_total_r_vaxx 
    unsuccessful_r_vaxx = (1 .- p.VE) .* actual_total_r_vaxx 
    
    successful_b_vaxx_Vw = p.VE .* actual_total_b_vaxx_Vw 
    
    if t <= 3 && t >= 1
        p.vx_c.=1
    else
        p.vx_c.=0
    end

    successful_c_vaxx = p.VE .* p.vx_c .* p.pvx_c .* S
    unsuccessful_c_vaxx = (1 .- p.VE) .* p.vx_c .* p.pvx_c .* S

    # 2. AGING AND MORTALITY
    dS .= p.Bi_burn .* [N; zeros(ages-1)] .+ 
        (1 .- p.pu_in_burn) .* p.a_in .* [0; circshift(S, 1)[2:ages]] .- 
        successful_r_vaxx .- 
        unsuccessful_r_vaxx .- 
        successful_c_vaxx .- 
        unsuccessful_c_vaxx .- 
        p.a_out .* (1 .- p.pu_burn) .* S .- p.u_burn .* S  
   
    dV .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(V, 1)[2:ages]] .+  
        successful_c_vaxx .+
        successful_r_vaxx .+ 
        successful_b_vaxx_Vw .- 
        p.a_out .* (1 .- p.pu_burn) .* V .- p.u_burn .* V    

    dV2 .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(V2, 1)[2:ages]] .- 
         p.a_out .* (1 .- p.pu_burn) .* V2 .- p.u_burn .* V2
   
    dVw .= (1 .- p.pu_in_burn) .* p.a_in .* [0;circshift(Vw, 1)[2:ages]] .+
           unsuccessful_c_vaxx .+ 
           unsuccessful_r_vaxx .-
           successful_b_vaxx_Vw .- 
           p.a_out .* (1 .- p.pu_burn) .* Vw .- p.u_burn .* Vw 
  
    aging_factor_in = (1 .- p.pu_in_burn) .* p.a_in
    aging_factor_out = p.a_out .* (1 .- p.pu_burn) .+ p.u_burn
    
    # Is counts aging
    dIs_never .+= aging_factor_in .* [0; circshift(Is_never, 1)[2:end]] .- aging_factor_out .* Is_never
    dIs_V .+= aging_factor_in .* [0; circshift(Is_V, 1)[2:end]] .- aging_factor_out .* Is_V
    dIs_V2 .+= aging_factor_in .* [0; circshift(Is_V2, 1)[2:end]] .- aging_factor_out .* Is_V2
    dIs_Vw .+= aging_factor_in .* [0; circshift(Is_Vw, 1)[2:end]] .- aging_factor_out .* Is_Vw
    
    # Ia counts aging
    dIa_never .+= aging_factor_in .* [0; circshift(Ia_never, 1)[2:end]] .- aging_factor_out .* Ia_never
    dIa_V .+= aging_factor_in .* [0; circshift(Ia_V, 1)[2:end]] .- aging_factor_out .* Ia_V
    dIa_V2 .+= aging_factor_in .* [0; circshift(Ia_V2, 1)[2:end]] .- aging_factor_out .* Ia_V2
    dIa_Vw .+= aging_factor_in .* [0; circshift(Ia_Vw, 1)[2:end]] .- aging_factor_out .* Ia_Vw
    
    # C counts aging
    dC_never .+= aging_factor_in .* [0; circshift(C_never, 1)[2:end]] .- aging_factor_out .* C_never
    dC_V .+= aging_factor_in .* [0; circshift(C_V, 1)[2:end]] .- aging_factor_out .* C_V
    dC_V2 .+= aging_factor_in .* [0; circshift(C_V2, 1)[2:end]] .- aging_factor_out .* C_V2
    dC_Vw .+= aging_factor_in .* [0; circshift(C_Vw, 1)[2:end]] .- aging_factor_out .* C_Vw
    
  # R counts aging
    dR_never .+= aging_factor_in .* [0; circshift(R_never, 1)[2:end]] .- aging_factor_out .* R_never
    dR_V .+= aging_factor_in .* [0; circshift(R_V, 1)[2:end]] .- aging_factor_out .* R_V
    dR_V2 .+= aging_factor_in .* [0; circshift(R_V2, 1)[2:end]] .- aging_factor_out .* R_V2
    dR_Vw .+= aging_factor_in .* [0; circshift(R_Vw, 1)[2:end]] .- aging_factor_out .* R_Vw

    ddoses .= actual_total_r_vaxx_ALL .+ actual_total_b_vaxx_ALL .+ p.vx_c .* p.pvx_c .* (S  .+ Ia_never .+ C_never .+ R_never)
    ddoses_routine .= actual_total_r_vaxx_ALL
    ddoses_booster .= actual_total_b_vaxx_ALL
    ddoses_campaign .= p.vx_c .* p.pvx_c .* (S .+ Ia_never .+ C_never .+ R_never) #p.vx_c .* p.pvx_c .* (S)
        
   # 3. DISEASE TRANSMISSION
    Is_vec = fill(total_Is, ages) / N
    C_vec = fill(total_C, ages) / N
    Ia_never_vec = fill(sum(Ia_never)) / N
    Ia_V_vec = fill(sum(Ia_V)) / N
    Ia_V2_vec = fill(sum(Ia_V2)) / N
    Ia_Vw_vec = fill(sum(Ia_Vw)) / N
    
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
   
    # case tracking
    dcases .+= new_Is_from_S .+ new_Is_from_Vw
     
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
    dV .+=  recover_Ia_V_to_V.- waning_V_to_V2 
    dV2 .+= waning_V_to_V2 .+ recover_Ia_V2_to_V2 .- waning_V2_to_Vw 
    dVw .+= waning_V2_to_Vw .+ recover_Ia_Vw_to_Vw .+ recover_Is_Vw_to_Vw .+ waning_R_V_to_Vw .+ waning_R_V2_to_Vw .+ waning_R_Vw_to_Vw 
    dIs .= dIs_never .+ dIs_V .+ dIs_V2 .+ dIs_Vw
    dIa .= dIa_never .+ dIa_V .+ dIa_V2 .+ dIa_Vw  
    dC .= dC_never .+ dC_V .+ dC_V2 .+ dC_Vw
    dR .= dR_never .+ dR_V .+ dR_V2 .+ dR_Vw

end


@everywhere function clean_output_projections(sol)
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
    
    Ia_never_values = [u[12*m+1:13*m] for u in sol.u]
    Ia_V_values = [u[13*m+1:14*m] for u in sol.u]
    Ia_V2_values = [u[14*m+1:15*m] for u in sol.u]
    Ia_Vw_values = [u[15*m+1:16*m] for u in sol.u]
    
    C_never_values = [u[16*m+1:17*m] for u in sol.u]
    C_V_values = [u[17*m+1:18*m] for u in sol.u]
    C_V2_values = [u[18*m+1:19*m] for u in sol.u]
    C_Vw_values = [u[19*m+1:20*m] for u in sol.u]

    R_never_values = [u[20*m+1:21*m] for u in sol.u]
    R_V_values = [u[21*m+1:22*m] for u in sol.u]
    R_V2_values = [u[22*m+1:23*m] for u in sol.u]
    R_Vw_values = [u[23*m+1:24*m] for u in sol.u]
    
    cases_values = [u[24*m+1:25*m] for u in sol.u]

    doses_values = [u[25*m+1:26*m] for u in sol.u]
    doses_routine_values = [u[26*m+1:27*m] for u in sol.u]
    doses_booster_values = [u[27*m+1:28*m] for u in sol.u]
    doses_campaign_values = [u[28*m+1:29*m] for u in sol.u]

    model_df = DataFrame(time = time_points)
    
    for age in 1:ages
        age_label = string(age)
        
        model_df[!, "S_$age_label"] = [S[age] for S in S_values]
        model_df[!, "V_$age_label"] = [V[age] for V in V_values]
        model_df[!, "V2_$age_label"] = [V2[age] for V2 in V2_values]
        model_df[!, "Vw_$age_label"] = [Vw[age] for Vw in Vw_values]
        model_df[!, "Is_$age_label"] = [Is[age] for Is in Is_values]
        model_df[!, "Ia_$age_label"] = [Ia[age] for Ia in Ia_values]
        model_df[!, "C_$age_label"] = [C[age] for C in C_values]
        model_df[!, "R_$age_label"] = [R[age] for R in R_values]
        
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
    
        model_df[!, "cases_$age_label"] = [cases[age] for cases in cases_values]

        model_df[!, "doses_$age_label"] = [doses[age] for doses in doses_values]
        model_df[!, "doses_routine_$age_label"] = [doses_routine[age] for doses_routine in doses_routine_values]
        model_df[!, "doses_booster_$age_label"] = [doses_booster[age] for doses_booster in doses_booster_values]
        model_df[!, "doses_campaign_$age_label"] = [doses_campaign[age] for doses_campaign in doses_campaign_values]

    end
    
    return model_df
end


@everywhere function gen_initial_projections(current_S, current_Is, current_Ia, current_C, current_R)

    ages=17
    
    S_0 = current_S
    Is_0 = current_Is
    Ia_0 = current_Ia
    C_0 = current_C
    R_0 = current_R
    V_0 = zeros(ages)         # Actively protected vaccinated
    V2_0 = zeros(ages)            # Actively protected vaccinated V2
    Vw_0 = zeros(ages)    # Partially protected vaccinated
    
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

    cases_0 = zeros(ages)

    doses_0 = zeros(ages)
    doses_routine_0 = zeros(ages)
    doses_booster_0 = zeros(ages)
    doses_campaign_0 = zeros(ages)

    initial = vcat(
        S_0, V_0, V2_0, Vw_0,                          # Susceptible compartments
        Is_0, Ia_0, C_0, R_0,                    # Disease compartments
        Is_never, Is_V, Is_V2, Is_Vw,  # Is 
        Ia_never, Ia_V, Ia_V2, Ia_Vw,  # Ia 
        C_never, C_V, C_V2, C_Vw,   # C 
        R_never, R_V, R_V2, R_Vw,                  # R 
        cases_0, doses_0, doses_routine_0, doses_booster_0, doses_campaign_0       
    )
    
    return initial
end

@everywhere function calc_cases_projections_age_stratified(sol_df)
    all_cols = names(sol_df)
    
    case_cols_projections_cases_pediatric = [("cases_$i") for i in 1:6]
    case_cols_projections_cases_adult = [("cases_$i") for i in 7:17]
    
    n_months = size(sol_df, 1) - 1
    n_years = div(n_months, 12)
    
    annual_cases_pediatric = zeros(n_years)
    annual_cases_adult = zeros(n_years)
    annual_cases_by_age = zeros(n_years, 17)
    
    for year in 1:n_years
        start_idx = (year-1)*12 + 1
        end_idx = year*12 + 1
        
        annual_cases_pediatric[year] = sum(
            sol_df[end_idx, col] - sol_df[start_idx, col] 
            for col in case_cols_projections_cases_pediatric
        )
        annual_cases_adult[year] = sum(
            sol_df[end_idx, col] - sol_df[start_idx, col] 
            for col in case_cols_projections_cases_adult
        )
        for age in 1:17
            annual_cases_by_age[year, age] = sol_df[end_idx, "cases_$age"] - sol_df[start_idx, "cases_$age"]
        end
    end
    
    result_df = DataFrame(
        year = 1:n_years,
        pediatric_cases = annual_cases_pediatric,
        adult_cases = annual_cases_adult,
        total_cases = annual_cases_pediatric + annual_cases_adult
    )
    
    for age in 1:17
        result_df[!, "cases_age_$age"] = annual_cases_by_age[:, age]
    end
    
    return result_df
end

@everywhere function create_age_stratified_output_dataframe(no_vaxx_df, strategy_dfs)
    result_df = DataFrame(year = no_vaxx_df.year)
    
    strategy_names = ["routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                     "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                     "routine_9mos_boosters_5yrs_10yrs", "routine_15mos_boosters_5yrs_10yrs"]
    
    for age in 1:17
        col = "cases_age_$age"
        result_df[!, "$(col)_no_vaxx"] = no_vaxx_df[!, col]
        for (i, name) in enumerate(strategy_names)
            result_df[!, "$(col)_$(name)"] = strategy_dfs[i][!, col]
        end
    end
    
    return result_df
end

    
@everywhere function calc_inc_projections_annual(sol_df)
    all_cols = names(sol_df)
    
    case_cols = [("cases_$i") for i in 1:17]
    doses_cols = [("doses_$i") for i in 1:17]
    doses_routine_cols = [("doses_routine_$i") for i in 1:17]
    doses_booster_cols = [("doses_booster_$i") for i in 1:17]
    doses_campaign_cols = [("doses_campaign_$i") for i in 1:17]

    exclude_cols = vcat(case_cols, doses_cols, doses_routine_cols, doses_booster_cols, doses_campaign_cols, "time", 
        [("Is_never_$i") for i in 1:17],[("Is_V_$i") for i in 1:17],[("Is_V2_$i") for i in 1:17],[("Is_Vw_$i") for i in 1:17],
        [("Ia_never_$i") for i in 1:17],[("Ia_V_$i") for i in 1:17],[("Ia_V2_$i") for i in 1:17],[("Ia_Vw_$i") for i in 1:17],
        [("C_never_$i") for i in 1:17],[("C_V_$i") for i in 1:17],[("C_V2_$i") for i in 1:17],[("C_Vw_$i") for i in 1:17],
        [("R_never_$i") for i in 1:17],[("R_V_$i") for i in 1:17],[("R_V2_$i") for i in 1:17],[("R_Vw_$i") for i in 1:17]) # Exclude case, dose, time, and tracker columns 
     compartment_cols = filter(col -> !(col in exclude_cols), all_cols)

    n_months = size(sol_df, 1) - 1
    n_years = div(n_months, 12)
    
    annual_inc = zeros(n_years)
    
    for year in 1:n_years
        start_idx = (year-1)*12 + 1
        end_idx = year*12 + 1
        
        total_new_cases = sum(sol_df[end_idx, col] - sol_df[start_idx, col] for col in case_cols)
    
        start_population = sum(sol_df[start_idx, col] for col in compartment_cols)
        end_population = sum(sol_df[end_idx, col] for col in compartment_cols)
        avg_population = (start_population + end_population) / 2
        
        annual_inc[year] = avg_population > 0 ? (total_new_cases * 100000) / avg_population : 0
    end

    result_df = DataFrame(
        year = 1:n_years,
        annual_incidence = annual_inc)
    
    return result_df
end


@everywhere function run_projections_vaxx_scenario_doses(params_archetype_vaxx_scenario, current_S, current_Is, current_Ia, current_C, current_R, routine_or_booster_or_all)
    T_max = 20*12
    ts = range(0, stop=T_max, step=1)
    
    initial_projections_doses = gen_initial_projections(current_S, current_Is, current_Ia, current_C, current_R)
    
    prob_projections_doses = ODEProblem(typhoid_model_projections!, initial_projections_doses, (0.0, T_max), params_archetype_vaxx_scenario)
    sol_projections_doses = solve(prob_projections_doses, dense=false, save_everystep=false, tstops=ts, saveat=ts)
    
    cleaned_sol_projections_doses = clean_output_projections(sol_projections_doses)
    
    doses_cols_projections = [("doses_$i") for i in 1:17]
    doses_routine_cols_projections = [("doses_routine_$i") for i in 1:17]
    doses_booster_cols_projections = [("doses_booster_$i") for i in 1:17]
    doses_campaign_cols_projections = [("doses_campaign_$i") for i in 1:17]

    n_years = div(T_max, 12)
    
    annual_doses = zeros(n_years)
    annual_doses_routine = zeros(n_years)
    annual_doses_booster = zeros(n_years)
    annual_doses_campaign = zeros(n_years)

    for year in 1:n_years
        start_idx = (year-1)*12 + 1
        end_idx = year*12 + 1
        
        annual_doses[year] = sum(
            cleaned_sol_projections_doses[end_idx, col] - 
            cleaned_sol_projections_doses[start_idx, col] 
            for col in doses_cols_projections
        )
        
        annual_doses_routine[year] = sum(
            cleaned_sol_projections_doses[end_idx, col] - 
            cleaned_sol_projections_doses[start_idx, col] 
            for col in doses_routine_cols_projections
        )
        
        annual_doses_booster[year] = sum(
            cleaned_sol_projections_doses[end_idx, col] - 
            cleaned_sol_projections_doses[start_idx, col] 
            for col in doses_booster_cols_projections
        )

        annual_doses_campaign[year] = sum(
            cleaned_sol_projections_doses[end_idx, col] - 
            cleaned_sol_projections_doses[start_idx, col] 
            for col in doses_campaign_cols_projections
        )
    end
    
    if routine_or_booster_or_all == "all"
        # Create a DataFrame with all three types
        result_df = DataFrame(
            year = 1:n_years,
            total_doses = annual_doses,
            routine_doses = annual_doses_routine,
            booster_doses = annual_doses_booster,
            campaign_doses = annual_doses_campaign
        )
        return result_df
    elseif routine_or_booster_or_all == "routine"
        # Return DataFrame with routine doses only
        result_df = DataFrame(
            year = 1:n_years,
            routine_doses = annual_doses_routine
        )
        return result_df
    elseif routine_or_booster_or_all == "booster"
        # Return DataFrame with booster doses only
        result_df = DataFrame(
            year = 1:n_years,
            booster_doses = annual_doses_booster
        )
        return result_df
    else 
        # Return DataFrame with campaign doses only
        result_df = DataFrame(
            year = 1:n_years,
            campaign_doses = annual_doses_campaign
        )
        return result_df
    end
end

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

@everywhere function gen_initial_burn_in!(params, ages)
    #initialize starting compartment sizes
    Is_0 = (params.inc_age .* params.N_age_burn)./12 
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

@everywhere function create_vaccination_params(base_params, vaccination_schedule)
    params = copy(base_params)
    
    params.pvx_r = zeros(Float64, 17)
    params.pvx_b = zeros(Float64, 17)  
    params.pvx_c = zeros(Float64, 17)
    
    # apply vaccination schedule 
    for (vaccine_type, schedules) in vaccination_schedule
        for (age_idx, coverage) in schedules
            if vaccine_type == :routine
                params.pvx_r[age_idx] = coverage
            elseif vaccine_type == :booster
                params.pvx_b[age_idx] = coverage
            elseif vaccine_type == :campaign
                params.pvx_c[age_idx] = coverage
            end
        end
    end
    
    return params
end

@everywhere function update_params_vaxx_covg(local_params, strategy)
    # routine vaccinations only
    routine_9m_with_campaign = Dict(
         :routine => Dict(2 => local_params.pvx_r_9mo[1]),
         :campaign => Dict(i => local_params.pvx_c[1] for i in 2:6)
    )
    routine_15m_with_campaign = Dict(
         :routine => Dict(3 => local_params.pvx_r_15mo[1]),
         :campaign => Dict(i => local_params.pvx_c[1] for i in 3:6)
    )
    routine_2y_with_campaign = Dict(
         :routine => Dict(4 => local_params.pvx_r_2yr[1]),
         :campaign => Dict(i => local_params.pvx_c[1] for i in 4:6)
    )
    routine_5y_with_campaign = Dict(
         :routine => Dict(5 => local_params.pvx_r_5yr[1]),
         :campaign => Dict(i => local_params.pvx_c[1] for i in 5:6)
    )
     
    # routine with boosters
    routine_9m_booster_5y_with_campaign = Dict(
        :routine => Dict(2 => local_params.pvx_r_9mo[1]),
        :booster => Dict(5 => local_params.pvx_r_9mo[1]),
        :campaign => Dict(i => local_params.pvx_c[1] for i in 2:6)
    )
    routine_15m_booster_5y_with_campaign = Dict(
        :routine => Dict(3 => local_params.pvx_r_15mo[1]),
        :booster => Dict(5 => local_params.pvx_r_15mo[1]),
        :campaign => Dict(i => local_params.pvx_c[1] for i in 3:6)
    )
    routine_9m_boosters_5y_10y_with_campaign = Dict(
        :routine => Dict(2 => local_params.pvx_r_9mo[1]),
        :booster => Dict(5 => local_params.pvx_r_9mo[1], 6 => local_params.pvx_r_9mo[1]),
        :campaign => Dict(i => local_params.pvx_c[1] for i in 2:6)
    )

    routine_15m_boosters_5y_10y_with_campaign = Dict(
        :routine => Dict(3 => local_params.pvx_r_15mo[1]),
        :booster => Dict(5 => local_params.pvx_r_15mo[1], 6 => local_params.pvx_r_15mo[1]),
        :campaign => Dict(i => local_params.pvx_c[1] for i in 3:6)
    )
    
    if strategy == "routine at 9 months"
        return create_vaccination_params(local_params, routine_9m_with_campaign)
    elseif strategy == "routine at 15 months"
        return create_vaccination_params(local_params, routine_15m_with_campaign) 
    elseif strategy == "routine at 2 years"
        return create_vaccination_params(local_params, routine_2y_with_campaign) 
    elseif strategy == "routine at 5 years"
        return create_vaccination_params(local_params, routine_5y_with_campaign) 
    elseif strategy == "routine at 9 months, booster at 5 years"
        return create_vaccination_params(local_params, routine_9m_booster_5y_with_campaign) 
    elseif strategy == "routine at 15 months, booster at 5 years"
        return create_vaccination_params(local_params, routine_15m_booster_5y_with_campaign) 
    elseif strategy == "routine at 9 months, boosters at 5 and 10 years"
        return create_vaccination_params(local_params, routine_9m_boosters_5y_10y_with_campaign) 
    elseif strategy == "routine at 15 months, boosters at 5 and 10 years"
        return create_vaccination_params(local_params, routine_15m_boosters_5y_10y_with_campaign) 
    end
end


@everywhere begin
    params=CSV.read("params_latest_adapted_june14.csv", DataFrame)
    
    calibrated_demographic_params=CSV.read("demo_calib_output.csv", DataFrame)
    
    psa_samples = CSV.read("params_psa_full_complete_fast.csv", DataFrame)
    
    params_archetype1=update_params_demographics(params, calibrated_demographic_params, 1)
    params_archetype2=update_params_demographics(params, calibrated_demographic_params, 2)
    params_archetype3=update_params_demographics(params, calibrated_demographic_params, 3)

    psa_samples_archetype1=psa_samples[psa_samples.archetype.==1,:]
    psa_samples_archetype2=psa_samples[psa_samples.archetype.==2,:]
    psa_samples_archetype3=psa_samples[psa_samples.archetype.==3,:]

    psa_samples_archetype1_ids=psa_samples_archetype1.sample_id
    psa_samples_archetype2_ids=psa_samples_archetype2.sample_id
    psa_samples_archetype3_ids=psa_samples_archetype3.sample_id

    psa_lookup_1 = Dict(row.sample_id => row for row in eachrow(psa_samples_archetype1))
    psa_lookup_2 = Dict(row.sample_id => row for row in eachrow(psa_samples_archetype2))
    psa_lookup_3 = Dict(row.sample_id => row for row in eachrow(psa_samples_archetype3))
  
end

# run projections for all strategies with a given parameter set
@everywhere function run_projections_for_all_strategies(params_base, strategy_params, current_S, current_Is, current_Ia, current_C, current_R, iteration)
    T_max = 20*12
    ts = range(0, stop=T_max, step=1)
    
    initial_no_vaxx = gen_initial_projections(current_S, current_Is, current_Ia, current_C, current_R)
    prob_no_vaxx = ODEProblem(typhoid_model_projections!, initial_no_vaxx, (0.0, T_max), params_base)
    sol_no_vaxx = solve(prob_no_vaxx, dense=false, save_everystep=false, tstops=ts, saveat=ts)
    cleaned_sol_no_vaxx = clean_output_projections(sol_no_vaxx)
 
    no_vaxx_population = calc_population_sizes(cleaned_sol_no_vaxx, 1)
    no_vaxx_cases = calc_cases_projections_age_stratified(cleaned_sol_no_vaxx)
    no_vaxx_incidence = calc_inc_projections_annual(cleaned_sol_no_vaxx)
    no_vaxx_doses = run_projections_vaxx_scenario_doses(params_base, current_S, current_Is, current_Ia, current_C, current_R, "all")

    strategy_cases = Dict()
    strategy_incidence = Dict()
    strategy_doses = Dict()
    strategy_population = Dict()
    
    for strategy in 1:8
        initial_strategy = gen_initial_projections(current_S, current_Is, current_Ia, current_C, current_R)
        prob_strategy = ODEProblem(typhoid_model_projections!, initial_strategy, (0.0, T_max), strategy_params[strategy])
        sol_strategy = solve(prob_strategy, dense=false, save_everystep=false, tstops=ts, saveat=ts)
        cleaned_sol_strategy = clean_output_projections(sol_strategy)
        
        strategy_cases[strategy] = calc_cases_projections_age_stratified(cleaned_sol_strategy)
        strategy_incidence[strategy] = calc_inc_projections_annual(cleaned_sol_strategy)
        strategy_doses[strategy] = run_projections_vaxx_scenario_doses(strategy_params[strategy], current_S, current_Is, current_Ia, current_C, current_R, "all")
        strategy_population[strategy] = calc_population_sizes(cleaned_sol_strategy, 0)
    end
    
    total_cases_df = create_output_dataframe(no_vaxx_cases, strategy_cases, :total_cases, false)
    pediatric_cases_df = create_output_dataframe(no_vaxx_cases, strategy_cases, :pediatric_cases, false)
    adult_cases_df = create_output_dataframe(no_vaxx_cases, strategy_cases, :adult_cases, false)
    incidence_df = create_incidence_dataframe(no_vaxx_incidence, strategy_incidence)
    population_df = create_population_dataframe(no_vaxx_population, strategy_population)
    total_doses_df = create_output_dataframe(no_vaxx_doses, strategy_doses, :total_doses, false)
    routine_doses_df = create_output_dataframe(no_vaxx_doses, strategy_doses, :routine_doses, false)
    booster_doses_df = create_output_dataframe(no_vaxx_doses, strategy_doses, :booster_doses, true)
    campaign_doses_df = create_output_dataframe(no_vaxx_doses, strategy_doses, :campaign_doses, false)
    
    age_stratified_cases_df = create_age_stratified_output_dataframe(no_vaxx_cases, strategy_cases)

    total_cases_df[!, :iteration] .= iteration
    pediatric_cases_df[!, :iteration] .= iteration
    adult_cases_df[!, :iteration] .= iteration
    incidence_df[!, :iteration] .= iteration
    population_df[!, :iteration] .= iteration
    total_doses_df[!, :iteration] .= iteration
    routine_doses_df[!, :iteration] .= iteration
    booster_doses_df[!, :iteration] .= iteration
    campaign_doses_df[!, :iteration] .= iteration
    age_stratified_cases_df[!, :iteration] .= iteration  
   
    return (
        total_cases_df,
        pediatric_cases_df,
        adult_cases_df,
        incidence_df,
        population_df,
        total_doses_df,
        routine_doses_df,
        booster_doses_df,
        campaign_doses_df,
        age_stratified_cases_df  
    )
end

@everywhere function run_projections_for_all_archetypes(id_seq_archetype1,id_seq_archetype2,id_seq_archetype3)
    all_results = Dict()
    
    for archetype in 1:3
        println("Processing archetype $archetype")
        
        results = []
        
        if archetype == 1
           id_sequence = id_seq_archetype1
        elseif archetype == 2
           id_sequence = id_seq_archetype2
        else 
           id_sequence = id_seq_archetype3
        end

        for i in id_sequence
            println("Starting archetype $archetype, iteration $i")
                
            sample_use = get_psa_samples(archetype, i)
                
            local_params_fast_waning = update_params_psa_burn_all(get_base_params(archetype), sample_use, archetype, "fast")
            local_params = update_params_psa_burn(get_base_params(archetype), sample_use)
                                
            T = 1200
            ts = range(0, stop=T, step=1)
                
            initial = gen_initial_burn_in!(local_params, 17)
            prob = ODEProblem(typhoid_model!, initial, (0.0, T), local_params)
                
            sol = solve(prob, 
                           dense=false, save_everystep=false, tstops=ts, saveat=ts)
                
            cleaned_sol = clean_output_full_transmission_calib!(sol)
                
            current_S = extract_compartment_values(cleaned_sol, "S", T)
            current_Is = extract_compartment_values(cleaned_sol, "Is", T)
            current_Ia = extract_compartment_values(cleaned_sol, "Ia", T)
            current_C = extract_compartment_values(cleaned_sol, "C", T)
            current_R = extract_compartment_values(cleaned_sol, "R", T)
                
            # create strategy parameters
            strategy_params_fast_waning = create_strategy_params(local_params_fast_waning)
                
            # initialize for no-vaxx scenario
            local_params_fast_waning.pvx_b .= repeat([0], 17)
            local_params_fast_waning.pvx_c .= repeat([0], 17)
                
             # run projections
                fast_waning_results = run_projections_for_all_strategies(
                    local_params_fast_waning, 
                    strategy_params_fast_waning, 
                    current_S, current_Is, current_Ia, current_C, current_R, 
                    i
                )
                
             push!(results, fast_waning_results)
                 
        end
        
        all_results[archetype] = results
    end

    return all_results
end

@everywhere function extract_compartment_values(cleaned_sol, compartment, T)
    cols = ["$(compartment)_$i" for i in 1:17]
    return collect(select(cleaned_sol, cols)[T+1, :])
end

@everywhere function get_base_params(archetype)
    if archetype == 1
        return params_archetype1
    elseif archetype == 2
        return params_archetype2
    else
        return params_archetype3
    end
end

function get_psa_samples(archetype, sample_id)
    if archetype == 1
        return psa_lookup_1[sample_id]
    elseif archetype == 2
        return psa_lookup_2[sample_id]
    else
        return psa_lookup_3[sample_id]
    end
end

@everywhere function create_strategy_params(base_params)
    strategy_params = Dict()
    
    strategy_params[1] = update_params_vaxx_covg(base_params, "routine at 9 months")
    strategy_params[2] = update_params_vaxx_covg(base_params, "routine at 15 months")
    strategy_params[3] = update_params_vaxx_covg(base_params, "routine at 2 years")
    strategy_params[4] = update_params_vaxx_covg(base_params, "routine at 5 years")
    strategy_params[5] = update_params_vaxx_covg(base_params, "routine at 9 months, booster at 5 years")
    strategy_params[6] = update_params_vaxx_covg(base_params, "routine at 15 months, booster at 5 years")
    strategy_params[7] = update_params_vaxx_covg(base_params, "routine at 9 months, boosters at 5 and 10 years")
    strategy_params[8] = update_params_vaxx_covg(base_params, "routine at 15 months, boosters at 5 and 10 years")

    return strategy_params
end



# calculate population sizes by year
@everywhere function calc_population_sizes(sol_df, no_vaxx_indicator)
    all_cols = names(sol_df)
    
    # identify compartment columns that contain population
    if no_vaxx_indicator == 1
       compartment_prefixes = ["S_", "Is_", "Ia_", "C_", "R_", "V_", "Vw_", "Iv_", "Sv_"]
    else
       compartment_prefixes = ["S_", "Is_", "Ia_", "C_", "R_", "V_", "V2_", "Vw_"]
    end 

    population_cols = String[]
    
    for prefix in compartment_prefixes
        for i in 1:17
            col = "$(prefix)$(i)"
            if col in all_cols
                push!(population_cols, col)
            end
        end
    end
    
    n_months = size(sol_df, 1) - 1
    n_years = div(n_months, 12)
    
    total_population = zeros(n_years)
    age_group_population = zeros(n_years, 17)
    
    for year in 1:n_years
        end_idx = year*12 + 1
        
        # calculate total population across all compartments and age groups
        for age in 1:17
            age_pop = 0.0
            for prefix in compartment_prefixes
                col = "$(prefix)$(age)"
                if col in population_cols
                    age_pop += sol_df[end_idx, col]
                end
            end
            age_group_population[year, age] = age_pop
        end
        
        total_population[year] = sum(age_group_population[year, :])
    end
    
    # create age group population columns
    age_group_cols = Dict()
    for age in 1:17
        age_group_cols[Symbol("age_group_$(age)")] = age_group_population[:, age]
    end
    
    result_df = DataFrame(
        year = 1:n_years,
        total_population = total_population
    )
    
    # add age group population columns
    for (col_name, values) in age_group_cols
        result_df[!, col_name] = values
    end
    
    return result_df
end

# create population dataframe
@everywhere function create_population_dataframe(no_vaxx_df, strategy_dfs)
    result_df = select(no_vaxx_df, [:year, :total_population])
    
    rename!(result_df, :total_population => :no_vaxx)
    
    strategy_names = ["routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                     "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                     "routine_9mos_boosters_5yrs_10yrs", 
                     "routine_15mos_boosters_5yrs_10yrs"]
    
    for (i, strategy_name) in enumerate(strategy_names)
        result_df[!, strategy_name] = strategy_dfs[i][!, :total_population]
    end
    
    return result_df
end

@everywhere function create_output_dataframe(no_vaxx_df, strategy_dfs, column_name, is_booster)
    result_df = select(no_vaxx_df, [:year, column_name])
    
    rename!(result_df, column_name => :no_vaxx)
    
    strategy_names = [
        "routine_9mos", 
        "routine_15mos", 
        "routine_2yrs", 
        "routine_5yrs", 
        "routine_9mos_booster_5yrs", 
        "routine_15mos_booster_5yrs", 
        "routine_9mos_boosters_5yrs_10yrs", 
        "routine_15mos_boosters_5yrs_10yrs"
    ]
    
    for (i, strategy_name) in enumerate(strategy_names)
        result_df[!, strategy_name] = strategy_dfs[i][!, column_name]
    end
    
    return result_df
end

# create incidence dataframe
@everywhere function create_incidence_dataframe(no_vaxx_df, strategy_dfs)
    result_df = select(no_vaxx_df, [:year, :annual_incidence])
    
    rename!(result_df, :annual_incidence => :no_vaxx)
    
    strategy_names = ["routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                     "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                     "routine_9mos_boosters_5yrs_10yrs", "routine_15mos_boosters_5yrs_10yrs"]
    
    for (i, strategy_name) in enumerate(strategy_names)
        result_df[!, strategy_name] = strategy_dfs[i][!, :annual_incidence]
    end
    
    return result_df
end

# process results for all archetypes
all_archetype_results_with_campaign = run_projections_for_all_archetypes(psa_samples_archetype1_ids,psa_samples_archetype2_ids,
psa_samples_archetype3_ids)


# process and save results for each archetype
for archetype in 1:3
    archetype_results_with_campaign = all_archetype_results_with_campaign[archetype]
    
    fast_total_cases_list = [result[1] for result in archetype_results_with_campaign]
    fast_pediatric_cases_list = [result[2] for result in archetype_results_with_campaign]
    fast_adult_cases_list = [result[3] for result in archetype_results_with_campaign]
    fast_incidence_list = [result[4] for result in archetype_results_with_campaign]
    fast_population_list = [result[5] for result in archetype_results_with_campaign]
    fast_total_doses_list = [result[6] for result in archetype_results_with_campaign]
    fast_routine_doses_list = [result[7] for result in archetype_results_with_campaign]
    fast_booster_doses_list = [result[8] for result in archetype_results_with_campaign]
    fast_campaign_doses_list = [result[9] for result in archetype_results_with_campaign]
    fast_age_stratified_cases_list = [result[10] for result in archetype_results_with_campaign]  

    fast_total_cases_with_campaign = vcat(fast_total_cases_list...)
    fast_pediatric_cases_with_campaign = vcat(fast_pediatric_cases_list...)
    fast_adult_cases_with_campaign = vcat(fast_adult_cases_list...)
    fast_incidence_with_campaign = vcat(fast_incidence_list...)
    fast_population_with_campaign = vcat(fast_population_list...)
    fast_total_doses_with_campaign = vcat(fast_total_doses_list...)
    fast_routine_doses_with_campaign = vcat(fast_routine_doses_list...)
    fast_booster_doses_with_campaign = vcat(fast_booster_doses_list...)
    fast_campaign_doses_with_campaign = vcat(fast_campaign_doses_list...)
    fast_age_stratified_cases_with_campaign = vcat(fast_age_stratified_cases_list...)  

    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_total_cases_with_campaign_latest.csv", fast_total_cases_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_pediatric_cases_with_campaign_latest.csv", fast_pediatric_cases_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_adult_cases_with_campaign_latest.csv", fast_adult_cases_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_incidence_with_campaign_latest.csv", fast_incidence_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_population_with_campaign_latest.csv", fast_population_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_total_doses_with_campaign_latest.csv", fast_total_doses_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_routine_doses_with_campaign_latest.csv", fast_routine_doses_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_booster_doses_with_campaign_latest.csv", fast_booster_doses_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_campaign_doses_with_campaign_latest.csv", fast_campaign_doses_with_campaign)
    CSV.write("./projections_output_check/archetype$(archetype)_fast_waning_age_stratified_cases_with_campaign_latest.csv", fast_age_stratified_cases_with_campaign)  
end