using CSV, DataFrames

# read in costs PSA data
costs_psa_africa=CSV.read("costs_data_PSA_africa_region.csv",DataFrame)
costs_psa_africa.LE.=64.02
costs_psa_africa.vaccine_procurement_costs.=1.5
costs_psa_asia=CSV.read("costs_data_PSA_asia_wpr_region.csv",DataFrame)
costs_psa_asia.LE.=74.76;
costs_psa_asia.vaccine_procurement_costs.=1.5;


function convert_to_dataframe(values, n_rows, n_years, n_strategies, strategy_names, metric_name)
    df = DataFrame()
    
    df.iteration = repeat(1:n_rows, inner=n_years*n_strategies)
    df.year = repeat(repeat(1:n_years, inner=n_strategies), outer=n_rows)
    df.strategy = repeat(strategy_names, outer=n_rows*n_years)
    
    df[!, metric_name] = [values[i, j, k] for i in 1:n_rows for j in 1:n_years for k in 1:n_strategies]
    
    return df
end

function convert_wide_to_long(df, metric_name)
    strategy_names = ["no_vaxx", "routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                     "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                     "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs"]
    
    strategy_cols = Vector{Symbol}()
    strategy_idx = Dict{Symbol, Int}()
    
    for (i, strat_name) in enumerate(strategy_names)
        strat_sym = Symbol(strat_name)
        if strat_sym in propertynames(df)
            push!(strategy_cols, strat_sym)
            strategy_idx[strat_sym] = i
        end
    end
    
    result_df = DataFrame()
    result_rows = []
    
    for row in eachrow(df)
        year = row.year
        iter = row.iteration
        
        for strat_col in strategy_cols
            # Get the strategy name and value
            strat_name = String(strat_col)
            strat_value = row[strat_col]
            
            push!(result_rows, (iteration=iter, year=year, strategy=strat_name, value=strat_value))
        end
    end
    
    result_df = DataFrame(result_rows)
    
    rename!(result_df, :value => metric_name)
    
    return result_df
end

function reorganize_combined_dataframe(dfs, metric_names)
    combined_df = DataFrame()
    
    strategies = ["no_vaxx", "routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                 "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                 "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs"]
    
    first_df = dfs[1]
    iterations = unique(first_df.iteration)
    years = unique(first_df.year)
    
    all_strategies = String[]
    all_iterations = Int[]
    all_years = Int[]
    
    metric_values = Dict(metric => Any[] for metric in metric_names)
    
    data_lookup = Dict{Int, Dict{Int, Dict{String, Dict{Int, Any}}}}()
    
    for i in 1:length(dfs)
        df = dfs[i]
        metric = metric_names[i]
        
        lookup = Dict{Int, Dict{String, Dict{Int, Any}}}()
        
        for row in eachrow(df)
            strat = row.strategy
            iter = row.iteration
            yr = row.year
            val = row[metric]
            
            if !haskey(lookup, iter)
                lookup[iter] = Dict{String, Dict{Int, Any}}()
            end
            if !haskey(lookup[iter], strat)
                lookup[iter][strat] = Dict{Int, Any}()
            end
            
            lookup[iter][strat][yr] = val
        end
        
        data_lookup[i] = lookup
    end
    
    for iter in iterations
        for strat in strategies
            for yr in years
                push!(all_iterations, iter)
                push!(all_strategies, strat)
                push!(all_years, yr)
                
                for (i, metric) in enumerate(metric_names)
                    value = missing
                    if haskey(data_lookup, i) && 
                       haskey(data_lookup[i], iter) && 
                       haskey(data_lookup[i][iter], strat) && 
                       haskey(data_lookup[i][iter][strat], yr)
                        value = data_lookup[i][iter][strat][yr]
                    end
                    
                    push!(metric_values[metric], value)
                end
            end
        end
    end
    
    combined_df.iteration = all_iterations
    combined_df.strategy = all_strategies
    combined_df.year = all_years
    
    for metric in metric_names
        combined_df[!, metric] = metric_values[metric]
    end
    
    sort!(combined_df, [:iteration, :strategy, :year])
    
    return combined_df
end

function apply_discounting(values, discount_factors)
    n_rows, n_years, n_cols = size(values)
    discounted = similar(values)
    
    for i in 1:n_rows # iterations
        for j in 1:n_years # years
            for k in 1:n_cols # strategies
                discounted[i, j, k] = values[i, j, k] * discount_factors[j]
            end
        end
    end

    return discounted
end

function apply_discounting_DALYs(DALYs, deaths_by_age, outcome_discount_rate, life_exp, n_age_groups)
    
    n_rows, n_years, n_strategies = size(DALYs)

    disc_life_exp = zeros(n_years, n_age_groups)
    undisc_life_exp = zeros(n_years, n_age_groups)

    for year in 1:n_years
        for age_group in 1:n_age_groups
            LE_years = Int(round(life_exp[age_group]))
            undisc_life_exp[year, age_group] = LE_years
            for k in 1:LE_years
                future_year = year + k - 1
                disc_life_exp[year, age_group] += 1 / ((1 + outcome_discount_rate) ^ (future_year - 1))
            end
        end
    end

    discounted_DALYs = zeros(n_rows, n_years, n_strategies)

    for iter in 1:n_rows
        for year in 1:n_years
            for strat in 1:n_strategies
                disc_fac = 1 / ((1 + outcome_discount_rate) ^ (year - 1))
                
                disc_DALYs_deaths = 0.0
                undisc_DALYs_deaths = 0.0
                
                for age_group in 1:n_age_groups
                    deaths_this_age = deaths_by_age[iter, year, strat, age_group]
                    disc_DALYs_deaths += deaths_this_age * disc_life_exp[year, age_group]
                    undisc_DALYs_deaths += deaths_this_age * undisc_life_exp[year, age_group]
                end
                
                illness_DALYs = DALYs[iter, year, strat] - undisc_DALYs_deaths
                disc_illness_DALYs = illness_DALYs * disc_fac
                discounted_DALYs[iter, year, strat] = disc_illness_DALYs + disc_DALYs_deaths
            end
        end
    end
    
    return discounted_DALYs
end


function calculate_costs_no_campaign(cases_adult_output, cases_peds_output,
                cases_age_stratified_output,
        doses_routine_output, doses_booster_output, 
        params_psa_costs, LE)
    
    n_rows = 1000
    n_years = 20  # Number of years in the projection
    n_strategies = 9  # Number of strategies (including no-vaxx)
    n_booster_strategies = 4  # Number of strategies with boosters
    n_age_groups = 2  # Adult and pediatric age groups
    
    hospitalizations = zeros(n_rows, n_years, n_strategies)
    deaths = zeros(n_rows, n_years, n_strategies)
    deaths_by_age = zeros(n_rows, n_years, n_strategies, 17)  
    DALYs = zeros(n_rows, n_years, n_strategies)
    vaccine_costs_routine = zeros(n_rows, n_years, n_strategies)
    vaccine_costs_booster = zeros(n_rows, n_years, n_strategies)
    vaccine_costs_all = zeros(n_rows, n_years, n_strategies)
    treatment_costs = zeros(n_rows, n_years, n_strategies)
    
    booster_mapping = [0, 0, 0, 0, 5, 6, 7, 8]  # For strategies 1-8 (excluding no-vaxx)

     age_midpoints = [ (0+0.75)/2, (0.75+1.166666667)/2, (1.166666667+2)/2, (2+5)/2, (5+10)/2,
        (10+15)/2, (15+20)/2, (20+25)/2, (25+30)/2, (30+35)/2, (35+40)/2, (40+45)/2, (45+50)/2,
        (50+55)/2, (55+60)/2, (60+65)/2, (65+100)/2
      ]

    
    base_LE = LE
    life_exp_bands = [max(0, base_LE - m) for m in age_midpoints]  # 17-element vector

    strategy_names_col = ["no_vaxx", "routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                     "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                     "routine_9mos_boosters_5yrs_10yrs",   
                     "routine_15mos_boosters_5yrs_10yrs"]
 
    for iter in 1:n_rows
        seek_care = params_psa_costs.p_seek_care[iter]
        inpatient = params_psa_costs.p_inpatient[iter]
        complications = params_psa_costs.p_complications[iter]
        cfr = params_psa_costs.CFR[iter]
        duration_untreated = params_psa_costs.duration_untreated[iter]
        duration_severe = params_psa_costs.duration_severe[iter]
        duration_moderate = params_psa_costs.duration_moderate[iter]
        DW_untreated = params_psa_costs.DW_untreated[iter]
        DW_severe = params_psa_costs.DW_severe[iter]
        DW_severe_with_ip = params_psa_costs.DW_severe_with_ip[iter]
        DW_moderate = params_psa_costs.DW_moderate[iter]
        inpatient_cost_peds = params_psa_costs.inpatient_cost_peds[iter]
        outpatient_cost_peds = params_psa_costs.outpatient_cost_peds[iter]
        inpatient_cost_adult = params_psa_costs.inpatient_cost_adult[iter]
        outpatient_cost_adult = params_psa_costs.outpatient_cost_adult[iter]
        vaccine_procurement_costs = params_psa_costs.vaccine_procurement_costs[iter]
        vaccine_safety_costs = params_psa_costs.vaccine_safety_costs[iter]
        vaccine_delivery_cost_routine = params_psa_costs.vaccine_delivery_cost_routine[iter]
        vaccine_delivery_cost_booster = params_psa_costs.vaccine_delivery_cost_booster[iter]
        
        iter_cases_adult = cases_adult_output[cases_adult_output.iteration .== iter, :]
        iter_cases_peds = cases_peds_output[cases_peds_output.iteration .== iter, :]
        iter_doses_routine = doses_routine_output[doses_routine_output.iteration .== iter, :]
        iter_doses_booster = doses_booster_output[doses_booster_output.iteration .== iter, :]
        iter_cases_age = cases_age_stratified_output[cases_age_stratified_output.iteration .== iter, :]

        for year in 1:n_years
            year_adult_idx = iter_cases_adult.year .== year
            year_peds_idx = iter_cases_peds.year .== year
            year_routine_idx = iter_doses_routine.year .== year
            year_booster_idx = iter_doses_booster.year .== year
            year_age_idx = iter_cases_age.year .== year

            if sum(year_adult_idx) == 0 || sum(year_peds_idx) == 0
                continue  # Skip if no case data for this year
            end
            
            #
            year_adult_cases = Array(iter_cases_adult[year_adult_idx, 2:10])[1,:]  
            year_peds_cases = Array(iter_cases_peds[year_peds_idx, 2:10])[1,:]  
            
            # Extract the dose values for this year
            # Columns 2-9 in routine doses (strategies 1-8, excluding no-vaxx)
            if sum(year_routine_idx) > 0
                year_routine_doses = Array(iter_doses_routine[year_routine_idx, 2:9])[1,:]  
            else
                year_routine_doses = zeros(8)  
            end
            
            if sum(year_booster_idx) > 0
                year_booster_doses = Array(iter_doses_booster[year_booster_idx, 2:9])[1,:]  
            else
                year_booster_doses = zeros(8)  
            end
            
            
            for strat in 1:n_strategies
                total_cases = year_peds_cases[strat] + year_adult_cases[strat]
                
                hospitalizations[iter, year, strat] = seek_care * inpatient * total_cases
                deaths[iter, year, strat] = cfr * total_cases
                
                # YLD 
                untreated_dalys = (1 - seek_care) * duration_untreated/365 * 
                                  DW_untreated * total_cases
                severe_with_compl = seek_care * complications * inpatient * 
                                   duration_severe/365 * 
                                   DW_severe_with_ip * total_cases
                severe_no_compl = seek_care * (1 - complications) * inpatient * 
                                 duration_severe/365 * 
                                 DW_severe * total_cases
                moderate = seek_care * (1 - inpatient) * duration_moderate/365 * 
                          DW_moderate * total_cases

                # YLL 
                death_dalys = 0.0
                if sum(year_age_idx) > 0
                    for age in 1:17
                        col = "cases_age_$(age)_$(strategy_names_col[strat])"
                        age_cases = iter_cases_age[year_age_idx, col][1]
                        deaths_by_age[iter, year, strat, age] = cfr * age_cases
                        death_dalys += cfr * age_cases * life_exp_bands[age]
                    end
                end
                

                DALYs[iter, year, strat] = untreated_dalys + severe_with_compl + 
                                           severe_no_compl + moderate + death_dalys
                                
                if strat > 1  
                    routine_strat_idx = strat - 1  
                    vaccine_costs_routine[iter, year, strat] = year_routine_doses[routine_strat_idx] * 
                                                           (vaccine_procurement_costs + 
                                                            vaccine_safety_costs + 
                                                            vaccine_delivery_cost_routine)
                    
                    booster_idx = booster_mapping[routine_strat_idx]
                    if booster_idx > 0
                        vaccine_costs_booster[iter, year, strat] = year_booster_doses[booster_idx] * 
                                                               (vaccine_procurement_costs + 
                                                                vaccine_safety_costs + 
                                                                vaccine_delivery_cost_booster)
                    end
                end
                
                vaccine_costs_all[iter, year, strat] = vaccine_costs_routine[iter, year, strat] + 
                                                     vaccine_costs_booster[iter, year, strat] 
                
                inpatient_peds = seek_care * inpatient * inpatient_cost_peds * 
                                year_peds_cases[strat]
                outpatient_peds = seek_care * (1 - inpatient) * outpatient_cost_peds * 
                                 year_peds_cases[strat]
                
                inpatient_adult = seek_care * inpatient * inpatient_cost_adult * 
                                 year_adult_cases[strat]
                outpatient_adult = seek_care * (1 - inpatient) * outpatient_cost_adult * 
                                  year_adult_cases[strat]
                
                treatment_costs[iter, year, strat] = (inpatient_peds + outpatient_peds) + 
                                                   (inpatient_adult + outpatient_adult)

            end
        end
    end
    
    cost_discount_rate = 0.03  
    outcome_discount_rate = 0.03  =
    
    cost_discount_factors = [1 / ((1 + cost_discount_rate) ^ (year - 1)) for year in 1:n_years]
    outcome_discount_factors = [1 / ((1 + outcome_discount_rate) ^ (year - 1)) for year in 1:n_years]

    discounted_hospitalizations = apply_discounting(hospitalizations, outcome_discount_factors)
    
    discounted_DALYs = apply_discounting_DALYs(DALYs, deaths_by_age, outcome_discount_rate, life_exp_bands, 17)
    
    discounted_vaccine_costs = apply_discounting(vaccine_costs_all, cost_discount_factors)
    discounted_treatment_costs = apply_discounting(treatment_costs, cost_discount_factors)
    discounted_total_costs = discounted_vaccine_costs + discounted_treatment_costs
    
    strategy_names = ["no_vaxx", "routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                     "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                     "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs"]
    
    hospitalizations_df = convert_to_dataframe(hospitalizations, n_rows, n_years, n_strategies, strategy_names, "hospitalizations")
    deaths_df = convert_to_dataframe(deaths, n_rows, n_years, n_strategies, strategy_names, "deaths")
    DALYs_df = convert_to_dataframe(DALYs, n_rows, n_years, n_strategies, strategy_names, "DALYs")
    discounted_DALYs_df = convert_to_dataframe(discounted_DALYs, n_rows, n_years, n_strategies, strategy_names, "discounted_DALYs")
    
    treatment_costs_df = convert_to_dataframe(treatment_costs, n_rows, n_years, n_strategies, strategy_names, "treatment_costs")
    vaccine_costs_df = convert_to_dataframe(vaccine_costs_all, n_rows, n_years, n_strategies, strategy_names, "vaccine_costs_all")
      
    discounted_treatment_costs_df = convert_to_dataframe(discounted_treatment_costs, n_rows, n_years, n_strategies, strategy_names, "discounted_treatment_costs")
    discounted_vaccine_costs_df = convert_to_dataframe(discounted_vaccine_costs, n_rows, n_years, n_strategies, strategy_names, "discounted_vaccine_costs_all")
    discounted_total_costs_df = convert_to_dataframe(discounted_total_costs, n_rows, n_years, n_strategies, strategy_names, "discounted_total_costs")
     
    return (
        hospitalizations_df,
        deaths_df,
        DALYs_df,
        discounted_DALYs_df,
        treatment_costs_df,
        vaccine_costs_df,
        discounted_treatment_costs_df,
        discounted_vaccine_costs_df,
        discounted_total_costs_df,
        deaths_by_age  
    )
end



archetype1_fast_adult_cases_no_campaign=CSV.read("archetype1_fast_waning_adult_cases_no_campaign_jun1.csv",DataFrame)
archetype1_fast_peds_cases_no_campaign=CSV.read("archetype1_fast_waning_pediatric_cases_no_campaign_jun1.csv",DataFrame)
archetype1_fast_age_stratified_cases_no_campaign=CSV.read("archetype1_fast_waning_age_stratified_cases_no_campaign_jun1.csv",DataFrame)
archetype1_fast_routine_doses_no_campaign=CSV.read("archetype1_fast_waning_routine_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype1_fast_booster_doses_no_campaign=CSV.read("archetype1_fast_waning_booster_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)] #7:11
archetype1_fast_population_no_campaign=CSV.read("archetype1_fast_waning_population_no_campaign_jun1.csv",DataFrame)
rename!(archetype1_fast_population_no_campaign,vcat(names(archetype1_fast_population_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype1_fast_population_long_no_campaign = convert_wide_to_long(archetype1_fast_population_no_campaign, "population")
archetype1_fast_total_cases_no_campaign=CSV.read("archetype1_fast_waning_total_cases_no_campaign_jun1.csv",DataFrame)
rename!(archetype1_fast_total_cases_no_campaign,vcat(names(archetype1_fast_total_cases_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype1_fast_total_cases_long_no_campaign = convert_wide_to_long(archetype1_fast_total_cases_no_campaign, "total_cases")
archetype1_fast_total_doses_no_campaign=CSV.read("archetype1_fast_waning_total_doses_no_campaign_jun1.csv",DataFrame)
rename!(archetype1_fast_total_doses_no_campaign,vcat(names(archetype1_fast_total_doses_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype1_fast_total_doses_long_no_campaign = convert_wide_to_long(archetype1_fast_total_doses_no_campaign, "total_doses")

archetype1_slow_adult_cases_no_campaign=CSV.read("archetype1_slow_waning_adult_cases_no_campaign_jun1.csv",DataFrame)
archetype1_slow_peds_cases_no_campaign=CSV.read("archetype1_slow_waning_pediatric_cases_no_campaign_jun1.csv",DataFrame)
archetype1_slow_age_stratified_cases_no_campaign=CSV.read("archetype1_slow_waning_age_stratified_cases_no_campaign_jun1.csv",DataFrame)
archetype1_slow_routine_doses_no_campaign=CSV.read("archetype1_slow_waning_routine_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype1_slow_booster_doses_no_campaign=CSV.read("archetype1_slow_waning_booster_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)];
archetype1_slow_population_no_campaign=CSV.read("archetype1_slow_waning_population_no_campaign_jun1.csv",DataFrame)
rename!(archetype1_slow_population_no_campaign,vcat(names(archetype1_slow_population_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype1_slow_population_long_no_campaign = convert_wide_to_long(archetype1_slow_population_no_campaign, "population")
archetype1_slow_total_cases_no_campaign=CSV.read("archetype1_slow_waning_total_cases_no_campaign_jun1.csv",DataFrame)
rename!(archetype1_slow_total_cases_no_campaign,vcat(names(archetype1_slow_total_cases_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype1_slow_total_cases_long_no_campaign = convert_wide_to_long(archetype1_slow_total_cases_no_campaign, "total_cases")
archetype1_slow_total_doses_no_campaign=CSV.read("archetype1_slow_waning_total_doses_no_campaign_jun1.csv",DataFrame)
rename!(archetype1_slow_total_doses_no_campaign,vcat(names(archetype1_slow_total_doses_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype1_slow_total_doses_long_no_campaign = convert_wide_to_long(archetype1_slow_total_doses_no_campaign, "total_doses")

archetype2_fast_adult_cases_no_campaign=CSV.read("archetype2_fast_waning_adult_cases_no_campaign_jun1.csv",DataFrame)
archetype2_fast_peds_cases_no_campaign=CSV.read("archetype2_fast_waning_pediatric_cases_no_campaign_jun1.csv",DataFrame)
archetype2_fast_age_stratified_cases_no_campaign=CSV.read("archetype2_fast_waning_age_stratified_cases_no_campaign_jun1.csv",DataFrame)
archetype2_fast_routine_doses_no_campaign=CSV.read("archetype2_fast_waning_routine_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype2_fast_booster_doses_no_campaign=CSV.read("archetype2_fast_waning_booster_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype2_fast_population_no_campaign=CSV.read("archetype2_fast_waning_population_no_campaign_jun1.csv",DataFrame)
rename!(archetype2_fast_population_no_campaign,vcat(names(archetype2_fast_population_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype2_fast_population_long_no_campaign = convert_wide_to_long(archetype2_fast_population_no_campaign, "population")
archetype2_fast_total_cases_no_campaign=CSV.read("archetype2_fast_waning_total_cases_no_campaign_jun1.csv",DataFrame)
rename!(archetype2_fast_total_cases_no_campaign,vcat(names(archetype2_fast_total_cases_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype2_fast_total_cases_long_no_campaign = convert_wide_to_long(archetype2_fast_total_cases_no_campaign, "total_cases")
archetype2_fast_total_doses_no_campaign=CSV.read("archetype2_fast_waning_total_doses_no_campaign_jun1.csv",DataFrame)
rename!(archetype2_fast_total_doses_no_campaign,vcat(names(archetype2_fast_total_doses_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype2_fast_total_doses_long_no_campaign = convert_wide_to_long(archetype2_fast_total_doses_no_campaign, "total_doses")

archetype2_slow_adult_cases_no_campaign=CSV.read("archetype2_slow_waning_adult_cases_no_campaign_jun1.csv",DataFrame)
archetype2_slow_peds_cases_no_campaign=CSV.read("archetype2_slow_waning_pediatric_cases_no_campaign_jun1.csv",DataFrame)
archetype2_slow_age_stratified_cases_no_campaign=CSV.read("archetype2_slow_waning_age_stratified_cases_no_campaign_jun1.csv",DataFrame)
archetype2_slow_routine_doses_no_campaign=CSV.read("archetype2_slow_waning_routine_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype2_slow_booster_doses_no_campaign=CSV.read("archetype2_slow_waning_booster_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)];
archetype2_slow_population_no_campaign=CSV.read("archetype2_slow_waning_population_no_campaign_jun1.csv",DataFrame)
rename!(archetype2_slow_population_no_campaign,vcat(names(archetype2_slow_population_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype2_slow_population_long_no_campaign = convert_wide_to_long(archetype2_slow_population_no_campaign, "population")
archetype2_slow_total_cases_no_campaign=CSV.read("archetype2_slow_waning_total_cases_no_campaign_jun1.csv",DataFrame)
rename!(archetype2_slow_total_cases_no_campaign,vcat(names(archetype2_slow_total_cases_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype2_slow_total_cases_long_no_campaign = convert_wide_to_long(archetype2_slow_total_cases_no_campaign, "total_cases")
archetype2_slow_total_doses_no_campaign=CSV.read("archetype2_slow_waning_total_doses_no_campaign_jun1.csv",DataFrame)
rename!(archetype2_slow_total_doses_no_campaign,vcat(names(archetype2_slow_total_doses_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype2_slow_total_doses_long_no_campaign = convert_wide_to_long(archetype2_slow_total_doses_no_campaign, "total_doses")

archetype3_fast_adult_cases_no_campaign=CSV.read("archetype3_fast_waning_adult_cases_no_campaign_jun1.csv",DataFrame)
archetype3_fast_peds_cases_no_campaign=CSV.read("archetype3_fast_waning_pediatric_cases_no_campaign_jun1.csv",DataFrame)
archetype3_fast_age_stratified_cases_no_campaign=CSV.read("archetype3_fast_waning_age_stratified_cases_no_campaign_jun1.csv",DataFrame)
archetype3_fast_routine_doses_no_campaign=CSV.read("archetype3_fast_waning_routine_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype3_fast_booster_doses_no_campaign=CSV.read("archetype3_fast_waning_booster_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype3_fast_population_no_campaign=CSV.read("archetype3_fast_waning_population_no_campaign_jun1.csv",DataFrame)
rename!(archetype3_fast_population_no_campaign,vcat(names(archetype3_fast_population_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype3_fast_population_long_no_campaign = convert_wide_to_long(archetype3_fast_population_no_campaign, "population")
archetype3_fast_total_cases_no_campaign=CSV.read("archetype3_fast_waning_total_cases_no_campaign_jun1.csv",DataFrame)
rename!(archetype3_fast_total_cases_no_campaign,vcat(names(archetype3_fast_total_cases_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype3_fast_total_cases_long_no_campaign = convert_wide_to_long(archetype3_fast_total_cases_no_campaign, "total_cases")
archetype3_fast_total_doses_no_campaign=CSV.read("archetype3_fast_waning_total_doses_no_campaign_jun1.csv",DataFrame)
rename!(archetype3_fast_total_doses_no_campaign,vcat(names(archetype3_fast_total_doses_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype3_fast_total_doses_long_no_campaign = convert_wide_to_long(archetype3_fast_total_doses_no_campaign, "total_doses")

archetype3_slow_adult_cases_no_campaign=CSV.read("archetype3_slow_waning_adult_cases_no_campaign_jun1.csv",DataFrame)
archetype3_slow_peds_cases_no_campaign=CSV.read("archetype3_slow_waning_pediatric_cases_no_campaign_jun1.csv",DataFrame)
archetype3_slow_age_stratified_cases_no_campaign=CSV.read("archetype3_slow_waning_age_stratified_cases_no_campaign_jun1.csv",DataFrame)
archetype3_slow_routine_doses_no_campaign=CSV.read("archetype3_slow_waning_routine_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)]
archetype3_slow_booster_doses_no_campaign=CSV.read("archetype3_slow_waning_booster_doses_no_campaign_jun1.csv",DataFrame)[:,vcat(1,3:11)];
archetype3_slow_population_no_campaign=CSV.read("archetype3_slow_waning_population_no_campaign_jun1.csv",DataFrame)
rename!(archetype3_slow_population_no_campaign,vcat(names(archetype3_slow_population_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype3_slow_population_long_no_campaign = convert_wide_to_long(archetype3_slow_population_no_campaign, "population")
archetype3_slow_total_cases_no_campaign=CSV.read("archetype3_slow_waning_total_cases_no_campaign_jun1.csv",DataFrame)
rename!(archetype3_slow_total_cases_no_campaign,vcat(names(archetype3_slow_total_cases_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype3_slow_total_cases_long_no_campaign = convert_wide_to_long(archetype3_slow_total_cases_no_campaign, "total_cases")
archetype3_slow_total_doses_no_campaign=CSV.read("archetype3_slow_waning_total_doses_no_campaign_jun1.csv",DataFrame)
rename!(archetype3_slow_total_doses_no_campaign,vcat(names(archetype3_slow_total_doses_no_campaign)[1:8],
        "routine_9mos_booster_5yrs_10yrs", "routine_15mos_booster_5yrs_10yrs","iteration"))
archetype3_slow_total_doses_long_no_campaign = convert_wide_to_long(archetype3_slow_total_doses_no_campaign, "total_doses");


# archetype 1 

## fast

### Asia/WPR region
archetype1_fast_hosp_asia_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
        archetype1_fast_peds_cases_no_campaign,
        archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[1]
archetype1_fast_deaths_asia_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[2]
archetype1_fast_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[3]
archetype1_fast_discounted_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[4]
archetype1_fast_discounted_treatment_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[7]
archetype1_fast_discounted_vaccine_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[8]
rename!(archetype1_fast_discounted_vaccine_costs_df_asia_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype1_fast_discounted_total_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[9]

archetype1_fast_asia_no_campaign = [archetype1_fast_population_long_no_campaign,archetype1_fast_total_cases_long_no_campaign,
    archetype1_fast_hosp_asia_no_campaign, archetype1_fast_deaths_asia_no_campaign,
    archetype1_fast_DALYS_asia_no_campaign, archetype1_fast_discounted_DALYS_asia_no_campaign,
    archetype1_fast_total_doses_long_no_campaign, archetype1_fast_discounted_treatment_costs_df_asia_no_campaign,
    archetype1_fast_discounted_vaccine_costs_df_asia_no_campaign, archetype1_fast_discounted_total_costs_df_asia_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype1_fast_asia_df_no_campaign=reorganize_combined_dataframe(archetype1_fast_asia_no_campaign,all_metrics)
rename!(archetype1_fast_asia_df_no_campaign,vcat("run_id",names(archetype1_fast_asia_df_no_campaign)[2:end]))

### Africa region

archetype1_fast_hosp_africa_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[1]
archetype1_fast_deaths_africa_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[2]
archetype1_fast_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[3]
archetype1_fast_discounted_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[4]
archetype1_fast_discounted_treatment_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[7]
archetype1_fast_discounted_vaccine_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[8]
rename!(archetype1_fast_discounted_vaccine_costs_df_africa_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype1_fast_discounted_total_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype1_fast_adult_cases_no_campaign, 
    archetype1_fast_peds_cases_no_campaign,
            archetype1_fast_age_stratified_cases_no_campaign,
        archetype1_fast_routine_doses_no_campaign, archetype1_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[9]

archetype1_fast_africa_no_campaign = [archetype1_fast_population_long_no_campaign,archetype1_fast_total_cases_long_no_campaign,
    archetype1_fast_hosp_africa_no_campaign, archetype1_fast_deaths_africa_no_campaign,
    archetype1_fast_DALYS_africa_no_campaign, archetype1_fast_discounted_DALYS_africa_no_campaign,
    archetype1_fast_total_doses_long_no_campaign, archetype1_fast_discounted_treatment_costs_df_africa_no_campaign,
    archetype1_fast_discounted_vaccine_costs_df_africa_no_campaign, archetype1_fast_discounted_total_costs_df_africa_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype1_fast_africa_df_no_campaign=reorganize_combined_dataframe(archetype1_fast_africa_no_campaign,all_metrics)
rename!(archetype1_fast_africa_df_no_campaign,vcat("run_id",names(archetype1_fast_africa_df_no_campaign)[2:end]))

## slow

### Asia/WPR region
archetype1_slow_hosp_asia_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
        archetype1_slow_peds_cases_no_campaign,
        archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[1]
archetype1_slow_deaths_asia_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
        archetype1_slow_peds_cases_no_campaign,
        archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[2]
archetype1_slow_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
        archetype1_slow_peds_cases_no_campaign,
        archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[3]
archetype1_slow_discounted_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
        archetype1_slow_peds_cases_no_campaign,
        archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign,archetype1_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[4]
archetype1_slow_discounted_treatment_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
        archetype1_slow_peds_cases_no_campaign,
        archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[7]
archetype1_slow_discounted_vaccine_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
        archetype1_slow_peds_cases_no_campaign,
        archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[8]
rename!(archetype1_slow_discounted_vaccine_costs_df_asia_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype1_slow_discounted_total_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
            archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[9]

archetype1_slow_asia_no_campaign = [archetype1_slow_population_long_no_campaign,archetype1_slow_total_cases_long_no_campaign,
    archetype1_slow_hosp_asia_no_campaign, archetype1_slow_deaths_asia_no_campaign,
    archetype1_slow_DALYS_asia_no_campaign, archetype1_slow_discounted_DALYS_asia_no_campaign,
    archetype1_slow_total_doses_long_no_campaign, archetype1_slow_discounted_treatment_costs_df_asia_no_campaign,
    archetype1_slow_discounted_vaccine_costs_df_asia_no_campaign, archetype1_slow_discounted_total_costs_df_asia_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype1_slow_asia_df_no_campaign=reorganize_combined_dataframe(archetype1_slow_asia_no_campaign,all_metrics)
rename!(archetype1_slow_asia_df_no_campaign,vcat("run_id",names(archetype1_slow_asia_df_no_campaign)[2:end]))

### Africa region

archetype1_slow_hosp_africa_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
                archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
         costs_psa_africa,costs_psa_africa.LE[1])[1]
archetype1_slow_deaths_africa_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
                archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[2]
archetype1_slow_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
                archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[3]
archetype1_slow_discounted_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
                archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[4]
archetype1_slow_discounted_treatment_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
                archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[7]
archetype1_slow_discounted_vaccine_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
                archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[8]
rename!(archetype1_slow_discounted_vaccine_costs_df_africa_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype1_slow_discounted_total_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype1_slow_adult_cases_no_campaign, 
    archetype1_slow_peds_cases_no_campaign,
                archetype1_slow_age_stratified_cases_no_campaign,
        archetype1_slow_routine_doses_no_campaign, archetype1_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[9]

archetype1_slow_africa_no_campaign = [archetype1_slow_population_long_no_campaign,archetype1_slow_total_cases_long_no_campaign,
    archetype1_slow_hosp_africa_no_campaign, archetype1_slow_deaths_africa_no_campaign,
    archetype1_slow_DALYS_africa_no_campaign, archetype1_slow_discounted_DALYS_africa_no_campaign,
    archetype1_slow_total_doses_long_no_campaign, archetype1_slow_discounted_treatment_costs_df_africa_no_campaign,
    archetype1_slow_discounted_vaccine_costs_df_africa_no_campaign,archetype1_slow_discounted_total_costs_df_africa_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype1_slow_africa_df_no_campaign=reorganize_combined_dataframe(archetype1_slow_africa_no_campaign,all_metrics)
rename!(archetype1_slow_africa_df_no_campaign,vcat("run_id",names(archetype1_slow_africa_df_no_campaign)[2:end]));



# archetype 2

## fast

### Asia/WPR region
archetype2_fast_hosp_asia_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
                archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[1]
archetype2_fast_deaths_asia_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
    archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[2]
archetype2_fast_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
    archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[3]
archetype2_fast_discounted_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
    archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[4]
archetype2_fast_discounted_treatment_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign,
    archetype2_fast_peds_cases_no_campaign,
    archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[7]
archetype2_fast_discounted_vaccine_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
    archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[8]
rename!(archetype2_fast_discounted_vaccine_costs_df_asia_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype2_fast_discounted_total_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
        archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[9]

archetype2_fast_asia_no_campaign = [archetype2_fast_population_long_no_campaign,archetype2_fast_total_cases_long_no_campaign,
    archetype2_fast_hosp_asia_no_campaign, archetype2_fast_deaths_asia_no_campaign,
    archetype2_fast_DALYS_asia_no_campaign, archetype2_fast_discounted_DALYS_asia_no_campaign,
    archetype2_fast_total_doses_long_no_campaign, archetype2_fast_discounted_treatment_costs_df_asia_no_campaign,
    archetype2_fast_discounted_vaccine_costs_df_asia_no_campaign, archetype2_fast_discounted_total_costs_df_asia_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype2_fast_asia_df_no_campaign=reorganize_combined_dataframe(archetype2_fast_asia_no_campaign,all_metrics)
rename!(archetype2_fast_asia_df_no_campaign,vcat("run_id",names(archetype2_fast_asia_df_no_campaign)[2:end]))

### Africa region

archetype2_fast_hosp_africa_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
        archetype2_fast_peds_cases_no_campaign,
        archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[1]
archetype2_fast_deaths_africa_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
        archetype2_fast_peds_cases_no_campaign,
        archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[2]
archetype2_fast_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
        archetype2_fast_peds_cases_no_campaign,
        archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[3]
archetype2_fast_discounted_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
        archetype2_fast_peds_cases_no_campaign,
        archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[4]
archetype2_fast_discounted_treatment_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
       archetype2_fast_peds_cases_no_campaign,
        archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[7]
archetype2_fast_discounted_vaccine_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
        archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[8]
rename!(archetype2_fast_discounted_vaccine_costs_df_africa_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype2_fast_discounted_total_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype2_fast_adult_cases_no_campaign, 
    archetype2_fast_peds_cases_no_campaign,
            archetype2_fast_age_stratified_cases_no_campaign,
        archetype2_fast_routine_doses_no_campaign, archetype2_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[9]

archetype2_fast_africa_no_campaign = [archetype2_fast_population_long_no_campaign,archetype2_fast_total_cases_long_no_campaign,
    archetype2_fast_hosp_africa_no_campaign, archetype2_fast_deaths_africa_no_campaign,
    archetype2_fast_DALYS_africa_no_campaign, archetype2_fast_discounted_DALYS_africa_no_campaign,
    archetype2_fast_total_doses_long_no_campaign, archetype2_fast_discounted_treatment_costs_df_africa_no_campaign,
    archetype2_fast_discounted_vaccine_costs_df_africa_no_campaign, archetype2_fast_discounted_total_costs_df_africa_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype2_fast_africa_df_no_campaign=reorganize_combined_dataframe(archetype2_fast_africa_no_campaign,all_metrics)
rename!(archetype2_fast_africa_df_no_campaign,vcat("run_id",names(archetype2_fast_africa_df_no_campaign)[2:end]))

## slow

### Asia/WPR region
archetype2_slow_hosp_asia_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
        archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[1]
archetype2_slow_deaths_asia_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
            archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[2]
archetype2_slow_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[3]
archetype2_slow_discounted_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign,archetype2_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[4]
archetype2_slow_discounted_treatment_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[7]
archetype2_slow_discounted_vaccine_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[8]
rename!(archetype2_slow_discounted_vaccine_costs_df_asia_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype2_slow_discounted_total_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign,
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[9]

archetype2_slow_asia_no_campaign = [archetype2_slow_population_long_no_campaign,archetype2_slow_total_cases_long_no_campaign,
    archetype2_slow_hosp_asia_no_campaign, archetype2_slow_deaths_asia_no_campaign,
    archetype2_slow_DALYS_asia_no_campaign, archetype2_slow_discounted_DALYS_asia_no_campaign,
    archetype2_slow_total_doses_long_no_campaign, archetype2_slow_discounted_treatment_costs_df_asia_no_campaign,
    archetype2_slow_discounted_vaccine_costs_df_asia_no_campaign, archetype2_slow_discounted_total_costs_df_asia_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype2_slow_asia_df_no_campaign=reorganize_combined_dataframe(archetype2_slow_asia_no_campaign,all_metrics)
rename!(archetype2_slow_asia_df_no_campaign,vcat("run_id",names(archetype2_slow_asia_df_no_campaign)[2:end]))

### Africa region

archetype2_slow_hosp_africa_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[1]
archetype2_slow_deaths_africa_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[2]
archetype2_slow_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[3]
archetype2_slow_discounted_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[4]
archetype2_slow_discounted_treatment_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[7]
archetype2_slow_discounted_vaccine_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[8]
rename!(archetype2_slow_discounted_vaccine_costs_df_africa_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype2_slow_discounted_total_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype2_slow_adult_cases_no_campaign, 
    archetype2_slow_peds_cases_no_campaign,
                archetype2_slow_age_stratified_cases_no_campaign,
        archetype2_slow_routine_doses_no_campaign, archetype2_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[9]

archetype2_slow_africa_no_campaign = [archetype2_slow_population_long_no_campaign,archetype2_slow_total_cases_long_no_campaign,
    archetype2_slow_hosp_africa_no_campaign, archetype2_slow_deaths_africa_no_campaign,
    archetype2_slow_DALYS_africa_no_campaign, archetype2_slow_discounted_DALYS_africa_no_campaign,
    archetype2_slow_total_doses_long_no_campaign, archetype2_slow_discounted_treatment_costs_df_africa_no_campaign,
    archetype2_slow_discounted_vaccine_costs_df_africa_no_campaign,archetype2_slow_discounted_total_costs_df_africa_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype2_slow_africa_df_no_campaign=reorganize_combined_dataframe(archetype2_slow_africa_no_campaign,all_metrics)
rename!(archetype2_slow_africa_df_no_campaign,vcat("run_id",names(archetype2_slow_africa_df_no_campaign)[2:end]));



# archetype 3

## fast

### Asia/WPR region
archetype3_fast_hosp_asia_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
       archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[1]
archetype3_fast_deaths_asia_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[2]
archetype3_fast_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[3]
archetype3_fast_discounted_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[4]
archetype3_fast_discounted_treatment_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[7]
archetype3_fast_discounted_vaccine_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[8]
rename!(archetype3_fast_discounted_vaccine_costs_df_asia_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype3_fast_discounted_total_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_asia, costs_psa_asia.LE[1])[9]

archetype3_fast_asia_no_campaign = [archetype3_fast_population_long_no_campaign,archetype3_fast_total_cases_long_no_campaign,
    archetype3_fast_hosp_asia_no_campaign, archetype3_fast_deaths_asia_no_campaign,
    archetype3_fast_DALYS_asia_no_campaign, archetype3_fast_discounted_DALYS_asia_no_campaign,
    archetype3_fast_total_doses_long_no_campaign, archetype3_fast_discounted_treatment_costs_df_asia_no_campaign,
    archetype3_fast_discounted_vaccine_costs_df_asia_no_campaign, archetype3_fast_discounted_total_costs_df_asia_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype3_fast_asia_df_no_campaign=reorganize_combined_dataframe(archetype3_fast_asia_no_campaign,all_metrics)
rename!(archetype3_fast_asia_df_no_campaign,vcat("run_id",names(archetype3_fast_asia_df_no_campaign)[2:end]))

### Africa region
archetype3_fast_hosp_africa_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[1]
archetype3_fast_deaths_africa_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[2]
archetype3_fast_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[3]
archetype3_fast_discounted_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[4]
archetype3_fast_discounted_treatment_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[7]
archetype3_fast_discounted_vaccine_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign,
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[8]
rename!(archetype3_fast_discounted_vaccine_costs_df_africa_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype3_fast_discounted_total_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype3_fast_adult_cases_no_campaign, 
    archetype3_fast_peds_cases_no_campaign,
           archetype3_fast_age_stratified_cases_no_campaign,
        archetype3_fast_routine_doses_no_campaign, archetype3_fast_booster_doses_no_campaign,
        costs_psa_africa, costs_psa_africa.LE[1])[9]

archetype3_fast_africa_no_campaign = [archetype3_fast_population_long_no_campaign,archetype3_fast_total_cases_long_no_campaign,
    archetype3_fast_hosp_africa_no_campaign, archetype3_fast_deaths_africa_no_campaign,
    archetype3_fast_DALYS_africa_no_campaign, archetype3_fast_discounted_DALYS_africa_no_campaign,
    archetype3_fast_total_doses_long_no_campaign, archetype3_fast_discounted_treatment_costs_df_africa_no_campaign,
    archetype3_fast_discounted_vaccine_costs_df_africa_no_campaign, archetype3_fast_discounted_total_costs_df_africa_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype3_fast_africa_df_no_campaign=reorganize_combined_dataframe(archetype3_fast_africa_no_campaign,all_metrics)
rename!(archetype3_fast_africa_df_no_campaign,vcat("run_id",names(archetype3_fast_africa_df_no_campaign)[2:end]))

## slow

### Asia/WPR region
archetype3_slow_hosp_asia_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign,
    archetype3_slow_peds_cases_no_campaign,
           archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[1]
archetype3_slow_deaths_asia_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[2]
archetype3_slow_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[3]
archetype3_slow_discounted_DALYS_asia_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign,archetype3_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[4]
archetype3_slow_discounted_treatment_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[7]
archetype3_slow_discounted_vaccine_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[8]
rename!(archetype3_slow_discounted_vaccine_costs_df_asia_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype3_slow_discounted_total_costs_df_asia_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_asia,costs_psa_asia.LE[1])[9]

archetype3_slow_asia_no_campaign = [archetype3_slow_population_long_no_campaign,archetype3_slow_total_cases_long_no_campaign,
    archetype3_slow_hosp_asia_no_campaign, archetype3_slow_deaths_asia_no_campaign,
    archetype3_slow_DALYS_asia_no_campaign, archetype3_slow_discounted_DALYS_asia_no_campaign,
    archetype3_slow_total_doses_long_no_campaign, archetype3_slow_discounted_treatment_costs_df_asia_no_campaign,
    archetype3_slow_discounted_vaccine_costs_df_asia_no_campaign, archetype3_slow_discounted_total_costs_df_asia_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype3_slow_asia_df_no_campaign=reorganize_combined_dataframe(archetype3_slow_asia_no_campaign,all_metrics)
rename!(archetype3_slow_asia_df_no_campaign,vcat("run_id",names(archetype3_slow_asia_df_no_campaign)[2:end]))

### Africa region

archetype3_slow_hosp_africa_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[1]
archetype3_slow_deaths_africa_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[2]
archetype3_slow_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,            
    archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[3]
archetype3_slow_discounted_DALYS_africa_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[4]
archetype3_slow_discounted_treatment_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[7]
archetype3_slow_discounted_vaccine_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[8]
rename!(archetype3_slow_discounted_vaccine_costs_df_africa_no_campaign,
    :discounted_vaccine_costs_all => :discounted_vaccine_costs)
archetype3_slow_discounted_total_costs_df_africa_no_campaign=calculate_costs_no_campaign(archetype3_slow_adult_cases_no_campaign, 
    archetype3_slow_peds_cases_no_campaign,
               archetype3_slow_age_stratified_cases_no_campaign,
        archetype3_slow_routine_doses_no_campaign, archetype3_slow_booster_doses_no_campaign,
        costs_psa_africa,costs_psa_africa.LE[1])[9]

archetype3_slow_africa_no_campaign = [archetype3_slow_population_long_no_campaign,archetype3_slow_total_cases_long_no_campaign,
    archetype3_slow_hosp_africa_no_campaign, archetype3_slow_deaths_africa_no_campaign,
    archetype3_slow_DALYS_africa_no_campaign, archetype3_slow_discounted_DALYS_africa_no_campaign,
    archetype3_slow_total_doses_long_no_campaign, archetype3_slow_discounted_treatment_costs_df_africa_no_campaign,
    archetype3_slow_discounted_vaccine_costs_df_africa_no_campaign,archetype3_slow_discounted_total_costs_df_africa_no_campaign
]

all_metrics = ["population","total_cases","hospitalizations","deaths","DALYs", "discounted_DALYs",
    "total_doses","discounted_treatment_costs","discounted_vaccine_costs","discounted_total_costs"]

archetype3_slow_africa_df_no_campaign=reorganize_combined_dataframe(archetype3_slow_africa_no_campaign,all_metrics)
rename!(archetype3_slow_africa_df_no_campaign,vcat("run_id",names(archetype3_slow_africa_df_no_campaign)[2:end]));



# Helper function to fix strategy names (remove "_no_campaign" suffix)
function fix_strategy_names(df)
    if "strategy" in names(df)
        df.strategy = replace.(df.strategy, "_no_campaign" => "")
    end
    return df
end


# Apply the fix to all no_campaign dataframes
# Archetype 1 Fast
archetype1_fast_hosp_asia_no_campaign = fix_strategy_names(archetype1_fast_hosp_asia_no_campaign)
archetype1_fast_deaths_asia_no_campaign = fix_strategy_names(archetype1_fast_deaths_asia_no_campaign)
archetype1_fast_DALYS_asia_no_campaign = fix_strategy_names(archetype1_fast_DALYS_asia_no_campaign)
archetype1_fast_discounted_DALYS_asia_no_campaign = fix_strategy_names(archetype1_fast_discounted_DALYS_asia_no_campaign)
archetype1_fast_discounted_treatment_costs_df_asia_no_campaign = fix_strategy_names(archetype1_fast_discounted_treatment_costs_df_asia_no_campaign)
archetype1_fast_discounted_vaccine_costs_df_asia_no_campaign = fix_strategy_names(archetype1_fast_discounted_vaccine_costs_df_asia_no_campaign)

archetype1_fast_hosp_africa_no_campaign = fix_strategy_names(archetype1_fast_hosp_africa_no_campaign)
archetype1_fast_deaths_africa_no_campaign = fix_strategy_names(archetype1_fast_deaths_africa_no_campaign)
archetype1_fast_DALYS_africa_no_campaign = fix_strategy_names(archetype1_fast_DALYS_africa_no_campaign)
archetype1_fast_discounted_DALYS_africa_no_campaign = fix_strategy_names(archetype1_fast_discounted_DALYS_africa_no_campaign)
archetype1_fast_discounted_treatment_costs_df_africa_no_campaign = fix_strategy_names(archetype1_fast_discounted_treatment_costs_df_africa_no_campaign)
archetype1_fast_discounted_vaccine_costs_df_africa_no_campaign = fix_strategy_names(archetype1_fast_discounted_vaccine_costs_df_africa_no_campaign)

# Archetype 1 Slow
archetype1_slow_hosp_asia_no_campaign = fix_strategy_names(archetype1_slow_hosp_asia_no_campaign)
archetype1_slow_deaths_asia_no_campaign = fix_strategy_names(archetype1_slow_deaths_asia_no_campaign)
archetype1_slow_DALYS_asia_no_campaign = fix_strategy_names(archetype1_slow_DALYS_asia_no_campaign)
archetype1_slow_discounted_DALYS_asia_no_campaign = fix_strategy_names(archetype1_slow_discounted_DALYS_asia_no_campaign)
archetype1_slow_discounted_treatment_costs_df_asia_no_campaign = fix_strategy_names(archetype1_slow_discounted_treatment_costs_df_asia_no_campaign)
archetype1_slow_discounted_vaccine_costs_df_asia_no_campaign = fix_strategy_names(archetype1_slow_discounted_vaccine_costs_df_asia_no_campaign)

archetype1_slow_hosp_africa_no_campaign = fix_strategy_names(archetype1_slow_hosp_africa_no_campaign)
archetype1_slow_deaths_africa_no_campaign = fix_strategy_names(archetype1_slow_deaths_africa_no_campaign)
archetype1_slow_DALYS_africa_no_campaign = fix_strategy_names(archetype1_slow_DALYS_africa_no_campaign)
archetype1_slow_discounted_DALYS_africa_no_campaign = fix_strategy_names(archetype1_slow_discounted_DALYS_africa_no_campaign)
archetype1_slow_discounted_treatment_costs_df_africa_no_campaign = fix_strategy_names(archetype1_slow_discounted_treatment_costs_df_africa_no_campaign)
archetype1_slow_discounted_vaccine_costs_df_africa_no_campaign = fix_strategy_names(archetype1_slow_discounted_vaccine_costs_df_africa_no_campaign)

# Archetype 2 Fast
archetype2_fast_hosp_asia_no_campaign = fix_strategy_names(archetype2_fast_hosp_asia_no_campaign)
archetype2_fast_deaths_asia_no_campaign = fix_strategy_names(archetype2_fast_deaths_asia_no_campaign)
archetype2_fast_DALYS_asia_no_campaign = fix_strategy_names(archetype2_fast_DALYS_asia_no_campaign)
archetype2_fast_discounted_DALYS_asia_no_campaign = fix_strategy_names(archetype2_fast_discounted_DALYS_asia_no_campaign)
archetype2_fast_discounted_treatment_costs_df_asia_no_campaign = fix_strategy_names(archetype2_fast_discounted_treatment_costs_df_asia_no_campaign)
archetype2_fast_discounted_vaccine_costs_df_asia_no_campaign = fix_strategy_names(archetype2_fast_discounted_vaccine_costs_df_asia_no_campaign)

archetype2_fast_hosp_africa_no_campaign = fix_strategy_names(archetype2_fast_hosp_africa_no_campaign)
archetype2_fast_deaths_africa_no_campaign = fix_strategy_names(archetype2_fast_deaths_africa_no_campaign)
archetype2_fast_DALYS_africa_no_campaign = fix_strategy_names(archetype2_fast_DALYS_africa_no_campaign)
archetype2_fast_discounted_DALYS_africa_no_campaign = fix_strategy_names(archetype2_fast_discounted_DALYS_africa_no_campaign)
archetype2_fast_discounted_treatment_costs_df_africa_no_campaign = fix_strategy_names(archetype2_fast_discounted_treatment_costs_df_africa_no_campaign)
archetype2_fast_discounted_vaccine_costs_df_africa_no_campaign = fix_strategy_names(archetype2_fast_discounted_vaccine_costs_df_africa_no_campaign)

# Archetype 2 Slow
archetype2_slow_hosp_asia_no_campaign = fix_strategy_names(archetype2_slow_hosp_asia_no_campaign)
archetype2_slow_deaths_asia_no_campaign = fix_strategy_names(archetype2_slow_deaths_asia_no_campaign)
archetype2_slow_DALYS_asia_no_campaign = fix_strategy_names(archetype2_slow_DALYS_asia_no_campaign)
archetype2_slow_discounted_DALYS_asia_no_campaign = fix_strategy_names(archetype2_slow_discounted_DALYS_asia_no_campaign)
archetype2_slow_discounted_treatment_costs_df_asia_no_campaign = fix_strategy_names(archetype2_slow_discounted_treatment_costs_df_asia_no_campaign)
archetype2_slow_discounted_vaccine_costs_df_asia_no_campaign = fix_strategy_names(archetype2_slow_discounted_vaccine_costs_df_asia_no_campaign)

archetype2_slow_hosp_africa_no_campaign = fix_strategy_names(archetype2_slow_hosp_africa_no_campaign)
archetype2_slow_deaths_africa_no_campaign = fix_strategy_names(archetype2_slow_deaths_africa_no_campaign)
archetype2_slow_DALYS_africa_no_campaign = fix_strategy_names(archetype2_slow_DALYS_africa_no_campaign)
archetype2_slow_discounted_DALYS_africa_no_campaign = fix_strategy_names(archetype2_slow_discounted_DALYS_africa_no_campaign)
archetype2_slow_discounted_treatment_costs_df_africa_no_campaign = fix_strategy_names(archetype2_slow_discounted_treatment_costs_df_africa_no_campaign)
archetype2_slow_discounted_vaccine_costs_df_africa_no_campaign = fix_strategy_names(archetype2_slow_discounted_vaccine_costs_df_africa_no_campaign)

# Archetype 3 Fast
archetype3_fast_hosp_asia_no_campaign = fix_strategy_names(archetype3_fast_hosp_asia_no_campaign)
archetype3_fast_deaths_asia_no_campaign = fix_strategy_names(archetype3_fast_deaths_asia_no_campaign)
archetype3_fast_DALYS_asia_no_campaign = fix_strategy_names(archetype3_fast_DALYS_asia_no_campaign)
archetype3_fast_discounted_DALYS_asia_no_campaign = fix_strategy_names(archetype3_fast_discounted_DALYS_asia_no_campaign)
archetype3_fast_discounted_treatment_costs_df_asia_no_campaign = fix_strategy_names(archetype3_fast_discounted_treatment_costs_df_asia_no_campaign)
archetype3_fast_discounted_vaccine_costs_df_asia_no_campaign = fix_strategy_names(archetype3_fast_discounted_vaccine_costs_df_asia_no_campaign)

archetype3_fast_hosp_africa_no_campaign = fix_strategy_names(archetype3_fast_hosp_africa_no_campaign)
archetype3_fast_deaths_africa_no_campaign = fix_strategy_names(archetype3_fast_deaths_africa_no_campaign)
archetype3_fast_DALYS_africa_no_campaign = fix_strategy_names(archetype3_fast_DALYS_africa_no_campaign)
archetype3_fast_discounted_DALYS_africa_no_campaign = fix_strategy_names(archetype3_fast_discounted_DALYS_africa_no_campaign)
archetype3_fast_discounted_treatment_costs_df_africa_no_campaign = fix_strategy_names(archetype3_fast_discounted_treatment_costs_df_africa_no_campaign)
archetype3_fast_discounted_vaccine_costs_df_africa_no_campaign = fix_strategy_names(archetype3_fast_discounted_vaccine_costs_df_africa_no_campaign)

# Archetype 3 Slow
archetype3_slow_hosp_asia_no_campaign = fix_strategy_names(archetype3_slow_hosp_asia_no_campaign)
archetype3_slow_deaths_asia_no_campaign = fix_strategy_names(archetype3_slow_deaths_asia_no_campaign)
archetype3_slow_DALYS_asia_no_campaign = fix_strategy_names(archetype3_slow_DALYS_asia_no_campaign)
archetype3_slow_discounted_DALYS_asia_no_campaign = fix_strategy_names(archetype3_slow_discounted_DALYS_asia_no_campaign)
archetype3_slow_discounted_treatment_costs_df_asia_no_campaign = fix_strategy_names(archetype3_slow_discounted_treatment_costs_df_asia_no_campaign)
archetype3_slow_discounted_vaccine_costs_df_asia_no_campaign = fix_strategy_names(archetype3_slow_discounted_vaccine_costs_df_asia_no_campaign)

archetype3_slow_hosp_africa_no_campaign = fix_strategy_names(archetype3_slow_hosp_africa_no_campaign)
archetype3_slow_deaths_africa_no_campaign = fix_strategy_names(archetype3_slow_deaths_africa_no_campaign)
archetype3_slow_DALYS_africa_no_campaign = fix_strategy_names(archetype3_slow_DALYS_africa_no_campaign)
archetype3_slow_discounted_DALYS_africa_no_campaign = fix_strategy_names(archetype3_slow_discounted_DALYS_africa_no_campaign)
archetype3_slow_discounted_treatment_costs_df_africa_no_campaign = fix_strategy_names(archetype3_slow_discounted_treatment_costs_df_africa_no_campaign)
archetype3_slow_discounted_vaccine_costs_df_africa_no_campaign = fix_strategy_names(archetype3_slow_discounted_vaccine_costs_df_africa_no_campaign)

# Also fix the aggregated dataframes
archetype1_fast_asia_df_no_campaign = fix_strategy_names(archetype1_fast_asia_df_no_campaign)
archetype1_fast_africa_df_no_campaign = fix_strategy_names(archetype1_fast_africa_df_no_campaign)
archetype1_slow_asia_df_no_campaign = fix_strategy_names(archetype1_slow_asia_df_no_campaign)
archetype1_slow_africa_df_no_campaign = fix_strategy_names(archetype1_slow_africa_df_no_campaign)
archetype2_fast_asia_df_no_campaign = fix_strategy_names(archetype2_fast_asia_df_no_campaign)
archetype2_fast_africa_df_no_campaign = fix_strategy_names(archetype2_fast_africa_df_no_campaign)
archetype2_slow_asia_df_no_campaign = fix_strategy_names(archetype2_slow_asia_df_no_campaign)
archetype2_slow_africa_df_no_campaign = fix_strategy_names(archetype2_slow_africa_df_no_campaign)
archetype3_fast_asia_df_no_campaign = fix_strategy_names(archetype3_fast_asia_df_no_campaign)
archetype3_fast_africa_df_no_campaign = fix_strategy_names(archetype3_fast_africa_df_no_campaign)
archetype3_slow_asia_df_no_campaign = fix_strategy_names(archetype3_slow_asia_df_no_campaign)
archetype3_slow_africa_df_no_campaign = fix_strategy_names(archetype3_slow_africa_df_no_campaign)



CSV.write("output_MEDIUM_FAST_ASIA_no_catchup_Stanford_jun1.csv", archetype1_fast_asia_df_no_campaign)
CSV.write("output_MEDIUM_FAST_AFRICA_no_catchup_Stanford_jun1.csv", archetype1_fast_africa_df_no_campaign)
CSV.write("output_MEDIUM_SLOW_ASIA_no_catchup_Stanford_jun1.csv", archetype1_slow_asia_df_no_campaign)
CSV.write("output_MEDIUM_SLOW_AFRICA_no_catchup_Stanford_jun1.csv", archetype1_slow_africa_df_no_campaign)
CSV.write("output_HIGH_FAST_ASIA_no_catchup_Stanford_jun1.csv", archetype2_fast_asia_df_no_campaign)
CSV.write("output_HIGH_FAST_AFRICA_no_catchup_Stanford_jun1.csv", archetype2_fast_africa_df_no_campaign)
CSV.write("output_HIGH_SLOW_ASIA_no_catchup_Stanford_jun1.csv", archetype2_slow_asia_df_no_campaign)
CSV.write("output_HIGH_SLOW_AFRICA_no_catchup_Stanford_jun1.csv", archetype2_slow_africa_df_no_campaign)
CSV.write("output_VERY_HIGH_FAST_ASIA_no_catchup_Stanford_jun1.csv", archetype3_fast_asia_df_no_campaign)
CSV.write("output_VERY_HIGH_FAST_AFRICA_no_catchup_Stanford_jun1.csv", archetype3_fast_africa_df_no_campaign)
CSV.write("output_VERY_HIGH_SLOW_ASIA_no_catchup_Stanford_jun1.csv", archetype3_slow_asia_df_no_campaign)
CSV.write("output_VERY_HIGH_SLOW_AFRICA_no_catchup_Stanford_jun1.csv", archetype3_slow_africa_df_no_campaign)
