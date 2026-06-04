using CSV, DataFrames, Distributions
using Base.Threads

println("Using $(Threads.nthreads()) threads")

# Load costs data
costs_psa_africa = CSV.read("costs_data_PSA_africa_region.csv", DataFrame) 
costs_psa_africa.LE .= 64.02
costs_psa_africa.vaccine_procurement_costs .= 1.5

costs_psa_asia = CSV.read("costs_data_PSA_asia_wpr_region.csv", DataFrame)
costs_psa_asia.LE .= 74.76
costs_psa_asia.vaccine_procurement_costs .= 1.5

const CEA_METRICS = ["hospitalizations", "deaths", "DALYs", "discounted_DALYs",
                     "discounted_treatment_costs", "discounted_vaccine_costs",
                     "discounted_total_costs"]

const ALL_STRATEGIES = ["no_vaxx", "routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                       "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                       "routine_9mos_boosters_5yrs_10yrs", "routine_15mos_boosters_5yrs_10yrs"]

const SELECTED_STRATEGIES = ["no_vaxx", "routine_2yrs", "routine_5yrs", "routine_9mos", 
                            "routine_15mos", "routine_9mos_booster_5yrs"]

const CSV_COLUMN_ORDER = ["routine_9mos", "routine_15mos", "routine_2yrs", "routine_5yrs", 
                         "routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                         "routine_9mos_boosters_5yrs_10yrs", "routine_15mos_boosters_5yrs_10yrs"]

const SELECTED_STRATEGY_INDICES = [findfirst(==(s), ALL_STRATEGIES) for s in SELECTED_STRATEGIES]
const N_SELECTED = length(SELECTED_STRATEGIES)

const STRATEGY_NAMES_COL = ["no_vaxx", "routine_9mos", "routine_15mos", "routine_2yrs", 
                            "routine_5yrs", "routine_9mos_booster_5yrs", 
                            "routine_15mos_booster_5yrs", "routine_9mos_boosters_5yrs_10yrs",   
                            "routine_15mos_boosters_5yrs_10yrs"]

# ============================================================================
# FUNCTIONS
# ============================================================================

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


function convert_to_dataframe(values, n_rows, n_years, n_strategies, strategy_names, metric_name)
    total_rows = n_rows * n_years * n_strategies
    
    iterations = Vector{Int}(undef, total_rows)
    years = Vector{Int}(undef, total_rows)
    strategies = Vector{String}(undef, total_rows)
    metric_values = Vector{Float64}(undef, total_rows)
    
    idx = 1
    for i in 1:n_rows
        for j in 1:n_years
            for k in 1:n_strategies
                iterations[idx] = i
                years[idx] = j
                strategies[idx] = strategy_names[k]
                metric_values[idx] = values[i, j, k]
                idx += 1
            end
        end
    end
    
    df = DataFrame(iteration = iterations, year = years, strategy = strategies)
    df[!, metric_name] = metric_values
    return df
end

function convert_wide_to_long(df, metric_name)
    strategy_names = SELECTED_STRATEGIES
    
    n_rows = nrow(df)
    n_strats = length(strategy_names)
    total_rows = n_rows * n_strats
    
    iterations = Vector{Int}(undef, total_rows)
    years = Vector{Int}(undef, total_rows)
    strategies = Vector{String}(undef, total_rows)
    values = Vector{Float64}(undef, total_rows)
    
    idx = 1
    for row in eachrow(df)
        iter = row.iteration
        year = row.year
        for strat_name in strategy_names
            strat_sym = Symbol(strat_name)
            if strat_sym in propertynames(df)
                iterations[idx] = iter
                years[idx] = year
                strategies[idx] = strat_name
                values[idx] = row[strat_sym]
                idx += 1
            end
        end
    end
    
    if idx <= total_rows
        resize!(iterations, idx-1)
        resize!(years, idx-1)
        resize!(strategies, idx-1)
        resize!(values, idx-1)
    end
    
    result_df = DataFrame(iteration = iterations, year = years, strategy = strategies)
    result_df[!, metric_name] = values
    return result_df
end

function prepare_arrays_efficient(cases_adult, cases_peds, cases_age_stratified,
                                   doses_routine, doses_booster, doses_campaign)
    total_rows = nrow(cases_adult)
    n_years    = 10
    n_rows     = div(total_rows, n_years)
    
    adult_array     = zeros(Float64, n_rows, n_years, N_SELECTED)
    peds_array      = zeros(Float64, n_rows, n_years, N_SELECTED)
    age_strat_array = zeros(Float64, n_rows, n_years, N_SELECTED, 17)

    # build csv_col_indices for cases files
    csv_col_indices = Int[]
    for strat in SELECTED_STRATEGIES
        if strat == "no_vaxx"
            push!(csv_col_indices, 2)
        else
            csv_idx = findfirst(==(strat), CSV_COLUMN_ORDER)
            csv_idx !== nothing || error("Strategy $strat not found in CSV_COLUMN_ORDER")
            push!(csv_col_indices, csv_idx + 2)
        end
    end

    for (new_idx, strat) in enumerate(SELECTED_STRATEGIES)
        csv_col = csv_col_indices[new_idx]
        adult_col_data = cases_adult[:, csv_col]
        peds_col_data  = cases_peds[:, csv_col]

        # find the matching column index in STRATEGY_NAMES_COL for age_strat lookup
        strat_col_idx = findfirst(==(strat), STRATEGY_NAMES_COL)
        strat_col_idx !== nothing || error("Strategy $strat not found in STRATEGY_NAMES_COL")

        for i in 1:n_rows
            row_start = (i-1) * n_years + 1
            row_end   = i * n_years
            adult_array[i, :, new_idx] = adult_col_data[row_start:row_end]
            peds_array[i, :, new_idx]  = peds_col_data[row_start:row_end]
            for age in 1:17
                col = "cases_age_$(age)_$(STRATEGY_NAMES_COL[strat_col_idx])"
                age_strat_array[i, :, new_idx, age] = cases_age_stratified[row_start:row_end, col]
            end
        end
    end

    # doses (excluding no_vaxx)
    selected_dose_strategies = SELECTED_STRATEGIES[2:end]
    n_dose_strategies = length(selected_dose_strategies)
    
    routine_array  = zeros(Float64, n_rows, n_years, n_dose_strategies)
    booster_array  = zeros(Float64, n_rows, n_years, n_dose_strategies)
    campaign_array = zeros(Float64, n_rows, n_years, n_dose_strategies)
    
    for (new_idx, strat) in enumerate(selected_dose_strategies)
        csv_idx = findfirst(==(strat), CSV_COLUMN_ORDER)
        csv_idx !== nothing || error("Strategy $strat not found in CSV_COLUMN_ORDER")
        csv_col = csv_idx + 1

        routine_col_data  = doses_routine[:, csv_col]
        booster_col_data  = doses_booster[:, csv_col]
        campaign_col_data = doses_campaign[:, csv_col]
        
        for i in 1:n_rows
            row_start = (i-1) * n_years + 1
            row_end   = i * n_years
            routine_array[i, :, new_idx]  = routine_col_data[row_start:row_end]
            booster_array[i, :, new_idx]  = booster_col_data[row_start:row_end]
            campaign_array[i, :, new_idx] = campaign_col_data[row_start:row_end]
        end
    end
    
    return adult_array, peds_array, age_strat_array, routine_array, booster_array, campaign_array
end


function reorganize_combined_dataframe(dfs, metric_names)
    strategies = SELECTED_STRATEGIES

    first_df = dfs[1]
    iterations = unique(first_df.iteration)
    years      = unique(first_df.year)

    all_iterations = Int[]
    all_strategies = String[]
    all_years      = Int[]
    metric_values  = Dict(metric => Any[] for metric in metric_names)

    data_lookup = Dict{Int, Dict{Int, Dict{String, Dict{Int, Any}}}}()

    for i in 1:length(dfs)
        df     = dfs[i]
        metric = metric_names[i]
        lookup = Dict{Int, Dict{String, Dict{Int, Any}}}()

        for row in eachrow(df)
            strat = row.strategy
            iter  = row.iteration
            yr    = row.year
            val   = row[metric]

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

    combined_df = DataFrame(
        iteration = all_iterations,
        strategy  = all_strategies,
        year      = all_years
    )
    for metric in metric_names
        combined_df[!, metric] = metric_values[metric]
    end

    sort!(combined_df, [:iteration, :strategy, :year])
    return combined_df
end

function calculate_costs_optimized(
    adult_cases_array, peds_cases_array,
    age_stratified_array,
    routine_doses_array, booster_doses_array, campaign_doses_array,
    params_psa_costs
)
    n_rows, n_years, n_strategies = size(adult_cases_array)
    
    hospitalizations = zeros(n_rows, n_years, n_strategies)
    deaths           = zeros(n_rows, n_years, n_strategies)
    deaths_by_age    = zeros(n_rows, n_years, n_strategies, 17)
    DALYs            = zeros(n_rows, n_years, n_strategies)
    vaccine_costs_all = zeros(n_rows, n_years, n_strategies)
    treatment_costs  = zeros(n_rows, n_years, n_strategies)
    
    # booster mapping for selected dose strategies (excluding no_vaxx)
    booster_mapping = zeros(Int, N_SELECTED-1)
    selected_dose_strategies = SELECTED_STRATEGIES[2:end]
    booster_strategy_names = ["routine_9mos_booster_5yrs", "routine_15mos_booster_5yrs", 
                              "routine_9mos_boosters_5yrs_10yrs", "routine_15mos_boosters_5yrs_10yrs"]
    for (new_idx, strat) in enumerate(selected_dose_strategies)
        if strat in booster_strategy_names
            booster_position = findfirst(==(strat), selected_dose_strategies)
            booster_mapping[new_idx] = booster_position
        end
    end
    
    base_LE = params_psa_costs.LE[1]
    age_midpoints = [
        (0 + 0.75)/2, (0.75 + 1.166666667)/2, (1.166666667 + 2)/2,
        (2 + 5)/2, (5 + 10)/2, (10 + 15)/2, (15 + 20)/2, (20 + 25)/2,
        (25 + 30)/2, (30 + 35)/2, (35 + 40)/2, (40 + 45)/2, (45 + 50)/2,
        (50 + 55)/2, (55 + 60)/2, (60 + 65)/2, (65 + 100)/2
    ]
    life_exp_bands = [max(0.0, base_LE - m) for m in age_midpoints]
    
    Threads.@threads for iter in 1:n_rows
        sc   = params_psa_costs.p_seek_care[iter]
        ip   = params_psa_costs.p_inpatient[iter]
        comp = params_psa_costs.p_complications[iter]
        cf   = params_psa_costs.CFR[iter]
        du   = params_psa_costs.duration_untreated[iter]
        ds   = params_psa_costs.duration_severe[iter]
        dm   = params_psa_costs.duration_moderate[iter]
        dwu  = params_psa_costs.DW_untreated[iter]
        dws  = params_psa_costs.DW_severe[iter]
        dwsi = params_psa_costs.DW_severe_with_ip[iter]
        dwm  = params_psa_costs.DW_moderate[iter]
        icp  = params_psa_costs.inpatient_cost_peds[iter]
        ocp  = params_psa_costs.outpatient_cost_peds[iter]
        ica  = params_psa_costs.inpatient_cost_adult[iter]
        oca  = params_psa_costs.outpatient_cost_adult[iter]
        vpc  = params_psa_costs.vaccine_procurement_costs[iter]
        vsc  = params_psa_costs.vaccine_safety_costs[iter]
        vdr  = params_psa_costs.vaccine_delivery_cost_routine[iter]
        vdb  = params_psa_costs.vaccine_delivery_cost_booster[iter]
        vdc  = params_psa_costs.vaccine_delivery_cost_campaign[iter]
        
        for year in 1:n_years
            peds  = peds_cases_array[iter, year, :]
            adult = adult_cases_array[iter, year, :]
            total = peds .+ adult
            
            hospitalizations[iter, year, :] = sc * ip .* total
            deaths[iter, year, :]           = cf .* total
            
            yld_unt   = (1 - sc) * du/365 * dwu .* total
            yld_sev_c = sc * comp * ip * ds/365 * dwsi .* total
            yld_sev_n = sc * (1 - comp) * ip * ds/365 * dws .* total
            yld_mod   = sc * (1 - ip) * dm/365 * dwm .* total
            
            for strat in 1:n_strategies
                death_dalys = 0.0
                for age in 1:17
                    age_cases = age_stratified_array[iter, year, strat, age]
                    deaths_by_age[iter, year, strat, age] = cf * age_cases
                    death_dalys += cf * age_cases * life_exp_bands[age]
                end
                DALYs[iter, year, strat] = yld_unt[strat] + yld_sev_c[strat] + 
                                           yld_sev_n[strat] + yld_mod[strat] + death_dalys
            end
            
            for strat in 2:n_strategies
                r_idx = strat - 1
                vaccine_costs_all[iter, year, strat] =
                    routine_doses_array[iter, year, r_idx] * (vpc + vsc + vdr) +
                    campaign_doses_array[iter, year, r_idx] * (vpc + vsc + vdc)
                b_idx = booster_mapping[r_idx]
                if b_idx > 0
                    vaccine_costs_all[iter, year, strat] +=
                        booster_doses_array[iter, year, b_idx] * (vpc + vsc + vdb)
                end
            end
            
            treatment_costs[iter, year, :] =
                sc * ip       * icp .* peds  .+
                sc * (1 - ip) * ocp .* peds  .+
                sc * ip       * ica .* adult .+
                sc * (1 - ip) * oca .* adult
        end
    end
    
    cost_discount_rate = 0.03
    discount_factors   = [1 / ((1 + cost_discount_rate) ^ (y - 1)) for y in 1:n_years]
    disc_reshape       = reshape(discount_factors, 1, n_years, 1)
    
    discounted_vaccine_costs   = vaccine_costs_all .* disc_reshape
    discounted_treatment_costs = treatment_costs   .* disc_reshape
    discounted_total_costs     = discounted_vaccine_costs .+ discounted_treatment_costs
    discounted_DALYs = apply_discounting_DALYs(DALYs, deaths_by_age, 0.03, life_exp_bands, 17)
    
    strategy_names = SELECTED_STRATEGIES
    
    return (
        convert_to_dataframe(hospitalizations,            n_rows, n_years, n_strategies, strategy_names, "hospitalizations"),
        convert_to_dataframe(deaths,                      n_rows, n_years, n_strategies, strategy_names, "deaths"),
        convert_to_dataframe(DALYs,                       n_rows, n_years, n_strategies, strategy_names, "DALYs"),
        convert_to_dataframe(discounted_DALYs,            n_rows, n_years, n_strategies, strategy_names, "discounted_DALYs"),
        convert_to_dataframe(treatment_costs,             n_rows, n_years, n_strategies, strategy_names, "treatment_costs"),
        convert_to_dataframe(vaccine_costs_all,           n_rows, n_years, n_strategies, strategy_names, "vaccine_costs_all"),
        convert_to_dataframe(discounted_treatment_costs,  n_rows, n_years, n_strategies, strategy_names, "discounted_treatment_costs"),
        convert_to_dataframe(discounted_vaccine_costs,    n_rows, n_years, n_strategies, strategy_names, "discounted_vaccine_costs_all"),
        convert_to_dataframe(discounted_total_costs,      n_rows, n_years, n_strategies, strategy_names, "discounted_total_costs"),
        deaths_by_age
    )
end

function process_scenario_optimized(
    adult_cases_df, peds_cases_df, age_stratified_df,
    routine_df, booster_df, campaign_df,
    costs_psa, multiplier
)
    adult_arr, peds_arr, age_strat_arr, routine_arr, booster_arr, campaign_arr =
        prepare_arrays_efficient(adult_cases_df, peds_cases_df, age_stratified_df,
                                  routine_df, booster_df, campaign_df)
    
    adult_arr     .*= multiplier
    peds_arr      .*= multiplier
    age_strat_arr .*= multiplier
    
    results = calculate_costs_optimized(
        adult_arr, peds_arr, age_strat_arr,
        routine_arr, booster_arr, campaign_arr,
        costs_psa
    )
    
    vaccine_costs = copy(results[8])
    rename!(vaccine_costs, :discounted_vaccine_costs_all => :discounted_vaccine_costs)
    
    combined = [
        results[1], results[2], results[3], results[4],
        results[7], vaccine_costs, results[9]
    ]
    
    df = reorganize_combined_dataframe(combined, CEA_METRICS)
    rename!(df, vcat("run_id", names(df)[2:end]))
    return df
end

function calculate_optimal_strategies_by_wtp(result_df, wtp_range)
    multipliers = sort(unique(result_df.multiplier))
    
    aggregated = combine(groupby(result_df, [:multiplier, :run_id, :strategy]),
        :discounted_total_costs => sum => :total_costs,
        :discounted_DALYs       => sum => :total_DALYs
    )
    
    baseline_data = filter(row -> row.strategy == "no_vaxx", aggregated)
    rename!(baseline_data, :total_costs => :baseline_cost, :total_DALYs => :baseline_DALYs)
    select!(baseline_data, [:multiplier, :run_id, :baseline_cost, :baseline_DALYs])
    
    aggregated = leftjoin(aggregated, baseline_data, on=[:multiplier, :run_id])
    aggregated.incremental_cost = aggregated.total_costs .- aggregated.baseline_cost
    aggregated.DALYs_averted    = aggregated.baseline_DALYs .- aggregated.total_DALYs
    
    optimal_results = DataFrame(
        multiplier            = Float64[],
        WTP                   = Float64[],
        optimal_strategy      = String[],
        mean_NMB              = Float64[],
        mean_incremental_cost = Float64[],
        mean_DALYs_averted    = Float64[],
        sd_NMB                = Float64[]
    )
    
    for wtp in wtp_range
        wtp_data = copy(aggregated)
        wtp_data[!, :NMB] = wtp_data.DALYs_averted .* wtp .- wtp_data.incremental_cost
        
        strategy_means = combine(groupby(wtp_data, [:multiplier, :strategy]),
            :NMB              => mean => :mean_NMB,
            :NMB              => (x -> length(x) > 1 ? std(x) : 0.0) => :sd_NMB,
            :incremental_cost => mean => :mean_incremental_cost,
            :DALYs_averted    => mean => :mean_DALYs_averted
        )
        
        for mult in multipliers
            mult_strategies = filter(row -> row.multiplier == mult, strategy_means)
            if nrow(mult_strategies) > 0
                max_idx = argmax(mult_strategies.mean_NMB)
                push!(optimal_results, (
                    mult, wtp,
                    mult_strategies.strategy[max_idx],
                    mult_strategies.mean_NMB[max_idx],
                    mult_strategies.mean_incremental_cost[max_idx],
                    mult_strategies.mean_DALYs_averted[max_idx],
                    mult_strategies.sd_NMB[max_idx]
                ))
            end
        end
    end
    
    return optimal_results
end

function calculate_all_strategies_nmb(result_df, wtp_range)
    aggregated = combine(groupby(result_df, [:multiplier, :run_id, :strategy]),
        :discounted_total_costs => sum => :total_costs,
        :discounted_DALYs       => sum => :total_DALYs
    )
    
    baseline_data = filter(row -> row.strategy == "no_vaxx", aggregated)
    rename!(baseline_data, :total_costs => :baseline_cost, :total_DALYs => :baseline_DALYs)
    select!(baseline_data, [:multiplier, :run_id, :baseline_cost, :baseline_DALYs])
    
    aggregated = leftjoin(aggregated, baseline_data, on=[:multiplier, :run_id])
    aggregated.incremental_cost = aggregated.total_costs .- aggregated.baseline_cost
    aggregated.DALYs_averted    = aggregated.baseline_DALYs .- aggregated.total_DALYs
    
    all_results = DataFrame()
    
    for wtp in wtp_range
        wtp_data = copy(aggregated)
        wtp_data[!, :NMB] = wtp_data.DALYs_averted .* wtp .- wtp_data.incremental_cost
        wtp_data[!, :WTP] = fill(wtp, nrow(wtp_data))
        
        wtp_summary = combine(groupby(wtp_data, [:multiplier, :strategy]),
            :WTP              => first => :WTP,
            :NMB              => mean  => :mean_NMB,
            :NMB              => (x -> length(x) > 1 ? std(x) : 0.0) => :sd_NMB,
            :incremental_cost => mean  => :mean_incremental_cost,
            :DALYs_averted    => mean  => :mean_DALYs_averted
        )
        
        append!(all_results, wtp_summary)
    end
    
    return all_results
end

# process each scenario 
function process_single_scenario(arch, waning, region, costs_psa, multipliers, output_file)
    t_start = time()
    
    prefix = "./projections_output_may31/$(arch)_$(waning)_waning"
    
    adult_cases = filter(row -> row.year <= 10,
        CSV.read("$(prefix)_adult_cases_with_campaign_may31.csv", DataFrame))
    peds_cases = filter(row -> row.year <= 10,
        CSV.read("$(prefix)_pediatric_cases_with_campaign_may31.csv", DataFrame))
    age_strat_cases = filter(row -> row.year <= 10,
        CSV.read("$(prefix)_age_stratified_cases_with_campaign_may31.csv", DataFrame))
    routine_doses = filter(row -> row.year <= 10,
        CSV.read("$(prefix)_routine_doses_with_campaign_may31.csv", DataFrame)[:, vcat(1,3:11)])
    booster_doses = filter(row -> row.year <= 10,
        CSV.read("$(prefix)_booster_doses_with_campaign_may31.csv", DataFrame)[:, vcat(1,3:11)])
    campaign_doses = filter(row -> row.year <= 10,
        CSV.read("$(prefix)_campaign_doses_with_campaign_may31.csv", DataFrame)[:, vcat(1,3:11)])
    
      
    scenario_results = Vector{DataFrame}(undef, length(multipliers))
    
    Threads.@threads for i in 1:length(multipliers)
        mult = multipliers[i]
        df = process_scenario_optimized(
            adult_cases, peds_cases, age_strat_cases,
            routine_doses, booster_doses, campaign_doses,
            costs_psa, mult
        )
        df[!, :multiplier]       .= mult
        df[!, :archetype]        .= String(arch)
        df[!, :waning_scenario]  .= String(waning)
        df[!, :region]           .= String(region)
        scenario_results[i] = df
    end
    
    result = vcat(scenario_results...)
    
    wtp_range = range(0, step=50, stop=5000)
    
      
    optimal_strategies = calculate_optimal_strategies_by_wtp(result, wtp_range)
    
    all_strategies_nmb = calculate_all_strategies_nmb(result, wtp_range)
    
    mkpath("./threshold_analysis_output_jun1")
    cea_optimal_file = replace(output_file, ".csv" => "_CEA_optimal_by_mean_NMB_WIDE_strategy.csv")
    CSV.write(cea_optimal_file, optimal_strategies)
    
    cea_all_file = replace(output_file, ".csv" => "_CEA_all_strategies_NMB.csv")
    CSV.write(cea_all_file, all_strategies_nmb)
     
    scenario_results    = nothing
    result              = nothing
    optimal_strategies  = nothing
    all_strategies_nmb  = nothing
    adult_cases         = nothing
    peds_cases          = nothing
    age_strat_cases     = nothing
    routine_doses       = nothing
    booster_doses       = nothing
    campaign_doses      = nothing
    GC.gc()
    
    return nothing
end

# ============================================================================
# Main execution
# ============================================================================

costs_lookup = Dict(:asia => costs_psa_asia, :africa => costs_psa_africa)

using Random
Random.seed!(12345) 

multipliers_by_archetype = Dict(
    :archetype1 => rand(Uniform(0.19, 9.6),  250), # run three additional scripts each also drawing 250 multipliers by archetype, but with a different random seed (just above); then, combine using combining_threshold_analysis.jl
    :archetype2 => rand(Uniform(0.09, 2.33), 250),
    :archetype3 => rand(Uniform(0.4,  1.20), 250)
)

scenarios = [
    #(:archetype1, :fast, :asia,   "output_MEDIUM_FAST_ASIA_threshold_analysis.csv"),
    #(:archetype1, :fast, :africa, "output_MEDIUM_FAST_AFRICA_threshold_analysis.csv"),
    #(:archetype1, :slow, :asia,   "output_MEDIUM_SLOW_ASIA_threshold_analysis.csv"),
    #(:archetype1, :slow, :africa, "output_MEDIUM_SLOW_AFRICA_threshold_analysis.csv"),
    #(:archetype2, :fast, :asia,   "output_HIGH_FAST_ASIA_threshold_analysis.csv"),
    #(:archetype2, :fast, :africa, "output_HIGH_FAST_AFRICA_threshold_analysis.csv"),
    #(:archetype2, :slow, :asia,   "output_HIGH_SLOW_ASIA_threshold_analysis.csv"),
    #(:archetype2, :slow, :africa, "output_HIGH_SLOW_AFRICA_threshold_analysis.csv"),
    #(:archetype3, :fast, :asia,   "output_VERY_HIGH_FAST_ASIA_threshold_analysis.csv"),
    #(:archetype3, :fast, :africa, "output_VERY_HIGH_FAST_AFRICA_threshold_analysis.csv"),
    (:archetype3, :slow, :asia,   "output_VERY_HIGH_SLOW_ASIA_threshold_analysis.csv"),
    (:archetype3, :slow, :africa, "output_VERY_HIGH_SLOW_AFRICA_threshold_analysis.csv")
]


@time for (idx, (arch, waning, region, output_file)) in enumerate(scenarios)
    multipliers = multipliers_by_archetype[arch]
    costs_psa   = costs_lookup[region]
    output_path = "./threshold_analysis_output_jun1/$output_file"
    process_single_scenario(arch, waning, region, costs_psa, multipliers, output_path)
end

