using CSV, DataFrames

function reshape_optimal_to_wide_format(combined_df, scenario_info)
    
    # get all unique WTP values and sort them
    wtp_values = sort(unique(combined_df.WTP))
    wtp_min = minimum(wtp_values)
    wtp_max = maximum(wtp_values)
    
    # build column names for WTP range
    wtp_col_names = ["$(Int(round(wtp)))" for wtp in wtp_values]
    
    # get all unique multipliers
    multipliers = sort(unique(combined_df.multiplier))
    
    # initialize result
    all_rows = []
    
    for (scenario_name, incidence, waning, region) in scenario_info
        
        scenario_data = filter(row -> 
            row.incidence == incidence && 
            row.waning_scenario == waning && 
            row.region == region, 
            combined_df)
        
        if nrow(scenario_data) == 0
            println("  ⚠ No data for $(scenario_name)")
            continue
        end
        
        for mult in multipliers
            mult_data = filter(row -> row.multiplier == mult, scenario_data)
            
            if nrow(mult_data) == 0
                continue
            end
            
            # build row: one column per WTP threshold showing optimal strategy
            row_dict = Dict{String, Any}(
                "incidence"    => incidence,
                "region"       => region,
                "waning_scenario" => waning,
                "multiplier"   => mult,
                "WTP_range"    => "$(Int(round(wtp_min))) to $(Int(round(wtp_max)))"
            )
            
            for (wtp, col_name) in zip(wtp_values, wtp_col_names)
                wtp_row = filter(row -> isapprox(row.WTP, wtp, atol=1e-6), mult_data)
                row_dict[col_name] = nrow(wtp_row) > 0 ? wtp_row.optimal_strategy[1] : missing
            end
            
            push!(all_rows, row_dict)
        end
    end
    
    result_df = DataFrame(all_rows)
    
    # reorder columns: metadata first, then WTP columns in order
    meta_cols = ["incidence", "region", "waning_scenario", "multiplier"]
    col_order = vcat(meta_cols, wtp_col_names)
    col_order = filter(c -> c in names(result_df), col_order)
    
   select!(result_df, col_order)
    
    incidence_order = Dict("Medium" => 1, "High" => 2, "VeryHigh" => 3)
    region_order    = Dict("Africa" => 1, "Asia" => 2)
    waning_order    = Dict("Slow" => 1, "Fast" => 2)

    result_df[!, :incidence_sort]  = [incidence_order[x] for x in result_df.incidence]
    result_df[!, :region_sort]     = [region_order[x] for x in result_df.region]
    result_df[!, :waning_sort]     = [waning_order[x] for x in result_df.waning_scenario]

    sort!(result_df, [:incidence_sort, :region_sort, :waning_sort, :multiplier])
    select!(result_df, Not([:incidence_sort, :region_sort, :waning_sort]))

    return result_df
end


# List of scenario names
scenarios = [
    "MEDIUM_FAST_ASIA",
    "MEDIUM_FAST_AFRICA",
    "MEDIUM_SLOW_ASIA",
    "MEDIUM_SLOW_AFRICA",
    "HIGH_FAST_ASIA",
    "HIGH_FAST_AFRICA",
    "HIGH_SLOW_ASIA",
    "HIGH_SLOW_AFRICA",
    "VERY_HIGH_FAST_ASIA",
    "VERY_HIGH_FAST_AFRICA",
    "VERY_HIGH_SLOW_ASIA",
    "VERY_HIGH_SLOW_AFRICA"
]

input_dir = "./threshold_analysis_output_jun1"
output_dir = "./threshold_analysis_output_jun1/combined"
mkpath(output_dir)

for scenario in scenarios
    # Read part1, part2, part 3, and part WIDE files
    part1_file = "$(input_dir)/output_$(scenario)_threshold_analysis_CEA_optimal_by_mean_NMB_WIDE_strategy.csv"
    part2_file = "$(input_dir)/output_$(scenario)_threshold_analysis_part2_CEA_optimal_by_mean_NMB_WIDE_strategy.csv"
    part3_file = "$(input_dir)/output_$(scenario)_threshold_analysis_part3_CEA_optimal_by_mean_NMB_WIDE_strategy.csv"
    part4_file = "$(input_dir)/output_$(scenario)_threshold_analysis_part4_CEA_optimal_by_mean_NMB_WIDE_strategy.csv"
 
    if isfile(part1_file) && isfile(part2_file) && isfile(part3_file) && isfile(part4_file)
        println("Combining: $(scenario) WIDE strategy file")
        
        df1 = CSV.read(part1_file, DataFrame)
        df2 = CSV.read(part2_file, DataFrame)
        df3 = CSV.read(part3_file, DataFrame)
        df4 = CSV.read(part4_file, DataFrame)

        # Combine and sort by multiplier
        combined = vcat(df1, df2, df3, df4)
        sort!(combined, [:multiplier, :WTP])
        
        # Save combined file
        output_file = "$(output_dir)/output_$(scenario)_threshold_analysis_combined_CEA_optimal_by_mean_NMB_WIDE_strategy.csv"
        CSV.write(output_file, combined)
        
        println("  ✓ Saved: $(output_file)")
    else
        println("  ⚠ Missing files for: $(scenario)")
    end
end

println("\n✓ All WIDE files combined and sorted!")


# Scenario definitions with metadata
scenario_info = [
    ("MEDIUM_SLOW_AFRICA",   "Medium",   "Slow", "Africa"),
    ("MEDIUM_FAST_AFRICA",   "Medium",   "Fast", "Africa"),
    ("MEDIUM_SLOW_ASIA",     "Medium",   "Slow", "Asia"),
    ("MEDIUM_FAST_ASIA",     "Medium",   "Fast", "Asia"),
    ("HIGH_SLOW_AFRICA",     "High",     "Slow", "Africa"),
    ("HIGH_FAST_AFRICA",     "High",     "Fast", "Africa"),
    ("HIGH_SLOW_ASIA",       "High",     "Slow", "Asia"),
    ("HIGH_FAST_ASIA",       "High",     "Fast", "Asia"),
    ("VERY_HIGH_SLOW_AFRICA","VeryHigh", "Slow", "Africa"),
    ("VERY_HIGH_FAST_AFRICA","VeryHigh", "Fast", "Africa"),
    ("VERY_HIGH_SLOW_ASIA",  "VeryHigh", "Slow", "Asia"),
    ("VERY_HIGH_FAST_ASIA",  "VeryHigh", "Fast", "Asia")
]

input_dir = "./threshold_analysis_output_jun1/combined"
output_dir = "./threshold_analysis_output_jun1"

all_scenarios = DataFrame()
first_file = true

for (scenario_name, incidence, waning, region) in scenario_info
    file_path = "$(input_dir)/output_$(scenario_name)_threshold_analysis_combined_CEA_optimal_by_mean_NMB_WIDE_strategy.csv"
    
    if isfile(file_path)
        println("Reading: $(scenario_name)")
        df = CSV.read(file_path, DataFrame)
        sort!(df, [:multiplier, :WTP])
        df[!, :incidence]       .= incidence
        df[!, :region]          .= region
        df[!, :waning_scenario] .= waning
        global first_file, all_scenarios
        if first_file
            all_scenarios = df
            first_file = false
        else
            append!(all_scenarios, df, promote=true)
        end
        println("  ✓ Added $(nrow(df)) rows")
    else
        println("  ⚠ Missing file: $(scenario_name)")
    end
end

incidence_order = Dict("Medium" => 1, "High" => 2, "VeryHigh" => 3)
region_order    = Dict("Africa" => 1, "Asia" => 2)
waning_order    = Dict("Slow" => 1, "Fast" => 2)

all_scenarios[!, :incidence_sort]  = [incidence_order[x] for x in all_scenarios.incidence]
all_scenarios[!, :region_sort]     = [region_order[x] for x in all_scenarios.region]
all_scenarios[!, :waning_sort]     = [waning_order[x] for x in all_scenarios.waning_scenario]

sort!(all_scenarios, [:incidence_sort, :region_sort, :waning_sort, :multiplier, :WTP])
select!(all_scenarios, Not([:incidence_sort, :region_sort, :waning_sort]))

output_file = "$(output_dir)/ALL_SCENARIOS_CEA_optimal_WIDE_strategy.csv"
CSV.write(output_file, all_scenarios)

println("\n✓ Master file created: $(output_file)")
println("  Total rows: $(nrow(all_scenarios))")
println("  Columns: $(names(all_scenarios))")

wide_df = reshape_optimal_to_wide_format(all_scenarios, scenario_info)
CSV.write("$(output_dir)/ALL_SCENARIOS_CEA_optimal_WIDE_by_multiplier.csv", wide_df)
println("Wide format saved: $(nrow(wide_df)) rows × $(ncol(wide_df)) columns")
println("Columns: $(names(wide_df))")