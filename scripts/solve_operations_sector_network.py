# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch using the capacities of previous capacity expansion in rule :mod:`solve_network`.
Custom constraints and extra_functionality can be set in the config.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from scripts._benchmark import memory_logger
from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.solve_network import collect_kwargs, create_optimization_model
from pypsa.optimization.abstract import discretized_capacity

logger = logging.getLogger(__name__)


def update_dict(
    original_dict: dict, 
    new_dict: dict, 
    depth: int = 0,
) -> dict:
    """
    Recursively updates the original dictionary with the new dictionary.
    Adds leading dots to the log message based on recursion depth.
    """
    prefix = "." * (depth * 2)  # Two dots per depth level

    for key, value in new_dict.items():
        if depth == 0:
            logger.info("Updating config with column-specific options.")
        logger.info(f"{prefix}Updating key: ['{key}'] with value: ['{value}']")
        if isinstance(value, dict) and isinstance(original_dict.get(key), dict):
            original_dict[key] = update_dict(original_dict.get(key, {}), value, depth + 1)
        else:
            original_dict[key] = value

    return original_dict


def set_minimum_investment(
    n: pypsa.Network,
    planning_horizons: str,
    comps: list=["Generator", "Link", "Store", "Line"],
) -> None:
    """
    Sets a minimum investment for a given carrier in the network, allows for extendable components of the planning horizon.
    """
    logger.info(f"Fixing optimal capacities for components before the investment run.")
    logger.info("Setting minimum capacities of components based on results from investment run.")
    logger.info(f"Components: {comps}")

    planning_horizons = int(planning_horizons)

    for c in comps:
        ext_i = n.get_extendable_i(c)
        attrs = n.component_attrs[c]
        nominal_attr = attrs.loc[attrs.index.str.endswith("_nom")].index.values[0]

        c_in_build_year = n.static(c).loc[ext_i, "build_year"] == planning_horizons

        mask = ext_i[c_in_build_year]

        if mask.any():
            # For case where optimal capacity is slightly higher than maximum capacity due to solver tolerances
            n.static(c).loc[mask, nominal_attr+"_opt"] = np.minimum(
                n.static(c).loc[mask, nominal_attr+"_opt"],
                n.static(c).loc[mask, nominal_attr+"_max"],
            )

            b_reached_max = n.static(c).loc[mask, nominal_attr+"_opt"] == n.static(c).loc[mask, nominal_attr+"_max"]


            # If maximum potential is reached:
            n.static(c).loc[b_reached_max[b_reached_max].index, nominal_attr] = n.static(c).loc[b_reached_max[b_reached_max].index, nominal_attr+"_opt"]

            n.static(c).loc[b_reached_max[b_reached_max].index, nominal_attr+"_extendable"] = False

            # If maximum potential is not reached:
            n.static(c).loc[b_reached_max[~b_reached_max].index, nominal_attr+"_min"] = n.static(c).loc[b_reached_max[~b_reached_max].index, nominal_attr+"_opt"]

            n.static(c).loc[b_reached_max[~b_reached_max].index, nominal_attr+"_extendable"] = True


def fix_all_optimal_capacities(
    n: pypsa.Network,
    comps: list=["Generator", "Line", "Link", "Store"],
) -> None:
    """
    Fixes the optimal capacities of extendable components in the network.
    """
    logger.info("Fixing optimal capacities of components based on results from investment run.")
    logger.info(f"Components to fix: {comps}")
    for c in comps:
        ext_i = n.get_extendable_i(c)
        attrs = n.component_attrs[c]
        nominal_attr = attrs.loc[attrs.index.str.endswith("_nom")].index.values[0]

        if ext_i.any():
            n.static(c).loc[ext_i, nominal_attr] = n.static(c).loc[ext_i, nominal_attr+"_opt"]
            n.static(c).loc[ext_i, nominal_attr+"_extendable"] = False


def fix_optimal_pipeline_capacities(
    n: pypsa.Network,
) -> None:
    """
    Fixes the optimal capacities of pipelines in the network.
    """
    logger.info("Fixing optimal capacities of pipelines")
    if "CO2 pipeline" in n.links.carrier.values:
        logger.info("Disabling extendability of CO2 pipelines.")
        n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom"] = n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom_opt"]
        n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom_extendable"] = False

    if "H2 pipeline" in n.links.carrier.values:
        logger.info("Disabling extendability of H2 pipelines.")
        n.links.loc[n.links.carrier == "H2 pipeline", "p_nom"] = n.links.loc[n.links.carrier == "H2 pipeline", "p_nom_opt"]
        n.links.loc[n.links.carrier == "H2 pipeline", "p_nom_extendable"] = False


def add_load_shedding(
    n: pypsa.Network,
    marginal_cost: float=10000,
) -> None:
    """
    Adds load shedding to the network.
    """
    n.add("Carrier", "load", color="#dd2e23", nice_name="Load Shedding")
    buses_i = pd.Index(n.loads.bus.unique())

    logger.info(f"Adding load shedding to buses with carriers {n.buses.carrier[buses_i].unique()}.")
    logger.info(f"Load shedding marginal cost: {marginal_cost} EUR/MWh.")
    n.add(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        marginal_cost=marginal_cost,
        p_nom_extendable=True,
    )    

    n.add(
        "Generator",
        buses_i,
        " load negative",
        bus=buses_i,
        carrier="load",
        marginal_cost=-marginal_cost,
        p_nom_extendable=True,
        p_min_pu=-1,
        p_max_pu=0,
    )    

def adjust_industry_demand(
        n: pypsa.Network,
        factor: float,
) -> None:
    logger.info(f"Adjusting industry loads by factor {factor}")
    # get loads
    industry_carrier = ['solid biomass for industry',
                        'gas for industry',
                        'H2 for industry',
                        'industry methanol',
                        'naphtha for industry',
                        'low-temperature heat for industry',
                        'industry electricity',
                        'coal for industry',
                        'NH3',
                        'steel',
                        'cement',
                        ]
    index = n.loads[n.loads.carrier.isin(industry_carrier)].index
    n.loads.loc[index, "p_set"] *= factor


def adjust_seq_limit(
        n: pypsa.Network,
        limit: float,
) -> None:
    n.global_constraints.drop("co2_sequestration_limit", inplace=True)
    n.add(
        "GlobalConstraint",
        "co2_sequestration_limit",
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=np.nan,
    )


def allow_meoh_import(
        n: pypsa.Network,
) -> None:
    logger.info(f"Adding methanol import from non-European countries")

    co2_intensity = 0.2482
    n.add("Carrier", "import methanol")
    n.add(
        "Link",
        "EU methanol",
        suffix=" import",
        carrier="import methanol",
        bus0="co2 atmosphere",
        bus1="EU methanol",
        reversed=False,
        efficiency=1 / co2_intensity,
        marginal_cost= 120 / co2_intensity,
        p_nom=1e7,
    )



if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_sector_network",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2040",
            column="ops_meoh_import",
            run="endogenous",
            configfiles="config/co2.config.yaml",
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    params = snakemake.params
    config = snakemake.config
    solving = snakemake.params.solving

    planning_horizons = snakemake.wildcards.get("planning_horizons", None)
    column = snakemake.wildcards.get("column", None)
    solve_operations_col = snakemake.params.solve_operations["definitions"][column]

    np.random.seed(solving.get("seed", 123))

    # Initialise additional settings to be passed to solve_network function
    additional_settings = dict()
    additional_settings["capacity_constraints"] = False
    additional_settings["co2_atmosphere_constraint"] = True

    n = pypsa.Network(snakemake.input.network)

    # Update solving options, where needed
    solving["options"]["noisy_costs"] = False # Only apply noisy_costs once to enable the same solution space
    planning_horizons = "2040"
    # operations optimization
    if solve_operations_col.get("fix_minimum_investments", False):
        set_minimum_investment(n, planning_horizons)
    if solve_operations_col.get("fix_optimal_pipeline_capacities", False):
        fix_optimal_pipeline_capacities(n)
    if solve_operations_col.get("allow_load_shedding", False):
        config["solving"]["options"]["load_shedding"] = True
        marginal_cost = solve_operations_col["allow_load_shedding"]*1
        add_load_shedding(n, marginal_cost)
    # change short-term developments
    if solve_operations_col.get("adjust_industry_demand", False):
        adjust_industry_demand(n, solve_operations_col["adjust_industry_demand"])
    if solve_operations_col.get("adjust_seq_limit", False):
        adjust_seq_limit(n, solve_operations_col["adjust_seq_limit"])
    if solve_operations_col.get("allow_meoh_import", False):
        allow_meoh_import(n)
        # add constraint config
        logger.info(f"Restricting MeOH import up to {solve_operations_col["allow_meoh_import"]} TWh")
        config["sector"]["imports"]["enable"] = True
        config["sector"]["imports"]["limit"] = solve_operations_col["allow_meoh_import"]
        config["sector"]["imports"]["limit_sense"] = '<='
        snakemake.config["sector"]["imports"]["enable"] = True
        snakemake.config["sector"]["imports"]["limit"] = 200
        snakemake.config["sector"]["imports"]["sense"] = '<='

    # if solve_operations_col["options"].get("fix_all_capacities", False):
    #     fix_all_optimal_capacities(n)

    # Overwrite individual config options
    if solve_operations_col.get("overwrite_config", {}):
        config = update_dict(
            config, solve_operations_col["overwrite_config"]
        )

    # Store updated params and config in network file
    n.params = params
    n.config = config

    # Run the re-optimisation of the model
    logger.info("---")
    logger.info(f"Running re-optimisation for column ['{column}'] and year ['{planning_horizons}']")

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        model_kwargs, solve_kwargs = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="single",
            )
        create_optimization_model(
                n,
                config=snakemake.config,
                params=snakemake.params,
                model_kwargs=model_kwargs,
                solve_kwargs=solve_kwargs,
                planning_horizons=planning_horizons,
                additional_settings=additional_settings,
            )

        logger.info("Solving model...")
        status, condition = n.optimize.solve_model(**solve_kwargs)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])