import pyomo.environ as pyo
import numpy as np


def build_model(batteries: list[dict], scenarios: dict, settings_dict: dict) -> pyo.ConcreteModel:
    """
    Build the multi-service battery dispatch MIP.

    Optimises charge/discharge scheduling across day-ahead, Dynamic Containment,
    and Balancing Mechanism markets over a 96-period (48-hour) horizon under
    scenario-based price uncertainty.

    Decision variables
    ------------------
    Scenario-independent (committed before uncertainty resolves):
        da_charge, da_discharge  -- day-ahead physical schedule (MW)
        dc_low, dc_high          -- DC capacity committed per EFA block (MW)

    Scenario-dependent (decided after uncertainty resolves):
        bm_offer                 -- BM dispatch accepted by ESO (MW)
        soc                      -- state of charge (fraction of capacity)

    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    required_keys = {"da", "bm", "dc_low", "dc_high"}
    missing_keys = required_keys - scenarios.keys()
    if missing_keys:
        raise ValueError(f"scenarios dict is missing required keys: {missing_keys}")

    n_scenarios, n_periods = scenarios["da"].shape
    if n_scenarios == 0:
        raise ValueError("scenarios must contain at least one scenario (n_scenarios=0)")
    if n_periods == 0:
        raise ValueError("scenarios must contain at least one period (n_periods=0)")

    shape_mismatches = {
        k: scenarios[k].shape
        for k in required_keys
        if scenarios[k].shape != (n_scenarios, n_periods)
    }
    if shape_mismatches:
        raise ValueError(
            f"All scenario arrays must have shape ({n_scenarios}, {n_periods}). "
            f"Mismatches: {shape_mismatches}"
        )

    if not batteries:
        raise ValueError("batteries list must not be empty")

    m = pyo.ConcreteModel()

    T = range(n_periods)
    T_SOC = range(n_periods + 1)  # for SoC dynamics indexing (t-1)
    B = [b["id"] for b in batteries]
    S = range(n_scenarios)
    bat = {b["id"]: b for b in batteries}

    rep_cost = float(settings_dict.get("replacement_cost_per_mwh", 175000))
    dc_min_h = int(settings_dict.get("dc_efa_block_min_hours", 4))
    dc_block = dc_min_h * 2  # number of 30-min periods per EFA block
    dc_response_duration_h = float(settings_dict.get("dc_response_duration_h", 0.5))
    cvar_alpha  = float(settings_dict.get("cvar_alpha",  1.0))
    cvar_lambda = float(settings_dict.get("cvar_lambda", 1.0))
    use_cvar    = cvar_lambda < 1.0


    m.T = pyo.Set(initialize=T)
    m.T_SOC = pyo.Set(initialize=T_SOC)
    m.B = pyo.Set(initialize=B)
    m.S = pyo.Set(initialize=S)

    # Scenario-dependent variables
    m.soc       = pyo.Var(m.T_SOC, m.B, m.S, within=pyo.NonNegativeReals)
    m.bm_offer  = pyo.Var(m.T, m.B, m.S, within=pyo.NonNegativeReals)

    # Scenario-independent variables -- committed before uncertainty resolves
    m.da_charge    = pyo.Var(m.T, m.B, within=pyo.NonNegativeReals)
    m.da_discharge = pyo.Var(m.T, m.B, within=pyo.NonNegativeReals)
    m.dc_low       = pyo.Var(m.T, m.B, within=pyo.NonNegativeReals)
    m.dc_high      = pyo.Var(m.T, m.B, within=pyo.NonNegativeReals)
    m.charge_mode = pyo.Var(m.T, m.B, within=pyo.Binary)

    if use_cvar:
        m.eta       = pyo.Var(within=pyo.Reals)
        m.shortfall = pyo.Var(m.S, within=pyo.NonNegativeReals)


    # ------------------------------------------------------------------
    # Per-Scenario revenue
    # ------------------------------------------------------------------

    def _scenario_net_rev(s):
        da_r   = sum(scenarios["da"][s,t]      * (m.da_discharge[t,b] - m.da_charge[t,b]) * 0.5 for t in T for b in B)
        bm_r   = sum(scenarios["bm"][s,t]      * m.bm_offer[t,b,s]  * 0.5                        for t in T for b in B)
        dcl_r  = sum(scenarios["dc_low"][s,t]  * m.dc_low[t,b]       * 0.5                        for t in T for b in B)
        dch_r  = sum(scenarios["dc_high"][s,t] * m.dc_high[t,b]      * 0.5                        for t in T for b in B)
        deg_bm = sum(m.bm_offer[t,b,s] * 0.5 * (rep_cost / (2 * bat[b]["design_cycle_life"]))     for t in T for b in B)
        deg_da = sum((m.da_charge[t,b] + m.da_discharge[t,b]) * 0.5 * (rep_cost / (2 * bat[b]["design_cycle_life"])) for t in T for b in B)
        return da_r + bm_r + dcl_r + dch_r - deg_bm - deg_da


    # ------------------------------------------------------------------
    # Objective: maximise expected net revenue across scenarios
    # ------------------------------------------------------------------
    # def revenue(m):
    #     n_s = len(S)

    #     da_rev  = sum(
    #         scenarios["da"][s,t] * (m.da_discharge[t,b] - m.da_charge[t,b]) * 0.5
    #         for t in T for b in B for s in S
    #     ) / n_s

    #     bm_rev  = sum(
    #         scenarios["bm"][s,t] * m.bm_offer[t,b,s] * 0.5
    #         for t in T for b in B for s in S
    #     ) / n_s

    #     dcl_rev = sum(
    #         scenarios["dc_low"][s,t] * m.dc_low[t,b] * 0.5
    #         for t in T for b in B for s in S
    #     ) / n_s

    #     dch_rev = sum(
    #         scenarios["dc_high"][s,t] * m.dc_high[t,b] * 0.5
    #         for t in T for b in B for s in S
    #     ) / n_s

    #     # Degradation: cost_per_mwh_throughput x MWh per period (MW x 0.5h)
    #     # Scenario-dependent throughput averaged across scenarios
    #     deg_cost_scenarios = sum(
    #         (m.bm_offer[t,b,s])
    #         * 0.5 * (rep_cost / (2 * bat[b]["design_cycle_life"]))
    #         for t in T for b in B for s in S
    #     ) / n_s

    #     # DA throughput is scenario-independent -- summed over T and B only
    #     # to avoid overcounting by n_s
    #     deg_cost_da = sum(
    #         (m.da_charge[t,b] + m.da_discharge[t,b])
    #         * 0.5 * (rep_cost / (2 * bat[b]["design_cycle_life"]))
    #         for t in T for b in B
    #     )

    #     return da_rev + bm_rev + dcl_rev + dch_rev - deg_cost_scenarios - deg_cost_da

    def revenue(m):
        n_s = len(S)
        mean_rev = sum(_scenario_net_rev(s) for s in S) / n_s
        if not use_cvar:
            return mean_rev
        cvar = m.eta - (1.0 / ((1.0 - cvar_alpha) * n_s)) * sum(m.shortfall[s] for s in S)
        return cvar_lambda * mean_rev + (1.0 - cvar_lambda) * cvar


    m.obj = pyo.Objective(rule=revenue, sense=pyo.maximize)

    if use_cvar:
        def shortfall_con(m, s):
            return m.shortfall[s] >= m.eta - _scenario_net_rev(s)
        m.shortfall_con = pyo.Constraint(m.S, rule=shortfall_con)



    # ------------------------------------------------------------------
    # SoC dynamics -- energy balance with asymmetric round-trip efficiency.
    # Charging at eta < 1 stores less energy than consumed.
    # Discharging at eta < 1 removes more stored energy than delivered.
    # All physical activity (DA, real-time, BM) affects SoC.
    # ------------------------------------------------------------------
    def soc_init(m, t, b, s):
        if t == 0:
            return m.soc[t,b,s] == bat[b]["current_soc"]
        # One-way efficiency per side so that eta_one_way^2 = round_trip_efficiency.
        # Applying the full RTE value to both sides would give RTE^2 effective efficiency.
        eta = bat[b].get("round_trip_efficiency", 0.85) ** 0.5
        return m.soc[t,b,s] == (
            m.soc[t-1,b,s]
            + ((m.da_charge[t-1,b]     * eta     * 0.5)) / bat[b]["capacity_mwh"]
            - ((m.da_discharge[t-1,b]  * (1/eta) * 0.5)
            + (m.bm_offer[t-1,b,s]  * (1/eta) * 0.5)) / bat[b]["capacity_mwh"]
        )
    m.soc_dynamics = pyo.Constraint(m.T_SOC, m.B, m.S, rule=soc_init)

    # Hard SoC bounds
    def soc_lb(m, t, b, s): return m.soc[t,b,s] >= 0.1
    def soc_ub(m, t, b, s): return m.soc[t,b,s] <= 0.9
    m.soc_lower = pyo.Constraint(m.T_SOC, m.B, m.S, rule=soc_lb)
    m.soc_upper = pyo.Constraint(m.T_SOC, m.B, m.S, rule=soc_ub)

    # ------------------------------------------------------------------
    # Power capacity -- all simultaneous uses cannot exceed rated power.
    # DA schedule, real-time dispatch, BM, and DC all draw from the same
    # physical power limit in every scenario.
    # ------------------------------------------------------------------
    def power_limit(m, t, b, s):
        return (
            (m.da_charge[t,b] + m.da_discharge[t,b])
            + m.bm_offer[t,b,s]
            + m.dc_low[t,b] + m.dc_high[t,b]
            <= bat[b]["max_power_mw"]
        )
    m.power_cap = pyo.Constraint(m.T, m.B, m.S, rule=power_limit)


    def charge_mode_ub(m, t, b):
        return m.da_charge[t, b] <= bat[b]["max_power_mw"] * m.charge_mode[t, b]

    def discharge_mode_ub(m, t, b):
        return m.da_discharge[t, b] <= bat[b]["max_power_mw"] * (1 - m.charge_mode[t, b])
    
    def bm_discharge_mode_ub(m, t, b, s):
        return m.bm_offer[t, b, s] <= bat[b]["max_power_mw"] * (1 - m.charge_mode[t, b])

    m.charge_mode_limit       = pyo.Constraint(m.T, m.B, rule=charge_mode_ub)
    m.discharge_mode_limit    = pyo.Constraint(m.T, m.B, rule=discharge_mode_ub)
    m.bm_discharge_mode_limit = pyo.Constraint(m.T, m.B, m.S, rule=bm_discharge_mode_ub)



    # ------------------------------------------------------------------
    # DC EFA block consistency -- DC capacity is committed at auction per
    # 4-hour EFA block and cannot change within a block.
    # ------------------------------------------------------------------

    dc_period_pairs = [
        (t0, t) for t0 in range(0, len(T), dc_block)
        for t in range(t0+1, min(t0 + dc_block, len(T)))
    ]

    def dc_low_block(m, t0, t, b):
        return m.dc_low[t, b] == m.dc_low[t0, b]

    def dc_high_block(m, t0, t, b):
        return m.dc_high[t, b] == m.dc_high[t0, b]

    m.dc_low_consistency  = pyo.Constraint(dc_period_pairs, B, rule=dc_low_block)
    m.dc_high_consistency = pyo.Constraint(dc_period_pairs, B, rule=dc_high_block)

    # ------------------------------------------------------------------
    # DC capacity cap -- limits total DC commitment to a configurable
    # fraction of rated power, reserving headroom for DA and BM.
    # Realistic: operators typically do not fully commit to a single market.
    # ------------------------------------------------------------------
    dc_max_fraction = float(settings_dict.get("dc_max_fraction", 0.5))

    def dc_cap(m, t, b):
        return m.dc_low[t,b] + m.dc_high[t,b] <= bat[b]["max_power_mw"] * dc_max_fraction
    m.dc_capacity_cap = pyo.Constraint(m.T, m.B, rule=dc_cap)

    # ------------------------------------------------------------------
    # DC SoC window -- energy headroom constraints.
    # When DC capacity is committed, the battery must hold enough SoC
    # headroom to deliver the required response in either direction.
    # dc_low (downward response) requires discharge headroom above soc_min.
    # dc_high (upward response) requires charge headroom below soc_max.
    # Efficiency is applied asymmetrically: discharge removes more stored
    # energy than delivered (divide by eta); charge stores less than
    # consumed (multiply by eta). Response duration is configurable via
    # dc_response_duration_h (default 0.5h = one settlement period).
    # ------------------------------------------------------------------

    
    def dc_soc_headroom_lb(m, t, b, s):
        eta = bat[b].get("round_trip_efficiency", 0.85) ** 0.5
        return m.soc[t,b,s] >= 0.1 + (m.dc_low[t,b] * dc_response_duration_h) / (eta * bat[b]["capacity_mwh"])

    def dc_soc_headroom_ub(m, t, b, s):
        eta = bat[b].get("round_trip_efficiency", 0.85) ** 0.5
        return m.soc[t,b,s] <= 0.9 - (m.dc_high[t,b] * dc_response_duration_h * eta) / bat[b]["capacity_mwh"]

    m.dc_soc_lower = pyo.Constraint(m.T, m.B, m.S, rule=dc_soc_headroom_lb)
    m.dc_soc_upper = pyo.Constraint(m.T, m.B, m.S, rule=dc_soc_headroom_ub)

    terminal_soc_min = float(settings_dict.get("terminal_soc_min", 0.1))
    # Terminal SoC -- prevent end-of-horizon battery depletion.
    # Uses a configurable floor (terminal_soc_min) rather than initial SoC,
    # allowing profitable discharge across the horizon without forcing the
    # battery back to its initial state.
    def terminal_soc(m, b, s):
        return m.soc[len(T_SOC)-1, b, s] >= terminal_soc_min # prevents end-of-horizon depletion
    m.terminal_soc = pyo.Constraint(m.B, m.S, rule=terminal_soc)

    return m
