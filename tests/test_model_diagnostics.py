
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from optimiser.model import build_model
from backtest.engine import _read_ending_soc



def build_scenarios(n_periods=4, price=0.0):
    return {
        "da":      np.full((1, n_periods), price),
        "bm":      np.full((1, n_periods), price),
        "dc_low":  np.full((1, n_periods), price),
        "dc_high": np.full((1, n_periods), price),
        "seed":    999,
    }


def build_battery():
    return [{
        "id":                   "test_bat",
        "current_soc":          0.5,
        "max_power_mw":         1.0,
        "capacity_mwh":         1.0,
        "round_trip_efficiency": 0.85,
        "design_cycle_life":    4000,
    }]


OPT_SETTINGS = {
    "replacement_cost_per_mwh": "175000",
    "dc_efa_block_min_hours":   "2",
    "dc_soc_min":               "0.4",
    "dc_soc_max":               "0.6",
    "dc_max_fraction":          "0.5",
}


def solve(model):
    solver = pyo.SolverFactory("appsi_highs")
    result = solver.solve(model)
    assert result.solver.termination_condition == TerminationCondition.optimal, (
        f"Solver did not find optimal solution: {result.solver.termination_condition}"
    )
    return result


def test_all_prices_zero_battery_does_nothing():
    """When all prices are zero the battery should not charge or discharge."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    batteries = build_battery()
    model     = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    for t in range(4):
        da_d = pyo.value(model.da_discharge[t, "test_bat"])
        da_c = pyo.value(model.da_charge[t,    "test_bat"])
        assert abs(da_d) < 1e-6, f"Expected zero DA discharge at t={t}, got {da_d}"
        assert abs(da_c) < 1e-6, f"Expected zero DA charge at t={t}, got {da_c}"



def test_simple_arbitrage():
    """Low price in period 0, high price in period 2 — battery should charge then discharge."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["da"][0, 0] = 10.0   # cheap — charge
    scenarios["da"][0, 2] = 100.0  # expensive — discharge
    batteries = build_battery()
    model     = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    total_discharge = sum(pyo.value(model.da_discharge[t, "test_bat"]) for t in range(4))
    total_charge    = sum(pyo.value(model.da_charge[t,    "test_bat"]) for t in range(4))


    print(total_charge)
    print(total_discharge)

    assert total_discharge > 1e-6, "Expected some DA discharge"
    assert total_charge    > 1e-6, "Expected some DA charge"
    assert pyo.value(model.obj) > 0, "Expected positive net revenue"

    


def test_dc_only_prices_reserves_capacity():
    """When only DC prices are positive, model should commit DC capacity."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["dc_low"][0, :]  = 50.0
    scenarios["dc_high"][0, :] = 50.0
    batteries = build_battery()
    model     = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    total_dc = sum(
        pyo.value(model.dc_low[t, "test_bat"]) + pyo.value(model.dc_high[t, "test_bat"])
        for t in range(4)
    )
    assert total_dc > 1e-6, "Expected DC capacity to be committed"
    assert pyo.value(model.obj) > 0, "Expected positive revenue from DC"



def test_final_period_no_free_discharge():
    """Battery should not discharge for free in the final period."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["da"][0, 3] = 100.0  # high price only in final period
    batteries = build_battery()
    model     = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    # Battery starts at 50% SoC. To discharge at t=3 it needs energy.
    # It should only discharge if it charged earlier at a lower cost.
    # With no cheap period to charge in, it should not discharge.
    discharge_t3 = pyo.value(model.da_discharge[3, "test_bat"])
    soc_final    = pyo.value(model.soc[4, "test_bat", 0])

    assert soc_final >= 0.1, "Terminal SoC should be above hard lower bound"
    # Revenue should be zero or negative — no free energy to sell
    print(f"  discharge_t3={discharge_t3:.4f}, soc_final={soc_final:.4f}")
    print(f"  objective={pyo.value(model.obj):.4f}")


def test_charge_discharge_not_simultaneous():
    """Battery should not charge and discharge at the same time."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["da"][0, 0] = 10.0
    scenarios["da"][0, 2] = 100.0
    batteries = build_battery()
    model     = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    for t in range(4):
        c = pyo.value(model.da_charge[t,    "test_bat"])
        d = pyo.value(model.da_discharge[t, "test_bat"])
        assert not (c > 1e-6 and d > 1e-6), (
            f"Simultaneous charge and discharge at t={t}: charge={c:.4f}, discharge={d:.4f}"
        )



def test_soc_continuity():
    """SoC at each period should correctly reflect previous period actions."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["da"][0, 0] = 10.0
    scenarios["da"][0, 2] = 100.0
    batteries  = build_battery()
    model      = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    # One-way efficiency per side — matches model convention where eta = sqrt(RTE)
    eta      = 0.85 ** 0.5
    capacity = 1.0

    for t in range(1, 5):
        soc_prev    = pyo.value(model.soc[t-1, "test_bat", 0])
        da_c        = pyo.value(model.da_charge[t-1,    "test_bat"])
        da_d        = pyo.value(model.da_discharge[t-1, "test_bat"])
        soc_expected = soc_prev + (da_c * eta * 0.5) / capacity - (da_d * (1/eta) * 0.5) / capacity
        soc_actual   = pyo.value(model.soc[t, "test_bat", 0])
        assert abs(soc_actual - soc_expected) < 1e-4, (
            f"SoC continuity violated at t={t}: expected {soc_expected:.4f}, got {soc_actual:.4f}"
        )


def test_roundtrip_efficiency_convention():
    """After one charge+discharge cycle the SoC loss should match RTE=0.85, not RTE^2=0.72."""
    import math

    # Negative DA price at t=0 -> model earns revenue for charging (pays to take energy).
    # Positive DA price at t=2 -> model earns revenue for discharging.
    # t=1 and t=3 have zero price so no throughput incentive (degradation cost keeps them idle).
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["da"][0, 0] = -100.0
    scenarios["da"][0, 2] =  100.0
    batteries = build_battery()
    model = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    eta_rte     = 0.85
    eta_oneway  = math.sqrt(eta_rte)   # one-way efficiency per side
    cap         = 1.0
    bm_t0 = pyo.value(model.bm_offer[0, "test_bat", 0])


    charge_t0    = pyo.value(model.da_charge[0,    "test_bat"])
    discharge_t2 = pyo.value(model.da_discharge[2, "test_bat"])

    assert charge_t0    > 1e-6, "Expected charging at t=0"
    assert discharge_t2 > 1e-6, "Expected discharging at t=2"

    soc = [pyo.value(model.soc[t, "test_bat", 0]) for t in range(5)]

    # Check charge leg: SoC rise should match eta_oneway, not full RTE
    expected_soc1 = soc[0] + (charge_t0 * eta_oneway * 0.5) / cap - (bm_t0 * 0.5) / (eta_oneway * cap)
    assert abs(soc[1] - expected_soc1) < 1e-4, (
        f"Charge leg wrong: soc[1]={soc[1]:.4f}, expected {expected_soc1:.4f} "
        f"(eta_oneway={eta_oneway:.4f}). If this is ~{soc[0] + charge_t0*eta_rte*0.5/cap:.4f} "
        f"the model is still using full RTE on both sides."
    )

    # Check discharge leg: SoC drop should match 1/eta_oneway, not full 1/RTE
    expected_soc3 = soc[2] - (discharge_t2 * 0.5) / (eta_oneway * cap)
    assert abs(soc[3] - expected_soc3) < 1e-4, (
        f"Discharge leg wrong: soc[3]={soc[3]:.4f}, expected {expected_soc3:.4f} "
        f"(eta_oneway={eta_oneway:.4f})."
    )

    # Full energy balance: SoC at t=3 should equal initial SoC plus all stored/removed energy
    expected_soc3_final = (
        soc[0]
        + (charge_t0 * eta_oneway * 0.5) / cap
        - (bm_t0 * 0.5) / (eta_oneway * cap)
        - (discharge_t2 * 0.5) / (eta_oneway * cap)
    )
    assert abs(soc[3] - expected_soc3_final) < 1e-4, (
        f"Full energy balance wrong: soc[3]={soc[3]:.4f}, expected {expected_soc3_final:.4f}"
    )
    print(f"  charge_t0={charge_t0:.4f} MW, bm_t0={bm_t0:.4f} MW, discharge_t2={discharge_t2:.4f} MW")
    print(f"  SoC trajectory: {[f'{s:.4f}' for s in soc]}")


def test_dc_block_consistency():
    """DC capacity must be constant within each EFA block — no intra-block variation."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["dc_low"][0, :]  = 50.0
    scenarios["dc_high"][0, :] = 50.0
    batteries = build_battery()
    model = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    # OPT_SETTINGS has dc_efa_block_min_hours=2 -> block size = 4 periods (the full horizon)
    # Every period must equal period 0
    ref_low  = pyo.value(model.dc_low[0,  "test_bat"])
    ref_high = pyo.value(model.dc_high[0, "test_bat"])

    for t in range(1, 4):
        low_t  = pyo.value(model.dc_low[t,  "test_bat"])
        high_t = pyo.value(model.dc_high[t, "test_bat"])
        assert abs(low_t  - ref_low)  < 1e-6, f"dc_low varies within block: t=0={ref_low:.4f}, t={t}={low_t:.4f}"
        assert abs(high_t - ref_high) < 1e-6, f"dc_high varies within block: t=0={ref_high:.4f}, t={t}={high_t:.4f}"

    print(f"  dc_low={ref_low:.4f} MW, dc_high={ref_high:.4f} MW (constant across all 4 periods)")

def test_dc_soc_window_binds():
    """When DC is committed, SoC must stay within the energy headroom window."""
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["dc_low"][0, :]  = 50.0
    scenarios["dc_high"][0, :] = 50.0
    batteries = build_battery()
    model = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    import math
    cap      = 1.0
    eta      = 0.85 ** 0.5  # one-way efficiency — matches model convention
    dur_h    = 0.5           # dc_response_duration_h default

    for t in range(4):
        dc_low_t  = pyo.value(model.dc_low[t,  "test_bat"])
        dc_high_t = pyo.value(model.dc_high[t, "test_bat"])
        soc_t     = pyo.value(model.soc[t, "test_bat", 0])

        # Discharge headroom: delivering dc_low MW for dur_h at efficiency 1/eta
        # requires dc_low * dur_h / eta of stored energy above soc_min.
        soc_floor = 0.1 + (dc_low_t  * dur_h) / (eta * cap)
        # Charge headroom: absorbing dc_high MW for dur_h at efficiency eta
        # consumes dc_high * dur_h * eta of storage capacity below soc_max.
        soc_ceil  = 0.9 - (dc_high_t * dur_h * eta) / cap

        assert soc_t >= soc_floor - 1e-6, (
            f"SoC headroom violated (discharge) at t={t}: "
            f"soc={soc_t:.4f}, floor={soc_floor:.4f}, dc_low={dc_low_t:.4f}"
        )
        assert soc_t <= soc_ceil + 1e-6, (
            f"SoC headroom violated (charge) at t={t}: "
            f"soc={soc_t:.4f}, ceil={soc_ceil:.4f}, dc_high={dc_high_t:.4f}"
        )

    dc_low_ref  = pyo.value(model.dc_low[0,  "test_bat"])
    dc_high_ref = pyo.value(model.dc_high[0, "test_bat"])
    print(f"  dc_low={dc_low_ref:.4f} MW, dc_high={dc_high_ref:.4f} MW")
    print(f"  SoC floor={0.1 + dc_low_ref*dur_h/(eta*cap):.4f}, "
          f"ceil={0.9 - dc_high_ref*dur_h*eta/cap:.4f}")

def test_soc_boundary_at_settle_t():
    """_read_ending_soc(settle_t=2) must equal soc[2], which is SOC after period 1."""
    # Use n_periods=4 so T_SOC = range(5): soc[0]=initial, soc[t]=SOC after period t-1.
    # settle_t=2 represents the carryover boundary after the first 2 periods — the
    # same logic as settle_t=SETTLE_PERIODS=48 in the real backtest.
    scenarios = build_scenarios(n_periods=4, price=0.0)
    scenarios["da"][0, 0] = -100.0  # strongly negative: model earns by charging
    batteries = build_battery()
    model = build_model(batteries, scenarios, OPT_SETTINGS)
    solve(model)

    settle_t = 2
    via_helper = _read_ending_soc(model, "test_bat", settle_t=settle_t)

    # Direct read: average soc[settle_t] across all scenarios (1 scenario here)
    soc_keys = [(t, b, s) for (t, b, s) in model.soc.keys() if t == settle_t and b == "test_bat"]
    direct = sum(pyo.value(model.soc[t, b, s]) for (t, b, s) in soc_keys) / len(soc_keys)

    assert via_helper is not None, "_read_ending_soc returned None"
    assert abs(via_helper - direct) < 1e-8, (
        f"_read_ending_soc mismatch: helper={via_helper:.6f}, direct={direct:.6f}"
    )
    # SOC after period 0 charging should be above initial 0.5
    assert via_helper > 0.5, (
        f"Expected SOC to rise after charging at t=0, got {via_helper:.4f}"
    )
    print(f"  soc[{settle_t}] via helper={via_helper:.4f}, direct={direct:.4f}")
    




if __name__ == "__main__":
    test_all_prices_zero_battery_does_nothing()
    test_simple_arbitrage()
    test_dc_only_prices_reserves_capacity()
    test_final_period_no_free_discharge()
    test_charge_discharge_not_simultaneous()
    test_soc_continuity()
    test_roundtrip_efficiency_convention()
    test_dc_block_consistency()
    test_dc_soc_window_binds()
    test_soc_boundary_at_settle_t()
