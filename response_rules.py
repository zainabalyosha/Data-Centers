def make_plan(risk, load_mw, cooling_kw):
    plan = []
    if risk >= 0.7:
        plan.append("Pre-cool facility 3 hours ahead of forecasted peak heat.")
        plan.append("Shift 25% of non-critical compute to cooler regions (US-East/US-North).")
        plan.append("Charge BESS to 90% state-of-charge by 02:00 local time.")
        plan.append("Throttle GPU-intensive batch jobs by 20% during 14:00–18:00.")
        plan.append("Increase chilled water loop flow and inspect CRAH filters pre-event.")
    elif risk >= 0.4:
        plan.append("Increase chilled water setpoint safety margin by 1°C.")
        plan.append("Schedule non-urgent maintenance outside peak window.")
        plan.append("Prepare BESS to 70% SOC; alert ops team for possible load shed.")
    else:
        plan.append("Operate normally; continue 3-hourly monitoring.")
    return plan
