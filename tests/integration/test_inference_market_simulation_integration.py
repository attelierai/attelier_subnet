# ruff: noqa: E501, E701, E702
import json
import math
import subprocess
import sys

from scripts.inference_market_simulation import Fleet, simulate


def test_market_simulation_heterogeneous_round_robin_and_accounting(tmp_path):
    output=tmp_path/"market"
    subprocess.run([sys.executable,"scripts/inference_market_simulation.py","--out",str(output)],check=True,cwd=".",env={"PYTHONPATH":"src"},capture_output=True,text=True)
    report=json.loads((output/"summary.json").read_text()); baseline=report["baseline"]
    assert report["schema"].endswith("/v3") and len(report["sensitivity"])==225
    for row in baseline["seeds"]:
        assert row["successful"]<=row["arrivals"]
        assert math.isclose(row["totalGpuCostUsd"],row["productiveGpuCostUsd"]+row["idleGpuCostUsd"])
        assert row["revenueUsd"]+row["unallocatedUsd"]==baseline["budgetUsd"]
        assert all(0<=x["utilization"]<=1 for x in row["perFleet"].values())

def test_low_demand_is_equal_idle_round_robin_but_saturation_favors_fast_capacity():
    fleets=[Fleet("fast","fast",1,10,1),Fleet("slow","slow",1,100,1)]
    low=simulate([0.,1000.],fleets,900,10,8,2)
    assert low["perFleet"]["fast"]["successful"]==low["perFleet"]["slow"]["successful"]==1
    high=simulate([float(x) for x in range(200)],fleets,900,200,8,200)
    assert high["perFleet"]["fast"]["successfulShare"]>high["perFleet"]["slow"]["successfulShare"]
