#!/usr/bin/env python3
# ruff: noqa: E501, E701, E702
"""CPU-only discrete-event M1 sensitivity model; never calls GPU/API/chain/billing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

DAY=86400.0; SCHEMA="forge/inference-market-simulation/v3"
DEFAULT_SCENARIO={"label":"hypothetical-heterogeneous-not-measured","emissionUnitsPerDay":.88,"valueUsdPerUnit":260.,"demandPoissonPerDay":1307,"demandGridPerDay":[300,1307,5000],"emissionUnitsGridPerDay":[.4,.88,1.4],"slaDeadlineSeconds":900,"maxQueue":256,"seeds":list(range(20)),"gpuTiers":[{"id":"assumed-fast-96gb","serviceSeconds":150.,"hourlyUsd":1.25,"evidence":{"status":"assumed","sampleCount":0}},{"id":"assumed-economy","serviceSeconds":240.,"hourlyUsd":.75,"evidence":{"status":"assumed","sampleCount":0}}],"fleets":[{"id":"owner","gpuTier":"assumed-fast-96gb","count":4},{"id":"competitor","gpuTier":"assumed-economy","count":4}],"enumerateCounts":[0,1,2,3,4],"observations":[{"gpuTier":"96gb-first-worker","evidence":{"status":"measured","sampleCount":1},"serviceSeconds":205.598,"custodyTtvSeconds":208.172,"cache":"cold-none","useForSteadyState":False},{"gpuTier":"b200","evidence":{"status":"allocated_not_measured","sampleCount":0}},{"gpuTier":"rtx-5090","evidence":{"status":"unavailable","sampleCount":0}}]}
@dataclass(frozen=True)
class Fleet: id:str; gpu_tier:str; count:int; service:float; hourly:float
def req(x:bool,s:str):
 if not x: raise ValueError(s)
def num(x:Any,s:str,positive=False,nonnegative=False)->float:
 req(type(x) in (int,float) and math.isfinite(float(x)),s); x=float(x); req((not positive or x>0)and(not nonnegative or x>=0),s); return x
def canon(x:Any)->bytes:return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def parse(v:Any)->tuple[dict[str,Any],list[Fleet]]:
 keys={"label","emissionUnitsPerDay","valueUsdPerUnit","demandPoissonPerDay","demandGridPerDay","emissionUnitsGridPerDay","slaDeadlineSeconds","maxQueue","seeds","gpuTiers","fleets","enumerateCounts","observations"};req(isinstance(v,dict)and set(v)==keys,"scenario_fields")
 for k in("emissionUnitsPerDay","valueUsdPerUnit","demandPoissonPerDay"):num(v[k],k,True)
 for k in("demandGridPerDay","emissionUnitsGridPerDay"):
  req(isinstance(v[k],list)and v[k],k)
  for x in v[k]:num(x,k,True)
 req(type(v["slaDeadlineSeconds"])is int and v["slaDeadlineSeconds"]>0,"sla");req(type(v["maxQueue"])is int and v["maxQueue"]>=0,"queue");req(isinstance(v["seeds"],list)and v["seeds"]and all(type(x)is int for x in v["seeds"]),"seeds");req(isinstance(v["enumerateCounts"],list)and all(type(x)is int and x>=0 for x in v["enumerateCounts"]),"counts")
 tiers={}
 for t in v["gpuTiers"]:
  req(isinstance(t,dict)and set(t)=={"id","serviceSeconds","hourlyUsd","evidence"}and isinstance(t["id"],str)and t["id"]not in tiers,"tier")
  e=t["evidence"];req(isinstance(e,dict)and set(e)=={"status","sampleCount"}and e["status"]in{"assumed","measured"}and type(e["sampleCount"])is int and e["sampleCount"]>=0,"evidence");tiers[t["id"]]=(num(t["serviceSeconds"],"service",True),num(t["hourlyUsd"],"hourly",nonnegative=True))
 req(isinstance(v["fleets"],list)and len(v["fleets"])==2,"fleets");out=[]
 for f in v["fleets"]:
  req(isinstance(f,dict)and set(f)=={"id","gpuTier","count"}and isinstance(f["id"],str)and f["gpuTier"]in tiers and type(f["count"])is int and f["count"]>=0,"fleet");s,h=tiers[f["gpuTier"]];out.append(Fleet(f["id"],f["gpuTier"],f["count"],s,h))
 req(len({x.id for x in out})==2 and isinstance(v["observations"],list),"fleet_ids");return v,out
def arrivals(rate:float,seed:int)->list[float]:
 r=random.Random(seed);t=0.;out=[]
 while True:
  t+=r.expovariate(rate/DAY)
  if t>=DAY:return out
  out.append(t)
def pct(a:list[float],p:float)->float|None:
 return sorted(a)[min(len(a)-1,math.ceil(p*len(a))-1)] if a else None
def simulate(a:list[float],fs:list[Fleet],deadline:float,maxq:int,budget:float,target:float)->dict[str,Any]:
 slots=[(f.id,i)for f in fs for i in range(f.count)];by={f.id:f for f in fs};free=set(slots);done=[];wait=[];cursor=0;last=0.;timeout=reject=0;ok={f.id:0 for f in fs};assigned={f.id:0 for f in fs};work={f.id:0. for f in fs};ttv=[]
 def expire(now:float):
  nonlocal timeout
  while wait and wait[0][0]<now:heapq.heappop(wait);timeout+=1
 def choose():
  nonlocal cursor
  for off in range(len(slots)):
   z=slots[(cursor+off)%len(slots)]
   if z in free:cursor=(slots.index(z)+1)%len(slots);return z
  raise RuntimeError("slot")
 def dispatch(now:float):
  expire(now)
  while free and wait:
   end,at,_=heapq.heappop(wait)
   if end<now:continue
   z=choose();free.remove(z);f=by[z[0]];heapq.heappush(done,(now+f.service,z,at));assigned[f.id]+=1;work[f.id]+=f.service
 def finish(x):
  nonlocal last,timeout
  end,z,at=x;last=end;free.add(z)
  if end<=at+deadline:ok[z[0]]+=1;ttv.append(end-at)
  else:timeout+=1
  dispatch(end)
 for i,at in enumerate(a):
  while done and done[0][0]<=at:finish(heapq.heappop(done))
  expire(at)
  if len(wait)>=maxq and not free:reject+=1;continue
  heapq.heappush(wait,(at+deadline,at,i));dispatch(at)
 while done:finish(heapq.heappop(done))
 expire(DAY+deadline+max(f.service for f in fs));timeout+=len(wait);n=len(ttv);horizon=max(DAY,last);unit=budget/n if n else 0.;pf={}
 for f in fs:
  cost=f.count*f.hourly*horizon/3600;prod=work[f.id]*f.hourly/3600;rev=ok[f.id]*unit;pf[f.id]={"gpuTier":f.gpu_tier,"count":f.count,"assigned":assigned[f.id],"successful":ok[f.id],"successfulShare":ok[f.id]/n if n else 0.,"productiveSeconds":work[f.id],"totalCostUsd":cost,"productiveCostUsd":prod,"idleCostUsd":cost-prod,"revenueUsd":rev,"profitUsd":rev-cost,"utilization":work[f.id]/(f.count*horizon)if f.count else 0.}
 total=sum(x["totalCostUsd"]for x in pf.values());prod=sum(x["productiveCostUsd"]for x in pf.values())
 return {"arrivals":len(a),"routing":"persistent_round_robin_free_capacity","paymentPolicy":"fixed_budget_proportional_to_served","accountingHorizonSeconds":horizon,"targetUnitRewardUsd":budget/target,"actualUnitRewardUsd":unit,"arrivalHash":hashlib.sha256(canon(a)).hexdigest(),"successful":n,"timedOut":timeout,"rejectedQueueFull":reject,"completedFraction":n/max(1,len(a)),"p50TtvSeconds":pct(ttv,.5),"p95TtvSeconds":pct(ttv,.95),"totalGpuCostUsd":total,"productiveGpuCostUsd":prod,"idleGpuCostUsd":total-prod,"revenueUsd":budget if n else 0.,"unallocatedUsd":budget if not n else 0.,"perFleet":pf}
def avg(rows:list[dict[str,Any]])->dict[str,Any]:
 keys=["arrivals","successful","timedOut","rejectedQueueFull","completedFraction","totalGpuCostUsd","productiveGpuCostUsd","idleGpuCostUsd","revenueUsd","unallocatedUsd"];v={k:mean(float(r[k])for r in rows)for k in keys}
 for k in("p50TtvSeconds","p95TtvSeconds"):q=[r[k]for r in rows if r[k]is not None];v[k]=mean(q)if q else None
 v["perFleet"]={i:{k:mean(float(r["perFleet"][i][k])for r in rows)for k in("count","assigned","successful","successfulShare","totalCostUsd","productiveCostUsd","idleCostUsd","revenueUsd","profitUsd","utilization")}for i in rows[0]["perFleet"]};return v
def evaluate(s:dict[str,Any],fs:list[Fleet],d:float|None=None,e:float|None=None)->dict[str,Any]:
 d=float(s["demandPoissonPerDay"]if d is None else d);e=float(s["emissionUnitsPerDay"]if e is None else e);b=e*float(s["valueUsdPerUnit"]);rows=[simulate(arrivals(d,seed),fs,float(s["slaDeadlineSeconds"]),int(s["maxQueue"]),b,d)for seed in s["seeds"]];return {"routing":"persistent_round_robin_free_capacity","demandPoissonPerDay":d,"emissionUnitsPerDay":e,"budgetUsd":b,"seeds":rows,"mean":avg(rows)}
def grid(s:dict[str,Any],fs:list[Fleet])->list[dict[str,Any]]:
 out=[]
 for d in s["demandGridPerDay"]:
  for e in s["emissionUnitsGridPerDay"]:
   for a in s["enumerateCounts"]:
    for b in s["enumerateCounts"]:
     x=[Fleet(fs[0].id,fs[0].gpu_tier,a,fs[0].service,fs[0].hourly),Fleet(fs[1].id,fs[1].gpu_tier,b,fs[1].service,fs[1].hourly)];out.append({"demandPoissonPerDay":d,"emissionUnitsPerDay":e,"counts":{x[0].id:a,x[1].id:b},"tiers":{x[0].id:x[0].gpu_tier,x[1].id:x[1].gpu_tier},"result":evaluate(s,x,d,e)["mean"]})
 return out
def write(out:Path,s:dict[str,Any],base:dict[str,Any],rows:list[dict[str,Any]]):
 out.mkdir(parents=True,exist_ok=False,mode=0o700);report={"schema":SCHEMA,"scenario":s,"assumptions":["Tier inputs marked assumed are future-scenario inputs, not benchmarks.","Observations are retained in scenario.observations and do not establish steady-state performance.","Capacity simulation assumes each provider is already admitted and every simulated completion has conforming output, custody, the frozen profile, and passed reliability; it does not validate payout eligibility.","This is discrete persistent round-robin; fluid capacity-proportional shares are only an upper bound.","Fixed E is divided across eligible successful volume; zero success is unallocated. No chain/billing action occurs.","Not realized profit, a Nash proof, or a result for unlisted GPU tiers."],"baseline":base,"sensitivity":rows};(out/"summary.json").write_bytes(canon(report)+b"\n")
 with(out/"sensitivity.csv").open("x",newline="",encoding="utf8")as f:
  w=csv.DictWriter(f,fieldnames=["demandPoissonPerDay","emissionUnitsPerDay","ownerTier","competitorTier","ownerCount","competitorCount","ownerSuccessfulShare","competitorSuccessfulShare","ownerProfitUsd","competitorProfitUsd","completedFraction"]);w.writeheader();o,c=s["fleets"][0]["id"],s["fleets"][1]["id"]
  for r in rows:q=r["result"];w.writerow({"demandPoissonPerDay":r["demandPoissonPerDay"],"emissionUnitsPerDay":r["emissionUnitsPerDay"],"ownerTier":r["tiers"][o],"competitorTier":r["tiers"][c],"ownerCount":r["counts"][o],"competitorCount":r["counts"][c],"ownerSuccessfulShare":q["perFleet"][o]["successfulShare"],"competitorSuccessfulShare":q["perFleet"][c]["successfulShare"],"ownerProfitUsd":q["perFleet"][o]["profitUsd"],"competitorProfitUsd":q["perFleet"][c]["profitUsd"],"completedFraction":q["completedFraction"]})
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scenario",type=Path);p.add_argument("--out",type=Path,default=Path(".codex/market-simulation-v3"));a=p.parse_args();s,fs=parse(json.loads(a.scenario.read_text())if a.scenario else DEFAULT_SCENARIO);base=evaluate(s,fs);write(a.out,s,base,grid(s,fs));print(json.dumps({"schema":SCHEMA,"output":str(a.out),"budgetUsd":base["budgetUsd"]},sort_keys=True))
if __name__=="__main__":main()
