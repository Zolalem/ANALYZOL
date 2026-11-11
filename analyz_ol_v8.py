#!/usr/bin/env python3
"""ANALYZOL V8.0.1-DEV. Corrected home/away outcome probability labels."""
import argparse, json, hashlib, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

REQ = ["date","home_team","away_team","home_goals","away_goals"]

def tau(x,y,l,m,r):
    if x==0 and y==0: return 1-l*m*r
    if x==0 and y==1: return 1+l*r
    if x==1 and y==0: return 1+m*r
    if x==1 and y==1: return 1-r
    return 1.0

class PoissonDC:
    def fit(self, d):
        self.teams = sorted(set(d.home_team.astype(str)) | set(d.away_team.astype(str)))
        n=len(self.teams); ix={t:i for i,t in enumerate(self.teams)}
        hi=d.home_team.astype(str).map(ix).to_numpy()
        ai=d.away_team.astype(str).map(ix).to_numpy()
        hg=d.home_goals.to_numpy(int); ag=d.away_goals.to_numpy(int)
        self.hm=max(float(hg.mean()),.05); self.am=max(float(ag.mean()),.05)
        def loss(z):
            a=z[:n]-z[:n].mean(); de=z[n:2*n]-z[n:2*n].mean()
            ha,r=z[-2],z[-1]
            lh=np.exp(np.clip(np.log(self.hm)+ha+a[hi]-de[ai],-4,3))
            la=np.exp(np.clip(np.log(self.am)+a[ai]-de[hi],-4,3))
            ll=poisson.logpmf(hg,lh).sum()+poisson.logpmf(ag,la).sum()
            for x,y,l,m in zip(hg,ag,lh,la):
                q=tau(int(x),int(y),float(l),float(m),float(r))
                if q<=0: return 1e12
                ll += np.log(q)
            return -ll+.01*(np.square(a).sum()+np.square(de).sum())
        z0=np.zeros(2*n+2); z0[-2]=.1; z0[-1]=-.05
        res=minimize(loss,z0,method="L-BFGS-B",
                     bounds=[(-2.5,2.5)]*(2*n)+[(-.8,.8),(-.15,.15)],
                     options={"maxiter":1500})
        if not np.isfinite(res.fun): raise RuntimeError("Fit failed: "+str(res.message))
        z=res.x; a=z[:n]-z[:n].mean(); de=z[n:2*n]-z[n:2*n].mean()
        self.attack=dict(zip(self.teams,a)); self.defence=dict(zip(self.teams,de))
        self.home_adv=float(z[-2]); self.rho=float(z[-1])
        return self

    def score_matrix(self,h,a,max_goals=10):
        lh,la=self.expected_goals(h,a)
        k=np.arange(max_goals+1)
        m=np.outer(poisson.pmf(k,lh),poisson.pmf(k,la))
        for x in (0,1):
            for y in (0,1): m[x,y]*=max(tau(x,y,lh,la,self.rho),1e-12)
        return m/m.sum()

    def expected_goals(self,h,a):
        lh=np.exp(np.log(self.hm)+self.home_adv+self.attack.get(h,0)-self.defence.get(a,0))
        la=np.exp(np.log(self.am)+self.attack.get(a,0)-self.defence.get(h,0))
        return float(lh),float(la)

    def predict(self,h,a,max_goals=10):
        m=self.score_matrix(h,a,max_goals)
        # Rows are home goals, columns away goals.
        ph=float(np.tril(m,-1).sum()); pd=float(np.trace(m)); pa=float(np.triu(m,1).sum())
        x,y=np.unravel_index(np.argmax(m),m.shape)
        return {"p_home":ph,"p_draw":pd,"p_away":pa,"top_score":f"{x}-{y}",
                "lambda_home":self.expected_goals(h,a)[0],
                "lambda_away":self.expected_goals(h,a)[1]}

def metrics(p,y):
    p=np.clip(np.asarray(p,float),1e-15,1); p=p/p.sum(axis=1,keepdims=True)
    y=np.asarray(y,int); one=np.eye(3)[y]
    return {"n":len(y),"accuracy":float((p.argmax(1)==y).mean()),
            "brier_sum":float(((p-one)**2).sum(1).mean()),
            "brier_half":float(((p-one)**2).sum(1).mean()/2),
            "log_loss":float(-np.log(p[np.arange(len(y)),y]).mean())}

def outcome(h,a): return 0 if h>a else (1 if h==a else 2)

def monte_carlo(model,h,a,n=50000,seed=20261105,max_goals=10):
    m=model.score_matrix(h,a,max_goals); rng=np.random.default_rng(seed)
    draws=rng.choice(m.size,size=n,p=m.ravel())
    scores=np.column_stack(np.unravel_index(draws,m.shape))
    out=np.where(scores[:,0]>scores[:,1],0,np.where(scores[:,0]==scores[:,1],1,2))
    return {"n":n,"seed":seed,"p_home":float((out==0).mean()),
            "p_draw":float((out==1).mean()),"p_away":float((out==2).mean())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--out",default="analyz_ol_v8_output")
    ap.add_argument("--simulations",type=int,default=50000); ap.add_argument("--seed",type=int,default=20261105)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    d=pd.read_csv(args.data)
    missing=[c for c in REQ if c not in d]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    d["date"]=pd.to_datetime(d.date,utc=True,errors="coerce")
    d=d.dropna(subset=REQ).sort_values("date").reset_index(drop=True)
    d["home_goals"]=d.home_goals.astype(int); d["away_goals"]=d.away_goals.astype(int)
    if (d[["home_goals","away_goals"]]<0).any().any(): raise ValueError("Negative goals found.")
    if d.duplicated(["date","home_team","away_team"]).any(): raise ValueError("Duplicate fixtures found.")
    if len(d)<350: raise ValueError("Need at least 350 matches for this starter evaluation.")
    # Expanding-window walk-forward: each test block is predicted only from earlier rows.
    rows=[]
    for start in range(250,len(d),100):
        tr=d.iloc[:start]; te=d.iloc[start:start+100]
        model=PoissonDC().fit(tr); pp=[]; yy=[]; ge=[]
        for r in te.itertuples():
            z=model.predict(str(r.home_team),str(r.away_team))
            pp.append([z["p_home"],z["p_draw"],z["p_away"]]); yy.append(outcome(r.home_goals,r.away_goals))
            ge += [abs(z["lambda_home"]-r.home_goals),abs(z["lambda_away"]-r.away_goals)]
        m=metrics(pp,yy); m["goal_mae"]=float(np.mean(ge))
        m["test_start"]=str(te.date.min()); m["test_end"]=str(te.date.max()); rows.append(m)
    pd.DataFrame(rows).to_csv(os.path.join(args.out,"walk_forward_results.csv"),index=False)
    # Final chronological 20% holdout. No tuning on this set.
    split=int(len(d)*.8); tr,te=d.iloc[:split],d.iloc[split:]
    model=PoissonDC().fit(tr); pp=[]; yy=[]; ge=[]
    for r in te.itertuples():
        z=model.predict(str(r.home_team),str(r.away_team))
        pp.append([z["p_home"],z["p_draw"],z["p_away"]]); yy.append(outcome(r.home_goals,r.away_goals))
        ge += [abs(z["lambda_home"]-r.home_goals),abs(z["lambda_away"]-r.away_goals)]
    report=metrics(pp,yy); report["goal_mae"]=float(np.mean(ge))
    report["test_start"]=str(te.date.min()); report["test_end"]=str(te.date.max())
    report["data_sha256"]=hashlib.sha256(open(args.data,"rb").read()).hexdigest()
    with open(os.path.join(args.out,"holdout_metrics.json"),"w") as f: json.dump(report,f,indent=2)
    # Fit all available past data for an illustrative upcoming fixture prediction.
    final=PoissonDC().fit(d)
    params={"teams":final.teams,"attack":final.attack,"defence":final.defence,
            "home_adv_log":final.home_adv,"rho":final.rho,"home_goal_mean":final.hm,"away_goal_mean":final.am}
    with open(os.path.join(args.out,"model_parameters.json"),"w") as f: json.dump(params,f,indent=2)
    # Reproducible MC demonstration for the most recent observed pairing (not a future forecast).
    last=d.iloc[-1]; mc=monte_carlo(final,str(last.home_team),str(last.away_team),args.simulations,args.seed)
    with open(os.path.join(args.out,"monte_carlo_reproducibility.json"),"w") as f: json.dump(mc,f,indent=2)
    print(json.dumps({"holdout":report,"monte_carlo_example":mc,"output_folder":args.out},indent=2))

if __name__=="__main__": main()
