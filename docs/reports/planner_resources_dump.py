import glob, json, os, sys

root = os.path.expanduser("~/scratch/rtool")
out = []


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


for ap in sorted(glob.glob(root + "/**/artifacts/experiment_plan", recursive=True)):
    if "/hf/" in ap or "/models/" in ap:
        continue
    run = ap[: -len("/artifacts/experiment_plan")]
    vs = sorted(glob.glob(ap + "/v*.json"))
    rec = {"run": run.replace(root + "/", ""), "plan_versions": len(vs)}
    rj = load(run + "/run.json") or load(run + "/state/run.json") or {}
    rec["status"] = rj.get("status")
    rec["topic"] = (rj.get("topic") or rj.get("config", {}).get("topic") or "")[:200] if isinstance(rj, dict) else ""
    # brief
    b = sorted(glob.glob(run + "/artifacts/research_brief/v*.json"))
    if b:
        bp = load(b[-1])["payload"]
        rec["brief"] = {k: (v if isinstance(v, str) else json.dumps(v))[:500] for k, v in bp.items() if k in ("topic", "question", "title", "goal", "constraints", "summary")}
    h = sorted(glob.glob(run + "/artifacts/selected_hypothesis/v*.json"))
    if h:
        hp = load(h[-1])["payload"]
        rec["hyp"] = json.dumps(hp)[:600]
    plans = []
    for v in vs:
        d = load(v)
        if not d:
            continue
        p = d["payload"]
        plans.append(
            {
                "v": os.path.basename(v),
                "producer": d.get("producer"),
                "hypothesis": p.get("hypothesis"),
                "replicates": p.get("replicates"),
                "resources": [
                    {k: r.get(k) for k in ("kind", "name", "source", "identifier", "subset", "split", "purpose", "verified", "license", "size_gb", "note")}
                    for r in p.get("resources", [])
                ],
                "issues": p.get("issues"),
                "compute": p.get("compute"),
                "procedure": p.get("procedure"),
                "conditions": [c.get("name") for c in p.get("conditions", [])],
                "metrics": [m.get("name") for m in p.get("metrics", [])],
            }
        )
    rec["plans"] = plans
    # stages
    st = {}
    for sp in sorted(glob.glob(run + "/state/stages/*.json")):
        s = load(sp)
        if not s:
            continue
        sid = s.get("stage_id")
        e = {"status": s.get("status"), "attempt": s.get("attempt"), "failures": s.get("failures"), "last_error": (s.get("last_error") or "")[:600] if isinstance(s.get("last_error"), str) else s.get("last_error")}
        fb = [f.get("text", "")[:400] for f in s.get("feedback", [])]
        if fb:
            e["feedback"] = fb
        if sid == "fetch":
            e["fetched"] = {k: {kk: (str(vv)[:200]) for kk, vv in v.items() if kk in ("kind", "source", "identifier", "status", "note", "bytes")} for k, v in (s.get("agent_state", {}).get("fetched", {}) or {}).items()}
        st[sid] = e
    rec["stages"] = st
    # resource manifest
    m = sorted(glob.glob(run + "/artifacts/resource_manifest/v*.json"))
    if m:
        mp = load(m[-1])["payload"]
        rec["manifest"] = [{k: r.get(k) for k in ("name", "kind", "source", "identifier", "status", "note")} for r in mp.get("resources", [])]
    # results
    rec["has_results"] = bool(glob.glob(run + "/artifacts/results*") or glob.glob(run + "/work/**/results.json", recursive=True))
    out.append(rec)

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(len(out))
