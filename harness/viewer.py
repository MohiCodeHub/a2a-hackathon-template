"""Run viewer: a small live dashboard over the a2a-hack results dirs.

    uv run python viewer.py            # serves http://localhost:8050
    uv run python viewer.py --port 8060 --results results

Reads results/<run>/simulations/*.json straight off disk on every request, so
it updates live while a run is still writing. No build step — the SPA is inline
and polls the API every few seconds.
"""

import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

RESULTS_DIR = Path(os.environ.get("A2A_HACK_RESULTS", Path(__file__).parent / "results"))

app = FastAPI()


# ---------------------------------------------------------------- parsing ----

def _args_view(name: str, arguments: dict) -> tuple[str, str]:
    """Render a tool call's (display-name, args-string). Unwraps the
    discoverable-tool indirection so the real bank op + its args show."""
    if not isinstance(arguments, dict):
        return name, str(arguments)
    inner = arguments.get("agent_tool_name") or arguments.get("discoverable_tool_name")
    if inner:
        raw = arguments.get("arguments", "")
        return inner, raw if isinstance(raw, str) else json.dumps(raw)
    return name, json.dumps(arguments)


def _transcript(messages: list) -> list[dict]:
    """Flatten a merged trajectory into display items."""
    items = []
    for m in messages or []:
        role = m.get("role")
        tcs = m.get("tool_calls")
        if tcs:
            for tc in tcs:
                disp, args = _args_view(tc.get("name", "?"), tc.get("arguments") or {})
                scope = "user" if tc.get("requestor") == "user" else "agent"
                items.append({"kind": "tool", "scope": scope, "name": disp,
                              "raw": tc.get("name", ""), "args": args})
        elif role == "tool":
            items.append({"kind": "result", "text": (m.get("content") or "")[:1400]})
        elif m.get("content"):
            who = "user" if role == "user" else "agent"
            items.append({"kind": "say", "who": who, "text": (m.get("content") or "")[:2000]})
    return items


def _load_sim(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (ValueError, OSError):
        return None  # file mid-write; skip this tick
    info = d.get("info") or {}
    ri = d.get("reward_info") or {}
    return {
        "task_id": d.get("task_id"),
        "reward": ri.get("reward"),
        "termination": d.get("termination_reason"),
        "env_calls": info.get("num_env_tool_calls"),
        "transcript": _transcript(d.get("messages")),
        "leg2": [{"role": r.get("role"), "text": (r.get("content") or "")[:1400]}
                 for r in (info.get("leg2") or [])],
    }


def _run_dir(name: str) -> Path:
    return RESULTS_DIR / name


def _list_sims(name: str) -> list[Path]:
    d = _run_dir(name) / "simulations"
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.is_dir() else []


# ----------------------------------------------------------------- routes ----

@app.get("/api/runs")
def api_runs():
    runs = []
    if RESULTS_DIR.is_dir():
        for d in sorted(RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            sims = [_load_sim(p) for p in _list_sims(d.name)]
            sims = [s for s in sims if s]
            rewards = [s["reward"] for s in sims if s["reward"] is not None]
            runs.append({
                "name": d.name,
                "n": len(sims),
                "mean": round(sum(rewards) / len(rewards), 3) if rewards else None,
                "tasks": [{"id": s["task_id"], "reward": s["reward"]} for s in sims],
            })
    return JSONResponse(runs)


@app.get("/api/run/{name}")
def api_run(name: str):
    sims = [_load_sim(p) for p in _list_sims(name)]
    sims = [s for s in sims if s]
    rewards = [s["reward"] for s in sims if s["reward"] is not None]
    return JSONResponse({
        "name": name,
        "mean": round(sum(rewards) / len(rewards), 3) if rewards else None,
        "n": len(sims),
        "tasks": sims,
    })


@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>a2a-hack · run console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0a0c0d; --panel:#101315; --panel2:#15191c; --line:#222a2e;
    --ink:#e9eef1; --dim:#7e8c93; --faint:#4c585d;
    --amber:#ffb454; --green:#56d27e; --red:#f06a6a; --blue:#5aa9f0; --violet:#b78cf0;
    --mono:'JetBrains Mono',ui-monospace,monospace; --serif:'Instrument Serif',Georgia,serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:var(--bg); color:var(--ink); font-family:var(--mono); font-size:13px; line-height:1.55;
    background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);
    background-size:46px 46px; background-position:-1px -1px;
  }
  body::before{content:"";position:fixed;inset:0;pointer-events:none;
    background:radial-gradient(120% 90% at 80% -10%,rgba(255,180,84,.07),transparent 60%),
               radial-gradient(80% 70% at -10% 110%,rgba(90,169,240,.06),transparent 55%);}
  .wrap{display:grid;grid-template-columns:288px 1fr;height:100vh;position:relative}
  /* sidebar */
  aside{border-right:1px solid var(--line);background:linear-gradient(180deg,rgba(16,19,21,.9),rgba(10,12,13,.9));
    backdrop-filter:blur(4px);padding:22px 18px;overflow:auto}
  .brand{font-family:var(--serif);font-size:30px;letter-spacing:.3px;line-height:1}
  .brand em{color:var(--amber);font-style:italic}
  .tag{color:var(--dim);font-size:11px;letter-spacing:.18em;text-transform:uppercase;margin-top:8px}
  .runs{margin-top:26px;display:flex;flex-direction:column;gap:7px}
  .runlbl{color:var(--faint);font-size:10px;letter-spacing:.2em;text-transform:uppercase;margin:0 0 4px}
  .run{border:1px solid var(--line);border-radius:7px;padding:10px 12px;cursor:pointer;background:var(--panel);
    transition:border-color .15s,transform .15s}
  .run:hover{border-color:var(--faint);transform:translateX(2px)}
  .run.active{border-color:var(--amber);background:var(--panel2)}
  .run .nm{font-weight:500;word-break:break-all}
  .run .meta{color:var(--dim);font-size:11px;margin-top:4px;display:flex;gap:10px;align-items:center}
  .dot{font-size:11px}
  /* main */
  main{overflow:auto;padding:26px 30px 80px}
  .head{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:8px}
  .h-title{font-family:var(--serif);font-size:34px;line-height:1}
  .gauge{margin-left:auto;display:flex;align-items:baseline;gap:10px}
  .gauge .num{font-family:var(--serif);font-size:46px;line-height:.9}
  .gauge .lab{color:var(--dim);font-size:11px;letter-spacing:.16em;text-transform:uppercase}
  .bar{height:4px;background:var(--panel2);border-radius:3px;overflow:hidden;width:160px;margin-top:6px}
  .bar i{display:block;height:100%;background:linear-gradient(90deg,var(--red),var(--amber),var(--green))}
  .cards{margin-top:20px;display:flex;flex-direction:column;gap:14px}
  .card{border:1px solid var(--line);border-radius:10px;background:linear-gradient(180deg,var(--panel),rgba(16,19,21,.6));
    overflow:hidden;opacity:0;transform:translateY(8px);animation:rise .5s cubic-bezier(.2,.7,.2,1) forwards}
  @keyframes rise{to{opacity:1;transform:none}}
  .crow{display:flex;align-items:center;gap:16px;padding:14px 18px;cursor:pointer;user-select:none}
  .crow:hover{background:rgba(255,255,255,.015)}
  .tid{font-weight:700;font-size:14px;letter-spacing:.02em}
  .badge{font-weight:700;font-size:12px;padding:3px 11px;border-radius:999px;border:1px solid}
  .b-pass{color:var(--green);border-color:rgba(86,210,126,.5);box-shadow:0 0 18px rgba(86,210,126,.25);background:rgba(86,210,126,.08)}
  .b-fail{color:var(--red);border-color:rgba(240,106,106,.45);background:rgba(240,106,106,.07)}
  .b-part{color:var(--amber);border-color:rgba(255,180,84,.5);background:rgba(255,180,84,.08)}
  .b-none{color:var(--dim);border-color:var(--line)}
  .chip{color:var(--dim);font-size:11px;border:1px solid var(--line);border-radius:5px;padding:2px 8px}
  .chip b{color:var(--ink);font-weight:500}
  .caret{margin-left:auto;color:var(--faint);transition:transform .2s}
  .card.open .caret{transform:rotate(90deg)}
  .body{display:none;border-top:1px solid var(--line);padding:6px 0 4px}
  .card.open .body{display:grid;grid-template-columns:1fr 1fr;gap:0}
  .leg{padding:14px 18px;min-width:0}
  .leg+.leg{border-left:1px solid var(--line)}
  .leg h4{margin:0 0 12px;font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--faint);font-weight:500}
  .turn{margin-bottom:11px;word-wrap:break-word;overflow-wrap:anywhere}
  .who{font-size:10px;letter-spacing:.14em;text-transform:uppercase;display:block;margin-bottom:2px}
  .who.user{color:var(--blue)} .who.agent{color:var(--amber)} .who.cs{color:var(--violet)} .who.personal{color:var(--green)}
  .say{color:var(--ink);white-space:pre-wrap}
  .tool{border-left:2px solid var(--faint);padding:5px 10px;margin:8px 0;background:rgba(255,255,255,.02);border-radius:0 6px 6px 0}
  .tool.user{border-color:var(--blue)} .tool.agent{border-color:var(--violet)}
  .tool .nm{color:var(--ink);font-weight:700}
  .tool .ar{color:var(--dim);white-space:pre-wrap;font-size:12px;margin-top:2px}
  .result{color:var(--dim);font-size:12px;white-space:pre-wrap;border-left:2px solid var(--line);padding:3px 10px;margin:6px 0}
  .empty{color:var(--faint);padding:40px;text-align:center}
  .pulse{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(86,210,126,.6);animation:pulse 1.8s infinite}
  @keyframes pulse{70%{box-shadow:0 0 0 9px rgba(86,210,126,0)}100%{box-shadow:0 0 0 0 rgba(86,210,126,0)}}
  ::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--line);border-radius:6px}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}aside{display:none}.card.open .body{grid-template-columns:1fr}.leg+.leg{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <div class="brand">a2a<em>·</em>hack</div>
    <div class="tag">run console <span class="pulse"></span></div>
    <div class="runs"><div class="runlbl">result sets</div><div id="runs"></div></div>
  </aside>
  <main>
    <div class="head">
      <div class="h-title" id="run-title">—</div>
      <div class="gauge">
        <div><div class="lab">mean reward</div><div class="bar"><i id="bar" style="width:0%"></i></div></div>
        <div class="num" id="mean">·</div>
      </div>
    </div>
    <div class="cards" id="cards"><div class="empty">select a result set</div></div>
  </main>
</div>
<script>
const $=s=>document.querySelector(s);
let current=null, openTasks=new Set();

function badge(r){
  if(r===null||r===undefined) return ['b-none','—'];
  if(r>=0.999) return ['b-pass','PASS 1.0'];
  if(r<=0.001) return ['b-fail','FAIL 0.0'];
  return ['b-part','PART '+(+r).toFixed(2)];
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function loadRuns(){
  const runs=await (await fetch('/api/runs')).json();
  const el=$('#runs'); el.innerHTML='';
  if(!current && runs.length) current=runs[0].name;
  runs.forEach(r=>{
    const d=document.createElement('div');
    d.className='run'+(r.name===current?' active':'');
    const [bc]=badge(r.mean);
    d.innerHTML=`<div class="nm">${esc(r.name)}</div>
      <div class="meta"><span class="dot ${bc}" style="color:var(--${r.mean>=.999?'green':r.mean>0?'amber':'red'})">●</span>
      mean ${r.mean===null?'·':r.mean} · ${r.n} sims</div>`;
    d.onclick=()=>{current=r.name; loadRuns(); loadRun();};
    el.appendChild(d);
  });
}

async function loadRun(){
  if(!current){return;}
  const run=await (await fetch('/api/run/'+encodeURIComponent(current))).json();
  $('#run-title').textContent=run.name;
  $('#mean').textContent=run.mean===null?'·':run.mean;
  $('#bar').style.width=((run.mean||0)*100)+'%';
  const cards=$('#cards'); cards.innerHTML='';
  if(!run.tasks.length){cards.innerHTML='<div class="empty">no simulations yet — run is warming up</div>';return;}
  run.tasks.forEach((t,i)=>{
    const [bc,btxt]=badge(t.reward);
    const open=openTasks.has(t.task_id);
    const c=document.createElement('div');
    c.className='card'+(open?' open':''); c.style.animationDelay=(i*55)+'ms';
    const leg1=t.transcript.map(it=>{
      if(it.kind==='say') return `<div class="turn"><span class="who ${it.who}">${it.who==='user'?'user sim':'agent'}</span><span class="say">${esc(it.text)}</span></div>`;
      if(it.kind==='tool') return `<div class="tool ${it.scope}"><span class="nm">▸ ${esc(it.name)}</span><div class="ar">${esc(it.args)}</div></div>`;
      return `<div class="result">↳ ${esc(it.text)}</div>`;
    }).join('')||'<div class="result">no leg-1 messages</div>';
    const leg2=t.leg2.map(r=>`<div class="turn"><span class="who ${r.role}">${r.role}</span><span class="say">${esc(r.text)}</span></div>`).join('')||'<div class="result">no personal↔CS messages</div>';
    c.innerHTML=`
      <div class="crow">
        <span class="tid">${esc(t.task_id||'?')}</span>
        <span class="badge ${bc}">${btxt}</span>
        <span class="chip">term <b>${esc((t.termination||'').replace('TerminationReason.',''))}</b></span>
        <span class="chip">env calls <b>${t.env_calls??'·'}</b></span>
        <span class="caret">▶</span>
      </div>
      <div class="body">
        <div class="leg"><h4>Leg 1 · user-sim ↔ personal · env tool calls</h4>${leg1}</div>
        <div class="leg"><h4>Leg 2 · personal ↔ customer service</h4>${leg2}</div>
      </div>`;
    c.querySelector('.crow').onclick=()=>{
      c.classList.toggle('open');
      c.classList.contains('open')?openTasks.add(t.task_id):openTasks.delete(t.task_id);
    };
    cards.appendChild(c);
  });
}

async function tick(){ try{ await loadRuns(); await loadRun(); }catch(e){} }
tick(); setInterval(tick, 4000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--results", default=None, help="results dir (default: ./results)")
    args = ap.parse_args()
    if args.results:
        RESULTS_DIR = Path(args.results)
    print(f"[viewer] serving {RESULTS_DIR}  ->  http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
