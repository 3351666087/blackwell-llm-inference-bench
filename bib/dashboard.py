"""Static, single-file HTML dashboard (task 6.2). No server, no CDN: data and code
are inlined, open `dashboard/index.html` directly from disk.

  python -m bib.dashboard --results results --out dashboard/index.html

Only VALID repetitions are shown (summaries already exclude invalid runs).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blackwell Inference Bench</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 16px}
.filters{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:0 0 16px;padding:10px 12px;
background:var(--surface);border:1px solid var(--ring);border-radius:10px}
.filters label{color:var(--ink2);font-size:13px}
select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:3px 6px}
.fw{display:inline-flex;align-items:center;gap:4px;margin-right:8px;color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,540px),1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:12px 12px 6px;min-width:0}
.card h2{font-size:14px;margin:0 0 2px}
.card p{margin:0 0 6px;color:var(--ink2);font-size:12px}
svg{width:100%;height:auto;display:block;overflow:visible}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.axt{fill:var(--ink2);font-size:11px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin:4px 0 6px}
.legend span{display:inline-flex;align-items:center;gap:5px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);
border-radius:8px;padding:6px 8px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.15);display:none;z-index:9;max-width:320px}
#tip b{font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:3px 6px;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
.tbl{overflow-x:auto}
.note{color:var(--muted);font-size:12px}
</style></head><body><main>
<h1>Blackwell Inference Bench</h1>
<p class="sub">vLLM · SGLang · TensorRT-LLM on one RTX PRO 6000 Blackwell Server Edition. Medians of valid repetitions; whiskers = min–max across reps. Generated __GENERATED__.</p>
<div class="filters" role="group" aria-label="Filters">
 <label>Model <select id="f-model"></select></label>
 <label>Workload <select id="f-wl"></select></label>
 <label>Precision <select id="f-prec"></select></label>
 <label>Plan <select id="f-plan"></select></label>
 <span id="f-fw"></span>
</div>
<div class="grid">
 <div class="card"><h2>Output throughput vs concurrency</h2><p>tokens/s, log₂ concurrency</p><div class="legend" id="lg1"></div><svg id="c-thr" viewBox="0 0 540 300"></svg></div>
 <div class="card"><h2>Latency vs concurrency</h2><p><select id="f-lat"><option value="ttft_p50_ms">TTFT p50</option><option value="ttft_p99_ms">TTFT p99</option><option value="tpot_p50_ms">TPOT p50</option><option value="tpot_p99_ms">TPOT p99</option><option value="itl_p99_ms">ITL p99</option><option value="e2e_p99_ms">E2E p99</option></select> ms, log scale</p><div class="legend" id="lg2"></div><svg id="c-lat" viewBox="0 0 540 300"></svg></div>
 <div class="card"><h2>Latency distribution (ECDF)</h2><p><select id="f-dist"><option value="ttft_ms">TTFT</option><option value="tpot_ms">TPOT</option><option value="e2e_ms">E2E</option></select> at concurrency <select id="f-conc"></select> — pooled sample of all valid reps</p><div class="legend" id="lg3"></div><svg id="c-dist" viewBox="0 0 540 300"></svg></div>
 <div class="card"><h2>Energy per output token</h2><p>J/token from the NVML energy counter</p><div class="legend" id="lg4"></div><svg id="c-en" viewBox="0 0 540 300"></svg></div>
 <div class="card"><h2>Throughput–latency trade-off</h2><p>x: TPOT p50 (ms) · y: output tokens/s · one point per concurrency</p><div class="legend" id="lg5"></div><svg id="c-tr" viewBox="0 0 540 300"></svg></div>
 <div class="card"><h2>Goodput (SLO attainment)</h2><p>output tokens/s from requests meeting the workload SLO</p><div class="legend" id="lg6"></div><svg id="c-gp" viewBox="0 0 540 300"></svg></div>
</div>
<div class="card" style="margin-top:14px"><h2>Table view</h2><p>All points for the current filter. <span class="note">source = run directory under results/&lt;plan&gt;/</span></p><div class="tbl"><table id="tbl"></table></div></div>
<p class="note">Invalid runs (sanity checks) are excluded upstream. Full raw data: results/*/&lt;server&gt;/&lt;workload&gt;/c*/rep*/requests.jsonl.gz.</p>
</main><div id="tip" role="tooltip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const FW = ['vllm','sglang','trtllm'], FWN={vllm:'vLLM',sglang:'SGLang',trtllm:'TensorRT-LLM'};
const COL = {vllm:'var(--s1)',sglang:'var(--s2)',trtllm:'var(--s3)'};
const DASH = {bf16:'',fp8:'6 4',nvfp4:'2 3'};
const $ = id => document.getElementById(id);
const uniq = a => [...new Set(a)];
const st = {fws:new Set(FW)};
function opts(sel, vals, all){ sel.innerHTML=''; (all?['all']:[]).concat(vals).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);}); }
opts($('f-model'), uniq(D.map(p=>p.model_id)).sort());
opts($('f-wl'), uniq(D.map(p=>p.workload)).sort()); if(D.some(p=>p.workload==='chat'))$('f-wl').value='chat';
opts($('f-prec'), uniq(D.map(p=>p.precision)).sort(), true);
opts($('f-plan'), uniq(D.map(p=>p.plan)).sort(), true);
$('f-plan').value = D.some(p=>p.plan==='main') ? 'main' : 'all';
FW.forEach(f=>{const l=document.createElement('label');l.className='fw';const c=document.createElement('input');c.type='checkbox';c.checked=true;
 c.onchange=()=>{c.checked?st.fws.add(f):st.fws.delete(f);render();};const sw=document.createElement('span');sw.style.cssText=`display:inline-block;width:14px;height:2px;background:${COL[f]}`;
 l.append(c,sw,document.createTextNode(FWN[f]));$('f-fw').appendChild(l);});
['f-model','f-wl','f-prec','f-plan','f-lat','f-dist','f-conc'].forEach(id=>$(id).onchange=render);
const med = a => {const b=a.filter(x=>x!=null).sort((x,y)=>x-y); return b.length? (b.length%2? b[(b.length-1)/2] : (b[b.length/2-1]+b[b.length/2])/2):null;};
function sel(){ const m=$('f-model').value,w=$('f-wl').value,pr=$('f-prec').value,pl=$('f-plan').value;
 return D.filter(p=>p.model_id===m&&p.workload===w&&(pr==='all'||p.precision===pr)&&(pl==='all'||p.plan===pl)&&st.fws.has(p.framework)&&p.n_valid>0);}
function series(pts){ const g={}; pts.forEach(p=>{const k=p.plan+'|'+p.server_id; (g[k]=g[k]||{key:k,fw:p.framework,prec:p.precision,plan:p.plan,sid:p.server_id,pc:p.prefix_cache,pts:[]}).pts.push(p);});
 return Object.values(g).map(s=>{s.pts.sort((a,b)=>a.concurrency-b.concurrency);return s;}).sort((a,b)=>FW.indexOf(a.fw)-FW.indexOf(b.fw)||a.prec.localeCompare(b.prec));}
const lab = s => FWN[s.fw]+' '+s.prec.toUpperCase()+(s.pc===false?' (no cache)':'')+(s.plan!=='main'?' ['+s.plan+']':'');
function legend(el, ss){ el.innerHTML=''; ss.forEach(s=>{const sp=document.createElement('span');const sv=document.createElementNS('http://www.w3.org/2000/svg','svg');
 sv.setAttribute('width','22');sv.setAttribute('height','8');sv.style.width='22px';const ln=document.createElementNS(sv.namespaceURI,'line');
 ln.setAttribute('x1','1');ln.setAttribute('x2','21');ln.setAttribute('y1','4');ln.setAttribute('y2','4');ln.setAttribute('stroke',COL[s.fw]);ln.setAttribute('stroke-width','2');
 if(DASH[s.prec])ln.setAttribute('stroke-dasharray',DASH[s.prec]);sv.appendChild(ln);sp.append(sv,document.createTextNode(lab(s)));el.appendChild(sp);});}
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
function fmt(v){ if(v==null)return '–'; const a=Math.abs(v); if(Number.isInteger(v)) return String(v); return a>=100? v.toFixed(0) : a>=10? v.toFixed(1): a>=1? v.toFixed(2): v.toPrecision(3);}
const tip=$('tip');
function showTip(ev, title, rows){ tip.innerHTML=''; const t=document.createElement('div');t.style.color='var(--ink2)';t.textContent=title;tip.appendChild(t);
 rows.forEach(r=>{const d=document.createElement('div');const b=document.createElement('b');b.textContent=r[0];const sw=document.createElement('span');
 sw.style.cssText=`display:inline-block;width:12px;height:2px;margin:0 6px 3px 0;background:${r[2]||'var(--ink2)'}`;d.append(sw,b,document.createTextNode(' '+r[1]));tip.appendChild(d);});
 tip.style.display='block'; const x=Math.min(ev.clientX+14, innerWidth-tip.offsetWidth-8); tip.style.left=x+'px'; tip.style.top=(ev.clientY+14)+'px';}
function hideTip(){tip.style.display='none';}
// generic XY line chart. getY(p)->{v,lo,hi}
function chart(svg, ss, getY, opt){
 svg.innerHTML=''; const W=540,H=300,L=56,R=12,T=10,B=36;
 const pts=ss.flatMap(s=>s.pts.map(p=>({s,p,x:opt.x(p),...getY(p)}))).filter(d=>d.v!=null&&d.x!=null&&(!opt.logy||d.v>0));
 if(!pts.length){const t=el('text',{x:W/2,y:H/2,'text-anchor':'middle',class:'axt'},svg);t.textContent='no valid data for this filter';return;}
 const xs=pts.map(d=>d.x), ys=pts.flatMap(d=>[d.v,d.lo??d.v,d.hi??d.v]).filter(v=>v!=null&&(!opt.logy||v>0));
 const lx=opt.logx?Math.log2:(v=>v), ly=opt.logy?Math.log10:(v=>v);
 let x0=lx(Math.min(...xs)),x1=lx(Math.max(...xs)); if(x0===x1){x0-=1;x1+=1;}
 let y0=opt.logy?ly(Math.min(...ys))-0.05:0, y1=ly(Math.max(...ys)); if(!opt.logy)y1*=1.08; else y1+=0.05; if(y0===y1)y1=y0+1;
 const X=v=>L+(lx(v)-x0)/(x1-x0)*(W-L-R), Y=v=>T+(1-(ly(v)-y0)/(y1-y0))*(H-T-B);
 // grid + ticks
 let yt=[]; if(opt.logy){for(let e=Math.floor(y0);e<=Math.ceil(y1);e++)[1,2,5].forEach(m=>{const v=m*10**e;if(ly(v)>=y0&&ly(v)<=y1)yt.push(v);});}
 else{const step=niceStep((y1-y0)/5);for(let v=0;v<=y1+1e-9;v+=step)yt.push(v);}
 yt.forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:'var(--grid)','stroke-width':1},svg);const t=el('text',{x:L-6,y:Y(v)+4,'text-anchor':'end',class:'tick'},svg);t.textContent=fmt(v);});
 let xt=opt.logx? uniq(xs).sort((a,b)=>a-b) : niceTicks(Math.min(...xs),Math.max(...xs));
 xt.forEach(v=>{const t=el('text',{x:X(v),y:H-B+16,'text-anchor':'middle',class:'tick'},svg);t.textContent=fmt(v);});
 el('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:'var(--axis)'},svg);
 const xl=el('text',{x:(L+W-R)/2,y:H-4,'text-anchor':'middle',class:'axt'},svg);xl.textContent=opt.xlabel;
 const placed=[];
 ss.forEach(s=>{const d=pts.filter(q=>q.s===s).sort((a,b)=>a.x-b.x); if(!d.length)return;
  d.forEach(q=>{if(q.lo!=null&&q.hi!=null&&q.hi>q.lo)el('line',{x1:X(q.x),x2:X(q.x),y1:Y(q.lo),y2:Y(q.hi),stroke:COL[s.fw],'stroke-width':1.5,opacity:.6},svg);});
  el('path',{d:d.map((q,i)=>(i?'L':'M')+X(q.x).toFixed(1)+','+Y(q.v).toFixed(1)).join(''),fill:'none',stroke:COL[s.fw],'stroke-width':2,'stroke-dasharray':DASH[s.prec]||'','stroke-linejoin':'round'},svg);
  d.forEach(q=>{el('circle',{cx:X(q.x),cy:Y(q.v),r:4,fill:COL[s.fw],stroke:'var(--surface)','stroke-width':2},svg);});
  const last=d[d.length-1]; if(ss.length<=4){let ly=Y(last.v)-6; const lx=Math.min(X(last.x)+6,W-R-2);
   // nudge labels apart when two series end close together (no overlapping direct labels)
   for(let k=0;k<8&&placed.some(p=>Math.abs(p.y-ly)<13&&Math.abs(p.x-lx)<120);k++) ly+=13;
   placed.push({x:lx,y:ly});
   const t=el('text',{x:lx,y:ly,'text-anchor':X(last.x)>W-90?'end':'start',class:'tick'},svg);t.textContent=lab(s).replace('TensorRT-LLM','TRT-LLM');}
 });
 // hover layer: nearest point (24px+ targets)
 const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'},svg);
 const cross=el('line',{y1:T,y2:H-B,stroke:'var(--axis)','stroke-width':1,visibility:'hidden'},svg);
 hit.addEventListener('pointermove',ev=>{const r=svg.getBoundingClientRect();const mx=(ev.clientX-r.left)*W/r.width;
  let best=null;pts.forEach(q=>{const dx=Math.abs(X(q.x)-mx);if(!best||dx<best.dx)best={dx,x:q.x};});
  cross.setAttribute('x1',X(best.x));cross.setAttribute('x2',X(best.x));cross.setAttribute('visibility','visible');
  const at=pts.filter(q=>q.x===best.x);showTip(ev,opt.xlabel+' = '+fmt(best.x),at.map(q=>[fmt(q.v)+(opt.unit||''),lab(q.s)+(q.lo!=null?` (min ${fmt(q.lo)}, max ${fmt(q.hi)})`:'' )+(q.extra?q.extra(q.p):''),COL[q.s.fw]]));});
 hit.addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');hideTip();});
}
function niceStep(r){const p=10**Math.floor(Math.log10(r||1));const m=r/p;return (m<1.5?1:m<3?2:m<7?5:10)*p;}
function niceTicks(a,b){const s=niceStep((b-a)/5||1);const out=[];for(let v=Math.ceil(a/s)*s;v<=b+1e-9;v+=s)out.push(+v.toFixed(6));return out;}
const rep=(p,k)=>({v:med(p.reps[k]||[]),lo:Math.min(...(p.reps[k]||[]).filter(x=>x!=null)),hi:Math.max(...(p.reps[k]||[]).filter(x=>x!=null))});
function ecdf(svg, ss, key, conc){
 svg.innerHTML=''; const W=540,H=300,L=56,R=12,T=10,B=36;
 const data=ss.map(s=>{const p=s.pts.find(p=>p.concurrency===conc);if(!p)return null;const v=p.samples.flatMap(r=>r[key]||[]).filter(x=>x!=null&&x>0).sort((a,b)=>a-b);return v.length?{s,v,p}:null;}).filter(Boolean);
 if(!data.length){const t=el('text',{x:W/2,y:H/2,'text-anchor':'middle',class:'axt'},svg);t.textContent='no samples at this concurrency';return;}
 const all=data.flatMap(d=>d.v);const x0=Math.log10(Math.min(...all))-0.02,x1=Math.log10(Math.max(...all))+0.02;
 const X=v=>L+(Math.log10(v)-x0)/(x1-x0)*(W-L-R),Y=q=>T+(1-q)*(H-T-B);
 [0,.25,.5,.75,.9,1].forEach(q=>{el('line',{x1:L,x2:W-R,y1:Y(q),y2:Y(q),stroke:'var(--grid)'},svg);const t=el('text',{x:L-6,y:Y(q)+4,'text-anchor':'end',class:'tick'},svg);t.textContent=q;});
 for(let e=Math.floor(x0);e<=Math.ceil(x1);e++)[1,2,5].forEach(m=>{const v=m*10**e;if(Math.log10(v)>=x0&&Math.log10(v)<=x1){const t=el('text',{x:X(v),y:H-B+16,'text-anchor':'middle',class:'tick'},svg);t.textContent=fmt(v);}});
 el('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:'var(--axis)'},svg);const xl=el('text',{x:(L+W-R)/2,y:H-4,'text-anchor':'middle',class:'axt'},svg);xl.textContent=key.replace('_ms','')+' (ms, log)';
 data.forEach(d=>{const n=d.v.length;el('path',{d:d.v.map((v,i)=>(i?'L':'M')+X(v).toFixed(1)+','+Y((i+1)/n).toFixed(1)).join(''),fill:'none',stroke:COL[d.s.fw],'stroke-width':2,'stroke-dasharray':DASH[d.s.prec]||''},svg);});
 const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'},svg);
 hit.addEventListener('pointermove',ev=>{const r=svg.getBoundingClientRect();const q=1-((ev.clientY-r.top)*H/r.height-T)/(H-T-B);const qq=Math.max(0.01,Math.min(1,q));
  showTip(ev,'quantile '+qq.toFixed(2),data.map(d=>[fmt(d.v[Math.min(d.v.length-1,Math.floor(qq*d.v.length))])+' ms',lab(d.s)+' (n='+d.v.length+')',COL[d.s.fw]]));});
 hit.addEventListener('pointerleave',hideTip);
}
function table(pts){const t=$('tbl');t.innerHTML='';const cols=[['server_id','server'],['concurrency','conc'],['n_valid','valid reps'],['output_throughput','tok/s'],['ttft_p50_ms','TTFT p50'],['ttft_p99_ms','TTFT p99'],['tpot_p50_ms','TPOT p50'],['tpot_p99_ms','TPOT p99'],['goodput_tok_s','goodput tok/s'],['j_per_output_token','J/tok'],['peak_mem_mib','peak MiB'],['sources','source']];
 const h=t.insertRow();cols.forEach(c=>{const th=document.createElement('th');th.textContent=c[1];h.appendChild(th);});
 pts.slice().sort((a,b)=>a.server_id.localeCompare(b.server_id)||a.concurrency-b.concurrency).forEach(p=>{const r=t.insertRow();cols.forEach(c=>{const td=r.insertCell();
  let v=p[c[0]]; if(c[0] in (p.reps||{}))v=med(p.reps[c[0]]); if(c[0]==='sources')v=(p.sources||'').split(';')[0].replace(/\/rep\d+$/,'/rep*'); td.textContent=typeof v==='number'?fmt(v):(v??'–');});});}
function render(){
 const pts=sel(), ss=series(pts);
 [1,2,3,4,5,6].forEach(i=>legend($('lg'+i),ss));
 const concs=uniq(pts.map(p=>p.concurrency)).sort((a,b)=>a-b); const cur=+$('f-conc').value; opts($('f-conc'),concs.map(String)); if(concs.includes(cur))$('f-conc').value=cur;
 chart($('c-thr'),ss,p=>rep(p,'output_throughput'),{x:p=>p.concurrency,logx:true,xlabel:'concurrency',unit:' tok/s'});
 const lk=$('f-lat').value; chart($('c-lat'),ss,p=>rep(p,lk),{x:p=>p.concurrency,logx:true,logy:true,xlabel:'concurrency',unit:' ms'});
 ecdf($('c-dist'),ss,$('f-dist').value,+$('f-conc').value);
 chart($('c-en'),ss,p=>rep(p,'j_per_output_token'),{x:p=>p.concurrency,logx:true,logy:true,xlabel:'concurrency',unit:' J/tok'});
 chart($('c-tr'),ss,p=>rep(p,'output_throughput'),{x:p=>med(p.reps.tpot_p50_ms||[]),xlabel:'TPOT p50 (ms)',unit:' tok/s',extra:p=>' · c='+p.concurrency});
 chart($('c-gp'),ss,p=>rep(p,'goodput_tok_s'),{x:p=>p.concurrency,logx:true,xlabel:'concurrency',unit:' tok/s'});
 table(pts);
}
render();
</script></body></html>
"""


def build(results: str, out: str):
    pts = []
    for f in sorted(glob.glob(os.path.join(results, "*", "summary", "points.json"))):
        if os.path.basename(os.path.dirname(os.path.dirname(f))).startswith("_"):
            continue  # test plans (e.g. results/_mock)
        for p in json.load(open(f)):
            p.pop("fingerprint", None)
            pts.append(p)
    data = json.dumps(pts, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data).replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(html)
    print(f"wrote {out} ({len(pts)} points, {len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="dashboard/index.html")
    a = ap.parse_args()
    build(a.results, a.out)
