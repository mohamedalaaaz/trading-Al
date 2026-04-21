import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

// ─── colour tokens ────────────────────────────────────────────────────────────
const C = {
  bg:      "#050a0f",
  panel:   "#0a1520",
  border:  "#0d2035",
  accent:  "#00d4ff",
  green:   "#00ff88",
  red:     "#ff3860",
  yellow:  "#ffe566",
  purple:  "#a78bfa",
  muted:   "#4a6a8a",
  text:    "#c8e0f0",
  dim:     "#3a5a7a",
};

// ─── GBM synthetic price data ────────────────────────────────────────────────
function generateBars(sym, n = 200) {
  const seeds = { BTCUSDT:[65000,0.0002,0.020,17], ETHUSDT:[3200,0.0003,0.024,31],
                  BNBUSDT:[580,0.0001,0.018,43],    SOLUSDT:[170,0.0004,0.028,59] };
  const [start, drift, vol, seed] = seeds[sym] || [100, 0.0002, 0.022, 7];
  // deterministic seeded random (xorshift)
  let s = seed * 13 + 7;
  const rand = () => { s ^= s<<13; s ^= s>>17; s ^= s<<5; return (s>>>0)/0xffffffff; };
  const gauss = () => Math.sqrt(-2*Math.log(rand()+1e-9))*Math.cos(2*Math.PI*rand());
  const bars = []; let price = start, mom = 0;
  const now = Date.now();
  for (let i = 0; i < n; i++) {
    mom = 0.88*mom + 0.12*gauss();
    const ret = drift + vol*gauss() + mom*0.004;
    const o = price, c = price*Math.exp(ret);
    const range = price*vol*(0.4+0.6*rand());
    const h = Math.max(o,c)+range*rand(), l = Math.min(o,c)-range*rand();
    bars.push({ ts: now-(n-i)*3600000, o, h, l, c, v: 500+rand()*4500 });
    price = c;
  }
  return bars;
}

// ─── math helpers ─────────────────────────────────────────────────────────────
const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
const std  = a => { const m=mean(a); return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/a.length); };
const ema  = (a,p) => { let k=2/(p+1),e=a[0]; for(let i=1;i<a.length;i++) e=a[i]*k+e*(1-k); return e; };
const emaFull = (a,p) => { let k=2/(p+1),e=[a[0]]; for(let i=1;i<a.length;i++) e.push(a[i]*k+e[i-1]*(1-k)); return e; };
const sma  = (a,p) => { if(a.length<p) return mean(a); let s=0; for(let i=a.length-p;i<a.length;i++) s+=a[i]; return s/p; };
const rsi  = (c,p=14) => {
  if(c.length<=p) return 50; let ag=0,al=0;
  for(let i=1;i<=p;i++){const g=c[i]-c[i-1];g>0?ag+=g:al-=g;} ag/=p;al/=p;
  for(let i=p+1;i<c.length;i++){const g=c[i]-c[i-1];ag=(ag*(p-1)+Math.max(g,0))/p;al=(al*(p-1)+Math.max(-g,0))/p;}
  return al===0?100:100-100/(1+ag/al);
};
const bb   = (a,p=20,ns=2) => { const m=sma(a,p),sl=a.slice(-p); const s=std(sl); return [m,m+ns*s,m-ns*s]; };
const macd = (a,f=12,s=26,sg=9) => {
  const fe=emaFull(a,f),se=emaFull(a,s);
  const ml=a.map((_,i)=>fe[i]-se[i]); const sl=emaFull(ml,sg);
  return {line:ml[ml.length-1],signal:sl[sl.length-1],hist:ml[ml.length-1]-sl[sl.length-1]};
};
const atr  = (h,l,c,p=14) => {
  if(c.length<2) return 0;
  const tr=c.slice(1).map((_,i)=>Math.max(h[i+1]-l[i+1],Math.abs(h[i+1]-c[i]),Math.abs(l[i+1]-c[i])));
  return sma(tr,Math.min(p,tr.length));
};
const hurstExp = (prices) => {
  const n=prices.length; if(n<20) return 0.5;
  const logs=prices.slice(1).map((p,i)=>Math.log(p/prices[i]));
  const pts=[]; const periods=[4,8,16,32];
  for(const period of periods){
    if(period>logs.length) continue;
    let rsSum=0,cnt=0;
    for(let s=0;s+period<=logs.length;s+=period){
      const seg=logs.slice(s,s+period),m=mean(seg),cum=[0];
      for(let i=0;i<period;i++) cum.push(cum[cum.length-1]+seg[i]-m);
      const r=Math.max(...cum)-Math.min(...cum),sv=std(seg);
      if(sv>0){rsSum+=r/sv;cnt++;}
    }
    if(cnt>0) pts.push([Math.log(period),Math.log(rsSum/cnt)]);
  }
  if(pts.length<2) return 0.5;
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const mx=mean(xs),my=mean(ys);
  let num=0,den=0; pts.forEach(([x,y])=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});
  return den===0?0.5:num/den;
};
const garchVol=(rets)=>{let h=std(rets)**2;for(const r of rets)h=0.00001+0.09*r*r+0.9*h;return Math.sqrt(Math.max(h,0));};
const monteCarlo=(price,mu,sig,n=500)=>{
  let up=0; const finals=[];
  for(let i=0;i<n;i++){
    let s=price;
    for(let t=0;t<24;t++) s*=Math.exp((mu-0.5*sig*sig)/24+sig/Math.sqrt(24)*(Math.sqrt(-2*Math.log(Math.random()+1e-9))*Math.cos(2*Math.PI*Math.random())));
    finals.push(s); if(s>price) up++;
  }
  finals.sort((a,b)=>a-b);
  return {mean:mean(finals),prob:up/n,var95:(price-finals[Math.floor(0.05*n)])/price};
};

// ─── simple ML score: logistic-like feature combination ──────────────────────
function mlScore(closes, highs, lows, i) {
  if(i<30) return 0.5;
  const c=closes.slice(0,i+1),h=highs.slice(0,i+1),l=lows.slice(0,i+1);
  const price=c[i],e9=ema(c,9),e21=ema(c,21);
  const rv=rsi(c,14),b=bb(c,20,2),mc=macd(c),at=atr(h,l,c,14);
  const ret5=c.slice(-6,-1).map((_,k,a)=>k>0?(a[k]-a[k-1])/a[k-1]:0).filter((_,k)=>k>0);
  const mom=mean(ret5);
  // feature weights (hand-tuned logistic)
  const f0=(e9-e21)/price * 20;   // EMA spread
  const f1=(rv-50)/50;            // RSI
  const f2=mc.hist/price * 50;    // MACD
  const f3=mom * 30;              // momentum
  const f4=((price-b[2])/(b[1]-b[2]+1e-9))*2-1; // BB pos
  const raw = f0*0.35+f1*0.25+f2*0.20+f3*0.15+f4*0.05;
  return 1/(1+Math.exp(-raw*3));  // sigmoid
}

const SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"];

// ─── main component ───────────────────────────────────────────────────────────
export default function CryptoDashboard() {
  const [activeSym, setActiveSym]   = useState("BTCUSDT");
  const [allBars, setAllBars]       = useState({});
  const [stats, setStats]           = useState({});
  const [signals, setSignals]       = useState({});
  const [aiAnalysis, setAiAnalysis] = useState("");
  const [aiLoading, setAiLoading]   = useState(false);
  const [equity, setEquity]         = useState([]);
  const [trades, setTrades]         = useState([]);
  const [tick, setTick]             = useState(0);
  const abortRef = useRef(null);

  // ── initialise data ──────────────────────────────────────────────────────
  useEffect(() => {
    const bars = {}; const st = {}; const sig = {};
    SYMBOLS.forEach(sym => {
      const b = generateBars(sym, 200);
      bars[sym] = b;
      const C = b.map(x=>x.c), H = b.map(x=>x.h), L = b.map(x=>x.l);
      const re = C.slice(1).map((_,i)=>(C[i+1]-C[i])/C[i]);
      const rv = rsi(C,14), at = atr(H,L,C,14);
      const hu = hurstExp(C);
      const gv = garchVol(re);
      const mc = monteCarlo(C[C.length-1], mean(re), std(re));
      const b20 = bb(C,20,2), mac = macd(C);
      const score = mlScore(C,H,L,C.length-1);
      st[sym] = { price:C[C.length-1], rsi:rv, atr:at, hurst:hu,
                  garchVol:gv*100, mcProb:mc.prob*100, mcVar:mc.var95*100,
                  skew:skewness(re), kurt:kurtosis(re), var95:var95pct(re),
                  cvar95:cvar95pct(re), ema9:ema(C,9), ema21:ema(C,21),
                  bb:b20, macd:mac, returns:re };
      sig[sym] = { mlScore:score, direction: score>0.58?1:score<0.42?-1:0 };
    });
    setAllBars(bars); setStats(st); setSignals(sig);

    // run simple backtest on BTC for equity curve
    const { eq, tr } = runBacktest(bars["BTCUSDT"], 10000);
    setEquity(eq); setTrades(tr);
  }, []);

  // ── live tick ────────────────────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      setTick(t => t+1);
      // micro-update last price
      setStats(prev => {
        if(!prev["BTCUSDT"]) return prev;
        const next = {...prev};
        SYMBOLS.forEach(sym => {
          if(!next[sym]) return;
          const drift = (Math.random()-0.499)*0.003;
          next[sym] = {...next[sym], price: next[sym].price*(1+drift)};
        });
        return next;
      });
    }, 2000);
    return () => clearInterval(id);
  }, []);

  // ── AI analysis ──────────────────────────────────────────────────────────
  const fetchAI = useCallback(async () => {
    if(!stats[activeSym]) return;
    if(abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    setAiLoading(true); setAiAnalysis("");
    const s = stats[activeSym], sg = signals[activeSym];
    const prompt = `You are a quantitative crypto analyst. Analyze ${activeSym} with these live metrics:

Price: $${s.price?.toFixed(4)}
RSI(14): ${s.rsi?.toFixed(2)}
EMA9 vs EMA21: ${s.ema9 > s.ema21 ? "Bullish crossover" : "Bearish crossover"}
MACD Histogram: ${s.macd?.hist?.toFixed(6)} (${s.macd?.hist > 0 ? "positive" : "negative"})
Bollinger Band position: price ${s.price > s.bb?.[1] ? "ABOVE upper band" : s.price < s.bb?.[2] ? "BELOW lower band" : "inside bands"}
ATR(14): ${s.atr?.toFixed(4)} (${(s.atr/s.price*100)?.toFixed(2)}% of price)
Hurst Exponent: ${s.hurst?.toFixed(3)} (${s.hurst > 0.55 ? "trending" : s.hurst < 0.45 ? "mean-reverting" : "random walk"})
GARCH(1,1) Hourly Vol: ${s.garchVol?.toFixed(2)}%
Monte Carlo P(up 24h): ${s.mcProb?.toFixed(1)}%
VaR 95%: ${s.var95?.toFixed(2)}%
CVaR 95%: ${s.cvar95?.toFixed(2)}%
Return Skewness: ${s.skew?.toFixed(3)}
Excess Kurtosis: ${s.kurt?.toFixed(3)}
ML Bayesian Score: ${(sg?.mlScore*100)?.toFixed(1)}% bullish
ML Signal: ${sg?.direction === 1 ? "LONG" : sg?.direction === -1 ? "SHORT" : "NEUTRAL"}

Provide a concise quant analysis (3-4 sentences) covering:
1. Pattern detection & regime (trending/mean-reverting)
2. Probability assessment from statistics
3. Risk assessment (vol, drawdown risk)
4. Trading recommendation with math justification`;

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method:"POST", signal: abortRef.current.signal,
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          model:"claude-sonnet-4-20250514", max_tokens:1000,
          messages:[{role:"user",content:prompt}]
        })
      });
      const data = await res.json();
      const text = data.content?.find(b=>b.type==="text")?.text || "No response";
      setAiAnalysis(text);
    } catch(e) {
      if(e.name!=="AbortError") setAiAnalysis("Analysis unavailable.");
    }
    setAiLoading(false);
  }, [activeSym, stats, signals]);

  useEffect(() => { fetchAI(); }, [activeSym]);

  // ── derived chart data ────────────────────────────────────────────────────
  const bars    = allBars[activeSym] || [];
  const st      = stats[activeSym] || {};
  const sg      = signals[activeSym] || {};
  const chartBars = bars.slice(-80).map((b,i,a) => {
    const closes = a.slice(0,i+1).map(x=>x.c);
    const highs  = a.map(x=>x.h); const lows = a.map(x=>x.l);
    const rsiVal = closes.length>14?rsi(closes,14):50;
    const mc2    = closes.length>26?macd(closes):{hist:0};
    const [bMid,bUp,bLo] = closes.length>=20?bb(closes,20,2):[b.c,b.c,b.c];
    const score  = mlScore(a.map(x=>x.c),a.map(x=>x.h),a.map(x=>x.l),i);
    return {
      t: new Date(b.ts).toLocaleTimeString("en",{hour:"2-digit",minute:"2-digit"}),
      price: +b.c.toFixed(2), high:+b.h.toFixed(2), low:+b.l.toFixed(2),
      rsi:+rsiVal.toFixed(1), macdHist:+mc2.hist.toFixed(6),
      bbUp:+bUp.toFixed(2),bbMid:+bMid.toFixed(2),bbLo:+bLo.toFixed(2),
      ml:+(score*100).toFixed(1), vol:+b.v.toFixed(0)
    };
  });

  const priceChange = bars.length>1 ? (bars[bars.length-1].c-bars[bars.length-2].c)/bars[bars.length-2].c*100 : 0;

  return (
    <div style={{background:C.bg,minHeight:"100vh",color:C.text,fontFamily:"'JetBrains Mono',monospace",fontSize:13}}>

      {/* header */}
      <div style={{borderBottom:`1px solid ${C.border}`,padding:"14px 24px",display:"flex",alignItems:"center",gap:24,background:"#060e18"}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <span style={{color:C.accent,fontSize:18,fontWeight:700,letterSpacing:2}}>◈ QUANT</span>
          <span style={{color:C.muted,fontSize:11}}>CRYPTO ENGINE</span>
        </div>
        <div style={{flex:1}}/>
        <Pill color={C.green}>● LIVE</Pill>
        <span style={{color:C.muted,fontSize:11}}>ML · STATS · PROBABILITY</span>
      </div>

      {/* symbol tabs */}
      <div style={{padding:"10px 24px",borderBottom:`1px solid ${C.border}`,display:"flex",gap:8}}>
        {SYMBOLS.map(sym => {
          const s=stats[sym]||{}, sg2=signals[sym]||{};
          const pc=allBars[sym]?.length>1?(allBars[sym].at(-1).c-allBars[sym].at(-2).c)/allBars[sym].at(-2).c*100:0;
          return (
            <button key={sym} onClick={()=>setActiveSym(sym)} style={{
              background: activeSym===sym ? C.panel : "transparent",
              border:`1px solid ${activeSym===sym?C.accent:C.border}`,
              color: activeSym===sym?C.accent:C.muted,
              padding:"8px 16px", borderRadius:6, cursor:"pointer",
              display:"flex",flexDirection:"column",alignItems:"flex-start",gap:2
            }}>
              <div style={{fontWeight:700,fontSize:12}}>{sym.replace("USDT","")}</div>
              <div style={{fontSize:10,color:C.text}}>${s.price?.toFixed(2)||"—"}</div>
              <div style={{fontSize:10,color:pc>=0?C.green:C.red}}>{pc>=0?"+":""}{pc.toFixed(2)}%</div>
              <div style={{fontSize:10,color:sg2.direction===1?C.green:sg2.direction===-1?C.red:C.muted}}>
                {sg2.direction===1?"▲ LONG":sg2.direction===-1?"▼ SHORT":"● NEUTRAL"}
              </div>
            </button>
          );
        })}
      </div>

      <div style={{padding:20, display:"grid", gridTemplateColumns:"1fr 320px", gap:16}}>

        {/* left column */}
        <div style={{display:"flex",flexDirection:"column",gap:14}}>

          {/* price + stats row */}
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr 1fr",gap:10}}>
            {[
              {label:"PRICE", val:`$${st.price?.toFixed(2)||"—"}`, color: priceChange>=0?C.green:C.red, sub:`${priceChange>=0?"+":""}${priceChange.toFixed(3)}%`},
              {label:"RSI(14)", val:(st.rsi||50).toFixed(1),
                color:st.rsi<=30?C.green:st.rsi>=70?C.red:C.yellow},
              {label:"GARCH VOL", val:`${(st.garchVol||0).toFixed(2)}%`, color:C.purple},
              {label:"HURST EXP", val:(st.hurst||0.5).toFixed(3),
                color:st.hurst>0.55?C.green:st.hurst<0.45?C.red:C.yellow,
                sub:st.hurst>0.55?"TRENDING":st.hurst<0.45?"MEAN-REV":"RANDOM"},
              {label:"ML SIGNAL", val:`${(sg.mlScore||0.5)*100|0}%`,
                color:sg.direction===1?C.green:sg.direction===-1?C.red:C.yellow,
                sub:sg.direction===1?"LONG":sg.direction===-1?"SHORT":"NEUTRAL"},
            ].map(({label,val,color,sub})=>(
              <div key={label} style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:"12px 14px"}}>
                <div style={{color:C.muted,fontSize:10,marginBottom:4}}>{label}</div>
                <div style={{color,fontSize:18,fontWeight:700}}>{val}</div>
                {sub&&<div style={{color,fontSize:10,marginTop:2}}>{sub}</div>}
              </div>
            ))}
          </div>

          {/* price chart */}
          <ChartPanel title="PRICE · BOLLINGER BANDS · ML SCORE">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={chartBars}>
                <defs>
                  <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={C.accent} stopOpacity={0.15}/>
                    <stop offset="95%" stopColor={C.accent} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                <XAxis dataKey="t" tick={{fill:C.muted,fontSize:9}} interval={14}/>
                <YAxis tick={{fill:C.muted,fontSize:9}} domain={["auto","auto"]}/>
                <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:6,fontSize:11}}
                         labelStyle={{color:C.muted}} itemStyle={{color:C.text}}/>
                <Line type="monotone" dataKey="bbUp" stroke={C.dim} strokeWidth={1} dot={false} strokeDasharray="4 2"/>
                <Line type="monotone" dataKey="bbMid" stroke={C.muted} strokeWidth={1} dot={false} strokeDasharray="2 2"/>
                <Line type="monotone" dataKey="bbLo" stroke={C.dim} strokeWidth={1} dot={false} strokeDasharray="4 2"/>
                <Area type="monotone" dataKey="price" stroke={C.accent} strokeWidth={2} fill="url(#pg)" dot={false}/>
              </AreaChart>
            </ResponsiveContainer>
          </ChartPanel>

          {/* RSI + MACD row */}
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <ChartPanel title="RSI(14) — MOMENTUM">
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={chartBars}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="t" tick={{fill:C.muted,fontSize:9}} interval={19}/>
                  <YAxis domain={[0,100]} tick={{fill:C.muted,fontSize:9}}/>
                  <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:6,fontSize:11}}/>
                  <ReferenceLine y={70} stroke={C.red} strokeDasharray="3 3"/>
                  <ReferenceLine y={30} stroke={C.green} strokeDasharray="3 3"/>
                  <ReferenceLine y={50} stroke={C.muted} strokeDasharray="2 2"/>
                  <Line type="monotone" dataKey="rsi" stroke={C.yellow} strokeWidth={2} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </ChartPanel>
            <ChartPanel title="MACD HISTOGRAM — DIVERGENCE">
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={chartBars.slice(-50)}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="t" tick={{fill:C.muted,fontSize:9}} interval={9}/>
                  <YAxis tick={{fill:C.muted,fontSize:9}}/>
                  <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:6,fontSize:11}}/>
                  <ReferenceLine y={0} stroke={C.muted}/>
                  <Bar dataKey="macdHist" fill={C.purple}
                       shape={(props)=>{
                         const{x,y,width,height,value}=props;
                         return <rect x={x} y={y} width={width} height={Math.abs(height)}
                           fill={value>=0?C.green:C.red} opacity={0.8}/>;
                       }}/>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>
          </div>

          {/* ML score + equity */}
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            <ChartPanel title="ML BAYESIAN PROBABILITY SCORE">
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={chartBars}>
                  <defs>
                    <linearGradient id="mlg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.purple} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={C.purple} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="t" tick={{fill:C.muted,fontSize:9}} interval={19}/>
                  <YAxis domain={[0,100]} tick={{fill:C.muted,fontSize:9}}/>
                  <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:6,fontSize:11}}/>
                  <ReferenceLine y={58} stroke={C.green} strokeDasharray="3 3"/>
                  <ReferenceLine y={42} stroke={C.red} strokeDasharray="3 3"/>
                  <ReferenceLine y={50} stroke={C.muted} strokeDasharray="2 2"/>
                  <Area type="monotone" dataKey="ml" stroke={C.purple} strokeWidth={2} fill="url(#mlg)" dot={false}/>
                </AreaChart>
              </ResponsiveContainer>
            </ChartPanel>
            <ChartPanel title="EQUITY CURVE — SIMULATED $10K">
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={equity}>
                  <defs>
                    <linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.green} stopOpacity={0.2}/>
                      <stop offset="95%" stopColor={C.green} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="i" tick={{fill:C.muted,fontSize:9}} interval={19}/>
                  <YAxis tick={{fill:C.muted,fontSize:9}} domain={["auto","auto"]}/>
                  <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:6,fontSize:11}}
                    formatter={v=>`$${v.toFixed(2)}`}/>
                  <ReferenceLine y={10000} stroke={C.muted} strokeDasharray="3 3"/>
                  <Area type="monotone" dataKey="eq" stroke={C.green} strokeWidth={2} fill="url(#eqg)" dot={false}/>
                </AreaChart>
              </ResponsiveContainer>
            </ChartPanel>
          </div>
        </div>

        {/* right column */}
        <div style={{display:"flex",flexDirection:"column",gap:14}}>

          {/* stat grid */}
          <ChartPanel title="STATISTICAL INDICATORS">
            {[
              ["VaR 95%", `${(st.var95||0).toFixed(2)}%`, C.red],
              ["CVaR 95%", `${(st.cvar95||0).toFixed(2)}%`, C.red],
              ["MC P(↑24h)", `${(st.mcProb||50).toFixed(1)}%`, st.mcProb>55?C.green:st.mcProb<45?C.red:C.yellow],
              ["ATR/Price", `${st.atr&&st.price?(st.atr/st.price*100).toFixed(2):0}%`, C.purple],
              ["Skewness", (st.skew||0).toFixed(3), st.skew>0?C.green:C.red],
              ["Kurtosis", (st.kurt||0).toFixed(3), Math.abs(st.kurt||0)>1?C.yellow:C.text],
              ["EMA Cross", st.ema9>st.ema21?"BULLISH":"BEARISH", st.ema9>st.ema21?C.green:C.red],
              ["MACD", st.macd?.hist>0?"POSITIVE":"NEGATIVE", st.macd?.hist>0?C.green:C.red],
            ].map(([k,v,c])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"7px 0",borderBottom:`1px solid ${C.border}`}}>
                <span style={{color:C.muted,fontSize:11}}>{k}</span>
                <span style={{color:c,fontSize:11,fontWeight:600}}>{v}</span>
              </div>
            ))}
          </ChartPanel>

          {/* AI analysis */}
          <ChartPanel title="◈ AI QUANT ANALYSIS" extra={
            <button onClick={fetchAI} style={{
              background:"transparent",border:`1px solid ${C.accent}`,
              color:C.accent,padding:"3px 10px",borderRadius:4,cursor:"pointer",fontSize:10
            }}>REFRESH</button>
          }>
            <div style={{minHeight:120,fontSize:12,lineHeight:1.7,color:C.text}}>
              {aiLoading ? (
                <div style={{color:C.muted,display:"flex",alignItems:"center",gap:8}}>
                  <span style={{animation:"spin 1s linear infinite",display:"inline-block"}}>◈</span>
                  Analyzing patterns…
                </div>
              ) : aiAnalysis ? (
                <div style={{whiteSpace:"pre-wrap"}}>{aiAnalysis}</div>
              ) : (
                <div style={{color:C.muted}}>Click REFRESH to run AI analysis</div>
              )}
            </div>
          </ChartPanel>

          {/* trades */}
          <ChartPanel title="RECENT TRADES">
            {trades.slice(-6).reverse().map((t,i)=>(
              <div key={i} style={{
                display:"flex",justifyContent:"space-between",alignItems:"center",
                padding:"6px 0",borderBottom:`1px solid ${C.border}`,fontSize:11
              }}>
                <div>
                  <span style={{color:t.pnl>0?C.green:C.red,fontWeight:700}}>
                    {t.dir===1?"▲":"▼"} {t.strat}
                  </span>
                  <div style={{color:C.muted,fontSize:10}}>{t.exitDate}</div>
                </div>
                <div style={{textAlign:"right"}}>
                  <div style={{color:t.pnl>0?C.green:C.red}}>
                    {t.pnl>0?"+":""}{t.pnl.toFixed(2)}
                  </div>
                  <div style={{color:C.muted,fontSize:10}}>ML {(t.conf*100).toFixed(0)}%</div>
                </div>
              </div>
            ))}
          </ChartPanel>

          {/* probability gauge */}
          <ChartPanel title="BAYESIAN SIGNAL GAUGE">
            <div style={{padding:"8px 0"}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:6,fontSize:11}}>
                <span style={{color:C.red}}>SHORT</span>
                <span style={{color:C.muted,fontWeight:700}}>{((sg.mlScore||0.5)*100).toFixed(1)}% BULLISH</span>
                <span style={{color:C.green}}>LONG</span>
              </div>
              <div style={{background:C.border,borderRadius:8,height:12,overflow:"hidden"}}>
                <div style={{
                  width:`${(sg.mlScore||0.5)*100}%`,
                  height:"100%",borderRadius:8,
                  background:`linear-gradient(90deg,${C.red},${C.yellow},${C.green})`,
                  transition:"width 0.8s ease"
                }}/>
              </div>
              <div style={{position:"relative",marginTop:4}}>
                <div style={{
                  position:"absolute",
                  left:`${(sg.mlScore||0.5)*100}%`,
                  transform:"translateX(-50%)",
                  color:C.accent,fontSize:16,marginTop:-2
                }}>▲</div>
              </div>
            </div>
          </ChartPanel>
        </div>
      </div>
      <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

// ── helpers ────────────────────────────────────────────────────────────────────
function ChartPanel({title,children,extra}){
  return (
    <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:10,overflow:"hidden"}}>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${C.border}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <span style={{color:C.accent,fontSize:10,fontWeight:700,letterSpacing:1}}>{title}</span>
        {extra}
      </div>
      <div style={{padding:12}}>{children}</div>
    </div>
  );
}
function Pill({color,children}){
  return <span style={{background:color+"22",border:`1px solid ${color}44`,color,padding:"3px 10px",borderRadius:20,fontSize:10}}>{children}</span>;
}
function skewness(a){const m=mean(a),s=std(a);if(!s)return 0;return a.reduce((t,x)=>t+((x-m)/s)**3,0)/a.length;}
function kurtosis(a){const m=mean(a),s=std(a);if(!s)return 0;return a.reduce((t,x)=>t+((x-m)/s)**4,0)/a.length-3;}
function var95pct(rets){const s=[...rets].sort((a,b)=>a-b);return -s[Math.floor(0.05*s.length)]*100;}
function cvar95pct(rets){const s=[...rets].sort((a,b)=>a-b);const c=Math.floor(0.05*s.length);return c>0?-s.slice(0,c).reduce((a,x)=>a+x,0)/c*100:0;}

function runBacktest(bars, capital){
  const eq=[{i:0,eq:capital}]; const tr=[]; let cash=capital;
  const C=bars.map(x=>x.c),H=bars.map(x=>x.h),L=bars.map(x=>x.l);
  let pos=null; let idx=1;
  for(let i=50;i<bars.length;i++){
    const bar=bars[i];
    const c=C.slice(0,i+1),h=H.slice(0,i+1),l=L.slice(0,i+1);
    const at=atr(h,l,c,14);
    if(pos){
      if((pos.dir===1&&bar.l<=pos.stop)||(pos.dir===-1&&bar.h>=pos.stop)){
        const pnl=(pos.stop-pos.ep)*pos.dir*pos.qty; cash+=pos.qty*pos.stop;
        tr.push({...pos,exitDate:new Date(bar.ts).toLocaleDateString(),exit:pos.stop,pnl,why:"SL"}); pos=null;
      } else if((pos.dir===1&&bar.h>=pos.tp)||(pos.dir===-1&&bar.l<=pos.tp)){
        const pnl=(pos.tp-pos.ep)*pos.dir*pos.qty; cash+=pos.qty*pos.tp;
        tr.push({...pos,exitDate:new Date(bar.ts).toLocaleDateString(),exit:pos.tp,pnl,why:"TP"}); pos=null;
      }
    }
    if(!pos){
      const score=mlScore(C,H,L,i);
      const rv=rsi(c,14);
      const e9=ema(c,9),e21=ema(c,21);
      const mc2=macd(c);
      let dir=0,strat="";
      if(score>0.62&&rv<65&&e9>e21&&mc2.hist>0){dir=1;strat="ML+TECH";}
      else if(score<0.38&&rv>35&&e9<e21&&mc2.hist<0){dir=-1;strat="ML+TECH";}
      else if(score>0.68){dir=1;strat="ML-LONG";}
      else if(score<0.32){dir=-1;strat="ML-SHORT";}
      if(dir!==0){
        const stop=bar.c-dir*1.8*at, tp=bar.c+dir*2.8*at;
        const qty=(capital*0.12)/bar.c;
        if(qty*bar.c<cash){
          cash-=qty*bar.c;
          pos={sym:"BTC",dir,strat,ep:bar.c,stop,tp,qty,conf:score,entryDate:new Date(bar.ts).toLocaleDateString()};
        }
      }
    }
    eq.push({i:idx++,eq:+(cash+(pos?pos.qty*bar.c:0)).toFixed(2)});
  }
  return {eq,tr};
}
