import java.net.URI;
import java.net.http.*;
import java.time.*;
import java.util.*;
import java.util.stream.*;

/**
 * ═══════════════════════════════════════════════════════════════════════════
 *   CRYPTO QUANTITATIVE TRADING ENGINE  —  Live Binance + ML + Stats
 * ═══════════════════════════════════════════════════════════════════════════
 *  Live Data  : Binance REST API  (no API key)
 *  ML Models  : Linear Reg · Logistic Reg · MLP Neural Net · KNN
 *  Statistics : GARCH · Hurst · Skewness · Kurtosis · VaR · CVaR
 *  Probability: Bayesian Ensemble · Kelly Criterion · Monte Carlo · EV Filter
 *  Strategies : EMA · RSI · Bollinger · MACD · ROC  +  ML composite
 * ═══════════════════════════════════════════════════════════════════════════
 */
public class CryptoQuantTrader {

    static final String[] SYMBOLS          = {"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"};
    static final String   INTERVAL         = "1h";
    static final int      BARS_FETCH       = 300;
    static final double   INITIAL_CAPITAL  = 10_000.0;
    static final double   MAX_POS_PCT      = 0.15;
    static final double   COMMISSION       = 0.001;
    static final double   MAX_DD_HALT      = 0.20;
    static final double   MIN_BAYES_PROB   = 0.58;
    static final int      KNN_K            = 7;
    static final int      FEAT_WINDOW      = 14;
    static final int      WARMUP           = 50;

    record Bar(long ts, double o, double h, double l, double c, double v) {
        String date() {
            return Instant.ofEpochMilli(ts).atZone(ZoneId.of("UTC"))
                .format(java.time.format.DateTimeFormatter.ofPattern("MM-dd HH:mm"));
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // LIVE DATA
    // ════════════════════════════════════════════════════════════════════════
    static List<Bar> fetchBars(String sym) {
        String url = "https://api.binance.com/api/v3/klines?symbol=" + sym
                   + "&interval=" + INTERVAL + "&limit=" + BARS_FETCH;
        try {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(url))
                .header("Accept","application/json").GET().build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) throw new RuntimeException("HTTP "+resp.statusCode());
            return parseKlines(resp.body());
        } catch (Exception e) {
            System.err.println("    ⚠  Binance unavailable (" + e.getMessage()
                + ") — using GBM simulation");
            return syntheticBars(sym, BARS_FETCH);
        }
    }

    static List<Bar> parseKlines(String json) {
        List<Bar> out = new ArrayList<>();
        json = json.trim().substring(1, json.length()-1);
        for (String row : json.split("\\],\\s*\\[")) {
            String[] f = row.replace("[","").replace("]","").split(",");
            if (f.length < 6) continue;
            out.add(new Bar(
                Long.parseLong(f[0].trim()),
                Double.parseDouble(f[1].trim().replace("\"","")),
                Double.parseDouble(f[2].trim().replace("\"","")),
                Double.parseDouble(f[3].trim().replace("\"","")),
                Double.parseDouble(f[4].trim().replace("\"","")),
                Double.parseDouble(f[5].trim().replace("\"",""))
            ));
        }
        return out;
    }

    static List<Bar> syntheticBars(String sym, int n) {
        Map<String,double[]> cfg = Map.of(
            "BTCUSDT", new double[]{65000, 0.0002, 0.020, 17},
            "ETHUSDT", new double[]{3200,  0.0003, 0.024, 31},
            "BNBUSDT", new double[]{580,   0.0001, 0.018, 43},
            "SOLUSDT", new double[]{170,   0.0004, 0.028, 59}
        );
        double[] p = cfg.getOrDefault(sym, new double[]{100,0.0002,0.022,7});
        Random rng = new Random((long)p[3]*13);
        List<Bar> bars = new ArrayList<>();
        double price = p[0];
        long ts = System.currentTimeMillis() - (long)n * 3_600_000;
        double mom = 0;
        for (int i = 0; i < n; i++) {
            mom = 0.88 * mom + 0.12 * rng.nextGaussian();
            double ret = p[1] + p[2] * rng.nextGaussian() + mom * 0.004;
            double o = price, c = price * Math.exp(ret);
            double range = price * p[2] * (0.4 + 0.6*rng.nextDouble());
            double h = Math.max(o,c) + range * rng.nextDouble();
            double lo = Math.min(o,c) - range * rng.nextDouble();
            bars.add(new Bar(ts, o, h, lo, c, 500+rng.nextDouble()*4500));
            price = c; ts += 3_600_000;
        }
        return bars;
    }

    // ════════════════════════════════════════════════════════════════════════
    // STATISTICS LIBRARY
    // ════════════════════════════════════════════════════════════════════════
    static double mean(double[] a) { return Arrays.stream(a).average().orElse(0); }
    static double variance(double[] a) {
        double m=mean(a); return Arrays.stream(a).map(x->(x-m)*(x-m)).average().orElse(0);
    }
    static double std(double[] a) { return Math.sqrt(variance(a)); }

    static double skewness(double[] a) {
        double m=mean(a), s=std(a); if(s==0) return 0;
        double sum=0; for(double x:a) sum+=Math.pow((x-m)/s,3);
        return sum/a.length;
    }
    static double kurtosis(double[] a) {
        double m=mean(a), s=std(a); if(s==0) return 0;
        double sum=0; for(double x:a) sum+=Math.pow((x-m)/s,4);
        return sum/a.length-3;
    }
    static double autocorr(double[] a, int lag) {
        if(a.length<=lag) return 0;
        double m=mean(a); double num=0,den=0;
        for(int i=lag;i<a.length;i++) num+=(a[i]-m)*(a[i-lag]-m);
        for(double x:a) den+=(x-m)*(x-m);
        return den==0?0:num/den;
    }
    static double hurstExponent(double[] prices) {
        int n=prices.length; if(n<20) return 0.5;
        double[] logs=new double[n-1];
        for(int i=1;i<n;i++) logs[i-1]=Math.log(prices[i]/prices[i-1]);
        List<double[]> pts=new ArrayList<>();
        for(int period:new int[]{4,8,16,32,Math.min(64,n/4)}) {
            if(period>logs.length) continue;
            double rsSum=0; int cnt=0;
            for(int s=0;s+period<=logs.length;s+=period) {
                double[] seg=Arrays.copyOfRange(logs,s,s+period);
                double m2=mean(seg);
                double[] cum=new double[period]; cum[0]=seg[0]-m2;
                for(int i=1;i<period;i++) cum[i]=cum[i-1]+seg[i]-m2;
                double r=Arrays.stream(cum).max().orElse(0)-Arrays.stream(cum).min().orElse(0);
                double s2=std(seg); if(s2>0){rsSum+=r/s2;cnt++;}
            }
            if(cnt>0) pts.add(new double[]{Math.log(period),Math.log(rsSum/cnt)});
        }
        if(pts.size()<2) return 0.5;
        double[] xs=pts.stream().mapToDouble(p->p[0]).toArray();
        double[] ys=pts.stream().mapToDouble(p->p[1]).toArray();
        return slope(xs,ys);
    }
    static double slope(double[] x, double[] y) {
        int n=Math.min(x.length,y.length); if(n<2) return 0;
        double mx=mean(x),my=mean(y),num=0,den=0;
        for(int i=0;i<n;i++){num+=(x[i]-mx)*(y[i]-my);den+=(x[i]-mx)*(x[i]-mx);}
        return den==0?0:num/den;
    }
    static double garchVol(double[] rets) {
        double lv=variance(rets),h=lv;
        for(double r:rets) h=0.00001+0.09*r*r+0.90*h;
        return Math.sqrt(Math.max(h,0));
    }
    static double var95(double[] rets) {
        double[] s=rets.clone(); Arrays.sort(s);
        return -s[(int)(0.05*s.length)];
    }
    static double cvar95(double[] rets) {
        double[] s=rets.clone(); Arrays.sort(s);
        int c=(int)(0.05*s.length); double sum=0;
        for(int i=0;i<c;i++) sum+=s[i];
        return c>0?-sum/c:0;
    }
    static double sharpe(double[] rets) {
        double m=mean(rets),s=std(rets);
        return s==0?0:m/s*Math.sqrt(24*365);
    }
    static double maxDD(double[] eq) {
        double pk=eq[0],md=0;
        for(double e:eq){if(e>pk)pk=e;md=Math.max(md,(pk-e)/pk);}
        return md;
    }
    static double[] monteCarlo(double price,double mu,double sig,int n) {
        Random rng=new Random(42); double[] finals=new double[n];
        for(int p2=0;p2<n;p2++){
            double s=price;
            for(int t=0;t<24;t++)
                s*=Math.exp((mu-0.5*sig*sig)/24+sig/Math.sqrt(24)*rng.nextGaussian());
            finals[p2]=s;
        }
        Arrays.sort(finals);
        double up=(double)Arrays.stream(finals).filter(f->f>price).count()/n;
        double var=(price-finals[(int)(0.05*n)])/price;
        return new double[]{mean(finals),std(finals),up,var};
    }

    // ─── Technical indicators ────────────────────────────────────────────────
    static double sma(double[] d,int p) {
        if(d.length<p) return mean(d);
        double s=0; for(int i=d.length-p;i<d.length;i++) s+=d[i]; return s/p;
    }
    static double[] emaFull(double[] d,int p) {
        double k=2.0/(p+1); double[] e=new double[d.length]; e[0]=d[0];
        for(int i=1;i<d.length;i++) e[i]=d[i]*k+e[i-1]*(1-k); return e;
    }
    static double ema(double[] d,int p){double[]f=emaFull(d,p);return f[f.length-1];}
    static double rsi(double[] c,int p) {
        if(c.length<=p) return 50; double ag=0,al=0;
        for(int i=1;i<=p;i++){double g=c[i]-c[i-1];if(g>0)ag+=g;else al-=g;}
        ag/=p;al/=p;
        for(int i=p+1;i<c.length;i++){double g=c[i]-c[i-1];
            ag=(ag*(p-1)+Math.max(g,0))/p;al=(al*(p-1)+Math.max(-g,0))/p;}
        return al==0?100:100-100/(1+ag/al);
    }
    static double[] bb(double[] d,int p,double ns) {
        double mid=sma(d,p);
        double[] sl=Arrays.copyOfRange(d,d.length-p,d.length);
        double s=std(sl); return new double[]{mid,mid+ns*s,mid-ns*s};
    }
    static double[] macd(double[] d,int f,int s,int sg) {
        double[] fe=emaFull(d,f),se=emaFull(d,s);
        double[] ml=new double[d.length];
        for(int i=0;i<d.length;i++) ml[i]=fe[i]-se[i];
        double[] sl2=emaFull(ml,sg); double mv=ml[ml.length-1],sv=sl2[sl2.length-1];
        return new double[]{mv,sv,mv-sv};
    }
    static double atr(double[] h,double[] l,double[] c,int p) {
        int n=c.length; if(n<2) return 0;
        double[] tr=new double[n-1];
        for(int i=1;i<n;i++)
            tr[i-1]=Math.max(h[i]-l[i],Math.max(Math.abs(h[i]-c[i-1]),Math.abs(l[i]-c[i-1])));
        return sma(tr,Math.min(p,tr.length));
    }
    static double roc(double[] d,int p) {
        int n=d.length; if(n<=p) return 0; return (d[n-1]-d[n-1-p])/d[n-1-p]*100;
    }

    // ════════════════════════════════════════════════════════════════════════
    // FEATURE EXTRACTION  (12 features)
    // ════════════════════════════════════════════════════════════════════════
    static double[] features(double[] c,double[] h,double[] l,int i) {
        if(i < WARMUP) return null;
        double price = c[i];
        double e9  = ema(c,9), e21=ema(c,21);
        double rv  = rsi(c,14);
        double[] b = bb(c,20,2.0);
        double[] mc= macd(c,12,26,9);
        double rc  = roc(c,10);
        double at  = atr(h,l,c,14);
        // 5-bar returns
        int R=Math.min(5,i);
        double[] r5=new double[R];
        for(int k=0;k<R;k++) r5[k]=(c[i-k]-c[i-k-1])/c[i-k-1];
        double mom=mean(r5), vl=std(r5);
        double bbPos=(price-b[2])/Math.max(b[1]-b[2],1e-9);
        double sm50=sma(c,Math.min(50,c.length));

        return new double[]{
            safe((e9-e21)/price),          // 0 EMA spread
            safe((rv-50)/50),              // 1 RSI
            safe(bbPos*2-1),               // 2 BB position
            safe(mc[2]/price),             // 3 MACD hist
            safe(rc/100),                  // 4 ROC
            safe(mom),                     // 5 momentum
            safe(vl),                      // 6 realised vol
            safe(at/price),                // 7 ATR/price
            safe(autocorr(r5,1)),          // 8 autocorr
            safe((price-sm50)/price),      // 9 vs SMA50
            safe(R>1?r5[0]:0),             // 10
            safe(R>2?r5[1]:0)              // 11
        };
    }
    static double safe(double x) {
        return Double.isFinite(x) ? Math.max(-5, Math.min(5, x)) : 0;
    }

    // ════════════════════════════════════════════════════════════════════════
    // ML MODELS
    // ════════════════════════════════════════════════════════════════════════

    static class LogisticReg {
        double[] w; double b;
        static double sig(double x){return 1/(1+Math.exp(-Math.max(-15,Math.min(15,x))));}
        void train(double[][] X,double[] y) {
            int f=X[0].length; w=new double[f]; b=0; double lr=0.05;
            for(int it=0;it<2000;it++){
                double[] gw=new double[f]; double gb=0;
                for(int i=0;i<X.length;i++){
                    double raw=b; for(int j=0;j<f;j++) raw+=w[j]*X[i][j];
                    double err=sig(raw)-y[i];
                    for(int j=0;j<f;j++) gw[j]+=err*X[i][j]; gb+=err;
                }
                for(int j=0;j<f;j++) w[j]-=lr*gw[j]/X.length;
                b-=lr*gb/X.length;
            }
        }
        double prob(double[] x){
            double raw=b; for(int j=0;j<w.length;j++) raw+=w[j]*x[j]; return sig(raw);
        }
    }

    static class NeuralNet {
        // 12 → 10 → 5 → 1
        double[][] w1,w2,w3; double[] b1,b2,b3;
        Random rng=new Random(42);
        NeuralNet(){
            w1=he(10,12);b1=new double[10];
            w2=he(5,10); b2=new double[5];
            w3=he(1,5);  b3=new double[1];
        }
        double[][] he(int r,int c){
            double[][] m=new double[r][c]; double s=Math.sqrt(2.0/c);
            for(double[] row:m) for(int j=0;j<c;j++) row[j]=rng.nextGaussian()*s;
            return m;
        }
        static double relu(double x){return Math.max(0,x);}
        static double sig(double x){return 1/(1+Math.exp(-Math.max(-15,Math.min(15,x))));}

        double[] fwd1(double[] x){double[] h=new double[b1.length];for(int i=0;i<h.length;i++){double s=b1[i];for(int j=0;j<x.length;j++)s+=w1[i][j]*x[j];h[i]=relu(s);}return h;}
        double[] fwd2(double[] x){double[] h=new double[b2.length];for(int i=0;i<h.length;i++){double s=b2[i];for(int j=0;j<x.length;j++)s+=w2[i][j]*x[j];h[i]=relu(s);}return h;}
        double fwdO(double[] x){double s=b3[0];for(int j=0;j<x.length;j++)s+=w3[0][j]*x[j];return sig(s);}

        double predict(double[] x){return fwdO(fwd2(fwd1(x)));}

        void train(double[][] X,double[] y){
            double lr=0.01; int ep=200,bs=32;
            List<Integer> idx=new ArrayList<>();
            for(int i=0;i<X.length;i++) idx.add(i);
            for(int e=0;e<ep;e++){
                Collections.shuffle(idx,rng);
                for(int b2=0;b2<X.length;b2+=bs){
                    int end=Math.min(b2+bs,X.length);
                    for(int ii=b2;ii<end;ii++){
                        int i=idx.get(ii);
                        double[] h1=fwd1(X[i]),h2=fwd2(h1);
                        double out=fwdO(h2),d3=out-y[i];
                        for(int j=0;j<h2.length;j++) w3[0][j]-=lr*d3*h2[j];
                        b3[0]-=lr*d3;
                        double[] d2=new double[h2.length];
                        for(int j=0;j<h2.length;j++) d2[j]=d3*w3[0][j]*(h2[j]>0?1:0);
                        for(int j=0;j<h2.length;j++) for(int k=0;k<h1.length;k++) w2[j][k]-=lr*d2[j]*h1[k];
                        for(int j=0;j<h2.length;j++) this.b2[j]-=lr*d2[j];
                        double[] d1=new double[h1.length];
                        for(int j=0;j<h1.length;j++){double s=0;for(int k=0;k<d2.length;k++)s+=d2[k]*w2[k][j];d1[j]=s*(h1[j]>0?1:0);}
                        for(int j=0;j<h1.length;j++) for(int k=0;k<X[i].length;k++) w1[j][k]-=lr*d1[j]*X[i][k];
                        for(int j=0;j<h1.length;j++) b1[j]-=lr*d1[j];
                    }
                }
            }
        }
    }

    static class KNN {
        double[][] tx; double[] ty;
        void train(double[][] X,double[] y){tx=X;ty=y;}
        double cos(double[] a,double[] b){
            double d=0,na=0,nb=0;
            for(int i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
            double dn=Math.sqrt(na)*Math.sqrt(nb); return dn==0?0:d/dn;
        }
        double prob(double[] x){
            double[] s=new double[tx.length];
            for(int i=0;i<tx.length;i++) s[i]=cos(x,tx[i]);
            int[] order=IntStream.range(0,tx.length)
                .boxed().sorted((a,b)->Double.compare(s[b],s[a]))
                .mapToInt(Integer::intValue).toArray();
            double pos=0; int K=Math.min(KNN_K,tx.length);
            for(int k=0;k<K;k++) pos+=ty[order[k]];
            return pos/K;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // ML ENGINE  — trains all models, stores normalisation params
    // ════════════════════════════════════════════════════════════════════════
    static class MLEngine {
        LogisticReg lr = new LogisticReg();
        NeuralNet   nn = new NeuralNet();
        KNN         kn = new KNN();
        double[]    mu, sg;
        boolean     ok = false;

        void train(List<Bar> bars){
            int n=bars.size();
            double[] c=bars.stream().mapToDouble(Bar::c).toArray();
            double[] h=bars.stream().mapToDouble(Bar::h).toArray();
            double[] lo=bars.stream().mapToDouble(Bar::l).toArray();
            List<double[]> XL=new ArrayList<>(); List<Double> yL=new ArrayList<>();
            for(int i=WARMUP;i<n-1;i++){
                double[] f=features(c,h,lo,i); if(f==null) continue;
                boolean anyNaN=false; for(double v:f) if(!Double.isFinite(v)){anyNaN=true;break;}
                if(anyNaN) continue;
                XL.add(f); yL.add((c[i+1]-c[i])/c[i]>0.0005?1.0:0.0);
            }
            if(XL.size()<30){System.out.println("    insufficient training data");return;}
            double[][] X=XL.toArray(new double[0][]);
            double[] y=yL.stream().mapToDouble(Double::doubleValue).toArray();
            // normalise
            int F=X[0].length; mu=new double[F]; sg=new double[F];
            for(int j=0;j<F;j++){
                double[] col=new double[X.length];
                for(int i=0;i<X.length;i++) col[i]=X[i][j];
                mu[j]=mean(col); sg[j]=Math.max(std(col),1e-9);
                for(int i=0;i<X.length;i++) X[i][j]=(X[i][j]-mu[j])/sg[j];
            }
            lr.train(X,y); nn.train(X,y); kn.train(X,y);
            ok=true;
            System.out.printf("    ML training OK  (%d samples, %d features)%n",XL.size(),F);
        }

        double[] query(double[] raw){
            if(!ok||raw==null) return new double[]{0,0.5,0};
            double[] f=raw.clone();
            for(int j=0;j<f.length;j++) f[j]=(f[j]-mu[j])/sg[j];
            double pLR=clamp(lr.prob(f));
            double pNN=clamp(nn.predict(f));
            double pKN=clamp(kn.prob(f));
            // Bayesian product
            double pos=1,neg=1;
            for(double p:new double[]{pLR,pNN,pKN}){pos*=p/0.5;neg*=(1-p)/0.5;}
            double bayes=clamp(pos/(pos+neg));
            // majority vote
            int v=(pLR>0.55?1:pLR<0.45?-1:0)+(pNN>0.55?1:pNN<0.45?-1:0)+(pKN>0.55?1:pKN<0.45?-1:0);
            int sig=v>=2?1:v<=-2?-1:0;
            double conf=Math.abs(bayes-0.5)*2;
            return new double[]{sig,bayes,conf,pLR,pNN,pKN};
        }
        static double clamp(double x){return Double.isFinite(x)?Math.max(0.01,Math.min(0.99,x)):0.5;}
    }

    // ════════════════════════════════════════════════════════════════════════
    // PORTFOLIO
    // ════════════════════════════════════════════════════════════════════════
    record Trade(String sym,String strat,String e0,String e1,int dir,
                 double ep,double xp,double qty,double pnl,String why,double conf){}
    static class Pos {
        String sym,strat,date; int dir;
        double ep,stop,tp,qty,risk,conf;
        Pos(String s,String st,String d,int dir,double ep,double stop,double tp,double qty,double risk,double conf){
            this.sym=s;this.strat=st;this.date=d;this.dir=dir;
            this.ep=ep;this.stop=stop;this.tp=tp;this.qty=qty;this.risk=risk;this.conf=conf;
        }
    }

    static class Portfolio {
        double cash,peak; boolean halt;
        List<Pos>   open=new ArrayList<>();
        List<Trade> closed=new ArrayList<>();
        List<Double> eq=new ArrayList<>();
        int wins,losses; double tw,tl;
        Portfolio(double cap){cash=cap;peak=cap;eq.add(cap);}

        double equity(Map<String,Double> px){
            return cash+open.stream().mapToDouble(p->p.qty*px.getOrDefault(p.sym,p.ep)).sum();
        }
        boolean enter(String sym,String strat,int dir,double price,double atrV,
                      double totalEq,String date,double conf){
            if(halt) return false;
            double stop=price-dir*1.8*atrV, tp=price+dir*2.8*atrV;
            double rps=Math.abs(price-stop); if(rps==0) return false;
            double wr=wins+losses>15?(double)wins/(wins+losses):0.5;
            double avgW=wins>0?tw/wins:rps*2.8, avgL=losses>0?tl/losses:rps;
            double b=avgW/Math.max(avgL,1e-9);
            double kelly=Math.max(0,(b*wr-(1-wr))/b)*conf;
            kelly=Math.min(kelly,MAX_POS_PCT);
            double qty=totalEq*kelly/price;
            if(qty*price*(1+COMMISSION)>cash||qty<0.0001) return false;
            cash-=qty*price*(1+COMMISSION);
            open.add(new Pos(sym,strat,date,dir,price,stop,tp,qty,qty*rps,conf));
            return true;
        }
        void checkExits(String sym,Bar bar,String date,Map<String,Double> prices){
            Iterator<Pos> it=open.iterator();
            while(it.hasNext()){
                Pos p=it.next(); if(!p.sym.equals(sym)) continue;
                double xp=0; String why="";
                if((p.dir==1&&bar.l()<=p.stop)||(p.dir==-1&&bar.h()>=p.stop)){xp=p.stop;why="SL";}
                else if((p.dir==1&&bar.h()>=p.tp)||(p.dir==-1&&bar.l()<=p.tp)){xp=p.tp;why="TP";}
                if(xp>0){
                    double pnl=(xp-p.ep)*p.dir*p.qty-xp*p.qty*COMMISSION;
                    cash+=xp*p.qty*(1-COMMISSION);
                    closed.add(new Trade(p.sym,p.strat,p.date,date,p.dir,p.ep,xp,p.qty,pnl,why,p.conf));
                    if(pnl>0){wins++;tw+=pnl;}else{losses++;tl-=pnl;}
                    eq.add(equity(prices)); it.remove();
                }
            }
            double e=equity(prices); if(e>peak)peak=e;
            if((peak-e)/peak>=MAX_DD_HALT){System.out.printf("%n  ⚠ MAX DD — HALT%n");halt=true;}
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // BACKTEST ENGINE
    // ════════════════════════════════════════════════════════════════════════
    static void backtest(String sym,List<Bar> bars,Portfolio pm,MLEngine ml){
        int n=bars.size();
        double[] C=bars.stream().mapToDouble(Bar::c).toArray();
        double[] H=bars.stream().mapToDouble(Bar::h).toArray();
        double[] L=bars.stream().mapToDouble(Bar::l).toArray();
        for(int i=WARMUP;i<n;i++){
            Bar bar=bars.get(i); String date=bar.date();
            Map<String,Double> px=Map.of(sym,bar.c());
            pm.checkExits(sym,bar,date,px);
            if(pm.halt) break;
            double[] c=Arrays.copyOfRange(C,0,i+1);
            double[] h=Arrays.copyOfRange(H,0,i+1);
            double[] lo=Arrays.copyOfRange(L,0,i+1);
            double atrV=atr(h,lo,c,14);
            // tech signals
            double[] prev=Arrays.copyOf(c,c.length-1);
            double e9=ema(c,9),e21=ema(c,21),pe9=ema(prev,9),pe21=ema(prev,21);
            int sE=(pe9<=pe21&&e9>e21)?1:(pe9>=pe21&&e9<e21)?-1:0;
            double rv=rsi(c,14);
            int sR=rv<=30?1:rv>=70?-1:0;
            double[] b=bb(c,20,2.0);
            int sB=bar.c()>b[1]?1:bar.c()<b[2]?-1:0;
            double[] mc=macd(c,12,26,9); int sM=mc[2]>0?1:-1;
            int sROC=roc(c,10)>1.5?1:roc(c,10)<-1.5?-1:0;
            int tv=sE+sR+sB+sM+sROC;
            // ML
            double[] feat=features(C,H,L,i);
            double[] ml2=ml.query(feat);
            int mlSig=(int)ml2[0]; double bayes=ml2[1],conf=ml2[2];
            double ev=bayes*(2.8*atrV/bar.c())-(1-bayes)*(1.8*atrV/bar.c());
            // decide
            int dir=0; String strat="";
            boolean alreadyOpen=pm.open.stream().anyMatch(p->p.sym.equals(sym));
            if(alreadyOpen) continue;
            if(tv>=3&&mlSig==1&&bayes>MIN_BAYES_PROB&&ev>0){dir=1;strat="ML+TECH↑";}
            else if(tv<=-3&&mlSig==-1&&(1-bayes)>MIN_BAYES_PROB&&ev>0){dir=-1;strat="ML+TECH↓";}
            else if(mlSig!=0&&conf>0.35&&ev>0){dir=mlSig;strat=mlSig==1?"ML-LONG":"ML-SHORT";}
            else if(Math.abs(tv)>=4){dir=tv>0?1:-1;strat="TECH-STRONG";}
            if(dir==0) continue;
            double eq=pm.equity(px);
            if(pm.enter(sym,strat,dir,bar.c(),atrV,eq,date,conf)){
                System.out.printf("  [%s] %-10s %-12s %s @ %9.4f  P(↑)=%4.1f%%  EV=%+.3f%%%n",
                    date,sym,strat,dir==1?"BUY ↑":"SELL↓",bar.c(),
                    dir==1?bayes*100:(1-bayes)*100,ev*100);
            }
        }
        // close remaining
        if(!bars.isEmpty()){
            Bar last=bars.get(n-1);
            pm.open.removeIf(p->{
                if(!p.sym.equals(sym)) return false;
                double pnl=(last.c()-p.ep)*p.dir*p.qty;
                pm.cash+=last.c()*p.qty;
                pm.closed.add(new Trade(p.sym,p.strat,p.date,last.date(),p.dir,
                    p.ep,last.c(),p.qty,pnl,"END",p.conf));
                if(pnl>0){pm.wins++;pm.tw+=pnl;}else{pm.losses++;pm.tl-=pnl;}
                return true;
            });
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // REPORT
    // ════════════════════════════════════════════════════════════════════════
    static void report(Portfolio pm){
        List<Trade> tr=pm.closed;
        double[] eq2=pm.eq.stream().mapToDouble(Double::doubleValue).toArray();
        // add final equity
        double finalEq=pm.cash;
        for(Pos p:pm.open) finalEq+=p.qty*p.ep;
        eq2=Arrays.copyOf(eq2,eq2.length+1); eq2[eq2.length-1]=finalEq;
        double[] rets=new double[eq2.length-1];
        for(int i=1;i<eq2.length;i++) rets[i-1]=(eq2[i]-eq2[i-1])/eq2[i-1];
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║        CRYPTO QUANT TRADER — FULL STATISTICAL REPORT             ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
        System.out.printf("  %-30s $%.2f%n","Initial Capital:", INITIAL_CAPITAL);
        System.out.printf("  %-30s $%.2f%n","Final Equity:", finalEq);
        System.out.printf("  %-30s %.2f%%%n","Total Return:", (finalEq-INITIAL_CAPITAL)/INITIAL_CAPITAL*100);
        System.out.println("  ─────────────────────────────────────────────────────────────────");
        System.out.printf("  %-30s %.3f%n","Sharpe Ratio (ann.):", rets.length>1?sharpe(rets):0);
        System.out.printf("  %-30s %.2f%%%n","Max Drawdown:", maxDD(eq2)*100);
        System.out.printf("  %-30s %.3f%n","Skewness:", rets.length>3?skewness(rets):0);
        System.out.printf("  %-30s %.3f%n","Excess Kurtosis:", rets.length>3?kurtosis(rets):0);
        if(rets.length>10){
            System.out.printf("  %-30s %.3f%%%n","VaR 95%%:", var95(rets)*100);
            System.out.printf("  %-30s %.3f%%%n","CVaR 95%%:", cvar95(rets)*100);
        }
        double h=hurstExponent(eq2);
        System.out.printf("  %-30s %.3f  (%s)%n","Hurst Exponent:",h,h>0.55?"Trending":"Mean-Reverting");
        if(rets.length>5)
            System.out.printf("  %-30s %.3f%%%n","GARCH(1,1) Hourly Vol:", garchVol(rets)*100);
        System.out.println("  ─────────────────────────────────────────────────────────────────");
        if(!tr.isEmpty()){
            long wins=tr.stream().filter(t->t.pnl()>0).count();
            double avgW=tr.stream().filter(t->t.pnl()>0).mapToDouble(Trade::pnl).average().orElse(0);
            double avgL=tr.stream().filter(t->t.pnl()<=0).mapToDouble(t->-t.pnl()).average().orElse(0);
            long los=tr.size()-wins;
            double pf=avgL>0&&los>0?avgW*wins/(avgL*los):999;
            System.out.printf("  %-30s %d%n","Total Trades:", tr.size());
            System.out.printf("  %-30s %d  (%.1f%%)%n","Win Rate:", wins, 100.0*wins/tr.size());
            System.out.printf("  %-30s $%.4f%n","Avg Win:", avgW);
            System.out.printf("  %-30s $%.4f%n","Avg Loss:", avgL);
            System.out.printf("  %-30s %.2f%n","Profit Factor:", pf);
            double avgConf=tr.stream().mapToDouble(Trade::conf).average().orElse(0);
            System.out.printf("  %-30s %.1f%%%n","Avg ML Confidence:", avgConf*100);
            System.out.println("  ─────────────────────────────────────────────────────────────────");
            System.out.println("  STRATEGY BREAKDOWN:");
            tr.stream().collect(Collectors.groupingBy(Trade::strat)).forEach((s2,ts)->{
                long w=ts.stream().filter(t->t.pnl()>0).count();
                double pnl=ts.stream().mapToDouble(Trade::pnl).sum();
                System.out.printf("    %-14s  trades=%3d  wins=%2d  P&L=$%.4f%n",s2,ts.size(),w,pnl);
            });
            System.out.println("  ─────────────────────────────────────────────────────────────────");
            System.out.println("  LAST 8 CLOSED TRADES:");
            System.out.printf("  %-14s %-10s %-12s %-6s %-10s %-10s %-10s %-5s%n",
                "Exit","Symbol","Strategy","Dir","Entry","Exit","P&L","ML%");
            System.out.println("  " + "─".repeat(78));
            int s=Math.max(0,tr.size()-8);
            for(int i=s;i<tr.size();i++){
                Trade t=tr.get(i);
                System.out.printf("  %-14s %-10s %-12s %-6s %-10.4f %-10.4f %-10.4f %-5.0f%%%n",
                    t.e1(),t.sym(),t.strat(),t.dir()==1?"LONG":"SHORT",
                    t.ep(),t.xp(),t.pnl(),t.conf()*100);
            }
        } else {
            System.out.println("  No trades closed — thresholds not met in this period.");
        }
        // equity curve
        System.out.println();
        System.out.println("  ┌─ EQUITY CURVE ────────────────────────────────────────────────┐");
        int W=64,H=11;
        double mn=Arrays.stream(eq2).min().orElse(0),mx=Arrays.stream(eq2).max().orElse(1);
        double[] sam=new double[W];
        for(int i=0;i<W;i++){int idx=(int)((double)i/(W-1)*(eq2.length-1));sam[i]=eq2[idx];}
        for(int row=H;row>=0;row--){
            System.out.print("  │");
            double thr=mn+(mx-mn)*row/H;
            for(double v:sam) System.out.print(v>=thr?"█":" ");
            if(row==H) System.out.printf(" $%.2f%n",mx);
            else if(row==0) System.out.printf(" $%.2f%n",mn);
            else System.out.println();
        }
        System.out.println("  └" + "─".repeat(W+1) + "┘");
        System.out.println();
        System.out.println("  ML  : Logistic Reg · MLP Neural Net (12→10→5→1) · KNN (cosine)");
        System.out.println("  Prob: Bayesian Ensemble · Kelly Criterion · Expected Value gate");
        System.out.println("  Stats: GARCH · Hurst · Skew · Kurt · VaR · CVaR · Monte Carlo");
        System.out.println("═".repeat(70));
    }

    // ════════════════════════════════════════════════════════════════════════
    // MAIN
    // ════════════════════════════════════════════════════════════════════════
    public static void main(String[] args) throws Exception {
        System.out.println("═".repeat(70));
        System.out.println("  CRYPTO QUANTITATIVE TRADING ENGINE");
        System.out.println("  Live Binance Data · ML Pattern Detection · Statistical Prob");
        System.out.println("═".repeat(70));
        System.out.printf("  Symbols:  %s%n", String.join(" · ", SYMBOLS));
        System.out.printf("  Interval: %s    Bars: %d    Capital: $%.0f USDT%n%n",
            INTERVAL, BARS_FETCH, INITIAL_CAPITAL);

        Portfolio pm = new Portfolio(INITIAL_CAPITAL);

        for(String sym : SYMBOLS){
            System.out.printf("  ► %-10s  Fetching %s bars from Binance…%n", sym, INTERVAL);
            List<Bar> bars = fetchBars(sym);
            System.out.printf("    %d bars  [%s → %s]%n",
                bars.size(), bars.get(0).date(), bars.get(bars.size()-1).date());

            // per-symbol stats
            double[] C=bars.stream().mapToDouble(Bar::c).toArray();
            double[] H=bars.stream().mapToDouble(Bar::h).toArray();
            double[] L=bars.stream().mapToDouble(Bar::l).toArray();
            double[] re=new double[C.length-1];
            for(int i=1;i<C.length;i++) re[i-1]=(C[i]-C[i-1])/C[i-1];

            double h=hurstExponent(C);
            double gv=garchVol(re)*100;
            double sk=skewness(re), kt=kurtosis(re);
            double rv=rsi(C,14);
            double[] mc=monteCarlo(C[C.length-1],mean(re),std(re),500);

            System.out.printf("    Price=$%.4f  RSI=%.1f  Hurst=%.3f  GARCH-Vol=%.2f%%%n",
                C[C.length-1],rv,h,gv);
            System.out.printf("    Skew=%.3f  Kurt=%.3f  VaR95=%.2f%%  MC P(↑24h)=%.1f%%%n",
                sk,kt,var95(re)*100,mc[2]*100);

            System.out.println("    Training ML (LogReg · MLP · KNN)…");
            MLEngine ml = new MLEngine();
            ml.train(bars);

            System.out.println("    Back-testing…");
            backtest(sym, bars, pm, ml);
            System.out.println();
        }

        report(pm);
    }
}
