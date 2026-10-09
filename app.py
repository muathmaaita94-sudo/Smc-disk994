"""SMC DESK - Python port of SMC_ICT_MultiStrategy_EA (EMA80 v2) as a 6-agent swarm."""
import argparse, asyncio, json, math, threading, time, os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd

@dataclass
class Cfg:  
    symbol: str = "BTCUSDm"; live: bool = False; magic: int = 20260719; sim_speed: int = 5
    risk_pct: float = 0.5; max_total_risk: float = 1.5; max_trades: int = 2
    min_dist_pts: float = 800; max_spread_pts: float = 300
    atr_n: int = 14; atr_sl: float = 2.5; rr: float = 3.5
    p1: tuple = (50, 30); p2: tuple = (75, 30)          
    trail_start: float = 50; trail_atr: float = 2.0; be_trigger: float = 30; be_off_pts: float = 20
    min_score: int = 70; full_score: int = 85; light_score: int = 55; light_lot: float = 0.5; light_rr: float = 0.6
    sr_lookback: int = 100; sr_zone_pts: float = 50
    e80_n: int = 80; e80_slope_bars: int = 5; e80_min_slope: float = 0.08; e80_min_dist: float = 0.10
    e80_align: int = 15; e80_counter: int = 18; e80_allow_counter: bool = False; e80_counter_min: int = 80
    daily_limit: float = 3.0; max_dd: float = 10.0; emergency_dd: float = 15.0; dd_cooldown_h: int = 24
    recovery_losses: int = 3; recovery_mult: float = 0.5; hours: tuple = (0, 24)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); u = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    v = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean(); return 100 - 100/(1 + u/v.replace(0, np.nan))
def atr(df, n=14):
    pc = df.close.shift(); tr = pd.concat([df.high-df.low, (df.high-pc).abs(), (df.low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

class SimFeed:
    point, contract, vmin, vmax, vstep = 0.01, 1.0, 0.01, 10.0, 0.01
    def __init__(s, n=90000):
        s.rng = np.random.default_rng()
        r = s.rng.normal(0, 3.5e-4, n) + np.sin(np.arange(n)/900)*5e-5
        c = 80000*np.exp(np.cumsum(r)); o = np.r_[c[0], c[:-1]]; w = np.abs(s.rng.normal(0, 2e-4, (2, n)))*c
        s.df = pd.DataFrame({"open": o, "high": np.maximum(o, c)+w[0], "low": np.minimum(o, c)-w[1], "close": c},
                            index=pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="min"))
    def step(s):
        l = s.df.iloc[-1]; c = l.close*math.exp(s.rng.normal(0, 3.5e-4) + math.sin(len(s.df)/900)*5e-5)
        w = abs(s.rng.normal(0, 2e-4, 2))*c
        s.df.loc[s.df.index[-1] + pd.Timedelta(minutes=1)] = [l.close, max(l.close, c)+w[0], min(l.close, c)-w[1], c]
    def rates(s, tf, n):
        if tf == "M1": return s.df.tail(n)
        return s.df.resample({"H1": "1h", "H4": "4h", "D1": "1D"}[tf]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().tail(n)
    def tick(s): b = float(s.df.close.iloc[-1]); return b, b + 30*s.point
    def spread(s): return 30
    def now(s): return s.df.index[-1]

class MT5Feed:
    def __init__(s, sym):
        import MetaTrader5 as m
        s.m, s.sym = m, sym
        if not m.initialize(): raise RuntimeError(f"MT5 init failed: {m.last_error()}")
        m.symbol_select(sym, True); i = m.symbol_info(sym)
        s.point, s.vmin, s.vmax, s.vstep = i.point, i.volume_min, i.volume_max, i.volume_step
        s.contract = i.trade_tick_value/i.trade_tick_size       
    def step(s): pass
    def rates(s, tf, n):
        m = s.m; c = {"M1": m.TIMEFRAME_M1, "H1": m.TIMEFRAME_H1, "H4": m.TIMEFRAME_H4, "D1": m.TIMEFRAME_D1}[tf]
        d = pd.DataFrame(m.copy_rates_from_pos(s.sym, c, 0, n)); d.index = pd.to_datetime(d.time, unit="s"); return d
    def tick(s): t = s.m.symbol_info_tick(s.sym); return t.bid, t.ask
    def spread(s): return s.m.symbol_info(s.sym).spread
    def now(s): return pd.Timestamp(s.m.symbol_info_tick(s.sym).time, unit="s")

class Paper:
    def __init__(s, feed, bal=10000.0): s.f, s.bal, s.pos, s.n = feed, bal, {}, 0
    def balance(s): return s.bal
    def _pnl(s, p): b, a = s.f.tick(); return ((b if p["dir"] == 1 else a)-p["entry"])*p["dir"]*p["vol"]*s.f.contract
    def equity(s): return s.bal + sum(s._pnl(p) for p in s.pos.values())
    def positions(s): return [dict(p, pnl=round(s._pnl(p), 2)) for p in s.pos.values()]
    def open(s, d, vol, sl, tp, tag):
        b, a = s.f.tick(); s.n += 1
        s.pos[s.n] = dict(id=s.n, dir=d, vol=vol, entry=a if d == 1 else b, sl=sl, tp=tp, tag=tag, t=str(s.f.now())); return s.n
    def modify(s, id, sl): s.pos[id]["sl"] = sl
    def close(s, id, vol=None, px=None):
        p = s.pos[id]; vol = min(vol or p["vol"], p["vol"]); b, a = s.f.tick(); px = px or (b if p["dir"] == 1 else a)
        pr = round((px-p["entry"])*p["dir"]*vol*s.f.contract, 2); s.bal += pr; p["vol"] = round(p["vol"]-vol, 8)
        if p["vol"] <= 1e-9: del s.pos[id]
        return dict(id=id, pnl=pr)
    def check_stops(s):
        out = []; b, a = s.f.tick()
        for p in list(s.pos.values()):
            px = b if p["dir"] == 1 else a
            if p["sl"] and (px-p["sl"])*p["dir"] <= 0: out.append(s.close(p["id"], px=p["sl"]))
            elif p["tp"] and (px-p["tp"])*p["dir"] >= 0: out.append(s.close(p["id"], px=p["tp"]))
        return out

class Live:
    def __init__(s, feed, magic):
        s.f, s.m, s.magic, s.seen = feed, feed.m, magic, set()
        s.t0 = pd.Timestamp.now() - pd.Timedelta(days=1); s.check_stops()
    def balance(s): return s.m.account_info().balance
    def equity(s): return s.m.account_info().equity
    def positions(s):
        return [dict(id=p.ticket, dir=1 if p.type == 0 else -1, vol=p.volume, entry=p.price_open, sl=p.sl, tp=p.tp,
                     tag=p.comment, t=str(pd.Timestamp(p.time, unit="s")), pnl=p.profit)
                for p in (s.m.positions_get(symbol=s.f.sym) or []) if p.magic == s.magic]
    def open(s, d, vol, sl, tp, tag):
        m = s.m; b, a = s.f.tick()
        r = m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=s.f.sym, volume=vol, deviation=30, magic=s.magic,
                              type=m.ORDER_TYPE_BUY if d == 1 else m.ORDER_TYPE_SELL, price=a if d == 1 else b,
                              sl=sl, tp=tp, comment=tag[:30], type_filling=m.ORDER_FILLING_IOC))
        return r.order if r.retcode == m.TRADE_RETCODE_DONE else None
    def modify(s, id, sl):
        p = next(p for p in s.positions() if p["id"] == id)
        s.m.order_send(dict(action=s.m.TRADE_ACTION_SLTP, position=id, symbol=s.f.sym, sl=sl, tp=p["tp"]))
    def close(s, id, vol=None, px=None):
        m = s.m; p = next(p for p in s.positions() if p["id"] == id); b, a = s.f.tick()
        m.order_send(dict(action=s.m.TRADE_ACTION_DEAL, symbol=s.f.sym, position=id, volume=vol or p["vol"], deviation=30,
                          magic=s.magic, type=m.ORDER_TYPE_SELL if p["dir"] == 1 else m.ORDER_TYPE_BUY,
                          price=b if p["dir"] == 1 else a, type_filling=m.ORDER_FILLING_IOC))
    def check_stops(s):       
        m, out = s.m, []
        for d in m.history_deals_get(s.t0.to_pydatetime(), (pd.Timestamp.now()+pd.Timedelta(days=1)).to_pydatetime()) or []:
            if d.magic == s.magic and d.entry == m.DEAL_ENTRY_OUT and d.ticket not in s.seen:
                s.seen.add(d.ticket); out.append(dict(id=d.position_id, pnl=round(d.profit+d.swap+d.commission, 2)))
        return out

AGENTS = [("spotter", "SPOTTER", "TAPE · D1 / H4 / H1", "#2ee6a6"), ("prior", "PRIOR", "STRUCTURE · S/R · OB · FVG", "#ff9f43"),
          ("edge", "EDGE", "EMA80 BIAS · SCORE", "#ff3ea5"), ("kelly", "KELLY", "SIZING · SL/TP · RISK", "#a45cff"),
          ("taker", "TAKER", "FILTERS · EXECUTION", "#3b6bff"), ("closer", "CLOSER", "MANAGE · LIMITS", "#ff3b3b")]
SG = {1: "▲", -1: "▼", 0: "•"}; DN = {1: "BUY", -1: "SELL", 0: "NONE"}

class Agent:
    def __init__(a, key, name, role, color): a.key, a.name, a.role, a.color = key, name, role, color; a.msg, a.last, a.runs, a.pulse = "booting", "", 0, 0.0

class Engine:
    def __init__(s, cfg):
        s.c = cfg; s.feed = MT5Feed(cfg.symbol) if cfg.live else SimFeed()
        s.br = Live(s.feed, cfg.magic) if cfg.live else Paper(s.feed)
        s.ag = {k: Agent(k, n, r, col) for k, n, r, col in AGENTS}
        s.logs, s.tickets, s.events, s.tk, s.flags, s.eq = deque(maxlen=80), deque(maxlen=40), deque(maxlen=30), {}, {}, deque(maxlen=300)
        s.enabled, s.last_bar, s.ctx, s.halt, s.cool, s.day, s.atr, s.eid, s.losses, s.dd = True, None, {}, "", None, None, 0.0, 0, 0, 0.0
        s.peak, s.day_bal, s.stats = s.br.equity(), s.br.balance(), dict(n=0, w=0, gp=0.0, gl=0.0, maxdd=0.0)
    def say(s, a, msg, log=None, kind="info"):
        a.msg = msg; a.runs += 1
        if log and log != a.last:
            a.last = log; a.pulse = time.time(); s.logs.appendleft(dict(t=str(s.feed.now())[11:19], a=a.key, m=log, k=kind))
    def event(s, typ, txt, **kw): s.eid += 1; s.events.append(dict(id=s.eid, type=typ, txt=txt, **kw))
    def close(s, id, vol=None, why=""):
        rec = s.br.close(id, vol)
        if rec: s.on_close(rec, why)
    def on_close(s, rec, why=""):
        st, pnl = s.stats, rec["pnl"]; st["n"] += 1
        if pnl >= 0: st["w"] += 1; st["gp"] += pnl; s.losses = 0
        else: st["gl"] += pnl; s.losses += 1
        t = s.tk.get(rec["id"])
        if t:
            t["pnl"] = round(t["pnl"]+pnl, 2)
            if rec["id"] not in {p["id"] for p in s.br.positions()}: t["status"] = "WIN" if t["pnl"] >= 0 else "LOSS"
        s.event("close", f"{'WIN' if pnl >= 0 else 'LOSS'} {pnl:+,.0f}$", pnl=pnl)
        s.say(s.ag["closer"], f"exit #{rec['id']} {pnl:+.2f}$ {why}", f"exit #{rec['id']} {pnl:+,.2f}$ {why}", "win" if pnl >= 0 else "loss")
    def tick(s):
        for _ in range(1 if s.c.live else s.c.sim_speed): s.feed.step()
        s.ctx = dict(b=0, s=0, facts={})
        for r in s.br.check_stops(): s.on_close(r, "SL/TP")
        s.closer_risk(); s.closer_manage()
        s.spotter(); s.prior(); s.edge(); s.kelly(); s.taker()
        s.eq.append(round(s.br.equity(), 2))

    def spotter(s):
        x, f = s.ctx, s.feed; d1 = f.rates("D1", 80).close; h4 = f.rates("H4", 120).close; h1 = f.rates("H1", 200)
        sma = d1.rolling(50).mean().iloc[-2]
        tr = 0 if np.isnan(sma) else int(np.sign(d1.iloc[-2]-sma))
        ma = int(np.sign(ema(h4, 9).iloc[-2]-ema(h4, 21).iloc[-2])); r = float(rsi(h4).iloc[-2])
        cd = int(np.sign(h1.close.iloc[-2]-h1.open.iloc[-2]))
        x["b"] += 25*(tr == 1)+20*(ma == 1)+15*(30 <= r <= 60)+15*(cd == 1)
        x["s"] += 25*(tr == -1)+20*(ma == -1)+15*(40 <= r <= 70)+15*(cd == -1)
        x["h1"] = h1; x["facts"].update(trend=tr, ma=ma, rsi=round(r, 1), candle=cd)
        s.say(s.ag["spotter"], f"D1 {SG[tr]} · H4 MA {SG[ma]} · RSI {r:.0f} · H1 {SG[cd]}", f"tape D1 {SG[tr]} / H4 {SG[ma]} / candle {SG[cd]}")

    def prior(s):
        c, x = s.c, s.ctx; h1 = x["h1"]; a = float(atr(h1, c.atr_n).iloc[-2]); s.atr = x["atr"] = a
        o, h, l, cl = (h1[k].values[::-1] for k in ("open", "high", "low", "close"))   
        px, z = cl[1], c.sr_zone_pts*s.feed.point
        res, sup = h[1:c.sr_lookback+1].max(), l[1:c.sr_lookback+1].min(); ob = fvg = 0
        for i in range(2, 20):
            lo, hi = min(o[i+1], cl[i+1]), max(o[i+1], cl[i+1])
            if cl[i] > o[i] and cl[i] > h[i+1]+3*a and cl[i+1] < o[i+1] and lo <= px <= hi: ob = 1; break
            if cl[i] < o[i] and cl[i] < l[i+1]-3*a and cl[i+1] > o[i+1] and lo <= px <= hi: ob = -1; break
        for i in range(1, 15):
            if h[i+2] < l[i] and h[i+2] <= px <= l[i]: fvg = 1; break
            if l[i+2] > h[i] and h[i] <= px <= l[i+2]: fvg = -1; break
        x["b"] += 10*(abs(px-sup) <= z)+10*(ob == 1)+5*(fvg == 1)
        x["s"] += 10*(abs(px-res) <= z)+10*(ob == -1)+5*(fvg == -1)
        x["facts"].update(ob=ob, fvg=fvg, near_sup=bool(abs(px-sup) <= z), near_res=bool(abs(px-res) <= z))
        s.say(s.ag["prior"], f"OB {SG[ob]} · FVG {SG[fvg]} · S/R {'sup' if abs(px-sup) <= z else 'res' if abs(px-res) <= z else 'mid'}",
              f"structure OB {SG[ob]} / FVG {SG[fvg]}")

    def edge(s):
        c, x = s.c, s.ctx; m1 = s.feed.rates("M1", 400); e = ema(m1.close, c.e80_n); a = float(atr(m1, 14).iloc[-2])
        px, em = float(m1.close.iloc[-2]), float(e.iloc[-2]); sl = em-float(e.iloc[-2-c.e80_slope_bars])
        sa, da = abs(sl)/a, abs(px-em)/a
        st = min(100., (min(1, sa/c.e80_min_slope)*.6 + min(1, da/c.e80_min_dist)*.4)*100)
        ok = sa >= c.e80_min_slope and da >= c.e80_min_dist
        bias = 1 if ok and px > em and sl > 0 else -1 if ok and px < em and sl < 0 else 0
        b, sc = x["b"], x["s"]
        if bias:
            al, ct = round(c.e80_align*st/100), round(c.e80_counter*st/100)
            if bias == 1: b += al; sc = max(0, sc-ct)
            else: sc += al; b = max(0, b-ct)
        d = 0 if b == sc else (1 if b > sc else -1); score = max(b, sc); note = ""
        if d and bias and d != bias:
            if not c.e80_allow_counter: d, note = 0, "EMA80 blocked counter-trend"
            elif score < c.e80_counter_min: d, note = 0, "EMA80 counter score too low"
        tier = "FULL" if score >= c.full_score else "STANDARD" if score >= c.min_score else "LIGHT" if score >= c.light_score else ""
        x.update(dir=d, score=score, tier=tier, bias=bias, strength=round(st), note=note, buy=b, sell=sc, slope_atr=round(sa, 2), dist_atr=round(da, 2),
                 chart=dict(p=m1.close.tail(120).round(2).tolist(), e=e.tail(120).round(2).tolist()))
        s.say(s.ag["edge"], f"EMA80 {DN[bias] if bias else 'NEUTRAL'} {st:.0f}/100 → {DN[d]} {score} {tier}",
              f"EMA80 {DN[bias] if bias else 'NEUTRAL'} · signal {DN[d]} {tier or '-'}" + (f" · {note}" if note else ""),
              "signal" if d and tier else "info")

    def kelly(s):
        c, x, f = s.c, s.ctx, s.feed; x["plan"] = None; A = s.ag["kelly"]
        if not x["dir"] or not x["tier"]: return s.say(A, "no setup to size")
        b, a = f.tick(); d = x["dir"]; px = a if d == 1 else b; light = x["tier"] == "LIGHT"
        slp = x["atr"]*c.atr_sl; tpd = slp*c.rr*(c.light_rr if light else 1)
        bal = s.br.balance(); risk = c.risk_pct*(c.light_lot if light else 1)*(c.recovery_mult if s.losses >= c.recovery_losses else 1)
        used = sum(abs(p["entry"]-p["sl"])*p["vol"]*f.contract for p in s.br.positions() if p["sl"])/bal*100
        risk = min(risk, c.max_total_risk-used)
        if risk <= 0: x["block"] = "total risk budget used"; return s.say(A, x["block"], x["block"], "warn")
        per_lot = slp*f.contract; lot = round(math.floor(bal*risk/100/per_lot/f.vstep+1e-9)*f.vstep, 8)
        if lot < f.vmin:
            if f.vmin*per_lot/bal*100 > risk: x["block"] = "min lot exceeds risk budget"; return s.say(A, x["block"], x["block"], "warn")
            lot = f.vmin
        lot = min(lot, f.vmax)
        x["plan"] = dict(dir=d, lot=lot, entry=px, sl=round(px-d*slp, 2), tp=round(px+d*tpd, 2), rr=round(tpd/slp, 2), risk=round(risk, 2))
        s.say(A, f"{DN[d]} {lot} lot · risk {risk:.2f}% · RR {tpd/slp:.1f}" + (" · recovery" if s.losses >= c.recovery_losses else ""))

    def taker(s):
        c, x, f = s.c, s.ctx, s.feed; A = s.ag["taker"]; bar = x["h1"].index[-1]
        if bar == s.last_bar: return s.say(A, "waiting for new H1 bar")
        s.last_bar = bar; pos = s.br.positions(); plan = x["plan"]; d = x["dir"]
        if not d or not x["tier"]: return s.say(A, f"no entry · score {x['score']} · {x['note'] or 'below light threshold'}")
        why = (s.halt or ("trading paused" if not s.enabled else "") or
               ("outside trading hours" if not c.hours[0] <= f.now().hour < c.hours[1] else "") or
               ("spread too high" if f.spread() > c.max_spread_pts else "") or
               ("max open trades" if len(pos) >= c.max_trades else "") or (x.get("block", "") if not plan else "") or
               ("too close to open trade" if plan and any(abs(p["entry"]-plan["entry"]) < c.min_dist_pts*f.point for p in pos) else ""))
        t = dict(dir=d, tier=x["tier"], score=x["score"], bias=x["bias"], t=str(f.now()), pnl=0.0, sl=0, tp=0, lot=0, entry=0,
                 trail={k: a.msg for k, a in s.ag.items() if k != "closer"})
        if not why:
            id = s.br.open(d, plan["lot"], plan["sl"], plan["tp"], f"SMC|{x['tier']}")
            if id is None: why = "order rejected by broker"
        if why:
            t.update(id=f"x{s.eid+1}", status="BLOCKED", why=why); s.tickets.appendleft(t); s.event("block", why)
            return s.say(A, f"blocked: {why}", f"{DN[d]} blocked: {why}", "warn")
        t.update(id=id, status="OPEN", lot=plan["lot"], entry=plan["entry"], sl=plan["sl"], tp=plan["tp"], rr=plan["rr"])
        s.tickets.appendleft(t); s.tk[id] = t; s.event("open", f"{DN[d]} {plan['lot']}", dir=d)
        s.say(A, f"{DN[d]} #{id} filled @ {plan['entry']:,.2f}", f"{DN[d]} #{id} {x['tier']} filled @ {plan['entry']:,.2f}", "trade")

    def closer_risk(s):
        c, br = s.c, s.br; eq = br.equity(); now = s.feed.now(); s.peak = max(s.peak, eq)
        s.dd = dd = (s.peak-eq)/s.peak*100; s.stats["maxdd"] = max(s.stats["maxdd"], dd)
        if now.date() != s.day: s.day, s.day_bal = now.date(), br.balance()
        if dd >= c.emergency_dd:
            for p in br.positions(): s.close(p["id"], None, "EMERGENCY")
        s.halt = ""
        if dd >= c.max_dd:
            s.cool = s.cool or now+pd.Timedelta(hours=c.dd_cooldown_h)
            if now >= s.cool: s.peak, s.cool = eq, None
            else: s.halt = f"drawdown {dd:.1f}% cooldown"
        else: s.cool = None
        if not s.halt and (s.day_bal-br.balance())/s.day_bal*100 >= c.daily_limit: s.halt = "daily loss limit"

    def closer_manage(s):
        c, f = s.c, s.feed; b, a = f.tick(); n = 0
        for p in s.br.positions():
            fl = s.flags.setdefault(p["id"], dict(v0=p["vol"], e=p["entry"], tp=p["tp"], p1=0, p2=0, be=0)); d = p["dir"]; sl = p["sl"]
            px = b if d == 1 else a
            if not fl["tp"] or (px-fl["e"])*d <= 0: continue
            prog = abs(px-fl["e"])/abs(fl["tp"]-fl["e"])*100; n += 1
            for k, (trg, pc) in (("p1", c.p1), ("p2", c.p2)):
                v = round(math.floor(fl["v0"]*pc/100/f.vstep+1e-9)*f.vstep, 8)
                if not fl[k] and prog >= trg and f.vmin <= v < p["vol"]: s.close(p["id"], v, f"partial {trg}%"); fl[k] = 1
            if not fl["be"] and prog >= c.be_trigger:
                nsl = fl["e"]+d*c.be_off_pts*f.point
                if not sl or (nsl-sl)*d > 0: s.br.modify(p["id"], round(nsl, 2)); sl = nsl; fl["be"] = 1
                s.say(s.ag["closer"], f"#{p['id']} breakeven", f"#{p['id']} moved to breakeven", "info")
            if s.atr and prog >= c.trail_start:
                nsl = round(px-d*s.atr*c.trail_atr, 2)
                if not sl or (nsl-sl)*d > 0: s.br.modify(p["id"], nsl)
        s.say(s.ag["closer"], f"{len(s.br.positions())} open · DD {s.dd:.1f}% · {s.halt or 'limits ok'}")

    def snapshot(s):
        x, pos = s.ctx, s.br.positions(); pnl = {p["id"]: p["pnl"] for p in pos}; st = s.stats; now = time.time()
        for t in s.tickets:
            if t["status"] == "OPEN" and t["id"] in pnl: t["live"] = round(pnl[t["id"]], 2)
        b, a = s.feed.tick(); eq = s.br.equity()
        return dict(mode="LIVE" if s.c.live else "PAPER", symbol=s.c.symbol if s.c.live else "BTC/USD·SIM", time=str(s.feed.now()), price=b,
                    enabled=s.enabled, halt=s.halt, equity=round(eq, 2), balance=round(s.br.balance(), 2), dd=round(s.dd, 2),
                    agents=[dict(key=a.key, name=a.name, role=a.role, color=a.color, msg=a.msg, age=round(now-a.pulse, 1)) for a in s.ag.values()],
                    sig={k: x.get(k) for k in ("dir", "score", "tier", "bias", "strength", "note", "buy", "sell", "slope_atr", "dist_atr", "facts", "plan")},
                    positions=pos, tickets=list(s.tickets), logs=list(s.logs)[:40], events=list(s.events), eq=list(s.eq), chart=x.get("chart"),
                    stats=dict(n=st["n"], wr=round(st["w"]/st["n"]*100, 1) if st["n"] else 0, pf=round(abs(st["gp"]/st["gl"]), 2) if st["gl"] else 0,
                               maxdd=round(st["maxdd"], 2), pnl=round(st["gp"]+st["gl"], 2)))

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMC DESK</title>
<style>
:root{--bg:#020306;--card:#07090e;--line:#141a26;--tx:#e8edf5;--mut:#5d6a80;--g:#2ee6a6;--r:#ff3b4d;--o:#ff9f43}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--tx);font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;padding:8px;max-width:1500px;margin:auto}
.k{color:var(--mut);font-size:8px;letter-spacing:1.4px;text-transform:uppercase}.v{font:700 15px system-ui}.up{color:var(--g)}.dn{color:var(--r)}
header{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.logo{display:flex;align-items:center;gap:8px;font:700 17px system-ui}.logo i{width:22px;height:22px;border-radius:50%;background:#fff;position:relative}
.logo i:after{content:'';position:absolute;left:7px;top:6px;width:3px;height:9px;background:#000;border-radius:2px;box-shadow:6px 0 #000}.logo b{color:var(--mut);font-weight:400}
.sp{flex:1}.clk{font:700 24px system-ui;letter-spacing:1px}
button{cursor:pointer;background:none;font:inherit;color:var(--g);padding:5px 10px;border-radius:6px;border:1px solid var(--g);font-size:9px;letter-spacing:1px}button.off{color:var(--r);border-color:var(--r)}
#strip{display:flex;flex-wrap:wrap;gap:4px 18px;padding:6px 4px;font-size:9px;color:var(--mut);letter-spacing:.8px}#strip b{color:var(--tx)}#strip .live{color:var(--g);border:1px solid #14533f;padding:0 6px;border-radius:3px}
#agents{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin:4px 0 8px}
.ag{border:1px solid var(--line);background:var(--card);border-radius:6px;padding:8px;min-width:0;transition:.3s}.ag.now{border-color:var(--c);box-shadow:0 0 14px -6px var(--c)}
.ag .t{display:flex;justify-content:space-between;font-size:7px;color:var(--mut);letter-spacing:1px}.ag .n{display:flex;align-items:center;gap:8px;margin:6px 0;font:700 13px system-ui}
.ag .bar{height:2px;background:#10151f}.ag .bar i{display:block;height:100%;background:var(--c);width:0;transition:width 1s linear}.ag p{color:#9aa7bd;font-size:9px;margin-top:5px;min-height:24px;word-break:break-word}
.card{border:1px solid var(--line);background:var(--card);border-radius:8px;padding:10px;min-width:0}.card h4{color:var(--mut);font-size:9px;letter-spacing:1.4px;font-weight:400;margin-bottom:6px;display:flex;justify-content:space-between}
canvas{width:100%;display:block}#hub{border-radius:6px}
.cols3,.cols2{display:grid;gap:8px;margin-top:8px}.cols3{grid-template-columns:repeat(3,1fr)}.cols2{grid-template-columns:1fr 1.3fr}
.big{font:800 34px system-ui;letter-spacing:-1px}.sig{font:800 30px system-ui}.bar2{height:5px;background:#10151f;border-radius:3px;margin:5px 0;overflow:hidden}.bar2>div{height:100%;transition:.4s}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}.chips span{border:1px solid var(--line);border-radius:4px;padding:1px 6px;color:#b8c3d6}
.tk{border:1px solid var(--line);border-left:3px solid var(--c);border-radius:6px;padding:8px;margin-bottom:6px;background:#05070b}.tk .t{display:flex;justify-content:space-between;font:700 12px system-ui}
.tk .m{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin:5px 0;color:#b8c3d6}.tk .m b{display:block;color:var(--mut);font-weight:400;font-size:8px}.tk .why{color:var(--o)}.tk.BLOCKED{opacity:.6}
.rail{height:4px;background:linear-gradient(90deg,#3a1219,#161d2b 50%,#0f3a2c);border-radius:3px;position:relative;margin:6px 0}.rail u{position:absolute;top:-3px;width:3px;height:10px;background:#fff;border-radius:2px}
.tr{color:var(--mut);font-size:9px;margin-top:3px;display:none}.tk:hover .tr,.tk:active .tr{display:block}
#log{max-height:150px;overflow:auto}#log div{padding:2px 0;border-bottom:1px solid #0b0f17;display:flex;gap:6px;font-size:9px}#log em{font-style:normal;color:var(--mut)}#log b{min-width:54px}
.win,.trade{color:var(--g)}.loss{color:var(--r)}.warn{color:var(--o)}.signal{color:#ff7ad0}
@media(max-width:900px){#agents{display:flex;overflow-x:auto}.ag{min-width:150px}.cols3,.cols2{grid-template-columns:1fr}.clk{font-size:18px}}
</style></head><body>
<header>
 <div class="logo"><i></i>SMC <b>DESK</b></div>
 <div><div class="k">Symbol</div><div class="v" id="sym">-</div></div>
 <div><div class="k">Price</div><div class="v" id="px">-</div></div>
 <div><div class="k">Equity</div><div class="v" id="eq">-</div></div>
 <div><div class="k">Drawdown</div><div class="v" id="dd">-</div></div>
 <div class="sp"></div><div class="clk" id="clk">--:--:--</div>
 <button id="snd" class="off">SOUND OFF</button><button id="tog">SWARM ONLINE</button>
</header>
<div id="strip"></div>
<div id="agents"></div>
<div class="card"><h4><span>◆ THE LENS · the tape goes in, entries come out</span><span id="halt" class="warn"></span></h4><canvas id="hub"></canvas></div>
<div class="cols2">
 <div class="card"><h4><span>CURRENT SIGNAL</span><span id="tier"></span></h4><div class="sig" id="sig">—</div>
  <div class="k">Score <span id="sc"></span></div><div class="bar2"><div id="scb"></div></div>
  <div class="k">EMA80 strength <span id="es"></span></div><div class="bar2"><div id="esb"></div></div><div class="chips" id="chips"></div></div>
 <div class="card"><h4><span>TRADE TICKETS</span><span id="nt"></span></h4><div id="tks"></div></div>
</div>
<div class="cols3">
 <div class="card"><h4><span>◆ SWARM PNL · one book, six agents</span><span class="up">LIVE</span></h4><div class="k" id="mode">PAPER</div><div class="big" id="pn">$0</div><div class="k" id="pns"></div></div>
 <div class="card"><h4><span>◆ BTC / USD · M1 + EMA80</span><span id="bias"></span></h4><div class="v" id="px2"></div><canvas id="c1" height="110"></canvas></div>
 <div class="card"><h4><span>◆ BAYESIAN UPDATE · buy vs sell</span><span id="bay"></span></h4><canvas id="c3" height="130"></canvas></div>
</div>
<div class="cols3">
 <div class="card"><h4><span>◆ BALANCE HISTORY</span><span id="st"></span></h4><canvas id="c2" height="120"></canvas></div>
 <div class="card"><h4><span>◆ ACTIVITY LOG</span></h4><div id="log"></div></div>
 <div class="card"><h4><span>◆ SIGNAL SPECTROGRAM</span></h4><canvas id="c4" height="120"></canvas></div>
</div>
<script>
const $=i=>document.getElementById(i),f2=n=>(+n).toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2}),SG={1:'▲',[-1]:'▼',0:'•'},DN={1:'BUY',[-1]:'SELL',0:'—'};
let S=null,lastEv=0,floats=[],eq0=null,hist=[],sound=false,AC=null,pk=[];
const SH={circle:'M12 2a10 10 0 1 0 .01 0z',tri:'M12 3 21.5 20H2.5z',drop:'M12 2C12 2 19.5 10.5 19.5 15.5a7.5 7.5 0 0 1-15 0C4.5 10.5 12 2 12 2z'},P2={};for(const k in SH)P2[k]=new Path2D(SH[k]);
const POS=[[.30,.36,'circle'],[.055,.52,'circle'],[.30,.72,'tri'],[.70,.36,'drop'],[.945,.52,'circle'],[.70,.72,'circle']],STEP=['01 · TAPE','02 · STRUCTURE','03 · EDGE','04 · SIZING','05 · EXECUTION','06 · SETTLEMENT'];
const svg=(c,s)=>`<svg viewBox="0 0 24 24" width="26" height="26"><path d="${SH[s]}" fill="${c}"/><path d="M8.6 11.5l1 3M15.4 11.5l-1 3" stroke="#fff" stroke-width="1.7" stroke-linecap="round"/></svg>`;
function beep(f,d,t0){if(!sound)return;try{AC=AC||new (window.AudioContext||window.webkitAudioContext)();const o=AC.createOscillator(),g=AC.createGain();o.frequency.value=f;o.type='sine';o.connect(g);g.connect(AC.destination);const t=AC.currentTime+(t0||0);g.gain.setValueAtTime(.07,t);g.gain.exponentialRampToValueAtTime(.0001,t+d);o.start(t);o.stop(t+d)}catch(e){}}
$('snd').onclick=()=>{sound=!sound;$('snd').textContent=sound?'SOUND ON':'SOUND OFF';$('snd').className=sound?'':'off';beep(660,.12)};
let ws;function conn(){ws=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws');ws.onmessage=e=>{S=JSON.parse(e.data);render()};ws.onclose=()=>setTimeout(conn,1500)}conn();
$('tog').onclick=()=>ws.send('toggle');
function render(){const s=S,g=s.sig,d=g.dir||0,st=s.stats;if(eq0===null)eq0=s.equity;const pl=s.equity-eq0,sw=Math.floor(Date.now()/1100)%6;
 hist.push({b:g.buy||0,s:g.sell||0,d,sc:g.score||0});if(hist.length>48)hist.shift();
 $('sym').textContent=s.symbol;$('px').textContent=f2(s.price);$('px2').textContent='$'+f2(s.price);$('eq').textContent='$'+f2(s.equity);$('dd').textContent=s.dd+'%';$('clk').textContent=s.time.slice(11,19)+' UTC';$('mode').textContent=s.mode+' · ACCOUNT';
 $('tog').textContent=s.enabled?'SWARM ONLINE':'SWARM PAUSED';$('tog').className=s.enabled?'':'off';$('halt').textContent=s.halt;
 $('pn').textContent='$'+f2(s.equity);$('pn').className='big '+(pl>=0?'up':'dn');$('pns').innerHTML=`${pl>=0?'▲':'▼'} ${f2(Math.abs(pl))} session · ${st.n} exits · WR ${st.wr}% · PF ${st.pf} · MAXDD ${st.maxdd}%`;
 $('strip').innerHTML=`<span class="live">● SIX AGENTS LIVE</span><span>HOLDER <b>${s.agents[sw].name}</b></span><span>HANDOFFS/MIN <b>${60+s.logs.length}</b></span><span>MODEL SCORE <b>${g.score||0}</b></span><span>EDGE <b class="${g.buy>g.sell?'up':'dn'}">${(g.buy-g.sell>=0?'+':'')+((g.buy||0)-(g.sell||0))}</b></span><span>OPEN <b>${s.positions.length} TICKETS</b></span><span>HUMAN INPUT <b>APPROVALS ONLY</b></span>`;
 if(!$('agents').children.length)$('agents').innerHTML=s.agents.map((a,i)=>`<div class="ag" id="a_${a.key}" style="--c:${a.color}"><div class="t"><span>${STEP[i]}</span><span class="bd">IDLE</span></div><div class="n">${svg(a.color,POS[i][2])}${a.name}</div><div class="bar"><i></i></div><p></p></div>`).join('');
 s.agents.forEach((a,i)=>{const e=$('a_'+a.key),now=i==sw||a.age<3;e.querySelector('p').textContent=a.msg;e.classList.toggle('now',now);e.querySelector('.bd').textContent=now?'● NOW':i==(sw+1)%6?'NEXT':'IDLE';e.querySelector('.bar i').style.width=now?'100%':'0'});
 $('sig').textContent=d?DN[d]+'  '+g.score:'NO SIGNAL';$('sig').style.color=d==1?'var(--g)':d==-1?'var(--r)':'var(--mut)';$('tier').textContent=g.tier\vert{}\vert{}'';$('sc').textContent=g.score+'/100 (buy '+g.buy+' · sell '+g.sell+')';
 $('scb').style.cssText=`width:${Math.min(100,g.score||0)}%;background:${d==1?'var(--g)':d==-1?'var(--r)':'#445'}`;$('es').textContent=g.strength+'/100 · slope '+g.slope_atr+' · dist '+g.dist_atr+' ATR';$('esb').style.cssText=`width:${g.strength}%;background:${g.bias==1?'var(--g)':g.bias==-1?'var(--r)':'#445'}`;
 $('bias').textContent='EMA80 '+(g.bias==1?'BULLISH':g.bias==-1?'BEARISH':'NEUTRAL');$('bias').className=g.bias==1?'up':g.bias==-1?'dn':'';$('bay').textContent=(g.buy>=g.sell?'BUY ':'SELL ')+Math.max(g.buy,g.sell);
 const F=g.facts||{};$('chips').innerHTML=[['D1',F.trend],['H4 MA',F.ma],['H1',F.candle],['OB',F.ob],['FVG',F.fvg]].map(([k,v])=>`<span>${k} ${SG[v||0]}</span>`).join('')+`<span>RSI ${F.rsi??'-'}</span>`+(g.note?`<span class="warn">${g.note}</span>`:'');
 $('st').textContent=`$${f2(s.balance)}`;$('nt').textContent=s.positions.length+' open';$('tks').innerHTML=s.tickets.slice(0,10).map(t=>{const c=t.status=='BLOCKED'?'#ff9f43':t.dir==1?'#2ee6a6':'#ff3b4d',pnl=t.status=='OPEN'?(t.live??0):t.pnl,pos=t.status=='OPEN'&&t.sl?Math.max(0,Math.min(100,(s.price-t.sl)/(t.tp-t.sl)*100)):50;
  return `<div class="tk ${t.status}" style="--c:${c}"><div class="t"><span>${DN[t.dir]} #${t.id} · ${t.tier}</span><span class="${t.status=='WIN'||pnl>0?'up':pnl<0?'dn':''}">${t.status=='BLOCKED'?'BLOCKED':t.status+'  '+(pnl>=0?'+':'')+f2(pnl)}</span></div>`+
  (t.status=='BLOCKED'?`<div class="why">${t.why}</div>`:`<div class="m"><span><b>ENTRY</b>${f2(t.entry)}</span><span><b>SL</b>${f2(t.sl)}</span><span><b>TP</b>${f2(t.tp)}</span><span><b>LOT</b>${t.lot}</span></div><div class="rail"><u style="left:${t.dir==1?pos:100-pos}%"></u></div>`)+
  `<div class="k">score ${t.score} · ${t.t.slice(5,16)}</div><div class="tr">${Object.entries(t.trail).map(([k,v])=>k.toUpperCase()+': '+v).join('<br>')}</div></div>`}).join('')||'<div class="k">waiting for the first entry signal…</div>';
 $('log').innerHTML=s.logs.map(l=>`<div><em>${l.t}</em><b style="color:${(s.agents.find(a=>a.key==l.a)||{}).color}">${l.a.toUpperCase()}</b><span class="${l.k}">${l.m}</span></div>`).join('');
 s.events.filter(e=>e.id>lastEv).forEach(e=>{lastEv=e.id;const c=e.type=='close'?(e.pnl>=0?'#2ee6a6':'#ff3b4d'):e.type=='open'?(e.dir==1?'#2ee6a6':'#ff3b4d'):'#ff9f43';floats.push({t:e.type=='open'?'ENTRY '+e.txt:e.txt,c,born:performance.now()});
  if(e.type=='open'){e.dir==1?(beep(520,.12),beep(780,.18,.12)):(beep(780,.12),beep(520,.18,.12))}else if(e.type=='close'){e.pnl>=0?(beep(660,.1),beep(880,.1,.1),beep(1320,.25,.2)):beep(200,.4)}else beep(300,.15)});
 const ch=s.chart||{};candles($('c1'),ch.p,ch.e);line($('c2'),[{d:s.eq,c:'#2ee6a6',fill:1}]);bayes($('c3'),g);spec($('c4'))}
function fit(cv,hh){const r=devicePixelRatio||1,w=cv.clientWidth,h=hh||+cv.getAttribute('height');if(hh)cv.style.height=h+'px';if(cv.width!=w*r||cv.height!=h*r){cv.width=w*r;cv.height=h*r}const x=cv.getContext('2d');x.setTransform(r,0,0,r,0,0);x.clearRect(0,0,w,h);return[x,w,h]}
function line(cv,ss){ss=ss.filter(s=>s.d&&s.d.length>1);if(!ss.length)return;const[x,w,h]=fit(cv),all=ss.flatMap(s=>s.d),lo=Math.min(...all),hi=Math.max(...all),y=v=>h-4-(v-lo)/((hi-lo)||1)*(h-8);
 ss.forEach(s=>{x.beginPath();s.d.forEach((v,i)=>x[i?'lineTo':'moveTo'](i/(s.d.length-1)*w,y(v)));x.strokeStyle=s.c;x.lineWidth=1.5;x.stroke();if(s.fill){x.lineTo(w,h);x.lineTo(0,h);const gr=x.createLinearGradient(0,0,0,h);gr.addColorStop(0,'#2ee6a655');gr.addColorStop(1,'#2ee6a600');x.fillStyle=gr;x.fill()}})}
function candles(cv,p,e){if(!p||p.length<8)return;const[x,w,h]=fit(cv),n=Math.floor(p.length/4),c=[];for(let i=0;i<n;i++){const a=p.slice(i*4,i*4+4);c.push([a[0],Math.max(...a),Math.min(...a),a[3]])}
 const lo=Math.min(...c.map(k=>k[2])),hi=Math.max(...c.map(k=>k[1])),y=v=>h-4-(v-lo)/((hi-lo)||1)*(h-8),bw=w/n;
 c.forEach((k,i)=>{const col=k[3]>=k[0]?'#2ee6a6':'#ff3b4d',cx=i*bw+bw/2;x.strokeStyle=x.fillStyle=col;x.beginPath();x.moveTo(cx,y(k[1]));x.lineTo(cx,y(k[2]));x.stroke();x.fillRect(cx-bw*.32,Math.min(y(k[0]),y(k[3])),bw*.64,Math.max(1.5,Math.abs(y(k[0])-y(k[3]))))});
 if(e){x.beginPath();e.slice(0,n*4).forEach((v,i)=>x[i?'lineTo':'moveTo'](i/(n*4)*w,y(v)));x.strokeStyle='#ff9f43';x.lineWidth=1;x.stroke()}}
function bayes(cv,g){const[x,w,h]=fit(cv),G=(m,sg,col,al)=>{x.beginPath();for(let i=0;i<=w;i+=3){const u=i/w,v=Math.exp(-Math.pow((u-m)/sg,2)/2);x[i?'lineTo':'moveTo'](i,h-6-v*(h-18))}x.strokeStyle=col;x.lineWidth=1.6;x.stroke();x.lineTo(w,h);x.lineTo(0,h);x.fillStyle=col+al;x.fill()};
 const st=(g.strength||0)/100;G((g.sell||0)/100,.14-.05*st,'#ff3b4d','33');G((g.buy||0)/100,.14-.05*st,'#2ee6a6','33');x.fillStyle='#5d6a80';x.font='8px ui-monospace';x.fillText('SELL '+(g.sell||0),4,10);x.textAlign='right';x.fillText('BUY '+(g.buy||0),w-4,10)}
function spec(cv){const[x,w,h]=fit(cv),n=48,R=14,cw=w/n,ch=h/R;hist.forEach((q,i)=>{for(let r=0;r<R;r++){const c=(r+.5)/R*100,v=Math.max(0,1-Math.abs(c-q.sc)/22);if(v>.05){x.fillStyle=q.d>=0&&q.b>=q.s?`rgba(46,230,166,${v})`:`rgba(255,159,67,${v})`;x.fillRect((n-hist.length+i)*cw,h-(r+1)*ch,cw-1,ch-1)}}})}
const hub=$('hub'),noise=[],fan=[];for(let i=0;i<280;i++)noise.push([Math.random(),Math.random()*2-1,Math.random()]);for(let i=0;i<70;i++)fan.push([Math.random(),Math.random()*6.28]);
function loop(t){requestAnimationFrame(loop);const w0=hub.clientWidth,[x,w,h]=fit(hub,w0<600?310:Math.min(560,w0*.5));x.fillStyle='#020306';x.fillRect(0,0,w,h);if(!S)return;
 const g=S.sig||{},d=g.dir||0,ox=w*.5,oy=h*.5,r=Math.min(w*.07,h*.17)*(1+.02*Math.sin(t/500)),aL=w*.16,aR=w*.84,top=h*.09,bot=h*.91,ph=t/1000,sw=Math.floor(t/1100)%6,p=(S.chart&&S.chart.p)||[],ph1=ph*1.7;
 const nb=34,rh=(bot-top)/nb;for(let i=0;i<nb;i++){const c=Math.abs((i+.5)/nb-.5)*2,len=w*.14*(.18+.82*c)*(.65+.35*Math.sin(ph1+i*.9));x.fillStyle=i<nb/2?'rgba(46,230,166,.55)':'rgba(255,59,77,.6)';x.fillRect(aL-len,top+i*rh+1,len,rh-2)}
 x.fillStyle='#2ee6a6';x.fillRect(aL-1,top,2,(bot-top)/2);x.fillStyle='#ff3b4d';x.fillRect(aL-1,oy,2,(bot-top)/2);
 x.lineWidth=.7;fan.forEach(([u,q])=>{const y0=top+u*(bot-top);x.strokeStyle=`rgba(255,255,255,${.1+.25*Math.abs(Math.sin(ph+q))})`;x.beginPath();x.moveTo(aL,y0);x.bezierCurveTo(w*.3,y0+Math.sin(ph*.8+q)*18,ox-r*2.6,oy+Math.cos(ph+q)*10,ox-r,oy+(u-.5)*r*.8);x.stroke()});
 const L=p.slice(0,80),Rr=p.slice(-40);if(L.length>2){const lo=Math.min(...L),hi=Math.max(...L);x.beginPath();L.forEach((v,i)=>x[i?'lineTo':'moveTo'](aL+i/(L.length-1)*(ox-r-aL),oy+((hi+lo)/2-v)/((hi-lo)||1)*h*.2));x.strokeStyle='rgba(255,255,255,.9)';x.lineWidth=1.3;x.stroke()}
 for(const[a,b,c]of noise){const nx=ox+r*1.3+a*(w*.8-ox-r*1.3),ny=oy+b*h*.3*(.3+.7*a)+Math.sin(ph+c*9)*3;x.fillStyle=`rgba(210,220,235,${.12+.5*Math.abs(Math.sin(ph*2+c*20))})`;x.fillRect(nx,ny,1.4,1.4)}
 [['#ff9f43',0],['#facc15',1],['#2ee6a6',2],['#ff9f43',3]].forEach(([c,k])=>{x.beginPath();for(let xx=w*.66;xx<=w*.86;xx+=3)x[xx==w*.66?'moveTo':'lineTo'](xx,oy+(k-1.5)*7+Math.sin(xx*.045+ph*(1+k*.3)+k)*h*.045*(1+k*.25));x.strokeStyle=c+'aa';x.lineWidth=1;x.stroke()});
 if(Rr.length>2){const lo=Math.min(...Rr),hi=Math.max(...Rr);x.save();x.shadowColor='#5eead4';x.shadowBlur=8;x.beginPath();Rr.forEach((v,i)=>x[i?'lineTo':'moveTo'](ox+r+i/(Rr.length-1)*(w*.78-ox-r),oy+(Rr[0]-v)/((hi-lo)||1)*h*.32));x.strokeStyle='#5eead4';x.lineWidth=1.8;x.stroke();x.restore()}
 for(let y=top;y<bot;y+=3){const wd=w*.06*Math.exp(-Math.pow((y-oy)/(h*.17),2))*(.85+.15*Math.sin(ph*2+y*.05));x.fillStyle=y<oy?'rgba(46,230,166,.4)':'rgba(255,59,77,.4)';x.fillRect(aR,y,wd,3)}
 x.fillStyle='#2ee6a6';x.fillRect(aR-1,top,2,oy-top);x.fillStyle='#ff3b4d';x.fillRect(aR-1,oy,2,bot-oy);
 x.fillStyle='#5d6a80';x.fillRect(w*.7,h*.36,1,h*.36);x.fillStyle='#fff';x.font='700 '+(w<600?12:18)+'px system-ui';x.textAlign='left';x.fillText('$'+f2(S.price),6,top-1);x.textAlign='right';x.fillText((g.score||0)+'/100',w-6,top-1);
 const P=S.agents.map((a,i)=>({a,i,cx:POS[i][0]*w,cy:POS[i][1]*h,hot:a.age<3||i==sw}));
 P.forEach(q=>{const mx=(q.cx+ox)/2,my=(q.cy+oy)/2+(q.cy<oy?-1:1)*h*.1;q.m=[mx,my];x.strokeStyle=q.a.color+(q.hot?'88':'28');x.lineWidth=q.hot?1.3:.8;x.beginPath();x.moveTo(q.cx,q.cy);x.quadraticCurveTo(mx,my,ox,oy);x.stroke();if(q.hot&&Math.random()<.4)pk.push({q,u:0,s:.012+Math.random()*.014})});
 pk=pk.filter(k=>k.u<1);pk.forEach(k=>{k.u+=k.s;const q=k.q,u=k.u,px=(1-u)**2*q.cx+2*(1-u)*u*q.m[0]+u*u*ox,py=(1-u)**2*q.cy+2*(1-u)*u*q.m[1]+u*u*oy;x.fillStyle=q.a.color;x.shadowColor=q.a.color;x.shadowBlur=8;x.beginPath();x.arc(px,py,2.2,0,7);x.fill();x.shadowBlur=0});
 const col=d==1?'#2ee6a6':d==-1?'#ff3b4d':'#ffffff';x.save();x.shadowColor=col;x.shadowBlur=36+8*Math.sin(ph*2);x.fillStyle='#fff';x.beginPath();x.arc(ox,oy,r,0,7);x.fill();x.restore();
 x.strokeStyle='rgba(255,255,255,.75)';x.lineWidth=1.4;x.beginPath();x.arc(ox,oy,r*1.17,ph*.9,ph*.9+1.3);x.stroke();x.strokeStyle='rgba(255,255,255,.3)';x.beginPath();x.arc(ox,oy,r*1.3,-ph*.6,-ph*.6+.9);x.stroke();
 const bl=(t%4200)<130?.1:1;x.fillStyle='#050505';[-1,1].forEach(sg=>{x.save();x.translate(ox+sg*r*.3+d*r*.05,oy);x.rotate(-sg*.3);x.scale(1,bl);x.beginPath();x.ellipse(0,0,r*.1,r*.3,0,0,7);x.fill();x.restore()});
 x.textAlign='center';x.fillStyle=d?col:'#5d6a80';x.font='700 '+(w<600?9:11)+'px ui-monospace';x.fillText(d?`${DN[d]} · ${g.tier||''} · ${g.score}`:'SCANNING THE TAPE',ox,oy+r*1.9);
 P.forEach(q=>{const sz=(w<600?14:21)*(1+(q.hot?.16:0)+.04*Math.sin(ph*2+q.i)),sh=POS[q.i][2];x.save();x.shadowColor=q.a.color;x.shadowBlur=q.hot?26:8;x.translate(q.cx-sz,q.cy-sz);x.scale(sz/12,sz/12);x.fillStyle=q.a.color;x.fill(P2[sh]);x.shadowBlur=0;
  x.strokeStyle='#fff';x.lineWidth=1.7;x.lineCap='round';const b2=(t%4200+q.i*300)%4200<120?.2:1;x.beginPath();x.moveTo(8.6,11.5);x.lineTo(9.6,11.5+3*b2);x.moveTo(15.4,11.5);x.lineTo(14.4,11.5+3*b2);x.stroke();x.restore();
  x.font='700 '+(w<600?7:9)+'px ui-monospace';x.fillStyle=q.a.color;x.textAlign='center';x.fillText(q.a.name,q.cx,q.cy+sz+11)});
 const now=performance.now();floats=floats.filter(f=>now-f.born<3800);floats.forEach(f=>{const a=(now-f.born)/3800;x.globalAlpha=Math.min(1,2-a*2);x.fillStyle=f.c;x.shadowColor=f.c;x.shadowBlur=12;x.font='800 '+(w<600?14:22)+'px system-ui';x.textAlign='left';x.fillText(f.t,w*.74,oy-6-a*34);x.shadowBlur=0;x.globalAlpha=1})}
requestAnimationFrame(loop);
</script></body></html>
"""

def build(cfg):
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    eng, app = Engine(cfg), FastAPI()
    def loop():
        while True:
            try: eng.tick()
            except Exception as ex: eng.logs.appendleft(dict(t="--", a="closer", m=f"ERROR {ex!r}", k="loss"))
            time.sleep(1)
    threading.Thread(target=loop, daemon=True).start()
    @app.get("/")
    def index(): return HTMLResponse(PAGE)
    @app.websocket("/ws")
    async def ws(w: WebSocket):
        await w.accept()
        async def rx():
            async for m in w.iter_text():
                if m == "toggle": eng.enabled = not eng.enabled
        task = asyncio.create_task(rx())
        try:
            while True:
                await w.send_text(json.dumps(eng.snapshot(), default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
                await asyncio.sleep(1)
        except Exception: pass
        finally: task.cancel()
    return app

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--live", action="store_true"); ap.add_argument("--symbol", default="BTCUSDm")
    ap.add_argument("--port", type=int, default=8000); a, _ = ap.parse_known_args()
    import uvicorn
    # قراءة البورت المخصص من Render تلقائياً أو الاعتماد على 10000 كقيمة افتراضية
    port = int(os.environ.get("PORT", a.port))
    print(f"\n>>> Running server on port: {port}\n")
    uvicorn.run(build(Cfg(symbol=a.symbol, live=a.live)), host="0.0.0.0", port=port, log_level="warning")
