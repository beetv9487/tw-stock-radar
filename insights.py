# ============================================================
# 📋 個股深入分析
#
#   ・結構價位：支撐／壓力分近端、中期、長期，附價格帶、依據與強度
#   ・現在的位置：觀察區／結構觀察區／偏熱區
#   ・今天為什麼動、重點整理
#   ・五面向評分（趨勢、量價、法人、籌碼、基本面），每一分都列出規則
#   ・籌碼（融資融券、借券賣出）、本益比、殖利率、月營收
#
# 所有價位、分數都是公式整理出來的參考，不構成買賣建議。
# ============================================================

import html
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

import screener
from screener import _get_json, _num, _clean

TZ = timezone(timedelta(hours=8))


# ------------------------------------------------------------
# 台股升降單位
# ------------------------------------------------------------

def tick(price):
    for limit, step in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1), (float("inf"), 5)):
        if price < limit:
            return step


def round_tick(x, mode="nearest"):
    if x is None or not np.isfinite(x) or x <= 0:
        return np.nan
    t = tick(x)
    f = {"nearest": np.round, "down": np.floor, "up": np.ceil}[mode]
    return round(float(f(x / t + (1e-9 if mode != "down" else -1e-9)) * t), 2)


def pct(a, b):
    return (a / b - 1) * 100 if b and np.isfinite(a) and np.isfinite(b) else np.nan


def fmt_price(v):
    return f"{v:,.2f}" if v is not None and np.isfinite(v) else "—"


def fmt_pct(v, digits=1):
    return f"{v:+.{digits}f}%" if v is not None and np.isfinite(v) else "—"


# ------------------------------------------------------------
# 技術指標
# ------------------------------------------------------------

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    u, v = float(up.iloc[-1]), float(dn.iloc[-1])
    if v == 0:
        return 100.0 if u > 0 else 50.0        # 一路上漲沒有下跌日 → RSI 100
    return 100 - 100 / (1 + u / v)


# ------------------------------------------------------------
# 結構價位：收集各種依據 → 合併成價格帶 → 分支撐／壓力、近端／中期／長期
# ------------------------------------------------------------

def _candidates(df):
    """回傳 [(價位, 依據)]"""
    out = []
    d = df.tail(250)
    w = 5
    win = 2 * w + 1
    hi = d["High"][d["High"] == d["High"].rolling(win, center=True).max()]
    lo = d["Low"][d["Low"] == d["Low"].rolling(win, center=True).min()]
    out += [(v, "波段高點") for v in hi.values]
    out += [(v, "波段低點") for v in lo.values]

    c = df["Close"]
    for n in (5, 10, 20, 60, 120):
        if len(c) >= n:
            out.append((float(c.rolling(n).mean().iloc[-1]), f"MA{n}"))

    # 成交量密集區：近 120 天
    d = df.tail(120)
    h, edges = np.histogram(d["Close"], bins=24, weights=d["Volume"])
    centers = (edges[:-1] + edges[1:]) / 2
    for i in np.argsort(h)[::-1][:3]:
        out.append((float(centers[i]), "成交量密集區"))

    # 爆量 K：近 120 天成交量 > 20 日均量 2 倍的那幾天
    v20 = df["Volume"].rolling(20).mean().shift(1)
    spikes = df[(df["Volume"] > v20 * 2)].tail(120)
    spikes = spikes[spikes.index >= df.index[-120]] if len(df) > 120 else spikes
    for _, r in spikes.iterrows():
        out.append((float(r["High"]), "爆量K高點"))
        out.append((float(r["Low"]), "爆量K低點"))

    # 跳空缺口：近 120 天、還沒被完全回補的
    d = df.tail(121)
    for i in range(1, len(d)):
        prev, cur = d.iloc[i - 1], d.iloc[i]
        later = d.iloc[i + 1:]
        if cur["Low"] > prev["High"]:                       # 向上跳空
            if later.empty or later["Low"].min() > prev["High"]:
                out.append((float(cur["Low"]), "缺口上緣"))
                out.append((float(prev["High"]), "缺口下緣"))
        elif cur["High"] < prev["Low"]:                     # 向下跳空
            if later.empty or later["High"].max() < prev["Low"]:
                out.append((float(cur["High"]), "缺口下緣"))
                out.append((float(prev["Low"]), "缺口上緣"))
    return [(p, s) for p, s in out if np.isfinite(p) and p > 0]


def _strength(n):
    return "極強" if n >= 4 else "強" if n == 3 else "中" if n == 2 else "弱"


def _cluster(cands, tol=0.015):
    cands = sorted(cands)
    groups = []
    for p, s in cands:
        if groups and p <= groups[-1]["lo"] * (1 + tol):
            g = groups[-1]
            g["hi"] = p
            g["items"].append(p)
            if s not in g["sources"]:
                g["sources"].append(s)
        else:
            groups.append({"lo": p, "hi": p, "items": [p], "sources": [s]})
    for g in groups:
        g["lo"], g["hi"] = round_tick(g["lo"], "down"), round_tick(g["hi"], "up")
        g["n"] = len(g["sources"])
        g["strength"] = _strength(g["n"])
    return groups


def structure(df, price):
    groups = _cluster(_candidates(df))
    sup, pres = [], []
    for g in groups:
        mid = (g["lo"] + g["hi"]) / 2
        if g["hi"] < price * 0.998 or (mid < price and g["lo"] < price * 0.998):
            g["price"] = min(g["hi"], round_tick(price * 0.998, "down"))     # 支撐取價格帶上緣
            g["side"] = "support"
            sup.append(g)
        elif g["lo"] > price * 1.002 or mid > price:
            g["price"] = max(g["lo"], round_tick(price * 1.002, "up"))       # 壓力取價格帶下緣
            g["side"] = "pressure"
            pres.append(g)
    for g in sup + pres:
        g["dist"] = pct(g["price"], price)
        a = abs(g["dist"])
        g["tier"] = "近端" if a <= 10 else "中期" if a <= 20 else "長期"
        g["price"] = round_tick(g["price"])
    sup.sort(key=lambda g: -g["price"])
    pres.sort(key=lambda g: g["price"])

    # 支撐：第一、第二只從近端／中期挑；超過 20% 列為長期結構
    near_sup = [g for g in sup if g["tier"] != "長期"]
    s1 = near_sup[0] if near_sup else None
    s2 = next((g for g in near_sup[1:] if s1 and g["hi"] < s1["lo"] * 0.995), None)
    long_sup = next((g for g in sup if g["tier"] == "長期"), None)

    # 壓力：第一壓力要在 +10% 內；沒有就留白、改列中期壓力
    within = [g for g in pres if g["tier"] == "近端"]
    p1 = within[0] if within else None
    mid_p = None if p1 else next(iter(pres), None)
    base = p1 or mid_p
    p2 = next((g for g in pres if base and g["lo"] > base["hi"] * 1.005), None)
    chosen = {id(x) for x in (p1, p2, mid_p) if x}
    others = [g for g in within if id(g) not in chosen][:3]
    longer = [g for g in pres if g["tier"] != "近端" and id(g) not in chosen][:3]

    limit_up = round_tick(price * 1.1, "down") if not pres else np.nan
    return {"s1": s1, "s2": s2, "long_sup": long_sup, "p1": p1, "mid_p": mid_p, "p2": p2,
            "others": others, "longer": longer, "limit_up": limit_up}


# ------------------------------------------------------------
# 第 2 批資料：籌碼、估值（都是全市場一張表，抓最近一個交易日）
# ------------------------------------------------------------

def _recent_days(n=7):
    d = datetime.now(TZ).date()
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def _find_col(fields, *words, nth=0, default=None):
    hits = [i for i, f in enumerate(fields) if all(w in str(f) for w in words)]
    return hits[nth] if len(hits) > nth else default


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def margin_table():
    """{代號: {"fin", "fin_prev", "short", "short_prev"}}（張）、資料日期"""
    out, got = {}, None
    for d in _recent_days():
        try:
            p = _get_json("https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
                          {"date": d.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"})
        except Exception:
            continue
        tables = p.get("tables") or []
        if p.get("stat") != "OK" or len(tables) < 2 or not tables[1].get("data"):
            continue
        f = tables[1].get("fields") or []
        i_fp, i_f = _find_col(f, "前日餘額", default=5), _find_col(f, "今日餘額", default=6)
        i_sp, i_s = _find_col(f, "前日餘額", nth=1, default=11), _find_col(f, "今日餘額", nth=1, default=12)
        for r in tables[1]["data"]:
            out[_clean(r[0])] = {"fin": _num(r[i_f]), "fin_prev": _num(r[i_fp]),
                                 "short": _num(r[i_s]), "short_prev": _num(r[i_sp])}
        got = d
        break
    for d in _recent_days():
        try:
            p = _get_json("https://www.tpex.org.tw/www/zh-tw/margin/balance",
                          {"date": d.strftime("%Y/%m/%d"), "id": "", "response": "json"})
        except Exception:
            continue
        t = (p.get("tables") or [{}])[0]
        if not t.get("data"):
            continue
        for r in t["data"]:
            if len(r) > 14:
                out[_clean(r[0])] = {"fin": _num(r[6]), "fin_prev": _num(r[2]),
                                     "short": _num(r[14]), "short_prev": _num(r[10])}
        got = got or d
        break
    return out, got


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def sbl_table():
    """借券賣出餘額 {代號: {"today", "prev"}}（張）、資料日期"""
    out, got = {}, None
    for d in _recent_days():
        try:
            p = _get_json("https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U",
                          {"date": d.strftime("%Y%m%d"), "response": "json"})
        except Exception:
            continue
        if p.get("stat") != "OK" or not p.get("data"):
            continue
        for r in p["data"]:
            if len(r) > 12:
                out[_clean(r[0])] = {"today": _num(r[12]) / 1000, "prev": _num(r[8]) / 1000}
        got = d
        break
    for d in _recent_days():
        try:
            p = _get_json("https://www.tpex.org.tw/www/zh-tw/margin/sbl",
                          {"date": d.strftime("%Y/%m/%d"), "response": "json"})
        except Exception:
            continue
        t = (p.get("tables") or [{}])[0]
        if not t.get("data"):
            continue
        for r in t["data"]:
            if len(r) > 12:
                out[_clean(r[0])] = {"today": _num(r[12]) / 1000, "prev": _num(r[8]) / 1000}
        got = got or d
        break
    return out, got


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def valuation_table():
    """{代號: {"pe", "yield", "pb"}}、資料日期"""
    out, got = {}, None
    for d in _recent_days():
        try:
            p = _get_json("https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d",
                          {"date": d.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"})
        except Exception:
            continue
        if p.get("stat") != "OK" or not p.get("data"):
            continue
        f = p.get("fields") or []
        i_y, i_pe, i_pb = _find_col(f, "殖利率", default=3), _find_col(f, "本益比", default=5), \
            _find_col(f, "淨值比", default=6)
        for r in p["data"]:
            out[_clean(r[0])] = {"pe": _num(r[i_pe]), "yield": _num(r[i_y]), "pb": _num(r[i_pb])}
        got = d
        break
    for d in _recent_days():
        try:
            p = _get_json("https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate",
                          {"date": d.strftime("%Y/%m/%d"), "response": "json"})
        except Exception:
            continue
        t = (p.get("tables") or [{}])[0]
        if not t.get("data"):
            continue
        for r in t["data"]:
            if len(r) > 6:
                out[_clean(r[0])] = {"pe": _num(r[2]), "yield": _num(r[5]), "pb": _num(r[6])}
        got = got or d
        break
    return out, got


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


# ------------------------------------------------------------
# 評分：每個面向從 50 分起算，逐條加減，最後限制在 0～100
# ------------------------------------------------------------

def _score(rules):
    """rules = [(說明, 分數)]；回傳 (分數, rules)"""
    if not rules:
        return None, []
    s = 50 + sum(p for _, p in rules)
    return int(max(0, min(100, s))), rules


WEIGHTS = {"趨勢面": 25, "量價": 15, "法人": 20, "籌碼": 15, "基本面": 25}


def grade(score):
    if score is None:
        return "資料不足"
    return ("偏多" if score >= 70 else "中性偏多" if score >= 55 else "中性" if score >= 45
            else "中性偏弱" if score >= 30 else "偏弱")


def build(code, hist, price):
    df = hist.copy()
    c = df["Close"]
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60, 120)}
    m = {n: float(ma[n].iloc[-1]) if len(c) >= n else np.nan for n in ma}
    prev_close = float(c.iloc[-2]) if len(c) >= 2 else np.nan
    chg = pct(price, prev_close)
    v20 = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else np.nan
    vol_ratio = float(df["Volume"].iloc[-1]) / v20 if v20 else np.nan
    r14 = rsi(c)
    hi20 = float(df["High"].iloc[-21:-1].max()) if len(df) > 21 else np.nan
    lo20 = float(df["Low"].tail(20).min())

    st_ = structure(df, price)
    dates = {"行情": df.index[-1].date() if hasattr(df.index[-1], "date") else None}

    # ---------- 法人 ----------
    fi_dates, fi = _safe(lambda: screener.recent_foreign(5), ([], {}))
    rec = fi.get(code)
    if fi_dates:
        dates["法人"] = fi_dates[0]
    # ---------- 籌碼 ----------
    mg_all, mg_date = _safe(margin_table, ({}, None))
    sb_all, sb_date = _safe(sbl_table, ({}, None))
    mg, sb = mg_all.get(code), sb_all.get(code)
    if mg_date or sb_date:
        dates["籌碼"] = mg_date or sb_date
    # ---------- 基本面 ----------
    val_all, val_date = _safe(valuation_table, ({}, None))
    val = val_all.get(code)
    if val_date:
        dates["估值"] = val_date
    rev = _safe(screener.load_revenue, {}).get(code)
    inc = _safe(screener.load_income, {}).get(code)

    scores = {}

    # 趨勢面
    r = []
    if np.isfinite(m[20]):
        r.append(("收盤在 MA20 之上" if price > m[20] else "收盤在 MA20 之下", 10 if price > m[20] else -10))
    if np.isfinite(m[60]):
        r.append(("收盤在 MA60（季線）之上" if price > m[60] else "收盤在 MA60（季線）之下",
                  10 if price > m[60] else -10))
    if len(c) >= 26:
        up = ma[20].iloc[-1] > ma[20].iloc[-6]
        r.append(("MA20 往上" if up else "MA20 往下", 10 if up else -10))
    if len(c) >= 81:
        up = ma[60].iloc[-1] > ma[60].iloc[-21]
        r.append(("MA60 往上" if up else "MA60 往下", 10 if up else -10))
    if all(np.isfinite([m[5], m[10], m[20]])):
        if m[5] > m[10] > m[20]:
            r.append(("MA5 > MA10 > MA20 多頭排列", 10))
        elif m[5] < m[10] < m[20]:
            r.append(("MA5 < MA10 < MA20 空頭排列", -10))
    scores["趨勢面"] = _score(r)

    # 量價
    r = []
    if np.isfinite(chg) and np.isfinite(vol_ratio):
        if chg > 0 and vol_ratio >= 1.2:
            r.append((f"價漲量增（量比 {vol_ratio:.2f} 倍）", 15))
        elif chg < 0 and vol_ratio >= 1.2:
            r.append((f"價跌量增（量比 {vol_ratio:.2f} 倍）", -15))
        elif chg > 0:
            r.append(("價漲量縮", 0))
        elif chg < 0:
            r.append(("價跌量縮", 5))
    if np.isfinite(r14):
        if r14 >= 80:
            r.append((f"RSI {r14:.0f} 過熱", -10))
        elif r14 >= 55:
            r.append((f"RSI {r14:.0f} 偏強", 10))
        elif r14 <= 20:
            r.append((f"RSI {r14:.0f} 超賣", -5))
        elif r14 < 45:
            r.append((f"RSI {r14:.0f} 偏弱", -10))
    if np.isfinite(hi20):
        if price > hi20:
            r.append(("突破前 20 日高點", 10))
    scores["量價"] = _score(r)

    # 法人
    r = []
    if rec:
        fo = [x for x in rec["nets"]]
        tr = [x for x in rec.get("trust", [])]
        f0 = fo[0] if fo and np.isfinite(fo[0]) else np.nan
        if np.isfinite(f0):
            r.append((f"外資今日{'買' if f0 > 0 else '賣'}超 {abs(f0) / 1000:,.0f} 張", 10 if f0 > 0 else -10))
        streak, sign = 0, np.sign(f0) if np.isfinite(f0) else 0
        for v in fo:
            if np.isfinite(v) and sign != 0 and np.sign(v) == sign:
                streak += 1
            else:
                break
        if streak >= 2:
            r.append((f"外資連 {streak} 天{'買' if sign > 0 else '賣'}超", int(sign) * min(15, 5 * streak)))
        tot = np.nansum(fo)
        if tot:
            r.append((f"外資近 {len(fi_dates)} 日合計{'買' if tot > 0 else '賣'}超 {abs(tot) / 1000:,.0f} 張",
                      10 if tot > 0 else -10))
        t0 = tr[0] if tr and np.isfinite(tr[0]) else np.nan
        if np.isfinite(t0) and t0 != 0:
            r.append((f"投信今日{'買' if t0 > 0 else '賣'}超 {abs(t0) / 1000:,.0f} 張", 10 if t0 > 0 else -10))
    scores["法人"] = _score(r)

    # 籌碼
    r = []
    if mg:
        dfin = mg["fin"] - mg["fin_prev"]
        if np.isfinite(dfin) and dfin != 0:
            if dfin < 0:
                r.append((f"融資減少 {abs(dfin):,.0f} 張（籌碼較安定）", 10))
            else:
                r.append((f"融資增加 {dfin:,.0f} 張（散戶追價）", -10 if (np.isfinite(chg) and chg <= 0) else -5))
        dsh = mg["short"] - mg["short_prev"]
        if np.isfinite(dsh) and dsh > 0:
            r.append((f"融券增加 {dsh:,.0f} 張", 5))
    if sb:
        db = sb["today"] - sb["prev"]
        if np.isfinite(db) and abs(db) >= 1:
            r.append((f"借券賣出餘額{'增加' if db > 0 else '減少'} {abs(db):,.0f} 張", -10 if db > 0 else 10))
    scores["籌碼"] = _score(r)

    # 基本面
    r = []
    if rev and np.isfinite(rev["yoy"]):
        y = rev["yoy"]
        r.append((f"月營收年增 {y:+.1f}%", 20 if y >= 20 else 10 if y > 0 else -15))
        if np.isfinite(rev.get("mom", np.nan)):
            r.append((f"月營收月增 {rev['mom']:+.1f}%", 5 if rev["mom"] > 0 else -5))
    if inc and np.isfinite(inc["net"]):
        r.append(("今年累計有獲利" if inc["net"] > 0 else "今年累計虧損", 10 if inc["net"] > 0 else -15))
    if val:
        pe, y = val["pe"], val["yield"]
        if np.isfinite(pe) and pe > 0:
            r.append((f"本益比 {pe:.1f}", 10 if pe < 15 else 0 if pe <= 30 else -10))
        if np.isfinite(y) and y > 0:
            r.append((f"殖利率 {y:.2f}%", 10 if y >= 4 else 5 if y >= 2 else 0))
    scores["基本面"] = _score(r)

    have = {k: v for k, v in scores.items() if v[0] is not None}
    total = None
    if have:
        wsum = sum(WEIGHTS[k] for k in have)
        total = int(round(sum(v[0] * WEIGHTS[k] for k, v in have.items()) / wsum))

    return {
        "code": code, "price": price, "chg": chg, "vol_ratio": vol_ratio, "rsi": r14, "ma": m,
        "hi20": hi20, "lo20": lo20, "struct": st_, "scores": scores, "total": total,
        "foreign": rec, "fi_dates": fi_dates, "margin": mg, "sbl": sb, "val": val,
        "rev": rev, "inc": inc, "dates": dates,
    }


# ------------------------------------------------------------
# 現在的位置
# ------------------------------------------------------------

def position(ins):
    st_ = ins["struct"]
    price, lo = ins["price"], ins["lo20"]
    top_g = st_["p1"] or st_["mid_p"]
    top = top_g["price"] if top_g else st_["limit_up"]
    pos = (price - lo) / (top - lo) if top and np.isfinite(top) and top > lo else 1.0
    pos = float(max(0, min(1, pos)))
    s1 = st_["s1"]
    near_strong = s1 and s1["n"] >= 2 and (price / s1["hi"] - 1) <= 0.03
    if pos >= 0.8 or (np.isfinite(ins["rsi"]) and ins["rsi"] >= 75):
        zone = "偏熱區"
    elif near_strong:
        zone = "結構觀察區"
    else:
        zone = "觀察區"
    return zone, pos, top


# ------------------------------------------------------------
# 畫面
# ------------------------------------------------------------

CSS = """
<style>
.ins-card { border-radius: 24px; padding: 19px 18px; margin-top: 15px;
            background: rgba(6,30,59,.92); border: 1px solid rgba(84,139,180,.25); }
.ins-kicker { color: #7d9ab7; font-size: 11px; letter-spacing: 3px; }
.ins-title { color: #fff; font-size: 19px; font-weight: 800; margin: 4px 0 12px; }
.ins-dates { color: #6f91ad; font-size: 11px; line-height: 1.9; margin-top: 8px; }
.score-wrap { display: flex; align-items: center; gap: 18px; }
.score-big { font-size: 46px; font-weight: 800; color: #fff; line-height: 1; }
.score-big small { font-size: 14px; color: #7d9ab7; margin-left: 4px; }
.score-grade { display: inline-block; margin-top: 8px; padding: 4px 12px; border-radius: 999px;
               border: 1px solid rgba(85,207,255,.5); color: #55cfff; font-size: 13px; font-weight: 800; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: 13px; color: #9db8cf; }
.bar-row .name { width: 52px; }
.bar-row .track { flex: 1; height: 8px; border-radius: 99px; background: rgba(105,150,185,.18); overflow: hidden; }
.bar-row .fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #0878bd, #4ee3d0); }
.bar-row .val { width: 34px; text-align: right; color: #edf7ff; font-weight: 700; }
.ins-list { margin: 0; padding-left: 18px; color: #d7e8f6; font-size: 14px; line-height: 1.9; }
.ins-list li { margin-bottom: 6px; }
.why { display: flex; gap: 10px; margin: 10px 0; }
.why .dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 7px; flex: none; }
.why .h { font-weight: 800; font-size: 15px; }
.why .t { color: #b9cfe2; font-size: 13px; margin-top: 2px; }
.zones { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }
.zone { padding: 6px 14px; border-radius: 999px; border: 1px solid rgba(105,150,185,.25);
        color: #6f91ad; font-size: 13px; font-weight: 700; }
.zone.on { border-color: #55cfff; color: #55cfff; background: rgba(85,207,255,.08); }
.slider { position: relative; height: 8px; border-radius: 99px; margin: 18px 4px 8px;
          background: linear-gradient(90deg, #ff6fae, #55cfff 50%, #ff9b60); }
.slider .knob { position: absolute; top: -8px; width: 24px; height: 24px; border-radius: 50%;
                background: #55cfff; border: 3px solid #07213f; transform: translateX(-50%); }
.slider-labels { display: flex; justify-content: space-between; color: #7d9ab7; font-size: 11px; }
.lv { padding: 10px 0; }
.lv + .lv { border-top: 1px solid rgba(105,150,185,.12); }
.lv-top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.lv-name { color: #9db8cf; font-size: 13px; }
.lv-price { color: #f5fbff; font-size: 26px; font-weight: 800; white-space: nowrap; }
.lv-sub { color: #6f91ad; font-size: 11px; line-height: 1.7; }
.lv-right { text-align: right; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 6px; font-size: 11px; font-weight: 800; margin-right: 6px; }
.tag.s4 { background: rgba(255,111,111,.15); color: #ff8a8a; }
.tag.s3 { background: rgba(255,155,96,.15); color: #ff9b60; }
.tag.s2 { background: rgba(255,209,102,.15); color: #ffd166; }
.tag.s1 { background: rgba(125,154,183,.15); color: #9db8cf; }
.rule { display: flex; justify-content: space-between; font-size: 13px; color: #b9cfe2; padding: 3px 0; }
.rule .p { font-weight: 800; }
.rule .p.pos { color: #4ee3d0; } .rule .p.neg { color: #ff8a8a; } .rule .p.zero { color: #7d9ab7; }
.kv { display: flex; justify-content: space-between; font-size: 13px; color: #9db8cf; padding: 4px 0; }
.kv b { color: #edf7ff; }
</style>
"""


def _sv(v):
    return "—" if v is None else v


def _e(s):
    return html.escape(str(s))


def _tag(g):
    k = {"極強": "s4", "強": "s3", "中": "s2", "弱": "s1"}[g["strength"]]
    return f'<span class="tag {k}">{g["strength"]}｜{g["n"]} 招共振</span>'


def _level(label, g, extra=""):
    if not g:
        return (f'<div class="lv"><div class="lv-top"><div class="lv-name">{_e(label)}</div>'
                f'<div class="lv-right"><div class="lv-price">尚未形成</div></div></div>'
                f'<div class="lv-sub">{_e(extra or "目前找不到對應的結構價位。")}</div></div>')
    band = f'價格帶 {fmt_price(g["lo"])}–{fmt_price(g["hi"])}' if g["hi"] > g["lo"] else ""
    return (f'<div class="lv"><div class="lv-top">'
            f'<div><div class="lv-name">{_e(label)}</div>'
            f'<div class="lv-sub">{_tag(g)}{_e("／".join(g["sources"]))}</div></div>'
            f'<div class="lv-right"><div class="lv-price">{fmt_price(g["price"])}</div>'
            f'<div class="lv-sub">{band}<br>距現價 {fmt_pct(g["dist"])}</div></div></div>'
            f'{f"<div class=lv-sub>{_e(extra)}</div>" if extra else ""}</div>')


def _rules_html(rules):
    if not rules:
        return '<div class="rule">資料不足</div>'
    out = []
    for txt, p in rules:
        cls = "pos" if p > 0 else "neg" if p < 0 else "zero"
        out.append(f'<div class="rule"><span>{_e(txt)}</span><span class="p {cls}">{p:+d}</span></div>')
    return "".join(out)


def render(ins, name):
    st.markdown(CSS, unsafe_allow_html=True)
    s, sc = ins["struct"], ins["scores"]
    price = ins["price"]

    # ---------- 資料日期 ----------
    dt = "　".join(f"{k}：{v:%Y/%m/%d}" for k, v in ins["dates"].items() if v)
    if ins["rev"] and ins["rev"].get("ym"):
        dt += f"　月營收：{ins['rev']['ym']}（民國年月）"
    st.markdown(f'<div class="ins-dates">{_e(dt)}</div>', unsafe_allow_html=True)

    # ---------- 綜合評分 ----------
    bars = []
    for k in WEIGHTS:
        v = sc[k][0]
        w = v if v is not None else 0
        bars.append(f'<div class="bar-row"><span class="name">{k}</span><div class="track">'
                    f'<div class="fill" style="width:{w}%"></div></div>'
                    f'<span class="val">{v if v is not None else "—"}</span></div>')
    total = ins["total"]
    missing = [k for k in WEIGHTS if sc[k][0] is None]
    miss_note = (f"<br>⚠️ {'、'.join(missing)}資料暫時抓不到，總分只用其他面向計算。" if missing else "")
    st.markdown(f"""
    <div class="ins-card">
      <div class="ins-kicker">RADAR SCORE</div>
      <div class="score-wrap">
        <div><div class="score-big">{total if total is not None else "—"}<small>/ 100</small></div>
             <div class="score-grade">{grade(total)}</div></div>
        <div style="flex:1">{''.join(bars)}</div>
      </div>
      <div class="ins-dates">五個面向各從 50 分起算、依規則加減；總分依權重平均
      （趨勢 25%、量價 15%、法人 20%、籌碼 15%、基本面 25%）。每一分怎麼來的，在下方「研究室」展開看。{miss_note}</div>
    </div>""", unsafe_allow_html=True)

    # ---------- 重點整理 ----------
    pts = []
    s1 = s["s1"]
    p_first = s["p1"] or s["mid_p"]
    a = f"距支撐 {fmt_price(s1['price'])}（{fmt_pct(s1['dist'])}）" if s1 else "下方支撐尚未形成"
    b = (f"距壓力 {fmt_price(p_first['price'])}（{fmt_pct(p_first['dist'])}）" if p_first
         else f"上方無歷史壓力，漲停參考價 {fmt_price(s['limit_up'])}")
    pts.append(f"{a}、{b}")
    m20 = ins["ma"][20]
    line = f"量能為近 20 日均量的 {ins['vol_ratio']:.2f} 倍" if np.isfinite(ins["vol_ratio"]) else "量能資料不足"
    if np.isfinite(ins["rsi"]):
        line += f"，RSI(14) {ins['rsi']:.1f}"
    if np.isfinite(m20):
        line += f"，收盤{'高' if price > m20 else '低'}於 20 日均線 {fmt_price(m20)}"
    pts.append(line)
    rv, vl = ins["rev"], ins["val"]
    f_parts = []
    if rv and np.isfinite(rv["yoy"]):
        t = f"{rv.get('ym', '')} 月營收年增 {rv['yoy']:+.2f}%"
        if np.isfinite(rv.get("mom", np.nan)):
            t += f"、月增 {rv['mom']:+.2f}%"
        f_parts.append(t)
    if vl:
        v = []
        if np.isfinite(vl["pe"]) and vl["pe"] > 0:
            v.append(f"本益比 {vl['pe']:.2f}")
        if np.isfinite(vl["yield"]):
            v.append(f"殖利率 {vl['yield']:.2f}%")
        if v:
            f_parts.append("、".join(v))
    if f_parts:
        pts.append("，".join(f_parts))
    have = {k: v[0] for k, v in sc.items() if v[0] is not None}
    if have:
        strong = [k for k, v in have.items() if v >= 60]
        weak = [k for k, v in have.items() if v < 45]
        t = f"綜合評分 {total}（{grade(total)}）"
        if strong:
            t += "；" + "、".join(f"{k}（{have[k]}）" for k in strong) + " 較強"
        if weak:
            t += "；" + "、".join(f"{k}（{have[k]}）" for k in weak) + " 偏低"
        pts.append(t)
    if p_first:
        pts.append(f"壓力共振強度「{p_first['strength']}」，依據：{'、'.join(p_first['sources'])}")
    st.markdown('<div class="ins-card"><div class="ins-kicker">KEY POINTS</div>'
                '<div class="ins-title">✨ 重點整理</div><ul class="ins-list">'
                + "".join(f"<li>{_e(x)}</li>" for x in pts) + "</ul>"
                '<div class="ins-dates">價位與評分是公式整理的參考，不是目標價或買賣建議。</div></div>',
                unsafe_allow_html=True)

    # ---------- 今天為什麼動 ----------
    whys = []
    chg = ins["chg"]
    if np.isfinite(chg):
        if chg > 0:
            whys.append(("#a87bff", "今日收紅", f"收盤 {fmt_price(price)}，{chg:+.2f}%。"))
        elif chg < 0:
            whys.append(("#4ee3d0", "今日收黑", f"收盤 {fmt_price(price)}，{chg:+.2f}%。"))
        else:
            whys.append(("#7d9ab7", "今日平盤", f"收盤 {fmt_price(price)}。"))
    vr = ins["vol_ratio"]
    if np.isfinite(vr):
        h = "成交量放大" if vr >= 1.5 else "成交量縮小" if vr < 0.8 else "成交量持平"
        whys.append(("#55cfff", h, f"成交量約為近 20 日均量的 {vr:.2f} 倍。"))
    fr = ins["foreign"]
    if fr and np.isfinite(fr["nets"][0]):
        f0 = fr["nets"][0] / 1000
        whys.append(("#ff6fae", "外資買超" if f0 > 0 else "外資賣超" if f0 < 0 else "外資持平",
                     f"外資（含陸資）合計 {abs(f0):,.0f} 張。"))
        t0 = fr.get("trust", [np.nan])[0]
        if np.isfinite(t0) and t0 != 0:
            whys.append(("#ffd166", "投信買超" if t0 > 0 else "投信賣超", f"投信合計 {abs(t0) / 1000:,.0f} 張。"))
    mg = ins["margin"]
    if mg and np.isfinite(mg["fin"] - mg["fin_prev"]) and mg["fin"] != mg["fin_prev"]:
        d = mg["fin"] - mg["fin_prev"]
        whys.append(("#ff9b60", "融資增加" if d > 0 else "融資減少", f"融資餘額 {d:+,.0f} 張，剩 {mg['fin']:,.0f} 張。"))
    st.markdown('<div class="ins-card"><div class="ins-kicker">今 天 為 什 麼 動 ？</div>'
                + "".join(f'<div class="why"><span class="dot" style="background:{c}"></span>'
                          f'<div><div class="h" style="color:{c}">{_e(h)}</div><div class="t">{_e(t)}</div></div></div>'
                          for c, h, t in whys) + "</div>", unsafe_allow_html=True)

    # ---------- 現在的位置 ----------
    zone, pos, top = position(ins)
    zs = "".join(f'<span class="zone{" on" if z == zone else ""}">{z}</span>'
                 for z in ("結構觀察區", "觀察區", "偏熱區"))
    desc = {"結構觀察區": "股價貼近有多種依據撐住的支撐區，是可以觀察的結構位置。",
            "觀察區": "股價在近 20 日低點到壓力之間，還沒到明確的支撐或壓力。",
            "偏熱區": "股價接近壓力或 RSI 偏高，短線位置偏熱。"}[zone]
    st.markdown(f"""
    <div class="ins-card"><div class="ins-kicker">現 在 的 位 置</div>
      <div class="ins-title">{_e(name)} 現在站在哪？</div>
      <div class="zones">{zs}</div>
      <div class="slider"><div class="knob" style="left:{pos * 100:.0f}%"></div></div>
      <div class="slider-labels"><span>近 20 日低點 {fmt_price(round_tick(ins['lo20']))}</span>
        <span>{'壓力' if (s['p1'] or s['mid_p']) else '漲停參考'} {fmt_price(top)}</span></div>
      <div class="ins-title" style="font-size:16px;color:#55cfff;margin-top:14px">
        以近 20 日低點到壓力區間衡量，價格位於區間約 {pos * 100:.0f}% 位置</div>
      <div class="ins-dates">{_e(desc)} 只描述位置，不代表買賣建議。</div>
    </div>""", unsafe_allow_html=True)

    # ---------- 支撐 ----------
    sup_html = (_level("近端支撐" if s["s1"] and s["s1"]["tier"] == "近端" else "第一支撐｜中期", s["s1"])
                + _level(f"第二支撐｜{s['s2']['tier']}" if s["s2"] else "第二支撐", s["s2"]))
    st.markdown(f'<div class="card support"><div class="card-title green-title">🛡️ 支撐觀察</div>{sup_html}'
                f'<div class="note">現價之下最接近的結構價位；跌破後結構轉弱。距現價超過 20% 的列在「長期結構」。</div></div>',
                unsafe_allow_html=True)

    # ---------- 壓力 ----------
    if s["p1"]:
        p_html = _level("⚡ 第一壓力", s["p1"], "現價上方 10% 以內最近的結構壓力。")
    elif s["mid_p"]:
        p_html = (_level("⚡ 第一壓力", None, "現價上方 10% 以內沒有結構壓力，近端留白。")
                  + _level(f"中期壓力｜距現價 {fmt_pct(s['mid_p']['dist'])}", s["mid_p"]))
    else:
        p_html = (f'<div class="lv"><div class="lv-top"><div><div class="lv-name">⚡ 漲停參考價</div>'
                  f'<div class="lv-sub">創新高、上方沒有歷史結構，以 10% 漲幅限制推算（非結構價位）</div></div>'
                  f'<div class="lv-right"><div class="lv-price">{fmt_price(s["limit_up"])}</div>'
                  f'<div class="lv-sub">距現價 {fmt_pct(pct(s["limit_up"], price))}</div></div></div></div>')
    if s["p2"]:
        p_html += _level(f"🔥 第二壓力{'｜' + s['p2']['tier'] if s['p2']['tier'] != '近端' else ''}",
                         s["p2"], "第一壓力帶之上、下一個獨立的結構壓力。")
    st.markdown(f'<div class="card pressure"><div class="card-title orange-title">🔵 壓力觀察</div>{p_html}</div>',
                unsafe_allow_html=True)

    # ---------- 更多結構價位 ----------
    more = []
    for g in s["others"]:
        more.append(_level("其他結構價位", g))
    for g in s["longer"]:
        more.append(_level(f"中長期結構價位（上方）", g))
    if s["long_sup"]:
        more.append(_level("長期結構價位（下方）", s["long_sup"]))
    if more:
        with st.expander("📐 更多結構價位"):
            st.markdown(f'<div class="ins-card" style="margin-top:0">{"".join(more)}'
                        f'<div class="ins-dates">招數＝這個價格帶同時符合幾種依據（波段高低點、均線、成交量密集區、'
                        f'爆量 K、跳空缺口）。4 招以上「極強」、3 招「強」、2 招「中」、1 招「弱」。'
                        f'價位依台股升降單位取整。</div></div>', unsafe_allow_html=True)

    # ---------- 研究室 ----------
    st.markdown('<div class="ins-kicker" style="margin-top:22px">RESEARCH</div>'
                '<div class="ins-title">🔬 研究室</div>', unsafe_allow_html=True)

    def kv(rows):
        return "".join(f'<div class="kv"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in rows)

    with st.expander(f"技術面（趨勢 {_sv(sc['趨勢面'][0])}／量價 {_sv(sc['量價'][0])}）"):
        mm = ins["ma"]
        st.markdown(kv([(f"MA{n}", fmt_price(mm[n])) for n in (5, 10, 20, 60, 120)]
                       + [("RSI(14)", f"{ins['rsi']:.1f}" if np.isfinite(ins['rsi']) else "—"),
                          ("量比（對 20 日均量）", f"{ins['vol_ratio']:.2f} 倍" if np.isfinite(ins['vol_ratio']) else "—"),
                          ("前 20 日最高", fmt_price(ins["hi20"]))])
                    + "<br><b style='color:#55cfff'>趨勢面評分規則</b>" + _rules_html(sc["趨勢面"][1])
                    + "<br><b style='color:#55cfff'>量價評分規則</b>" + _rules_html(sc["量價"][1]),
                    unsafe_allow_html=True)

    with st.expander(f"法人（{_sv(sc['法人'][0])}）"):
        fr = ins["foreign"]
        if fr and ins["fi_dates"]:
            tbl = pd.DataFrame({
                "日期": [d.strftime("%m/%d") for d in ins["fi_dates"]],
                "外資(張)": [round(x / 1000) if np.isfinite(x) else None for x in fr["nets"][:len(ins["fi_dates"])]],
                "投信(張)": [round(x / 1000) if np.isfinite(x) else None for x in fr.get("trust", [])[:len(ins["fi_dates"])]],
            })
            st.dataframe(tbl, hide_index=True, use_container_width=True)
        else:
            st.caption("抓不到法人資料。")
        st.markdown("<b style='color:#55cfff'>法人評分規則</b>" + _rules_html(sc["法人"][1]), unsafe_allow_html=True)

    with st.expander(f"籌碼（融資融券・借券）（{_sv(sc['籌碼'][0])}）"):
        mg, sb = ins["margin"], ins["sbl"]
        rows = []
        if mg:
            rows += [("融資餘額", f"{mg['fin']:,.0f} 張（{mg['fin'] - mg['fin_prev']:+,.0f}）"),
                     ("融券餘額", f"{mg['short']:,.0f} 張（{mg['short'] - mg['short_prev']:+,.0f}）")]
            if mg["fin"]:
                rows.append(("券資比", f"{mg['short'] / mg['fin'] * 100:.1f}%"))
        if sb:
            rows.append(("借券賣出餘額", f"{sb['today']:,.0f} 張（{sb['today'] - sb['prev']:+,.0f}）"))
        st.markdown((kv(rows) if rows else "<div class='kv'>抓不到籌碼資料</div>")
                    + "<br><b style='color:#55cfff'>籌碼評分規則</b>" + _rules_html(sc["籌碼"][1]),
                    unsafe_allow_html=True)

    with st.expander(f"基本面・營收（{_sv(sc['基本面'][0])}）"):
        rv, vl, ic = ins["rev"], ins["val"], ins["inc"]
        rows = []
        if rv:
            rows += [("營收月份", rv.get("ym") or "—"), ("月營收年增", fmt_pct(rv["yoy"], 2)),
                     ("月營收月增", fmt_pct(rv.get("mom", np.nan), 2))]
        if ic:
            rows += [("財報期間", f"{ic.get('period', '')}（年初至今累計）"),
                     ("毛利", f"{ic['gross'] / 1e5:,.2f} 億" if np.isfinite(ic["gross"]) else "—"),
                     ("淨利", f"{ic['net'] / 1e5:,.2f} 億" if np.isfinite(ic["net"]) else "—")]
        if vl:
            rows += [("本益比", f"{vl['pe']:.2f}" if np.isfinite(vl["pe"]) and vl["pe"] > 0 else "—（虧損或無資料）"),
                     ("殖利率", f"{vl['yield']:.2f}%" if np.isfinite(vl["yield"]) else "—"),
                     ("股價淨值比", f"{vl['pb']:.2f}" if np.isfinite(vl["pb"]) else "—")]
        st.markdown((kv(rows) if rows else "<div class='kv'>抓不到基本面資料</div>")
                    + "<br><b style='color:#55cfff'>基本面評分規則</b>" + _rules_html(sc["基本面"][1]),
                    unsafe_allow_html=True)
