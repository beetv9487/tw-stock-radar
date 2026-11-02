# ============================================================
# 🧠 四位大師會診
#
# 把四位投資人公開寫過的選股原則寫成規則，各自打 0～100 分：
#   🎩 巴菲特：好公司＋合理價（buffett.py）
#   📘 葛拉漢：便宜＋財務安全（《智慧型股票投資人》防禦型投資人的條件）
#   🔍 彼得・林區：成長＋價格合理（PEG、公司分類、存貨警訊）
#   🚀 歐尼爾：成長＋動能（CAN SLIM 七個字母）
#
# 同一檔股票，四個人常常意見不同——這正是要看的地方。
# 都是把原則公式化的參考，不代表本人的看法，也不構成投資建議。
# ============================================================

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import buffett
from buffett import _wide, _col, annual, verdict, eps_growth, NI_NAMES, MIN_COVER
from insights import _e, fmt_price

TZ = timezone(timedelta(hours=8))


# ------------------------------------------------------------
# 資料
# ------------------------------------------------------------

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _dividends(code):
    """每年發的股利（現金＋股票，元／股），用除權息資料日期的年份分組"""
    start = f"{datetime.now(TZ).year - 21}-01-01"      # 葛拉漢要 20 年配息紀錄
    d = buffett.fetch_rows("TaiwanStockDividend", code, start)
    return dividend_years(d)


def dividend_years(d):
    if d is None or d.empty or "date" not in d:
        return pd.Series(dtype=float)
    num = lambda c: (pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d
                     else pd.Series(0.0, index=d.index))
    total = num("CashEarningsDistribution") + num("CashStatutorySurplus") \
        + num("StockEarningsDistribution") + num("StockStatutorySurplus")
    return total.groupby(pd.to_datetime(d["date"]).dt.year).sum()


# 大盤：加權指數，抓不到改用 0050
@st.cache_data(ttl=3600, show_spinner=False)
def _market():
    for sym, name in (("^TWII", "加權指數"), ("0050.TW", "0050")):
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1d", auto_adjust=False)
            if df is not None and len(df) > 210:
                return df.dropna(subset=["Close"]), name
        except Exception:
            pass
    raise LookupError("抓不到大盤資料")


def market():
    try:
        return _market()
    except Exception:
        return None, None


@st.cache_data(ttl=1800, show_spinner=False)
def load_hist(tickers):
    """選股用：('2330.TW', ...) → {ticker: 近 2 年日 K（含成交量）}"""
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        try:
            df = yf.download(batch, period="2y", interval="1d", group_by="ticker",
                             auto_adjust=False, threads=True, progress=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for t in batch:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(subset=["Close"])
                if len(sub) >= 60:
                    out[t] = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
            except KeyError:
                pass
    return out


def quarters(inc):
    q = _wide(inc)
    if q.empty:
        return pd.DataFrame()
    q.index = pd.to_datetime(q.index)
    return pd.DataFrame({
        "revenue": _col(q, "Revenue"),
        "ni": _col(q, *NI_NAMES),
        "eps": _col(q, "EPS"),
    }).sort_index()


def balance(bs):
    b = _wide(bs)
    if b.empty:
        return pd.DataFrame()
    b.index = pd.to_datetime(b.index)
    out = pd.DataFrame({
        "equity": _col(b, "EquityAttributableToOwnersOfParent", "Equity"),
        "ca": _col(b, "CurrentAssets"),
        "cl": _col(b, "CurrentLiabilities"),
        "inv": _col(b, "Inventories"),
    })
    # 長期負債＝應付公司債＋長期借款；兩欄都沒有才算缺資料
    lt = pd.concat([_col(b, "BondsPayable"), _col(b, "LongtermBorrowings")], axis=1)
    out["ltd"] = lt.sum(axis=1, min_count=1)
    return out.sort_index()


def _shares(q):
    """用最近一季「淨利 ÷ EPS」推算股數（EPS 太接近 0 的季跳過）"""
    for _, r in q.iloc[::-1].iterrows():
        if np.isfinite(r["eps"]) and abs(r["eps"]) >= 0.1 and np.isfinite(r["ni"]):
            s = r["ni"] / r["eps"]
            if s > 0:
                return float(s)
    return np.nan


def _ttm_eps(q):
    e = q["eps"].dropna()
    return float(e.tail(4).sum()) if len(e) >= 4 else np.nan


def _yoy_quarter(series):
    """最近一季 vs 去年同一季"""
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan, None
    last_d = s.index[-1]
    prev = s[(s.index.year == last_d.year - 1) & (s.index.month == last_d.month)]
    if prev.empty:
        return float(s.iloc[-1]), np.nan, last_d
    return float(s.iloc[-1]), float(prev.iloc[-1]), last_d


def _growth(now, before):
    if not (np.isfinite(now) and np.isfinite(before)) or before <= 0:
        return np.nan
    return (now / before - 1) * 100


def is_financial(y):
    return bool(y["gross"].isna().all() and (y["de"].dropna() > 5).any())


def _finish(sections, cap=None, cap_note="", **extra):
    have = {k: v for k, v in sections.items() if v[0]}
    full = sum(w for _, (_, w) in have.items())
    if full < MIN_COVER:      # 能評的項目太少，分數會失真 → 不給分
        return None
    got = sum(p for _, (rules, _) in have.items() for _, p in rules)
    score = int(round(got / full * 100))
    capped = cap is not None and score > cap
    return {"score": min(score, cap) if capped else score, "raw_score": score, "sections": sections,
            "cap_note": cap_note if capped else "", **extra}


# ------------------------------------------------------------
# 📘 葛拉漢：防禦型投資人的七個條件（《智慧型股票投資人》第 14 章）
#   10 年年年獲利、20 年不中斷配息、10 年盈餘成長 1/3 以上（頭尾各取 3 年平均）、
#   本益比 ≤ 15（近 3 年平均 EPS）、淨值比 ≤ 1.5 或 本益比×淨值比 ≤ 22.5
# ------------------------------------------------------------

DIV_DATA_START = 2010      # FinMind 股利資料 2009 年前有缺漏（例如台積電 2009 年有配息卻查不到），從 2010 起算

def graham(y, q, b, divy, price, fin):
    s = {}
    eps = y["eps"].dropna()
    eps3 = eps.tail(3)
    avg3 = float(eps3.mean()) if len(eps3) == 3 else np.nan
    shares = _shares(q)
    eq = b["equity"].dropna() if not b.empty else pd.Series(dtype=float)
    bvps = float(eq.iloc[-1] / shares) if len(eq) and np.isfinite(shares) else np.nan

    # 1. 本益比（20）：股價 ÷ 近 3 年平均 EPS ≤ 15
    r, pe = [], np.nan
    if np.isfinite(avg3):
        if avg3 <= 0:
            r.append((f"近 3 年平均 EPS {avg3:.2f} 元（虧損），不符合", 0))
        else:
            pe = price / avg3
            r.append((f"股價 ÷ 近 3 年平均 EPS = {pe:.1f} 倍（標準 ≤ 15）", 20 if pe <= 15 else 8 if pe <= 20 else 0))
    s["本益比"] = (r, 20)

    # 2. 股價淨值比（20）：≤ 1.5，或 本益比 × 淨值比 ≤ 22.5（本益比夠低時，淨值比可以高一點；兩者過一個就算過）
    r, pb = [], np.nan
    if np.isfinite(bvps) and bvps > 0:
        pb = price / bvps
        prod = pe * pb if np.isfinite(pe) else np.nan
        ok = pb <= 1.5 or (np.isfinite(prod) and prod <= 22.5)
        txt = f"股價淨值比 {pb:.2f} 倍（≤ 1.5）"
        if np.isfinite(prod):
            txt += f"、本益比 × 淨值比 {prod:.1f}（≤ 22.5）"
        r.append((txt + "：兩者過一個就算過", 20 if ok else 0))
    elif np.isfinite(bvps):
        r.append(("每股淨值為負，不符合", 0))
    s["股價淨值比"] = (r, 20)

    # 3. 財務狀況（20）：流動比率 ≥ 2、長期負債 ≤ 營運資金（金融股不適用）
    r = []
    if not fin and not b.empty:
        last = b.iloc[-1]
        if np.isfinite(last["ca"]) and np.isfinite(last["cl"]) and last["cl"] > 0:
            cr = last["ca"] / last["cl"]
            r.append((f"流動比率 {cr:.2f}（標準 ≥ 2）", 10 if cr >= 2 else 5 if cr >= 1.5 else 0))
            wc = last["ca"] - last["cl"]
            ltd = last["ltd"] if np.isfinite(last["ltd"]) else 0.0
            r.append((f"長期負債 {ltd / 1e8:,.0f} 億 vs 營運資金 {wc / 1e8:,.0f} 億（標準：負債 ≤ 營運資金）",
                      10 if ltd <= wc else 0))
    s["財務狀況"] = (r, 20)

    # 4. 獲利穩定（15）：10 年年年賺錢
    r = []
    if len(eps) >= 3:
        pos = int((eps > 0).sum())
        short = "（資料只有這幾年）" if len(eps) < 10 else ""
        r.append((f"{eps.index[0]}–{eps.index[-1]} 共 {len(eps)} 年中 {pos} 年 EPS 為正{short}（標準：10 年都獲利）",
                  15 if pos == len(eps) else 0))
    s["獲利穩定"] = (r, 15)

    # 5. 股利紀錄（10）：20 年不中斷配息
    r = []
    if len(y):
        end = int(y.index.max())
        start = max(end - 19, DIV_DATA_START)
        yrs = list(range(start, end + 1))
        paid = sum(1 for yr in yrs if divy.get(yr, 0) > 0)
        note = f"（可靠的股利資料從 {DIV_DATA_START} 年起，只能看 {len(yrs)} 年）" if len(yrs) < 20 else ""
        r.append((f"{start}–{end} 年中 {paid} 年有發股利{note}（標準：20 年不中斷）",
                  10 if paid == len(yrs) else 5 if paid >= 0.8 * len(yrs) else 0))
    s["股利紀錄"] = (r, 10)

    # 6. 盈餘成長（10）：10 年間，最後 3 年平均 EPS 比最早 3 年平均多 1/3 以上
    r = []
    if len(eps) >= 6:
        a, z = eps.head(3).mean(), eps.tail(3).mean()
        g = _growth(z, a)
        span = f"{eps.index[0]}–{eps.index[2]} → {eps.index[-3]}–{eps.index[-1]}"
        if np.isfinite(g):
            gt = f"{g:+.0f}%" if g < 1000 else "超過 +1000%（早期幾乎沒賺錢）"
            r.append((f"3 年平均 EPS {span}：{gt}（標準 ≥ +33%）", 10 if g >= 33 else 5 if g > 0 else 0))
        else:
            r.append(("早期 3 年平均 EPS 為負，無法比較", 0))
    s["盈餘成長"] = (r, 10)

    # 7. 公司規模（5）：年營收 ≥ 100 億（台股版的「夠大」）
    r = []
    rev = y["revenue"].dropna()
    if len(rev):
        rv = rev.iloc[-1] / 1e8
        r.append((f"年營收 {rv:,.0f} 億（標準 ≥ 100 億）", 5 if rv >= 100 else 0))
    s["公司規模"] = (r, 5)

    gnum = np.sqrt(22.5 * avg3 * bvps) if np.isfinite(avg3) and np.isfinite(bvps) and avg3 > 0 and bvps > 0 else np.nan
    # 葛拉漢先看價格：本益比、淨值比都不及格，其他條件再好也不買 → 總分上限 40
    val_pts = sum(p for _, p in s["本益比"][0]) + sum(p for _, p in s["股價淨值比"][0])
    val_known = bool(s["本益比"][0] or s["股價淨值比"][0])
    return _finish(s, cap=40 if val_known and val_pts == 0 else None,
                   cap_note="本益比、股價淨值比都不及格：葛拉漢先看價格，太貴就不買，所以總分最多 40。",
                   graham_number=gnum, bvps=bvps, avg3=avg3)


# ------------------------------------------------------------
# 🔍 彼得・林區：公司分類＋PEG＋存貨警訊
# ------------------------------------------------------------

def lynch_category(eps, g):
    e = eps.dropna()
    if len(e) >= 3 and (e <= 0).any() and e.iloc[-1] > 0:
        return "轉機股"
    if len(e) >= 4 and e.mean() > 0 and e.std(ddof=0) / e.mean() > 0.6:
        return "景氣循環股"
    if not np.isfinite(g):
        return "無法分類"
    return "緩慢成長股" if g < 5 else "穩定成長股" if g < 15 else "快速成長股"


LYNCH_MAX_G = 25.0     # 林區：年成長超過 20～25% 很難長期維持


def lynch(y, q, b, price, dy, rev, fin):
    s = {}
    eps = y["eps"].dropna()
    g = eps_growth(eps.tail(6)) * 100       # 對數迴歸，不受頭尾兩年影響；有虧損年度 → nan
    cat = lynch_category(eps, g)
    ttm = _ttm_eps(q)
    pe = price / ttm if np.isfinite(ttm) and ttm > 0 else np.nan
    dy = dy if np.isfinite(dy) else 0.0

    # 1. PEG（40）：林區的「（長期成長率＋殖利率）÷ 本益比」：< 1 差、1.5 可以、≥ 2 才是要找的
    r, pegy = [], np.nan
    if not np.isfinite(ttm):
        pass
    elif ttm <= 0:
        r.append(("近 4 季 EPS 為負，PEG 不適用", 0))
    elif not np.isfinite(g):
        r.append((f"本益比 {pe:.1f} 倍；近幾年有虧損年度，算不出穩定的成長率，PEG 不適用", 0))
    elif g + dy <= 0:
        r.append((f"本益比 {pe:.1f} 倍，但盈餘沒有成長，PEG 不適用", 0))
    else:
        g_used = min(g, LYNCH_MAX_G)
        ratio = (g_used + dy) / pe
        pegy = 1 / ratio
        pts = 40 if ratio >= 2 else 30 if ratio >= 1.5 else 15 if ratio >= 1 else 0
        note = f"（以 {LYNCH_MAX_G:.0f}% 計，太高的成長難以持續）" if g > LYNCH_MAX_G else ""
        r.append((f"（EPS 年成長 {g:.1f}%{note}＋殖利率 {dy:.1f}%）÷ 本益比 {pe:.1f} = {ratio:.2f}"
                  f"（< 1 差、1.5 可以、≥ 2 好）", pts))
        if cat == "景氣循環股":
            r.append(("景氣循環股：本益比低的時候常是景氣高點，林區提醒別被騙，PEG 分數減半", -(pts // 2)))
    s["PEG"] = (r, 40)

    # 2. 成長穩定（15）
    r = []
    if len(eps) >= 3:
        ups = int((eps.diff().dropna() > 0).sum())
        n = len(eps) - 1
        r.append((f"EPS 比前一年成長的年數：{ups}／{n}", 15 if ups >= 0.6 * n else 5 if ups >= 0.4 * n else 0))
    s["成長穩定"] = (r, 15)

    # 3. 負債（15）：長期負債 ÷ 股東權益
    r = []
    if not fin and not b.empty:
        last = b.iloc[-1]
        ltd = last["ltd"] if np.isfinite(last["ltd"]) else 0.0
        if np.isfinite(last["equity"]) and last["equity"] > 0:
            d = ltd / last["equity"]
            r.append((f"長期負債 ÷ 股東權益 {d:.2f}（≤ 0.25 很好、≤ 0.5 可以）", 15 if d <= 0.25 else 8 if d <= 0.5 else 0))
    s["負債"] = (r, 15)

    # 4. 存貨（15）：存貨成長比營收快＝警訊
    r = []
    if not fin and not b.empty:
        inv_now, inv_prev, d = _yoy_quarter(b["inv"])
        rev_now, rev_prev, _ = _yoy_quarter(q["revenue"])
        ig, rg = _growth(inv_now, inv_prev), _growth(rev_now, rev_prev)
        if np.isfinite(ig) and np.isfinite(rg) and inv_now > 0:
            bad = ig > rg + 10
            r.append((f"存貨年增 {ig:+.0f}% vs 單季營收年增 {rg:+.0f}%"
                      + ("（存貨堆得比賣得快，林區的警訊）" if bad else ""), 0 if bad else 15))
    s["存貨"] = (r, 15)

    # 5. 最新月營收（15）：台股每月公布，最即時的成長證據
    r = []
    if rev and np.isfinite(rev.get("yoy", np.nan)):
        yy = rev["yoy"]
        r.append((f"{rev.get('ym', '')} 月營收年增 {yy:+.1f}%", 15 if yy >= 10 else 8 if yy > 0 else 0))
    s["月營收"] = (r, 15)

    return _finish(s, category=cat, pe=pe, g=g, pegy=pegy)


# ------------------------------------------------------------
# 🚀 歐尼爾：CAN SLIM
# ------------------------------------------------------------

def _ret(close, days=250):
    c = close.dropna()
    if len(c) <= days:
        return np.nan
    return (float(c.iloc[-1]) / float(c.iloc[-days - 1]) - 1) * 100


def oneil(y, q, hist, price, foreign, mkt, mkt_name, fin=False):
    s = {}

    # C 當季盈餘（20）：最近一季 EPS 年增 ≥ 25%、營收年增 ≥ 25%
    r = []
    e_now, e_prev, d = _yoy_quarter(q["eps"])
    if np.isfinite(e_now) and np.isfinite(e_prev):
        qlabel = f"{d.year} Q{(d.month - 1) // 3 + 1}"
        if e_prev <= 0 < e_now:
            r.append((f"{qlabel} EPS {e_now:.2f}，去年同季 {e_prev:.2f}：轉虧為盈", 8))
        else:
            gq = _growth(e_now, e_prev)
            if np.isfinite(gq):
                r.append((f"{qlabel} EPS 年增 {gq:+.0f}%（標準 ≥ 25%）", 12 if gq >= 25 else 4 if gq > 0 else 0))
            else:
                r.append((f"{qlabel} EPS {e_now:.2f}，仍在虧損", 0))
        rn, rp, _ = _yoy_quarter(q["revenue"])
        rg = _growth(rn, rp)
        if np.isfinite(rg) and not fin:
            r.append((f"{qlabel} 營收年增 {rg:+.0f}%（標準 ≥ 25%）", 8 if rg >= 25 else 4 if rg >= 10 else 0))
    s["C 當季盈餘"] = (r, 20)

    # A 年度盈餘（20）：近 3 年 EPS 每年成長 ≥ 25%、ROE ≥ 17%
    r = []
    eps = y["eps"].dropna()
    if len(eps) >= 4:
        last3 = [(_growth(eps.iloc[i], eps.iloc[i - 1])) for i in range(len(eps) - 3, len(eps))]
        good = sum(1 for g in last3 if np.isfinite(g) and g >= 25)
        txt = "、".join(f"{g:+.0f}%" if np.isfinite(g) else "—" for g in last3)
        r.append((f"近 3 年 EPS 年增 {txt}（標準：每年 ≥ 25%）", 12 if good == 3 else 6 if good == 2 else 0))
    roe = y["roe"].dropna()
    if len(roe):
        r.append((f"最近一年 ROE {roe.iloc[-1]:.1f}%（標準 ≥ 17%）", 8 if roe.iloc[-1] >= 17 else 0))
    s["A 年度盈餘"] = (r, 20)

    # N 新高（15）：股價接近一年高點
    r = []
    if hist is not None and len(hist) >= 120:
        hi = float(hist["High"].tail(250).max())
        gap = (price / hi - 1) * 100
        txt = (f"創一年新高（前高 {fmt_price(hi)}）" if gap >= 0
               else f"比一年最高價 {fmt_price(hi)} 低 {abs(gap):.1f}%（越接近新高越符合）")
        r.append((txt, 15 if gap >= -5 else 8 if gap >= -15 else 0))
    s["N 新高"] = (r, 15)

    # S 供需（10）：近 50 天上漲日的成交量 vs 下跌日
    r = []
    if hist is not None and len(hist) >= 51:
        h = hist.tail(51)
        chg = h["Close"].diff().iloc[1:]
        vol = h["Volume"].iloc[1:]
        up, dn = float(vol[chg > 0].sum()), float(vol[chg < 0].sum())
        if dn > 0:
            ud = up / dn
            r.append((f"近 50 天上漲日成交量 ÷ 下跌日成交量 = {ud:.2f}（> 1 代表買盤較強）",
                      10 if ud >= 1.2 else 5 if ud >= 1 else 0))
    s["S 供需"] = (r, 10)

    # L 領導股（15）：近一年漲幅贏大盤多少
    r = []
    if hist is not None and mkt is not None:
        sr, mr = _ret(hist["Close"]), _ret(mkt["Close"])
        if np.isfinite(sr) and np.isfinite(mr):
            diff = sr - mr
            r.append((f"近一年漲幅 {sr:+.0f}% vs {mkt_name} {mr:+.0f}%（{'贏' if diff >= 0 else '輸'} {abs(diff):.0f} 個百分點）",
                      15 if diff >= 20 else 8 if diff > 0 else 0))
    s["L 領導股"] = (r, 15)

    # I 法人認同（10）：近幾天外資＋投信合計買超
    r = []
    if foreign:
        fo = np.nansum(foreign.get("nets", []))
        tr = np.nansum(foreign.get("trust", []))
        tot = (fo + tr) / 1000
        n = len(foreign.get("nets", []))
        r.append((f"近 {n} 日外資＋投信合計{'買' if tot > 0 else '賣'}超 {abs(tot):,.0f} 張", 10 if tot > 0 else 0))
    s["I 法人認同"] = (r, 10)

    # M 大盤方向（10）：大盤在 50 日線之上、50 日線往上
    r = []
    if mkt is not None and len(mkt) >= 210:
        c = mkt["Close"]
        m50, m200 = c.rolling(50).mean(), c.rolling(200).mean()
        above50 = c.iloc[-1] > m50.iloc[-1]
        up50 = m50.iloc[-1] > m50.iloc[-11]
        above200 = c.iloc[-1] > m200.iloc[-1]
        if above50 and up50:
            r.append((f"{mkt_name}在 50 日線之上、50 日線往上：上升趨勢", 10))
        elif above200:
            r.append((f"{mkt_name}在 200 日線之上，但短線轉弱", 5))
        else:
            r.append((f"{mkt_name}在 200 日線之下：歐尼爾建議這時少買", 0))
    s["M 大盤方向"] = (r, 10)

    return _finish(s)


# ------------------------------------------------------------
# 全部算一次
# ------------------------------------------------------------

def build_all(code, price, hist, ins, data=None, divy=None, mkt=(None, None)):
    """回傳 {key: 結果 or None}, 財報最新季別"""
    try:
        inc, bs, cf = data if data is not None else buffett._load(code)
    except Exception:
        return None, None
    y_all = annual(inc, bs, cf)
    if len(y_all) < 3:
        return None, None
    y = y_all.tail(6)
    y10 = y_all.tail(10)
    q, b = quarters(inc), balance(bs)
    fin = is_financial(y)
    if divy is None:
        try:
            divy = _dividends(code)
        except Exception:
            divy = pd.Series(dtype=float)
    val = (ins or {}).get("val") or {}
    dy = val.get("yield", np.nan)
    rev = (ins or {}).get("rev")
    foreign = (ins or {}).get("foreign")

    res = {
        "buffett": buffett.evaluate(y, price, y10),
        "graham": graham(y10, q, b, divy, price, fin),
        "lynch": lynch(y, q, b, price, dy, rev, fin),
        "oneil": oneil(y, q, hist, price, foreign, mkt[0], mkt[1], fin),
    }
    last_q = q.index[-1] if len(q) else None
    return res, last_q


# ------------------------------------------------------------
# 畫面
# ------------------------------------------------------------

# 選股頁顯示的短名稱
SHORT = {"buffett": "巴菲特", "graham": "葛拉漢", "lynch": "林區", "oneil": "歐尼爾"}

MASTERS = [
    ("buffett", "🎩 巴菲特", "好公司＋合理價"),
    ("graham", "📘 葛拉漢", "便宜＋財務安全"),
    ("lynch", "🔍 彼得・林區", "成長＋價格合理"),
    ("oneil", "🚀 歐尼爾", "成長＋動能"),
]

CSS = """
<style>
.m-row { display: flex; align-items: center; gap: 10px; padding: 10px 0; }
.m-row + .m-row { border-top: 1px solid rgba(105,150,185,.12); }
.m-name { width: 104px; flex: none; }
.m-name b { color: #edf7ff; font-size: 14px; }
.m-name div { color: #6f91ad; font-size: 11px; margin-top: 2px; }
.m-track { flex: 1; height: 8px; border-radius: 99px; background: rgba(105,150,185,.18); overflow: hidden; }
.m-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #0878bd, #4ee3d0); }
.m-score { width: 30px; text-align: right; color: #fff; font-weight: 800; font-size: 18px; }
.m-verdict { width: 58px; text-align: right; color: #9db8cf; font-size: 11px; }
</style>
"""


def render(code, price, hist, ins):
    with st.spinner("四位大師讀財報中…"):
        res, last_q = build_all(code, price, hist, ins, mkt=market())

    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="ins-kicker" style="margin-top:22px">MASTERS</div>'
                '<div class="ins-title">🧠 四位大師怎麼看</div>', unsafe_allow_html=True)

    if res is None:
        st.caption("抓不到足夠的年度財報（需要至少 3 個完整年度），或財報資料暫時無法連線。")
        return

    rows, scored = [], {}
    for key, name, style in MASTERS:
        r = res.get(key)
        sc = r["score"] if r else None
        if key == "lynch" and r and r.get("category"):
            style = f"{style}｜{r['category']}"
        if sc is not None:
            scored[name] = sc
        rows.append(f'<div class="m-row"><div class="m-name"><b>{_e(name)}</b><div>{_e(style)}</div></div>'
                    f'<div class="m-track"><div class="m-fill" style="width:{sc or 0}%"></div></div>'
                    f'<div class="m-score">{sc if sc is not None else "—"}</div>'
                    f'<div class="m-verdict">{verdict(sc) if sc is not None else "資料不足"}</div></div>')

    summary = ""
    if len(scored) >= 2:
        hi = max(scored, key=scored.get)
        lo = min(scored, key=scored.get)
        n_ok = sum(1 for v in scored.values() if v >= 50)
        summary = (f"{len(scored)} 位中有 {n_ok} 位給到 50 分以上。"
                   f"最喜歡這檔的是{hi.split(' ')[1]}（{scored[hi]}），最不喜歡的是{lo.split(' ')[1]}（{scored[lo]}）。")
    q_txt = f"財報最新到 {last_q.year} Q{(last_q.month - 1) // 3 + 1}。" if last_q is not None else ""
    fin_txt = ("<br>金融股報表沒有毛利、流動資產這類項目，相關規則會略過，分數只用有資料的項目換算。"
               if res["buffett"] and res["buffett"].get("is_fin") else "")
    st.markdown(f"""
    <div class="ins-card">
      {''.join(rows)}
      <div class="ins-dates">{_e(summary)}<br>每位大師的原則寫成規則各自打分（0～100）。{_e(q_txt)}
      這是公式化的參考，不代表他們本人會怎麼看，也不是買賣建議。{fin_txt}</div>
    </div>""", unsafe_allow_html=True)

    # 每一位的明細
    for key, name, _ in MASTERS:
        r = res.get(key)
        title = f"{name}：{r['score']} 分｜每一分怎麼來的" if r else f"{name}：資料不足"
        with st.expander(title):
            if not r:
                st.caption("這位大師需要的資料暫時抓不到。")
                continue
            if key == "buffett":
                buffett.detail(r)
                continue
            if r.get("cap_note"):
                st.caption(f"⚠️ {r['cap_note']}（未設上限時是 {r['raw_score']} 分）")
            buffett.rules_block(r["sections"])
            if key == "graham" and np.isfinite(r.get("graham_number", np.nan)):
                gn = r["graham_number"]
                st.caption(f"葛拉漢數字 √(22.5 × 近 3 年平均 EPS {r['avg3']:.2f} × 每股淨值 {r['bvps']:.2f}) "
                           f"= {fmt_price(gn)}；股價低於這個數字，才同時符合本益比 15、淨值比 1.5 的上限。"
                           f"現價是它的 {price / gn:.1f} 倍。")
            if key == "lynch":
                st.caption({
                    "快速成長股": "快速成長股：林區最愛的類型，但要確認成長還能延續、價格沒有透支（看 PEG）。",
                    "穩定成長股": "穩定成長股：大型績優股，林區通常在漲 30～50% 時就換股，不期待十倍。",
                    "緩慢成長股": "緩慢成長股：成長慢，林區主要看殖利率，不期待股價大漲。",
                    "景氣循環股": "景氣循環股：獲利大起大落，買賣時機比本益比重要。",
                    "轉機股": "轉機股：曾經虧損、現在轉盈，要確認轉機是真的。",
                }.get(r.get("category"), ""))
            if key == "oneil":
                st.caption("CAN SLIM：C 當季盈餘、A 年度盈餘、N 新高、S 供需、L 領導股、I 法人認同、M 大盤方向。"
                           "歐尼爾也強調跌破買價 7～8% 就停損，這張卡只評選股條件。")
