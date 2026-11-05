# ============================================================
# 🧠 七位大師會診（依投資光譜分四區）
#
# 把七位投資人公開寫過的選股原則寫成規則，各自打 0～100 分：
#   深度價值  📘 葛拉漢：便宜＋財務安全（《智慧型股票投資人》防禦型投資人的條件）
#             📊 皮奧特洛斯基：F-score 九個訊號（2000 年論文），便宜股裡挑體質轉好的
#   品質價值  🎩 巴菲特：好公司＋合理價（buffett.py）
#             🧮 葛林布萊特：神奇公式，高資本報酬率＋高盈餘殖利率
#   合理價成長 🔍 彼得・林區：成長＋價格合理（PEG、公司分類、存貨警訊）
#   成長動能  🚀 歐尼爾：成長＋動能（CAN SLIM 七個字母）
#             📈 米奈爾維尼：趨勢樣板八條件（《超級績效》）
#
# 同一檔股票，不同流派常常意見不同——這正是要看的地方；同一區內意見一致才算「共識」。
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
def _div_rows(code):
    start = f"{datetime.now(TZ).year - 21}-01-01"      # 葛拉漢要 20 年配息紀錄
    return buffett.fetch_rows("TaiwanStockDividend", code, start)


def _dividends(code):
    """每年發的股利（現金＋股票，元／股），用除權息資料日期的年份分組"""
    return dividend_years(_div_rows(code))


def _div_num(d, *cols):
    out = pd.Series(0.0, index=d.index)
    for c in cols:
        if c in d:
            out = out + pd.to_numeric(d[c], errors="coerce").fillna(0)
    return out


def dividend_years(d):
    if d is None or d.empty or "date" not in d:
        return pd.Series(dtype=float)
    total = _div_num(d, "CashEarningsDistribution", "CashStatutorySurplus",
                     "StockEarningsDistribution", "StockStatutorySurplus")
    return total.groupby(pd.to_datetime(d["date"]).dt.year).sum()


def stock_dividend_years(d):
    """每年配的股票股利（元／股；面額 10 元，1 元＝配 10% 的股票）"""
    if d is None or d.empty or "date" not in d:
        return pd.Series(dtype=float)
    s = _div_num(d, "StockEarningsDistribution", "StockStatutorySurplus")
    return s.groupby(pd.to_datetime(d["date"]).dt.year).sum()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _industries():
    """{代號: 產業別}（FinMind TaiwanStockInfo，一次抓全部）"""
    d = buffett.fetch_rows("TaiwanStockInfo", "", "2000-01-01")
    if d.empty or "stock_id" not in d:
        return {}
    return dict(zip(d["stock_id"].astype(str), d["industry_category"].astype(str)))


def industry(code):
    try:
        return _industries().get(str(code), "")
    except Exception:
        return ""


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
        "op": _col(q, "OperatingIncome"),
        "gross": _col(q, "GrossProfit"),
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
        "ta": _col(b, "TotalAssets"),
        "cash": _col(b, "CashAndCashEquivalents"),
        "std": _col(b, "ShorttermBorrowings"),
        "ppe": _col(b, "PropertyPlantAndEquipment"),
        "cap": _col(b, "OrdinaryShare", "CapitalStock"),     # 普通股股本
    })
    # 長期負債＝應付公司債＋長期借款；兩欄都沒有才算缺資料
    lt = pd.concat([_col(b, "BondsPayable"), _col(b, "LongtermBorrowings")], axis=1)
    out["ltd"] = lt.sum(axis=1, min_count=1)
    return out.sort_index()


def year_end(b):
    """季資產負債表 → 每年 12/31 那一筆，index 是年份"""
    if b.empty:
        return b
    e = b[(b.index.month == 12) & (b.index.day == 31)].copy()
    e.index = e.index.year
    return e


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


def _finish(sections, cap=None, cap_note="", min_cover=MIN_COVER, **extra):
    have = {k: v for k, v in sections.items() if v[0]}
    full = sum(w for _, (_, w) in have.items())
    if full < min_cover:      # 能評的項目太少，分數會失真 → 不給分
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
# 🧮 葛林布萊特：神奇公式（《打敗大盤的獲利公式》The Little Book That Beats the Market）
#   資本報酬率 ROC ＝ EBIT ÷（淨營運資金＋淨固定資產）：好公司
#   盈餘殖利率 EY ＝ EBIT ÷ 企業價值（市值＋有息負債 − 多餘現金）：便宜
#   本人的做法是全市場兩項各自排名、名次相加，越小越好；不適用金融股、公用事業。
#   這裡沒有全市場排名，改用分段換成分數；6% 是他書中說的最低報酬門檻
#   （投資至少要贏 10 年期公債，公債低於 6% 時以 6% 計）。
# ------------------------------------------------------------

GB_HURDLE = 6.0


def _ttm(s):
    s = s.dropna()
    return float(s.tail(4).sum()) if len(s) >= 4 else np.nan


def greenblatt(q, b, price, fin, ind):
    if fin or ind in ("金融保險業", "油電燃氣業"):
        return {"score": None, "na": "金融股、公用事業不適用（葛林布萊特本人就把這兩類排除在外）"}
    s = {}
    ebit = _ttm(q["op"])          # 營業利益（近 4 季）當作 EBIT
    shares = _shares(q)
    last = b.iloc[-1] if not b.empty else None
    z = lambda v: float(v) if v is not None and np.isfinite(v) else 0.0
    ey = roc = np.nan

    # 1. 盈餘殖利率（50）
    r = []
    if np.isfinite(ebit) and np.isfinite(shares) and last is not None and np.isfinite(price):
        mcap = price * shares
        debt = z(last["std"]) + z(last["ltd"])
        cash = z(last["cash"])
        ev = mcap + debt - cash
        base = (f"近 4 季營業利益 {ebit / 1e8:,.0f} 億 ÷ 企業價值 {ev / 1e8:,.0f} 億"
                f"（市值 {mcap / 1e8:,.0f} ＋ 有息負債 {debt / 1e8:,.0f} − 現金 {cash / 1e8:,.0f}）")
        if ebit <= 0:
            r.append((f"近 4 季營業利益 {ebit / 1e8:,.0f} 億（虧損），不符合", 0))
        elif ev <= 0:
            r.append((base + "：現金比市值＋負債還多，盈餘殖利率極高", 50))
        else:
            ey = ebit / ev * 100
            pts = 50 if ey >= 15 else 35 if ey >= 10 else 20 if ey >= GB_HURDLE else 0
            r.append((base + f" = {ey:.1f}%（≥ 15% 很便宜；低於 {GB_HURDLE:.0f}% 不如買公債）", pts))
    s["盈餘殖利率（便宜）"] = (r, 50)

    # 2. 資本報酬率（50）
    r = []
    if np.isfinite(ebit) and last is not None and np.isfinite(last["ca"]) and np.isfinite(last["cl"]) \
            and np.isfinite(last["ppe"]):
        nwc = (last["ca"] - z(last["cash"])) - (last["cl"] - z(last["std"]))
        cap = nwc + last["ppe"]
        base = (f"近 4 季營業利益 {ebit / 1e8:,.0f} 億 ÷（淨營運資金 {nwc / 1e8:,.0f} 億 ＋ "
                f"不動產廠房設備 {last['ppe'] / 1e8:,.0f} 億）")
        if ebit <= 0:
            r.append((base + "：虧損，不符合", 0))
        elif cap <= 0:
            r.append((base + "：幾乎不用投入資本就能賺錢，資本報酬率極高", 50))
        else:
            roc = ebit / cap * 100
            pts = 50 if roc >= 50 else 35 if roc >= 25 else 20 if roc >= 15 else 0
            r.append((base + f" = {roc:.0f}%（≥ 50% 很好、≥ 25% 好）", pts))
    s["資本報酬率（好公司）"] = (r, 50)

    cheap_fail = bool(s["盈餘殖利率（便宜）"][0]) and np.isfinite(ey) and ey < GB_HURDLE
    return _finish(s, cap=40 if cheap_fail else None,
                   cap_note=f"盈餘殖利率低於 {GB_HURDLE:.0f}%：葛林布萊特說報酬贏不了公債就不值得買，所以總分最多 40。",
                   ey=ey, roc=roc)


# ------------------------------------------------------------
# 📊 皮奧特洛斯基：F-score（Piotroski 2000, Journal of Accounting Research）
#   九個 0/1 訊號，用最近一個完整年度和前一年比：
#   獲利 ROA>0、CFO>0、ΔROA>0、CFO>淨利；財務 長期負債比下降、流動比率上升、沒有發新股；
#   效率 毛利率上升、資產週轉率上升。ROA、CFO、週轉率都用「年初」總資產當分母。
#   原論文是在「股價淨值比低」的價值股裡用 F-score 挑出體質變好的。
# ------------------------------------------------------------

def piotroski(y, b, sdiv, fin):
    e = year_end(b)
    s = {"獲利能力": ([], 4), "財務結構": ([], 3), "營運效率": ([], 2)}
    yrs = [t for t in y.index if t in e.index and t - 1 in e.index]
    if not yrs:
        return None
    t = max(yrs)
    if t - 1 not in y.index:
        return None
    ta0, ta1 = e.loc[t - 1, "ta"], e.loc[t, "ta"]          # 年初、年底總資產
    ni, ni_p = y.loc[t, "ni"], y.loc[t - 1, "ni"]
    ocf = y.loc[t, "ocf"]
    na = []

    def sig(sec, ok, txt):
        if ok is None:
            return
        s[sec][0].append((("✔ " if ok else "✘ ") + txt, 1 if ok else 0))

    # ---- 獲利能力 ----
    roa = ni / ta0 * 100 if np.isfinite(ni) and np.isfinite(ta0) and ta0 > 0 else np.nan
    if np.isfinite(roa):
        sig("獲利能力", roa > 0, f"ROA（淨利 ÷ 年初總資產）{roa:.1f}% > 0")
    cfo = ocf / ta0 * 100 if np.isfinite(ocf) and np.isfinite(ta0) and ta0 > 0 else np.nan
    if np.isfinite(cfo):
        sig("獲利能力", cfo > 0, f"營業現金流 ÷ 年初總資產 {cfo:.1f}% > 0")
    if t - 2 in e.index and np.isfinite(roa):
        ta00 = e.loc[t - 2, "ta"]
        roa_p = ni_p / ta00 * 100 if np.isfinite(ni_p) and np.isfinite(ta00) and ta00 > 0 else np.nan
        if np.isfinite(roa_p):
            sig("獲利能力", roa > roa_p, f"ROA 比前一年進步：{roa_p:.1f}% → {roa:.1f}%")
    if np.isfinite(cfo) and np.isfinite(roa):
        sig("獲利能力", cfo > roa, f"營業現金流 {cfo:.1f}% > 淨利 {roa:.1f}%（獲利是真現金，應計項目少）")

    # ---- 財務結構（金融股不適用：存款、保單本來就是負債，也沒有流動資產這一項）----
    if fin:
        na = ["財務結構", "營運效率"]
    else:
        if t - 2 in e.index:
            ltd = lambda yr: e.loc[yr, "ltd"] if np.isfinite(e.loc[yr, "ltd"]) else 0.0
            avg = lambda yr: (e.loc[yr, "ta"] + e.loc[yr - 1, "ta"]) / 2
            if np.isfinite(avg(t)) and np.isfinite(avg(t - 1)):
                l1, l0 = ltd(t) / avg(t) * 100, ltd(t - 1) / avg(t - 1) * 100
                if l1 == 0 and l0 == 0:
                    sig("財務結構", True, "兩年都沒有長期負債（沒有增加槓桿）")
                else:
                    sig("財務結構", l1 < l0, f"長期負債 ÷ 平均總資產 下降：{l0:.1f}% → {l1:.1f}%")
        cr = lambda yr: e.loc[yr, "ca"] / e.loc[yr, "cl"] if e.loc[yr, "cl"] > 0 else np.nan
        c1, c0 = cr(t), cr(t - 1)
        if np.isfinite(c1) and np.isfinite(c0):
            sig("財務結構", c1 > c0, f"流動比率上升：{c0:.2f} → {c1:.2f}")
    k1, k0 = e.loc[t, "cap"], e.loc[t - 1, "cap"]
    if np.isfinite(k1) and np.isfinite(k0) and k0 > 0:
        g = (k1 / k0 - 1) * 100
        sd = float(sdiv.get(t, 0)) * 10        # 股票股利 1 元 ＝ 股本增加 10%
        issued = g > sd + 0.5
        txt = f"股本 {g:+.1f}%" + (f"（其中股票股利約 {sd:.1f}%，不算發新股）" if sd > 0 else "")
        if fin:
            s["財務結構"] = ([], 3)
            s["發行新股"] = ([], 1)
            sig("發行新股", not issued, ("沒有發行新股：" if not issued else "有發行新股（現增、轉換債或員工股）：") + txt)
        else:
            sig("財務結構", not issued, ("沒有發行新股：" if not issued else "有發行新股（現增、轉換債或員工股）：") + txt)

    # ---- 營運效率 ----
    if not fin:
        gm = y["gross"] / y["revenue"] * 100
        if np.isfinite(gm.get(t, np.nan)) and np.isfinite(gm.get(t - 1, np.nan)):
            sig("營運效率", gm[t] > gm[t - 1], f"毛利率上升：{gm[t - 1]:.1f}% → {gm[t]:.1f}%")
        if t - 2 in e.index:
            tu = lambda yr: y.loc[yr, "revenue"] / e.loc[yr - 1, "ta"] if yr in y.index else np.nan
            u1, u0 = tu(t), tu(t - 1)
            if np.isfinite(u1) and np.isfinite(u0):
                sig("營運效率", u1 > u0, f"資產週轉率（營收 ÷ 年初總資產）上升：{u0:.2f} → {u1:.2f}")

    n = sum(len(v[0]) for v in s.values())
    f = sum(p for v in s.values() for _, p in v[0])
    if n < 5:
        return None
    # 每個訊號等重，換成 0～100：F ÷ 能評的訊號數
    s = {k: (rules, len(rules) or w) for k, (rules, w) in s.items()}
    res = _finish(s, min_cover=0, fscore=f, n_signals=n, year=t)
    res["na_secs"] = na
    return res


# ------------------------------------------------------------
# 📈 米奈爾維尼：趨勢樣板（《超級績效》Trade Like a Stock Market Wizard 第 5 章）
#   八個條件都要符合，才算在「第二階段」上升趨勢，他才會考慮買：
#   1 股價 > 150 日與 200 日線  2 150 日線 > 200 日線  3 200 日線上揚至少 1 個月（最好 4～5 個月）
#   4 50 日線 > 150、200 日線  5 股價 > 50 日線  6 股價比 52 週低點高 30% 以上
#   7 股價距 52 週高點 25% 以內  8 相對強度（RS）排名 ≥ 70（最好 80、90 以上）
# ------------------------------------------------------------

MV_CAP = 70     # 不是 8 條全中就不算第二階段 → 總分最多 70


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _closes(day_str):
    """某一天上市＋上櫃全部股票收盤價 ({代號: 收盤}, 上櫃有沒有抓到)；抓不到丟錯誤"""
    import screener
    out, otc = {}, False
    try:
        p = screener._get_json("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
                               {"date": day_str, "type": "ALLBUT0999", "response": "json"})
        t = next((t for t in p.get("tables") or [] if (t.get("fields") or [""])[0] == "證券代號"), None)
        if t:
            i = t["fields"].index("收盤價")
            for r in t["data"]:
                out[screener._clean(r[0])] = screener._num(r[i])
    except Exception:
        pass
    try:
        d = datetime.strptime(day_str, "%Y%m%d")
        p = screener._get_json("https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
                               {"date": d.strftime("%Y/%m/%d"), "type": "EW", "response": "json"})
        t = (p.get("tables") or [{}])[0]
        if t.get("data"):
            i = t["fields"].index("收盤")
            for r in t["data"]:
                out[screener._clean(r[0])] = screener._num(r[i])
            otc = True
    except Exception:
        pass
    out = {k: v for k, v in out.items() if screener.STOCK_CODE.match(k) and np.isfinite(v) and v > 0}
    if not out:      # 還沒收盤或抓不到：丟錯誤，才不會把空結果快取 6 小時
        raise LookupError(f"{day_str} 沒有收盤資料")
    return out, otc


def rs_score(c0, c63, c126, c189, c252):
    """IBD 式相對強度：最近一季權重 40%，前三季各 20%"""
    return 0.4 * c0 / c63 + 0.2 * c0 / c126 + 0.2 * c0 / c189 + 0.2 * c0 / c252


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _rs_table(days):
    """days：(今天, 63, 126, 189, 252 個交易日前) 的日期字串 → {代號: RS 排名 1～99}、日期"""
    got = [_closes(d) for d in days]
    px = [g[0] for g in got]
    otc = all(g[1] for g in got)
    if any(len(p) < 500 for p in px):
        raise LookupError("全市場收盤價抓不齊")
    raw = {}
    for code, c0 in px[0].items():
        cs = [p.get(code) for p in px[1:]]
        if all(c is not None for c in cs):
            raw[code] = rs_score(c0, *cs)
    s = pd.Series(raw)
    rank = (s.rank(pct=True) * 99).clip(1, 99).round().astype(int)
    return rank.to_dict(), len(s), otc


def rs_table(mkt):
    """用大盤的交易日推出 0／63／126／189／252 天前是哪一天"""
    if mkt is None or len(mkt) < 253:
        return None
    idx = mkt.index
    for back in (0, 1):      # 盤中大盤已有今天的 K 線，但證交所收盤資料要收盤後才有 → 改用前一個交易日
        days = tuple(idx[-1 - back - n].strftime("%Y%m%d") for n in (0, 63, 126, 189, 252))
        try:
            rank, n, otc = _rs_table(days)
            return {"rank": rank, "n": n, "otc": otc, "date": idx[-1 - back]}
        except Exception:
            continue
    return None


def minervini(code, hist, price, rs):
    s = {}
    if hist is None or len(hist) < 260 or not np.isfinite(price):
        return None
    c = hist["Close"].dropna().copy()
    c.iloc[-1] = price
    ma = {n: c.rolling(n).mean() for n in (50, 150, 200)}
    m50, m150, m200 = (float(ma[n].iloc[-1]) for n in (50, 150, 200))
    yr = hist.tail(250)
    hi, lo = float(max(yr["High"].max(), price)), float(min(yr["Low"].min(), price))
    p = fmt_price

    def rule(name, ok, txt, w):
        s[name] = ([(("✔ " if ok else "✘ ") + txt, w if ok else 0)], w)

    rule("① 股價在長期均線上", price > m150 and price > m200,
         f"股價 {p(price)} vs 150 日線 {p(m150)}、200 日線 {p(m200)}（兩條都要在上方）", 13)
    rule("② 150 日線 > 200 日線", m150 > m200, f"150 日線 {p(m150)} vs 200 日線 {p(m200)}", 12)
    m200_1m, m200_5m = float(ma[200].iloc[-22]), float(ma[200].iloc[-106])
    up1, up5 = m200 > m200_1m, bool((ma[200].iloc[-106:].diff().dropna() >= 0).mean() >= 0.9)
    rule("③ 200 日線上揚", up1,
         f"200 日線 1 個月前 {p(m200_1m)} → 現在 {p(m200)}（至少上揚 1 個月）"
         + ("；近 5 個月都在往上，更好" if up1 and up5 else ""), 13)
    rule("④ 50 日線在最上面", m50 > m150 and m50 > m200,
         f"50 日線 {p(m50)} vs 150 日線 {p(m150)}、200 日線 {p(m200)}", 12)
    rule("⑤ 股價 > 50 日線", price > m50, f"股價 {p(price)} vs 50 日線 {p(m50)}", 12)
    above_lo = (price / lo - 1) * 100
    rule("⑥ 離 52 週低點夠遠", above_lo >= 30,
         f"比 52 週低點 {p(lo)} 高 {above_lo:.0f}%（標準 ≥ 30%，最強的常是 100% 以上）", 13)
    below_hi = (1 - price / hi) * 100
    rule("⑦ 接近 52 週高點", below_hi <= 25,
         f"距 52 週高點 {p(hi)} 還差 {below_hi:.0f}%（標準：25% 以內，越近越好）", 12)

    rk = (rs or {}).get("rank", {}).get(str(code))
    if rk is not None:
        rule("⑧ 相對強度 RS ≥ 70", rk >= 70,
             f"RS 排名 {rk}（在{'上市櫃' if rs.get('otc') else '上市'} {rs['n']} 檔股票中，"
             f"近一年加權漲幅勝過 {rk}% 的股票；最好 80、90 以上）", 13)
    else:
        s["⑧ 相對強度 RS ≥ 70"] = ([], 13)

    passed = sum(1 for rules, w in s.values() if rules and rules[0][1] > 0)
    known = sum(1 for rules, _ in s.values() if rules)
    all_ok = passed == known == 8
    why = (f"8 個條件只符合 {passed} 個" if passed < known
           else f"相對強度（RS）抓不到，無法確認 8 條全中")
    return _finish(s, cap=None if all_ok else MV_CAP,
                   cap_note=f"{why}：米奈爾維尼要全中才算第二階段上升趨勢，所以總分最多 {MV_CAP}。",
                   passed=passed, known=known, stage2=all_ok)


# ------------------------------------------------------------
# 全部算一次
# ------------------------------------------------------------

def build_all(code, price, hist, ins, data=None, divy=None, mkt=(None, None), rs=None, need=None):
    """回傳 {key: 結果 or None}, 財報最新季別。need：只算這幾位（選股時省 API 額度），None＝全部"""
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
    want = lambda k: need is None or k in need
    div_rows = None
    if divy is None and (want("graham") or want("piotroski")):
        try:
            div_rows = _div_rows(code)
        except Exception:
            div_rows = pd.DataFrame()
        divy = dividend_years(div_rows)
    if divy is None:
        divy = pd.Series(dtype=float)
    sdiv = stock_dividend_years(div_rows) if div_rows is not None else pd.Series(dtype=float)
    val = (ins or {}).get("val") or {}
    dy = val.get("yield", np.nan)
    rev = (ins or {}).get("rev")
    foreign = (ins or {}).get("foreign")
    if rs is None and want("minervini"):
        rs = rs_table(mkt[0])

    calc = {
        "buffett": lambda: buffett.evaluate(y, price, y10),
        "graham": lambda: graham(y10, q, b, divy, price, fin),
        "lynch": lambda: lynch(y, q, b, price, dy, rev, fin),
        "oneil": lambda: oneil(y, q, hist, price, foreign, mkt[0], mkt[1], fin),
        "greenblatt": lambda: greenblatt(q, b, price, fin, industry(code) if want("greenblatt") else ""),
        "piotroski": lambda: piotroski(y, b, sdiv, fin),
        "minervini": lambda: minervini(code, hist, price, rs),
    }
    res = {}
    for k, fn in calc.items():
        if not want(k):
            continue
        try:
            res[k] = fn()
        except Exception:          # 單一位大師算壞了，不要拖垮其他人
            res[k] = None
    if fin:      # 金融股：報表沒有的項目標成「金融股不適用」，不要寫「資料不足」
        for k, secs in (("graham", ("財務狀況",)), ("lynch", ("負債", "存貨"))):
            if res.get(k):
                res[k]["na_secs"] = secs
    last_q = q.index[-1] if len(q) else None
    return res, last_q


# ------------------------------------------------------------
# 畫面
# ------------------------------------------------------------

# 選股頁顯示的短名稱
SHORT = {"buffett": "巴菲特", "graham": "葛拉漢", "lynch": "林區", "oneil": "歐尼爾",
         "greenblatt": "葛林布萊特", "piotroski": "皮奧特洛斯基", "minervini": "米奈爾維尼"}

MASTERS = [
    ("graham", "📘 葛拉漢", "便宜＋財務安全"),
    ("piotroski", "📊 皮奧特洛斯基", "便宜股裡挑體質轉好"),
    ("buffett", "🎩 巴菲特", "好公司＋合理價"),
    ("greenblatt", "🧮 葛林布萊特", "高資本報酬＋便宜"),
    ("lynch", "🔍 彼得・林區", "成長＋價格合理"),
    ("oneil", "🚀 歐尼爾", "成長＋動能"),
    ("minervini", "📈 米奈爾維尼", "上升趨勢＋強勢"),
]
NAME = {k: n for k, n, _ in MASTERS}

# 光譜：從「只看便宜」到「只看趨勢」
ZONES = [
    ("deep", "深度價值", "買便宜、看財務安全", ["graham", "piotroski"]),
    ("quality", "品質價值", "好生意，價格合理", ["buffett", "greenblatt"]),
    ("garp", "合理價成長", "成長要看，價格也要看", ["lynch"]),
    ("momentum", "成長動能", "盈餘加速、股價強勢", ["oneil", "minervini"]),
]
ZONE_OF = {k: z for z, _, _, ks in ZONES for k in ks}


def score_of(r):
    return r.get("score") if r else None


def consensus(scores):
    """區內共識：分數 list（None＝沒評）→ (平均, 文字)"""
    have = [v for v in scores if v is not None]
    if not have:
        return None, "資料不足"
    avg = round(sum(have) / len(have))
    if len(have) == 1:
        return avg, ("只有一位大師，" + ("看好" if avg >= 50 else "不看好"))
    if all(v >= 50 for v in have):
        txt = "都看好"
    elif all(v < 50 for v in have):
        txt = "都不看好"
    else:
        txt = "意見分歧"
    if len(have) < len(scores):
        txt += f"（{len(scores) - len(have)} 位沒評）"
    return avg, txt


CSS = """
<style>
.z-head { display: flex; align-items: baseline; gap: 8px; margin: 14px 0 2px; }
.z-head:first-child { margin-top: 0; }
.z-head b { color: #55cfff; font-size: 13px; letter-spacing: .5px; }
.z-head span { color: #6f91ad; font-size: 11px; flex: 1; }
.z-head em { font-style: normal; color: #edf7ff; font-size: 12px; white-space: nowrap; }
.z-head em i { font-style: normal; color: #9db8cf; }
.z-bar { height: 3px; border-radius: 99px; margin-bottom: 4px;
         background: linear-gradient(90deg, #ffd166 0%, #4ee3d0 50%, #b48cff 100%); opacity: .35; }
.m-row { display: flex; align-items: center; gap: 10px; padding: 8px 0 8px 10px;
         border-left: 2px solid rgba(85,207,255,.25); }
.m-name { width: 116px; flex: none; }
.m-name b { color: #edf7ff; font-size: 14px; }
.m-name div { color: #6f91ad; font-size: 11px; margin-top: 2px; }
.m-track { flex: 1; height: 8px; border-radius: 99px; background: rgba(105,150,185,.18); overflow: hidden; }
.m-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #0878bd, #4ee3d0); }
.m-score { width: 30px; text-align: right; color: #fff; font-weight: 800; font-size: 18px; }
.m-verdict { width: 64px; text-align: right; color: #9db8cf; font-size: 11px; }
@media (max-width: 520px) { .m-name { width: 96px; } .m-verdict { width: 52px; } }
</style>
"""


def _row(key, name, style, r):
    sc = score_of(r)
    if key == "lynch" and r and r.get("category"):
        style = f"{style}｜{r['category']}"
    if key == "piotroski" and r:
        style = f"F-score {r['fscore']}／{r['n_signals']}"
    if key == "minervini" and r:
        style = f"趨勢樣板 {r['passed']}／8" + ("｜第二階段" if r.get("stage2") else "")
    na = r and r.get("na") and sc is None
    right = verdict(sc) if sc is not None else ("不適用" if na else "資料不足")
    return (f'<div class="m-row"><div class="m-name"><b>{_e(name)}</b><div>{_e(style)}</div></div>'
            f'<div class="m-track"><div class="m-fill" style="width:{sc or 0}%"></div></div>'
            f'<div class="m-score">{sc if sc is not None else "—"}</div>'
            f'<div class="m-verdict">{right}</div></div>')


def render(code, price, hist, ins):
    with st.spinner("七位大師讀財報中…"):
        res, last_q = build_all(code, price, hist, ins, mkt=market())

    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="ins-kicker" style="margin-top:22px">MASTERS</div>'
                '<div class="ins-title">🧠 七位大師怎麼看</div>', unsafe_allow_html=True)

    if res is None:
        st.caption("抓不到足夠的年度財報（需要至少 3 個完整年度），或財報資料暫時無法連線。")
        return

    meta = {k: (n, st_) for k, n, st_ in MASTERS}
    rows, zone_avg = [], {}
    for z, zname, zdesc, keys in ZONES:
        avg, txt = consensus([score_of(res.get(k)) for k in keys])
        if avg is not None:
            zone_avg[zname] = avg
        rows.append(f'<div class="z-head"><b>{_e(zname)}</b><span>{_e(zdesc)}</span>'
                    f'<em>{avg if avg is not None else "—"} <i>{_e(txt)}</i></em></div>')
        for k in keys:
            n, style = meta[k]
            rows.append(_row(k, n, style, res.get(k)))

    summary = ""
    if len(zone_avg) >= 2:
        hi = max(zone_avg, key=zone_avg.get)
        lo = min(zone_avg, key=zone_avg.get)
        summary = (f"四個流派中，最看好這檔的是「{hi}」（區內平均 {zone_avg[hi]}），"
                   f"最不看好的是「{lo}」（{zone_avg[lo]}）。")
    q_txt = f"財報最新到 {last_q.year} Q{(last_q.month - 1) // 3 + 1}。" if last_q is not None else ""
    fin_txt = ("<br>金融股報表沒有毛利、流動資產這類項目，相關規則標成「金融股不適用」，分數只用適用的項目換算。"
               if res.get("buffett") and res["buffett"].get("is_fin") else "")
    st.markdown(f"""
    <div class="ins-card">
      <div class="z-bar"></div>
      {''.join(rows)}
      <div class="ins-dates">{_e(summary)}<br>由上到下是投資光譜：越上面越重視「便宜」，越下面越重視「趨勢」。
      每位大師的原則寫成規則各自打分（0～100），區內共識看同一流派的大師是否意見一致。{_e(q_txt)}
      這是公式化的參考，不代表他們本人會怎麼看，也不是買賣建議。{fin_txt}</div>
    </div>""", unsafe_allow_html=True)

    # 每一位的明細
    for key, name, _ in MASTERS:
        r = res.get(key)
        sc = score_of(r)
        if sc is not None:
            title = f"{name}：{sc} 分｜每一分怎麼來的"
        elif r and r.get("na"):
            title = f"{name}：不適用"
        else:
            title = f"{name}：資料不足"
        with st.expander(title):
            if sc is None:
                st.caption(r["na"] if r and r.get("na") else "這位大師需要的資料暫時抓不到。")
                continue
            if key == "buffett":
                buffett.detail(r)
                continue
            if r.get("cap_note"):
                st.caption(f"⚠️ {r['cap_note']}（未設上限時是 {r['raw_score']} 分）")
            buffett.rules_block(r["sections"], na=r.get("na_secs", ()))
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
            if key == "greenblatt":
                st.caption("神奇公式本人的用法是：全市場（排除金融、公用事業）依「資本報酬率」與「盈餘殖利率」各排一次名，"
                           "兩個名次相加最小的 20～30 檔一起買、持有一年再換。這裡沒有全市場排名，改用分段給分；"
                           "EBIT 用近 4 季營業利益，淨營運資金扣掉現金與短期借款，固定資產用不動產廠房設備淨額。")
            if key == "piotroski":
                st.caption(f"F-score 用 {r['year']} 年報和前一年比，九個訊號各 1 分，分數＝符合數 ÷ 能評的訊號數。"
                           "皮奧特洛斯基的原始研究是在「股價淨值比最低的 20%」價值股裡，用 F-score 8～9 分挑贏家、"
                           "0～1 分避開輸家；高本益比的成長股拿高分，意義比較小。")
            if key == "minervini":
                st.caption("相對強度 RS：仿 IBD 算法，近一季漲幅權重 40%、前三季各 20%，再和全市場上市櫃股票比排名"
                           "（用未還原股價，除權息會讓高配息股略為吃虧）。米奈爾維尼只買 8 條全中的股票，"
                           "並且要再等出現波動收縮（VCP）的買點，這張卡只檢查趨勢樣板。")
