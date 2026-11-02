# ============================================================
# 🎩 巴菲特檢查
#
# 把巴菲特公開談過的選股原則寫成可檢查的規則，套到台股：
#   1. 股東權益報酬率（ROE）長期夠高
#   2. 獲利穩定、每股盈餘（EPS）大致往上
#   3. 負債不高
#   4. 毛利率高又穩（護城河的痕跡）
#   5. 賺的是真現金（自由現金流）
#   6. 價格低於估算的內在價值（安全邊際）
#
# 資料：FinMind 的季度損益表、資產負債表、現金流量表（近 7 年）。
# 只用「完整 4 季」的年度；金融股的報表結構不同，只評 ROE 與獲利穩定。
# 這是把原則寫成公式的參考，不是巴菲特本人的看法，也不構成投資建議。
# ============================================================

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from screener import HTTP
from insights import _rules_html, _e, fmt_price

TZ = timezone(timedelta(hours=8))
API = "https://api.finmindtrade.com/api/v4/data"

# 內在價值估算的假設（保守版）
DISCOUNT = 0.10        # 折現率 10%
TERMINAL_G = 0.025     # 10 年後的永續成長率
MAX_G = 0.10           # 前 10 年成長率上限
YEARS = 10


# ------------------------------------------------------------
# 抓資料
# ------------------------------------------------------------

def _token():
    """有 FinMind token 就帶上（額度比較高）；沒有也能用免費額度"""
    try:
        t = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        t = ""
    return t or os.environ.get("FINMIND_TOKEN", "")


def fetch_rows(dataset, code, start):
    """FinMind 原始資料（所有欄位）"""
    params = {"dataset": dataset, "data_id": code, "start_date": start}
    headers = {}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    r = HTTP.get(API, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(j.get("msg") or "FinMind 回傳錯誤")
    return pd.DataFrame(j.get("data") or [])


def _fetch(dataset, code, start):
    df = fetch_rows(dataset, code, start)
    return df.reindex(columns=["date", "type", "value"])


# 抓不到時丟出錯誤，這樣失敗結果不會被快取住
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _load(code):
    start = f"{datetime.now(TZ).year - 7}-01-01"
    inc = _fetch("TaiwanStockFinancialStatements", code, start)
    if inc.empty:
        raise LookupError("沒有財報資料")
    bs = _fetch("TaiwanStockBalanceSheet", code, start)
    cf = _fetch("TaiwanStockCashFlowsStatement", code, start)
    return inc, bs, cf


# ------------------------------------------------------------
# 季報 → 年度數字
# ------------------------------------------------------------

def _wide(df):
    if df.empty:
        return pd.DataFrame()
    df = df[["date", "type", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.pivot_table(index="date", columns="type", values="value", aggfunc="last")


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(np.nan, index=df.index)


def annual(inc, bs, cf):
    """回傳每年一列的 DataFrame（只留完整 4 季的年度）"""
    q = _wide(inc)
    if q.empty:
        return pd.DataFrame()
    q.index = pd.to_datetime(q.index)
    q = pd.DataFrame({
        "revenue": _col(q, "Revenue"),
        "gross": _col(q, "GrossProfit"),
        "op": _col(q, "OperatingIncome"),
        "ni": _col(q, "EquityAttributableToOwnersOfParent", "IncomeAfterTaxes"),
        "eps": _col(q, "EPS"),
    })
    q["year"] = q.index.year
    counts = q.groupby("year").size()
    full = counts[counts == 4].index
    # min_count=4：任一季缺值，該年就算缺，不要把 3 季加總當全年
    y = q[q["year"].isin(full)].groupby("year").sum(min_count=4)

    # 資產負債表：取 12/31 的數字
    b = _wide(bs)
    if not b.empty:
        b.index = pd.to_datetime(b.index)
        b = b[(b.index.month == 12) & (b.index.day == 31)]
        b.index = b.index.year
        y["equity"] = _col(b, "EquityAttributableToOwnersOfParent", "Equity").reindex(y.index)
        y["liab"] = _col(b, "Liabilities").reindex(y.index)
    else:
        y["equity"] = y["liab"] = np.nan

    # 現金流量表是「年初累計」，所以 12/31 那筆就是全年
    c = _wide(cf)
    if not c.empty:
        c.index = pd.to_datetime(c.index)
        c = c[(c.index.month == 12) & (c.index.day == 31)]
        c.index = c.index.year
        y["ocf"] = _col(c, "CashFlowsFromOperatingActivities", "NetCashInflowFromOperatingActivities").reindex(y.index)
        y["capex"] = _col(c, "PropertyAndPlantAndEquipment").reindex(y.index)   # 負數＝買設備
    else:
        y["ocf"] = y["capex"] = np.nan

    y["fcf"] = y["ocf"] + y["capex"]
    y["roe"] = y["ni"] / y["equity"] * 100
    y["gm"] = y["gross"] / y["revenue"] * 100
    y["opm"] = y["op"] / y["revenue"] * 100
    y["de"] = y["liab"] / y["equity"]
    y["shares"] = y["ni"] / y["eps"]
    return y.sort_index()


# ------------------------------------------------------------
# 內在價值：自由現金流折現（保守版）
# ------------------------------------------------------------

def _cagr(first, last, years):
    if years <= 0 or not (np.isfinite(first) and np.isfinite(last)) or first <= 0 or last <= 0:
        return np.nan
    return (last / first) ** (1 / years) - 1


def intrinsic_value(y):
    """回傳 (每股內在價值, 說明dict)；資料不夠回傳 (nan, {})"""
    recent = y.tail(3)
    fcf = recent["fcf"].dropna()
    shares = y["shares"].dropna()
    if len(fcf) < 2 or shares.empty or shares.iloc[-1] <= 0:
        return np.nan, {}
    base = fcf.mean() / shares.iloc[-1]          # 近 3 年平均每股自由現金流
    if not np.isfinite(base) or base <= 0:
        return np.nan, {"base": base}
    eps = y["eps"].dropna()
    g = _cagr(eps.iloc[0], eps.iloc[-1], len(eps) - 1) if len(eps) >= 3 else np.nan
    cyclical = eps_cv(eps) > 0.6
    g_used = 0.0 if cyclical else float(min(max(g if np.isfinite(g) else 0.0, 0.0), MAX_G))

    pv, cash = 0.0, base
    for t in range(1, YEARS + 1):
        cash *= 1 + g_used
        pv += cash / (1 + DISCOUNT) ** t
    terminal = cash * (1 + TERMINAL_G) / (DISCOUNT - TERMINAL_G)
    pv += terminal / (1 + DISCOUNT) ** YEARS
    return float(pv), {"base": float(base), "g": g, "g_used": g_used, "cyclical": cyclical}


def eps_cv(eps):
    """EPS 波動：標準差 ÷ 平均（平均 ≤ 0 視為波動極大）"""
    e = eps.dropna()
    if len(e) < 3:
        return np.nan
    m = e.mean()
    return float(e.std(ddof=0) / m) if m > 0 else np.inf


# ------------------------------------------------------------
# 評分：六項加總 100 分，每一分都列出規則
# ------------------------------------------------------------

def _pass(points, text):
    return (text, int(points))


def evaluate(y, price):
    is_fin = bool(y["gross"].isna().all() and (y["de"].dropna() > 5).any())
    sections = {}

    # 1. ROE（25 分）
    roe = y["roe"].dropna()
    r = []
    if len(roe):
        avg = roe.mean()
        good = int((roe >= 15).sum())
        frac = good / len(roe)
        pts = 25 if frac == 1 else 18 if frac >= 0.8 else 10 if frac >= 0.6 else 0
        r.append(_pass(pts, f"近 {len(roe)} 年有 {good} 年 ROE ≥ 15%（平均 {avg:.1f}%；標準：每年都要高）"))
    sections["ROE"] = (r, 25)

    # 2. 獲利穩定（15 分）
    eps = y["eps"].dropna()
    r = []
    if len(eps) >= 2:
        pos = int((eps > 0).sum())
        r.append(_pass(5 if pos == len(eps) else 0, f"{len(eps)} 年中 {pos} 年 EPS 為正"))
        ups = int((eps.diff().dropna() > 0).sum())
        need = int(np.ceil((len(eps) - 1) * 0.6))
        r.append(_pass(5 if ups >= need else 0, f"EPS 比前一年成長的年數：{ups}／{len(eps) - 1}"))
        cv = eps_cv(eps)
        if np.isfinite(cv):
            r.append(_pass(5 if cv <= 0.5 else 0,
                           f"EPS 波動（標準差÷平均）{cv:.2f}" + ("，獲利可預測" if cv <= 0.5 else "，大起大落不好預測")))
        elif len(eps) >= 3:
            r.append(_pass(0, "EPS 平均為負，獲利無法預測"))
    sections["獲利穩定"] = (r, 15)

    if is_fin:
        sections["負債"] = ([], 15)
        sections["護城河"] = ([], 15)
        sections["現金流"] = ([], 15)
        sections["安全邊際"] = ([], 15)
        iv, info = np.nan, {}
    else:
        # 3. 負債（15 分）
        de = y["de"].dropna()
        r = []
        if len(de):
            d = de.iloc[-1]
            r.append(_pass(15 if d < 0.5 else 8 if d < 1 else 0, f"負債／股東權益 {d:.2f} 倍"))
        sections["負債"] = (r, 15)

        # 4. 護城河：毛利率高又穩（15 分）
        gm = y["gm"].dropna()
        r = []
        if len(gm):
            avg, sd = gm.mean(), gm.std(ddof=0)
            r.append(_pass(10 if avg >= 40 else 5 if avg >= 25 else 0, f"平均毛利率 {avg:.1f}%"))
            if len(gm) >= 3:
                # 毛利率夠高（≥25%）又穩，才算護城河；低毛利或虧損時「穩」沒有意義
                r.append(_pass(5 if sd <= 3 and avg >= 25 else 0, f"毛利率波動 ±{sd:.1f} 個百分點"))
        sections["護城河"] = (r, 15)

        # 5. 現金流（15 分）
        f = y[["fcf", "ni"]].dropna()
        r = []
        if len(f):
            pos = int((f["fcf"] > 0).sum())
            r.append(_pass(10 if pos >= 0.8 * len(f) else 0, f"{len(f)} 年中 {pos} 年自由現金流為正"))
            conv = f["fcf"].sum() / f["ni"].sum() if f["ni"].sum() > 0 else np.nan
            if np.isfinite(conv):
                r.append(_pass(5 if conv >= 0.8 else 0, f"自由現金流／淨利 {conv:.2f} 倍"))
        sections["現金流"] = (r, 15)

        # 6. 安全邊際（15 分）
        iv, info = intrinsic_value(y)
        r = []
        if np.isfinite(iv) and iv > 0 and np.isfinite(price):
            mos = (iv - price) / iv * 100
            pts = 15 if mos >= 30 else 8 if mos >= 10 else 3 if mos >= 0 else 0
            r.append(_pass(pts, f"安全邊際 {mos:+.0f}%（內在價值 {fmt_price(iv)}）"))
        elif info.get("base") is not None and info["base"] <= 0:
            r.append(_pass(0, "近 3 年平均自由現金流為負，無法估內在價值"))
        sections["安全邊際"] = (r, 15)

    have = {k: v for k, v in sections.items() if v[0]}
    if not have:
        return None
    got = sum(p for k, (rules, _) in have.items() for _, p in rules)
    full = sum(w for _, (_, w) in have.items())
    score = int(round(got / full * 100))
    return {"score": score, "sections": sections, "is_fin": is_fin, "iv": iv, "iv_info": info,
            "years": list(y.index), "y": y}


def verdict(score):
    return ("高度符合" if score >= 75 else "部分符合" if score >= 50
            else "不太符合" if score >= 30 else "不符合")


def build(code, price):
    """回傳評估結果；抓不到資料回傳 None"""
    try:
        inc, bs, cf = _load(code)
    except Exception:
        return None
    y = annual(inc, bs, cf)
    if len(y) < 3:
        return None
    return evaluate(y.tail(6), price)


# ------------------------------------------------------------
# 畫面
# ------------------------------------------------------------

def render(code, price):
    with st.spinner("讀取近年財報…"):
        b = build(code, price)

    st.markdown('<div class="ins-kicker" style="margin-top:22px">BUFFETT CHECK</div>'
                '<div class="ins-title">🎩 巴菲特檢查</div>', unsafe_allow_html=True)

    if b is None:
        st.caption("抓不到足夠的年度財報（需要至少 3 個完整年度），或財報資料暫時無法連線。")
        return

    sc, secs = b["score"], b["sections"]
    rows = []
    for k, (rules, w) in secs.items():
        got = sum(p for _, p in rules)
        val = f"{got}／{w}" if rules else "不適用" if b["is_fin"] and k not in ("ROE", "獲利穩定") else "—"
        pctw = got / w * 100 if rules else 0
        rows.append(f'<div class="bar-row"><span class="name">{_e(k)}</span><div class="track">'
                    f'<div class="fill" style="width:{pctw:.0f}%"></div></div>'
                    f'<span class="val" style="width:58px">{val}</span></div>')
    yrs = b["years"]
    fin_note = ("<br>金融股的報表沒有毛利、資本支出，負債比也不能用一般標準看，所以只評 ROE 與獲利穩定，"
                "分數只用這兩項換算。" if b["is_fin"] else "")
    st.markdown(f"""
    <div class="ins-card">
      <div class="score-wrap">
        <div><div class="score-big">{sc}<small>/ 100</small></div>
             <div class="score-grade">{verdict(sc)}</div></div>
        <div style="flex:1">{''.join(rows)}</div>
      </div>
      <div class="ins-dates">用 {yrs[0]}–{yrs[-1]} 年共 {len(yrs)} 個完整年度的財報，
      把巴菲特常說的六個原則寫成規則打分。這是公式化的參考，不代表巴菲特本人會怎麼看。{fin_note}</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("🎩 巴菲特檢查：每一分怎麼來的"):
        detail(b)


def rules_block(secs):
    """各項規則與得分（四位大師共用）"""
    html_parts = []
    for k, (rules, w) in secs.items():
        html_parts.append(f"<b style='color:#55cfff'>{_e(k)}（滿分 {w}）</b>" + _rules_html(rules) + "<br>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def detail(b):
    rules_block(b["sections"])

    y = b["y"]
    tbl = pd.DataFrame({
        "年度": y.index,
        "ROE%": y["roe"].round(1),
        "EPS": y["eps"].round(2),
        "毛利率%": y["gm"].round(1),
        "負債/權益": y["de"].round(2),
        "自由現金流(億)": (y["fcf"] / 1e8).round(0),
    })
    st.dataframe(tbl, hide_index=True, width="stretch")

    info = b["iv_info"]
    if np.isfinite(b["iv"]):
        g = info.get("g")
        if info.get("cyclical"):
            st.caption(f"內在價值估法：近 3 年平均每股自由現金流 {info['base']:.2f} 元；"
                       "獲利大起大落，不假設未來會成長。景氣循環股的現金流會隨景氣大幅變動，這個估計特別不可靠。")
        else:
            st.caption(
                f"內在價值估法：近 3 年平均每股自由現金流 {info['base']:.2f} 元，"
                f"前 {YEARS} 年每年成長 {info['g_used'] * 100:.1f}%"
                f"（EPS 歷史年成長 {g * 100:.1f}%，上限 {MAX_G * 100:.0f}%）" if np.isfinite(g) else
                f"內在價值估法：近 3 年平均每股自由現金流 {info['base']:.2f} 元，假設不成長"
            )
        st.caption(f"折現率 {DISCOUNT * 100:.0f}%、永續成長 {TERMINAL_G * 100:.1f}%。"
                   "正在大量擴廠的公司，自由現金流會被資本支出壓低，內在價值容易低估。")
