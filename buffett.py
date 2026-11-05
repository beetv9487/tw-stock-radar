# ============================================================
# 🎩 巴菲特檢查
#
# 把巴菲特公開談過的選股原則寫成可檢查的規則，套到台股：
#   1. 股東權益報酬率（ROE）長期夠高
#   2. 獲利穩定、每股盈餘（EPS）大致往上
#   3. 負債不高
#   4. 毛利率高又穩（護城河的痕跡）
#   5. 賺的是真現金（股東盈餘＝營業現金流 − 維持性資本支出）
#   6. 價格低於估算的內在價值（安全邊際）
#
# 資料：FinMind 的季度損益表、資產負債表、現金流量表（近 12 年）。
# 只用「完整 4 季」的年度；金融股的報表結構不同，改評 ROE、獲利穩定、淨值比估值。
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
FIN_REQUIRED = 0.10    # 金融股：股東要求報酬率（推算合理淨值比用）
MIN_COVER = 50         # 有資料的項目配分合計至少 50 分才打分，避免只剩一兩項就變滿分


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
    params = {"dataset": dataset, "start_date": start}
    if code:
        params["data_id"] = code
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
    start = f"{datetime.now(TZ).year - 12}-01-01"      # 葛拉漢要 10 年
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


# 淨利的科目名稱：一般公司 vs 金融業（IncomeAfterTax 少一個 s）
NI_NAMES = ("EquityAttributableToOwnersOfParent", "IncomeAfterTaxes", "IncomeAfterTax",
            "TotalConsolidatedProfitForThePeriod")


def _sum_cols(df, *names):
    """幾個科目加總；全部都沒有才是缺值"""
    cols = [df[n] for n in names if n in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(cols, axis=1).sum(axis=1, min_count=1)


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
        "ni": _col(q, *NI_NAMES),
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
        # 有息負債＝短期借款＋長期借款＋應付公司債（應付帳款這類不用付利息的不算）
        y["debt"] = _sum_cols(b, "ShorttermBorrowings", "LongtermBorrowings", "BondsPayable").reindex(y.index)
        y["cash"] = _col(b, "CashAndCashEquivalents").reindex(y.index)
    else:
        y["equity"] = y["liab"] = y["debt"] = y["cash"] = np.nan

    # 現金流量表是「年初累計」，所以 12/31 那筆就是全年
    c = _wide(cf)
    if not c.empty:
        c.index = pd.to_datetime(c.index)
        c = c[(c.index.month == 12) & (c.index.day == 31)]
        c.index = c.index.year
        y["ocf"] = _col(c, "CashFlowsFromOperatingActivities", "NetCashInflowFromOperatingActivities").reindex(y.index)
        y["capex"] = _col(c, "PropertyAndPlantAndEquipment").reindex(y.index)   # 負數＝買設備
        y["dep"] = _sum_cols(c, "Depreciation", "AmortizationExpense").reindex(y.index)
    else:
        y["ocf"] = y["capex"] = y["dep"] = np.nan

    y["fcf"] = y["ocf"] + y["capex"]
    # 股東盈餘（巴菲特 1986 年股東信）：只扣「維持營運」的資本支出，擴廠的錢不扣。
    # 維持性支出用折舊攤銷估計（買設備的錢若比折舊少，就用實際金額）。
    maint = np.minimum((-y["capex"]).clip(lower=0), y["dep"])
    y["oe"] = (y["ocf"] - maint).where(maint.notna(), y["fcf"])
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


def eps_growth(eps):
    """EPS 年成長率（對數迴歸的斜率），比只看頭尾兩年穩；有虧損年度回傳 nan"""
    e = eps.dropna()
    if len(e) < 3 or (e <= 0).any():
        return np.nan
    x = np.arange(len(e), dtype=float)
    slope = np.polyfit(x, np.log(e.values.astype(float)), 1)[0]
    return float(np.exp(slope) - 1)


def intrinsic_value(y, y_long=None):
    """回傳 (每股內在價值, 說明dict)；資料不夠回傳 (nan, {})。y_long：景氣循環股用的較長年度"""
    eps = y["eps"].dropna()
    cyclical = eps_cv(eps) > 0.6
    # 一般公司看近 3 年；景氣循環股看整個循環（最多 10 年）的平均，避免剛好用到景氣高峰
    oe = ((y_long if y_long is not None else y) if cyclical else y.tail(3))["oe"].dropna()
    shares = y["shares"].dropna()
    if len(oe) < 2 or shares.empty or shares.iloc[-1] <= 0:
        return np.nan, {}
    base = oe.mean() / shares.iloc[-1]           # 平均每股股東盈餘
    if not np.isfinite(base) or base <= 0:
        return np.nan, {"base": base, "cyclical": cyclical, "n": len(oe)}
    g = eps_growth(eps)
    g_used = 0.0 if cyclical else float(min(max(g if np.isfinite(g) else 0.0, 0.0), MAX_G))

    pv, cash = 0.0, base
    for t in range(1, YEARS + 1):
        cash *= 1 + g_used
        pv += cash / (1 + DISCOUNT) ** t
    terminal = cash * (1 + TERMINAL_G) / (DISCOUNT - TERMINAL_G)
    pv += terminal / (1 + DISCOUNT) ** YEARS
    return float(pv), {"base": float(base), "g": g, "g_used": g_used, "cyclical": cyclical, "n": len(oe)}


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


CYCLE_CAP = 45     # 獲利不穩定（景氣循環、曾虧損）時的總分上限


def evaluate(y, price, y_long=None):
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
        # 金融業：負債就是生意本身（存款、保單），沒有毛利、資本支出，改用淨值比估值
        sections["負債"] = ([], 15)
        sections["護城河"] = ([], 15)
        sections["現金流"] = ([], 15)
        iv, info = np.nan, {}
        r = []
        eq, sh = y["equity"].dropna(), y["shares"].dropna()
        if len(roe) >= 3 and len(eq) and len(sh) and sh.iloc[-1] > 0 and np.isfinite(price):
            bvps = eq.iloc[-1] / sh.iloc[-1]
            pb = price / bvps
            fair = roe.mean() / 100 / FIN_REQUIRED      # ROE 10% 的銀行，合理淨值比約 1 倍
            pts = 15 if pb <= fair * 0.7 else 8 if pb <= fair else 0
            r.append(_pass(pts, f"股價淨值比 {pb:.2f} 倍；以平均 ROE {roe.mean():.1f}% 推算合理約 {fair:.2f} 倍"
                                f"（假設股東要求 {FIN_REQUIRED * 100:.0f}% 報酬，低於合理值 30% 以上才有安全邊際）"))
        sections["安全邊際"] = (r, 15)
    else:
        # 3. 負債（15 分）：有息負債（借款＋公司債）÷ 股東權益
        last = y.iloc[-1]
        r = []
        if np.isfinite(last["equity"]) and last["equity"] > 0:
            debt = last["debt"] if np.isfinite(last["debt"]) else 0.0
            d = debt / last["equity"]
            cash = last["cash"]
            net_cash = np.isfinite(cash) and cash >= debt
            pts = 15 if (d <= 0.3 or net_cash) else 8 if d <= 0.6 else 0
            txt = f"有息負債（借款＋公司債）÷ 股東權益 {d:.2f} 倍"
            if net_cash:
                txt += f"；現金 {cash / 1e8:,.0f} 億 ≥ 有息負債 {debt / 1e8:,.0f} 億（淨現金）"
            r.append(_pass(pts, txt))
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

        # 5. 現金流（15 分）：股東盈餘
        f = y[["oe", "ni"]].dropna()
        r = []
        if len(f):
            pos = int((f["oe"] > 0).sum())
            r.append(_pass(10 if pos >= 0.8 * len(f) else 0, f"{len(f)} 年中 {pos} 年股東盈餘為正"))
            conv = f["oe"].sum() / f["ni"].sum() if f["ni"].sum() > 0 else np.nan
            if np.isfinite(conv):
                r.append(_pass(5 if conv >= 0.8 else 0, f"股東盈餘／淨利 {conv:.2f} 倍（賺的是不是真現金）"))
        sections["現金流"] = (r, 15)

        # 6. 安全邊際（15 分）
        iv, info = intrinsic_value(y, y_long)
        r = []
        if np.isfinite(iv) and iv > 0 and np.isfinite(price):
            mos = (iv - price) / iv * 100
            pts = 15 if mos >= 30 else 8 if mos >= 10 else 3 if mos >= 0 else 0
            r.append(_pass(pts, f"安全邊際 {mos:+.0f}%（內在價值 {fmt_price(iv)}）"))
        elif info.get("base") is not None and info["base"] <= 0:
            r.append(_pass(0, f"近 {info.get('n', 3)} 年平均股東盈餘為負，無法估內在價值"))
        sections["安全邊際"] = (r, 15)

    have = {k: v for k, v in sections.items() if v[0]}
    full = sum(w for _, (_, w) in have.items())
    if full < MIN_COVER:      # 能評的項目太少，分數會失真 → 不給分
        return None
    got = sum(p for k, (rules, _) in have.items() for _, p in rules)
    score = int(round(got / full * 100))
    # 巴菲特收購條件第 2 條：要「穩定的獲利能力」，不碰轉機股 → 獲利大起大落或曾虧損，總分最多 45
    cv = eps_cv(eps)
    unstable = len(eps) >= 3 and ((eps <= 0).any() or not np.isfinite(cv) or cv > 0.6)
    cap_note = ""
    if unstable and score > CYCLE_CAP:
        cap_note = (f"獲利大起大落或曾虧損：巴菲特只買「獲利能力穩定」的公司（收購條件第 2 條），"
                    f"所以總分最多 {CYCLE_CAP}（未設上限時是 {score} 分）。")
        score = CYCLE_CAP
    return {"score": score, "sections": sections, "is_fin": is_fin, "iv": iv, "iv_info": info,
            "years": list(y.index), "y": y, "cap_note": cap_note}


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
    return evaluate(y.tail(6), price, y.tail(10))


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
        val = f"{got}／{w}" if rules else "不適用" if b["is_fin"] and k in ("負債", "護城河", "現金流") else "—"
        pctw = got / w * 100 if rules else 0
        rows.append(f'<div class="bar-row"><span class="name">{_e(k)}</span><div class="track">'
                    f'<div class="fill" style="width:{pctw:.0f}%"></div></div>'
                    f'<span class="val" style="width:58px">{val}</span></div>')
    yrs = b["years"]
    fin_note = ("<br>金融股的報表沒有毛利、資本支出，負債比也不能用一般標準看，所以改評 ROE、獲利穩定與淨值比估值，"
                "分數用這三項換算。" if b["is_fin"] else "")
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


FIN_NA = "金融股不適用"


def rules_block(secs, na=()):
    """各項規則與得分（大師共用）；na：這些項目沒有規則時寫「金融股不適用」而不是「資料不足」"""
    html_parts = []
    for k, (rules, w) in secs.items():
        if not rules and k in na:
            body = f'<div class="rule">{FIN_NA}（金融業報表沒有這類項目，不列入計分）</div>'
            head = f"<b style='color:#55cfff'>{_e(k)}</b>"
        else:
            body = _rules_html(rules)
            head = f"<b style='color:#55cfff'>{_e(k)}（滿分 {w}）</b>"
        html_parts.append(head + body + "<br>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def detail(b):
    if b.get("cap_note"):
        st.caption("⚠️ " + b["cap_note"])
    rules_block(b["sections"], na=("負債", "護城河", "現金流") if b.get("is_fin") else ())

    y = b["y"]
    tbl = pd.DataFrame({
        "年度": y.index,
        "ROE%": y["roe"].round(1),
        "EPS": y["eps"].round(2),
        "毛利率%": y["gm"].round(1),
        "有息負債/權益": (y["debt"].fillna(0) / y["equity"]).round(2),
        "股東盈餘(億)": (y["oe"] / 1e8).round(0),
    })
    st.dataframe(tbl, hide_index=True, width="stretch")

    info = b["iv_info"]
    if np.isfinite(b["iv"]):
        g = info.get("g")
        if info.get("cyclical"):
            st.caption(f"內在價值估法：獲利大起大落（景氣循環），改用近 {info.get('n', '')} 年整個循環的平均每股股東盈餘 "
                       f"{info['base']:.2f} 元，且不假設未來會成長。景氣循環股的現金流變動大，這個估計特別不可靠。")
        else:
            st.caption(
                f"內在價值估法：近 3 年平均每股股東盈餘 {info['base']:.2f} 元，"
                f"前 {YEARS} 年每年成長 {info['g_used'] * 100:.1f}%"
                f"（EPS 歷史年成長 {g * 100:.1f}%，上限 {MAX_G * 100:.0f}%）" if np.isfinite(g) else
                f"內在價值估法：近 3 年平均每股股東盈餘 {info['base']:.2f} 元，假設不成長"
            )
        st.caption(f"股東盈餘＝營業現金流 − 維持性資本支出（用折舊攤銷估計，擴廠的錢不扣）。"
                   f"折現率 {DISCOUNT * 100:.0f}%、永續成長 {TERMINAL_G * 100:.1f}%。")
