# ============================================================
# 🔎 選股功能
#
# 四個條件（每個都可以開關、調數字）：
#   1. 外資連續買超
#   2. K 線在 MA5／MA10／MA20 的糾結點
#   3. 股價在上升趨勢（季線之上、均線往上）
#   4. 營收成長、毛利為正、有賺錢
#
# 做法：先用「很快就能算完」的條件 1、4 刷掉大部分股票，
#       剩下的才去 Yahoo 抓 K 線算條件 2、3，全市場也能在短時間內跑完。
# ============================================================

import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

TZ = timezone(timedelta(hours=8))

HTTP = requests.Session()
HTTP.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128 Safari/537.36"
})

# 只看一般股票：4 碼、開頭不是 0（排除 ETF、權證等）
STOCK_CODE = re.compile(r"^[1-9]\d{3}$")

# K 線一次最多抓幾檔，避免免費主機跑太久
MAX_PRICE_FETCH = 400


# ------------------------------------------------------------
# 小工具
# ------------------------------------------------------------

def _num(x):
    """'1,234' → 1234.0；空值、'-'、轉不了 → NaN"""
    try:
        s = str(x).replace(",", "").strip()
        if s in ("", "-", "--", "None", "nan"):
            return np.nan
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _clean(s):
    return re.sub(r"<[^>]+>", "", str(s)).strip()


def _get_json(url, params=None, tries=2):
    last = None
    for i in range(tries):
        try:
            r = HTTP.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:   # 證交所偶爾會擋太頻繁的連線，稍等再試一次
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"連線失敗：{url}（{last}）")


def _pick_key(row, *keywords, exclude=()):
    """在一筆資料裡找「欄名包含所有關鍵字」的欄位，回傳欄名。"""
    for k in row:
        if all(w in k for w in keywords) and not any(x in k for x in exclude):
            return k
    return None


# ------------------------------------------------------------
# 條件 1：外資買賣超（證交所 T86 + 櫃買三大法人）
# ------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _twse_foreign(day_str):
    """上市某一天：{代號: (名稱, 外資買賣超股數)}。非交易日回 {}。"""
    p = _get_json("https://www.twse.com.tw/rwd/zh/fund/T86",
                  {"date": day_str, "selectType": "ALLBUT0999", "response": "json"})
    rows = p.get("data") or []
    if not rows or p.get("stat") != "OK":
        return {}
    fields = p.get("fields") or []
    # 第 4 欄：外陸資買賣超（不含外資自營商）；第 7 欄：外資自營商買賣超
    if len(fields) < 8 or "買賣超" not in fields[4] or "外資自營商" not in fields[7]:
        raise RuntimeError("證交所外資資料的欄位格式變了")
    out = {}
    for r in rows:
        code = _clean(r[0])
        out[code] = (_clean(r[1]), np.nan_to_num(_num(r[4])) + np.nan_to_num(_num(r[7])))
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _tpex_foreign(day_str):
    """上櫃某一天：{代號: (名稱, 外資買賣超股數)}。非交易日回 {}。"""
    d = datetime.strptime(day_str, "%Y%m%d")
    p = _get_json("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
                  {"type": "Daily", "sect": "EW", "date": d.strftime("%Y/%m/%d"),
                   "id": "", "response": "json"})
    tables = p.get("tables") or []
    rows = (tables[0].get("data") if tables else None) or []
    if not rows:
        return {}
    # 第 10 欄：外資及陸資合計買賣超
    out = {}
    for r in rows:
        if len(r) < 11:
            continue
        out[_clean(r[0])] = (_clean(r[1]), np.nan_to_num(_num(r[10])))
    return out


def recent_foreign(n_days, progress=None):
    """
    往回抓最近 n_days 個交易日的外資買賣超。
    回傳 (dates, data)；data = {代號: {"name", "market", "nets": [最新, 前一天, ...]}}
    """
    today = datetime.now(TZ).date()
    dates, data = [], {}
    day = today
    tries = 0
    while len(dates) < n_days and tries < n_days * 3 + 10:
        tries += 1
        if day.weekday() < 5:              # 跳過週六日
            ds = day.strftime("%Y%m%d")
            tw = _twse_foreign(ds)
            ot = _tpex_foreign(ds)
            if tw or ot:                   # 這天有資料才算一個交易日
                idx = len(dates)
                dates.append(day)
                for market, src in (("上市", tw), ("上櫃", ot)):
                    for code, (name, net) in src.items():
                        rec = data.setdefault(code, {"name": name, "market": market,
                                                     "nets": [np.nan] * n_days})
                        rec["nets"][idx] = net
                if progress:
                    progress(len(dates) / n_days, f"外資資料：已抓 {len(dates)}/{n_days} 個交易日")
                time.sleep(0.4)            # 別打太快，避免被證交所暫時擋掉
        day -= timedelta(days=1)
    return dates, data


# ------------------------------------------------------------
# 條件 4：月營收、損益表（證交所／櫃買 OpenAPI，只給最新一期）
# ------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_revenue():
    """{代號: {"name", "ym", "yoy"}}；yoy = 去年同月增減(%)"""
    out = {}
    for url in ("https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
                "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"):
        try:
            rows = _get_json(url)
        except Exception:
            continue
        if not rows:
            continue
        k_code = _pick_key(rows[0], "公司代號") or _pick_key(rows[0], "CompanyCode")
        k_name = _pick_key(rows[0], "公司名稱") or _pick_key(rows[0], "CompanyName")
        k_ym = _pick_key(rows[0], "資料年月")
        k_yoy = _pick_key(rows[0], "去年同月增減")
        if not (k_code and k_yoy):
            continue
        for r in rows:
            out[str(r.get(k_code, "")).strip()] = {
                "name": str(r.get(k_name, "")).strip() if k_name else "",
                "ym": str(r.get(k_ym, "")).strip() if k_ym else "",
                "yoy": _num(r.get(k_yoy)),
            }
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_income():
    """{代號: {"period", "gross", "net"}}；一般產業的綜合損益表（年初至今累計，單位千元）"""
    out = {}
    for url in ("https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ci",
                "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_ci"):
        try:
            rows = _get_json(url)
        except Exception:
            continue
        if not rows:
            continue
        r0 = rows[0]
        k_code = _pick_key(r0, "公司代號") or _pick_key(r0, "CompanyCode")
        k_year = _pick_key(r0, "年度") or _pick_key(r0, "Year")
        k_q = _pick_key(r0, "季別") or _pick_key(r0, "Season")
        k_gross = _pick_key(r0, "營業毛利", exclude=("已實現", "未實現"))
        k_net = (_pick_key(r0, "本期淨利", exclude=("歸屬",))
                 or _pick_key(r0, "本期稅後淨利", exclude=("歸屬",)))
        if not (k_code and k_gross and k_net):
            continue
        for r in rows:
            period = ""
            if k_year and k_q:
                period = f"{r.get(k_year, '')}Q{r.get(k_q, '')}"
            out[str(r.get(k_code, "")).strip()] = {
                "period": period,
                "gross": _num(r.get(k_gross)),
                "net": _num(r.get(k_net)),
            }
    return out


# ------------------------------------------------------------
# 條件 2、3：抓 K 線算均線（Yahoo Finance，一次抓一批）
# ------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(tickers):
    """tickers：('2330.TW', '6488.TWO', ...) → {ticker: DataFrame(Open/High/Low/Close)}"""
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 100):            # 100 檔一批
        batch = tickers[i:i + 100]
        try:
            df = yf.download(batch, period="9mo", interval="1d", group_by="ticker",
                             auto_adjust=False, threads=True, progress=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for t in batch:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(subset=["Close"])
                if len(sub) >= 85:
                    out[t] = sub[["Open", "High", "Low", "Close"]].copy()
            except KeyError:
                pass
    return out


def technicals(df, ma_tol, need_above):
    """回傳條件 2、3 的判斷與數值"""
    c = df["Close"]
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60)}
    last = df.iloc[-1]
    m5, m10, m20, m60 = (float(ma[n].iloc[-1]) for n in (5, 10, 20, 60))
    hi_ma, lo_ma = max(m5, m10, m20), min(m5, m10, m20)
    spread = (hi_ma - lo_ma) / lo_ma * 100

    # 條件 2：三條均線差距夠小，而且今天的 K 線（最高～最低）碰到這個範圍
    touch = float(last["Low"]) <= hi_ma and float(last["High"]) >= lo_ma
    cond2 = spread <= ma_tol and touch
    if need_above:
        cond2 = cond2 and float(last["Close"]) > hi_ma

    # 條件 3：收盤在季線上、季線比 20 天前高、月線比 5 天前高
    cond3 = (float(last["Close"]) > m60
             and m60 > float(ma[60].iloc[-21])
             and m20 > float(ma[20].iloc[-6]))

    return {"close": float(last["Close"]), "spread": spread,
            "cond2": bool(cond2), "cond3": bool(cond3),
            "date": df.index[-1].date() if hasattr(df.index[-1], "date") else None}


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def run_screen(opt, progress):
    steps = []          # 每一關剩幾檔，給畫面顯示
    notes = []

    # ---------- 股票名單與外資資料 ----------
    n_days = int(opt["fi_days"]) if opt["use_fi"] else 1
    dates, fi = recent_foreign(n_days, progress)
    if not fi:
        raise RuntimeError("抓不到證交所／櫃買的外資資料，可能是網站暫時擋連線，請過幾分鐘再試。")
    universe = {code: rec for code, rec in fi.items() if STOCK_CODE.match(code)}
    steps.append(("全部上市櫃股票", len(universe)))
    if dates:
        notes.append(f"外資資料最新日期：{dates[0]:%Y/%m/%d}")

    rows = {}
    for code, rec in universe.items():
        nets = rec["nets"]
        min_shares = float(opt["fi_min_lots"]) * 1000
        streak = 0
        for v in nets:
            if np.isfinite(v) and v > 0 and v >= min_shares:
                streak += 1
            else:
                break
        rows[code] = {
            "代號": code, "名稱": rec["name"], "市場": rec["market"],
            "外資連買(天)": streak,
            f"外資{len(dates)}日合計(張)": int(np.nansum(nets) / 1000),
        }

    # ---------- 條件 1：外資連續買超 ----------
    if opt["use_fi"]:
        rows = {c: r for c, r in rows.items() if r["外資連買(天)"] >= n_days}
        steps.append((f"① 外資連 {n_days} 天買超 ≥ {opt['fi_min_lots']} 張", len(rows)))

    # ---------- 條件 4：營收與獲利 ----------
    if opt["use_fund"]:
        progress(0.5, "抓月營收與財報…")
        rev, inc = load_revenue(), load_income()
        if not rev and not inc:
            raise RuntimeError("抓不到月營收與財報資料，請過幾分鐘再試。")
        keep = {}
        for c, r in rows.items():
            rv, ic = rev.get(c), inc.get(c)
            yoy = rv["yoy"] if rv else np.nan
            gross = ic["gross"] if ic else np.nan
            net = ic["net"] if ic else np.nan
            ok = np.isfinite(yoy) and yoy > float(opt["rev_yoy"])
            ok = ok and np.isfinite(gross) and gross > 0
            if opt["need_profit"]:
                ok = ok and np.isfinite(net) and net > 0
            if ok:
                r["營收年增%"] = round(yoy, 1)
                r["毛利(億)"] = round(gross / 1e5, 2)       # 千元 → 億元
                r["淨利(億)"] = round(net / 1e5, 2)
                keep[c] = r
        rows = keep
        label = f"④ 營收年增 > {opt['rev_yoy']}%、毛利 > 0" + ("、淨利 > 0" if opt["need_profit"] else "")
        steps.append((label, len(rows)))
        ym = next((v["ym"] for v in rev.values() if v.get("ym")), "")
        pr = next((v["period"] for v in inc.values() if v.get("period")), "")
        if ym or pr:
            notes.append(f"營收月份：{ym}（民國年月）；財報：{pr}（年初至今累計）")
        notes.append("金融股沒有「毛利」這一項，開啟條件 ④ 時不會出現。")

    # ---------- 條件 2、3：K 線與均線 ----------
    if opt["use_ma"] or opt["use_trend"]:
        codes = list(rows)
        if len(codes) > MAX_PRICE_FETCH:
            codes = sorted(codes, key=lambda c: -rows[c].get(f"外資{len(dates)}日合計(張)", 0))[:MAX_PRICE_FETCH]
            notes.append(f"剩下的股票超過 {MAX_PRICE_FETCH} 檔，只取外資買超最多的前 {MAX_PRICE_FETCH} 檔算技術面。"
                         "建議開啟條件 ① 或 ④ 先縮小範圍。")
        tick = {c: f"{c}.TW" if rows[c]["市場"] == "上市" else f"{c}.TWO" for c in codes}
        progress(0.7, f"抓 {len(codes)} 檔的 K 線…")
        prices = load_prices(tuple(sorted(tick.values())))
        keep = {}
        miss = 0
        for c in codes:
            df = prices.get(tick[c])
            if df is None:
                miss += 1
                continue
            t = technicals(df, float(opt["ma_tol"]), opt["need_above"])
            if opt["use_ma"] and not t["cond2"]:
                continue
            if opt["use_trend"] and not t["cond3"]:
                continue
            r = rows[c]
            r["收盤"] = round(t["close"], 2)
            r["均線糾結%"] = round(t["spread"], 2)
            keep[c] = r
        rows = keep
        if opt["use_ma"]:
            steps.append((f"② 均線糾結 ≤ {opt['ma_tol']}%" + ("、收在均線上" if opt["need_above"] else ""), None))
        if opt["use_trend"]:
            steps.append(("③ 上升趨勢", None))
        steps[-1] = (steps[-1][0], len(rows))
        if miss:
            notes.append(f"有 {miss} 檔抓不到 K 線（可能剛上市或 Yahoo 沒資料），已略過。")

    progress(1.0, "完成")
    df = pd.DataFrame(list(rows.values()))
    if not df.empty:
        df = df.sort_values(f"外資{len(dates)}日合計(張)", ascending=False).reset_index(drop=True)
    return df, steps, notes


# ------------------------------------------------------------
# 畫面
# ------------------------------------------------------------

def render(metric_html):
    st.markdown("#### 🔎 選股")

    with st.expander("⚙️ 篩選條件", expanded=True):
        c1, c2, c3 = st.columns([1.2, 1, 1])
        use_fi = c1.checkbox("① 外資連續買超", value=True, key="sc_use_fi")
        fi_days = c2.number_input("連續天數", 1, 10, 3, key="sc_fi_days")
        fi_min = c3.number_input("每天至少(張)", 0, 100000, 100, step=50, key="sc_fi_min")

        c1, c2, c3 = st.columns([1.2, 1, 1])
        use_ma = c1.checkbox("② K 線在 MA5/10/20 糾結點", value=True, key="sc_use_ma")
        ma_tol = c2.number_input("均線差距 ≤ (%)", 0.5, 10.0, 3.0, step=0.5, key="sc_ma_tol")
        need_above = c3.checkbox("收盤站上三條均線", value=False, key="sc_above")

        use_trend = st.checkbox("③ 上升趨勢（收盤 > MA60、MA60 與 MA20 往上）", value=True, key="sc_trend")

        c1, c2, c3 = st.columns([1.2, 1, 1])
        use_fund = c1.checkbox("④ 營收成長、毛利為正", value=True, key="sc_use_fund")
        rev_yoy = c2.number_input("營收年增 > (%)", -100.0, 500.0, 0.0, step=5.0, key="sc_rev")
        need_profit = c3.checkbox("淨利也要 > 0", value=True, key="sc_profit")

        go = st.button("🔎 開始選股", key="sc_go")

    if go:
        opt = dict(use_fi=use_fi, fi_days=fi_days, fi_min_lots=fi_min,
                   use_ma=use_ma, ma_tol=ma_tol, need_above=need_above,
                   use_trend=use_trend, use_fund=use_fund, rev_yoy=rev_yoy,
                   need_profit=need_profit)
        bar = st.progress(0.0, "準備中…")
        try:
            df, steps, notes = run_screen(opt, lambda p, t: bar.progress(min(max(p, 0.0), 1.0), t))
            st.session_state["sc_result"] = (df, steps, notes, datetime.now(TZ).strftime("%m/%d %H:%M"))
        except Exception as e:
            st.session_state.pop("sc_result", None)
            st.error(f"選股失敗：{e}")
        bar.empty()

    res = st.session_state.get("sc_result")
    if not res:
        st.caption("設定好條件後按「開始選股」。第一次跑大約需要 30 秒～1 分鐘。")
        return

    df, steps, notes, when = res

    # 每一關剩幾檔
    funnel = " → ".join(f"{name}：{n} 檔" for name, n in steps if n is not None)
    st.caption(f"{when} 篩選｜{funnel}")

    if df.empty:
        st.info("沒有股票同時符合所有條件，可以放寬一些數字再試。")
    else:
        st.markdown(metric_html("符合條件", f"{len(df)} 檔"), unsafe_allow_html=True)
        st.dataframe(df, hide_index=True, use_container_width=True)

        # 選一檔 → 跳到個股分析（下拉選單在手機上比點表格好按）
        options = [f"{r['代號']} {r['名稱']}" for _, r in df.iterrows()]
        c1, c2 = st.columns([2, 1])
        pick = c1.selectbox("選一檔看分析", options, key="sc_pick", label_visibility="collapsed")
        if c2.button("📊 看分析", key="sc_open") and pick:
            st.session_state["_goto"] = pick.split()[0]
            st.rerun()

    for n in notes:
        st.caption("・" + n)
