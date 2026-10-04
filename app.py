# ============================================================
# 第 2-1 格：基本設定 + 畫面樣式（建立 app.py）
# ============================================================

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st

APP_NAME = "我的台股雷達"
TZ = timezone(timedelta(hours=8))            # 台灣時區 UTC+8
DB_PATH = Path(__file__).parent / "stock_radar.db"   # 資料庫放在 app.py 旁邊（Colab 和雲端都通用）

# 頁面設定：必須是第一個 Streamlit 指令
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自訂 CSS：深藍色手機 App 風格
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }

/* 背景與版面 */
.stApp { background: radial-gradient(circle at 50% -20%, #173b67 0%, #071c37 38%, #020f22 80%); color: #edf7ff; }
.block-container { max-width: 760px; padding-top: 25px; padding-bottom: 100px; }

/* 頂部標題 */
.hero { padding: 5px 4px 18px; }
.brand { font-size: 12px; letter-spacing: 3px; color: #55d9ff; font-weight: 800; }
.hero h1 { font-size: 30px; margin: 3px 0; color: #fff; font-weight: 800; }
.hero p { color: #7d9ab7; font-size: 13px; }

/* 輸入框與按鈕 */
[data-testid="stTextInput"] input {
    background: #071d37 !important; color: white !important;
    border: 1px solid rgba(78,170,225,.45) !important; border-radius: 15px !important;
    height: 50px !important; font-size: 18px !important; font-weight: 700 !important;
}
.stButton button {
    width: 100%; border: none; border-radius: 15px; min-height: 46px;
    background: linear-gradient(135deg, #0878bd, #13b7e9); color: white; font-weight: 800;
}

/* 報價列 */
.quote { display: flex; justify-content: space-between; align-items: center; margin-top: 16px;
         padding: 16px 18px; border-radius: 20px; background: rgba(5,28,54,.82);
         border: 1px solid rgba(82,151,199,.25); }
.quote-name { color: #e7f5ff; font-weight: 800; font-size: 15px; }
.quote-code { color: #6e8eaa; font-size: 11px; margin-top: 3px; }
.quote-price { color: white; font-size: 28px; font-weight: 800; text-align: right; }
.quote-time { color: #6888a5; font-size: 10px; text-align: right; }

/* 小指標方塊 */
.metric { background: rgba(8,35,65,.75); border: 1px solid rgba(84,139,180,.20); border-radius: 17px; padding: 13px; }
.metric-title { color: #7697b3; font-size: 11px; }
.metric-value { color: #fff; font-size: 20px; font-weight: 800; margin-top: 3px; }

/* 三種分析卡片：支撐（綠）、突破（藍）、壓力（橘） */
.card { border-radius: 24px; padding: 19px 18px; margin-top: 15px; box-shadow: 0 15px 35px rgba(0,0,0,.20); }
.support { background: linear-gradient(145deg, rgba(8,54,72,.97), rgba(6,30,59,.98)); border: 1px solid rgba(61,215,201,.45); }
.breakout { background: linear-gradient(145deg, rgba(8,51,83,.98), rgba(6,29,60,.98)); border: 1px solid rgba(67,190,247,.45); }
.pressure { background: linear-gradient(145deg, rgba(43,42,60,.98), rgba(22,29,53,.98)); border: 1px solid rgba(244,151,88,.42); }
.card-title { font-size: 17px; font-weight: 800; margin-bottom: 15px; }
.green-title { color: #4ee3d0; }
.blue-title { color: #55cfff; }
.orange-title { color: #ff9b60; }

/* 卡片內每一列 */
.row { display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; padding: 8px 0; }
.label { color: #9db8cf; font-size: 13px; }
.sub { color: #6f91ad; font-size: 11px; line-height: 1.6; margin-top: 3px; }
.price { color: #f5fbff; font-size: 29px; font-weight: 800; white-space: nowrap; }
.distance { color: #7898b3; font-size: 11px; text-align: right; margin-top: 5px; }
.note { border-top: 1px solid rgba(105,150,185,.15); margin-top: 10px; padding-top: 12px;
        color: #a6bdd1; font-size: 12px; line-height: 1.8; }

/* 底部免責聲明 */
.warning { color: #6987a1; font-size: 10px; line-height: 1.8; padding: 15px 5px 5px; }
</style>
""", unsafe_allow_html=True)
# ============================================================
# 第 2-2 格：資料庫 + 清理代碼
# ============================================================


# ---------- 資料庫 ----------

# analyses：每次查詢的分析結果
# feedback：事後回填實際結果，給學習模式用
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            price REAL,
            near_support REAL,
            second_support REAL,
            breakout REAL,
            pressure REAL,
            second_pressure REAL,
            trend TEXT,
            score REAL,
            created_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            result REAL,
            created_at TEXT
        )
    """)
    con.commit()
    con.close()


init_db()


# ---------- 工具 ----------

# 清理代碼：去空白、轉大寫、只留英數字（' 2330 ' → '2330'）
def clean_code(code):
    return "".join(x for x in str(code).strip().upper() if x.isalnum())


# 文字轉數字；空值、'-'、'--' 或轉不了都回傳 NaN（'1,050.5' → 1050.5）
def number(x):
    try:
        if x in [None, "", "-", "--"]:
            return np.nan
        return float(str(x).replace(",", ""))
    except (ValueError, TypeError):
        return np.nan


# 某個價位距離現價幾 %（例：'距現價 +3.2%'）
def distance(price, level):
    if not (np.isfinite(level) and np.isfinite(price) and price > 0):
        return "距現價 —"
    p = (level / price - 1) * 100
    sign = "+" if p > 0 else ""
    return f"距現價 {sign}{p:.1f}%"


# ---------- 即時報價（證交所 MIS） ----------

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/128 Mobile Safari/537.36"
})


# 先試上市（tse），抓不到再試上櫃（otc）；全失敗回傳 None
def realtime(code):
    for market in ["tse", "otc"]:
        try:
            r = SESSION.get(
                "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
                params={"ex_ch": f"{market}_{code}.tw", "json": "1", "delay": "0"},
                timeout=7
            )
            arr = r.json().get("msgArray", [])
            if not arr:
                continue

            x = arr[0]

            # z = 最新成交價；沒成交時改用 pz
            p = number(x.get("z"))
            if not np.isfinite(p):
                p = number(x.get("pz"))

            if np.isfinite(p):
                return {
                    "name": x.get("n") or code,
                    "price": p,
                    "time": x.get("t") or "",
                    "market": "上市" if market == "tse" else "上櫃",
                    "source": "TWSE MIS",
                }
        except Exception:
            pass

    return None


# ---------- 歷史日 K（Yahoo Finance） ----------

# 近 2 年日 K，快取 10 分鐘；先試上市 .TW，再試上櫃 .TWO
@st.cache_data(ttl=600)
def get_history(code):
    for suffix in [".TW", ".TWO"]:
        try:
            df = yf.Ticker(f"{code}{suffix}").history(
                period="2y", interval="1d", auto_adjust=False
            )
            if df is not None and not df.empty:
                df = df.dropna(subset=["Close"])
                return df
        except Exception:
            pass

    return None
# ============================================================
# 第 2-3 格：技術分析（支撐／壓力／突破／趨勢／成交量／評分）
# ============================================================


# 找轉折點：某天的高（低）點是前後 w 天裡最高（最低）的，就算一個轉折
def find_pivots(df, w=5, lookback=250):
    d = df.tail(lookback)
    win = 2 * w + 1
    highs = d["High"][d["High"] == d["High"].rolling(win, center=True).max()]
    lows = d["Low"][d["Low"] == d["Low"].rolling(win, center=True).min()]
    return list(highs.values), list(lows.values)


# 找成交量密集區：近 120 天把價格切成 24 格，看哪幾格成交量最大
def volume_nodes(df, lookback=120, bins=24, top=4):
    d = df.tail(lookback)
    hist, edges = np.histogram(d["Close"], bins=bins, weights=d["Volume"])
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.argsort(hist)[::-1][:top]
    return list(centers[idx])


# 把很接近的價位（相差 1.5% 內）合併成一個「區」，並記下有幾種依據撐它
# 依據越多（轉折、均線、量能）＝共振越強
def cluster_levels(levels, tol=0.015):
    levels = sorted([(p, s) for p, s in levels if np.isfinite(p) and p > 0])
    groups = []
    for p, s in levels:
        if groups and p <= groups[-1]["price"] * (1 + tol):
            g = groups[-1]
            g["items"].append(p)
            g["sources"].add(s)
            g["price"] = float(np.mean(g["items"]))
        else:
            groups.append({"price": float(p), "items": [p], "sources": {s}})
    for g in groups:
        g["strength"] = len(g["sources"])
    return groups


def analyze(df, price):
    close = df["Close"]

    # 均線
    ma = {n: float(close.rolling(n).mean().iloc[-1]) for n in (5, 20, 60, 120) if len(close) >= n}

    # 收集所有候選價位
    highs, lows = find_pivots(df)
    levels = [(v, "轉折高") for v in highs] + [(v, "轉折低") for v in lows]
    levels += [(v, f"MA{n}") for n, v in ma.items()]
    levels += [(v, "量能密集") for v in volume_nodes(df)]

    groups = cluster_levels(levels)
    below = sorted([g for g in groups if g["price"] < price * 0.998], key=lambda g: -g["price"])
    above = sorted([g for g in groups if g["price"] > price * 1.002], key=lambda g: g["price"])

    # 🟢 近端支撐：現價下方最近的一區
    near_support = below[0] if below else None

    # 🟢 第二支撐：再往下、至少差 2% 的下一區
    second_support = None
    if near_support:
        second_support = next((g for g in below[1:] if g["price"] < near_support["price"] * 0.98), None)

    # 🔵 壓力共振區：上方 15% 內，依據 ≥2 種的最近一區；沒有就取最近一區
    strong_above = [g for g in above if g["strength"] >= 2 and g["price"] < price * 1.15]
    pressure = strong_above[0] if strong_above else (above[0] if above else None)

    # 🔥 第二壓力：再往上、至少差 2% 的下一區
    second_pressure = None
    if pressure:
        second_pressure = next((g for g in above if g["price"] > pressure["price"] * 1.02), None)

    # ⚡ 關鍵突破：前 20 個交易日（不含今天）的最高價
    breakout = float(df["High"].iloc[-21:-1].max()) if len(df) > 21 else np.nan

    # 📊 趨勢：看現價、MA20、MA60 的排列，以及 MA20 這 5 天是往上還是往下
    ma20, ma60 = ma.get(20, np.nan), ma.get(60, np.nan)
    ma20_series = close.rolling(20).mean()
    slope = ma20_series.iloc[-1] - ma20_series.iloc[-6] if len(close) >= 25 else 0
    if price > ma20 > ma60 and slope > 0:
        trend = "多頭"
    elif price < ma20 < ma60 and slope < 0:
        trend = "空頭"
    else:
        trend = "盤整"

    # 📊 成交量：今天量 ÷ 前 20 天平均量
    vol_today = float(df["Volume"].iloc[-1])
    vol_avg20 = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else np.nan
    vol_ratio = vol_today / vol_avg20 if vol_avg20 and np.isfinite(vol_avg20) else np.nan

    # 漲跌幅（跟昨天收盤比）
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else np.nan
    change_pct = (price / prev_close - 1) * 100 if np.isfinite(prev_close) else np.nan

    # 評分 0–100：只是把上面幾個條件加總的參考分數
    score = 50
    score += {"多頭": 20, "空頭": -20}.get(trend, 0)
    if np.isfinite(ma60):
        score += 10 if price > ma60 else -10
    if np.isfinite(vol_ratio) and np.isfinite(change_pct):
        if vol_ratio > 1.5 and change_pct > 0:
            score += 10
        elif vol_ratio > 1.5 and change_pct < 0:
            score -= 10
    if np.isfinite(breakout) and price > breakout:
        score += 10
    score = int(max(0, min(100, score)))

    return {
        "price": price,
        "ma": ma,
        "near_support": near_support,
        "second_support": second_support,
        "breakout": breakout,
        "pressure": pressure,
        "second_pressure": second_pressure,
        "trend": trend,
        "vol_today": vol_today,
        "vol_ratio": vol_ratio,
        "change_pct": change_pct,
        "score": score,
    }


# 把分析結果存進資料庫（學習模式之後會用到）
def save_analysis(code, a):
    lv = lambda g: g["price"] if g else None
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO analyses
           (code, price, near_support, second_support, breakout, pressure, second_pressure, trend, score, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (code, a["price"], lv(a["near_support"]), lv(a["second_support"]),
         a["breakout"] if np.isfinite(a["breakout"]) else None,
         lv(a["pressure"]), lv(a["second_pressure"]),
         a["trend"], a["score"], datetime.now(TZ).isoformat(timespec="seconds"))
    )
    con.commit()
    con.close()
# ============================================================
# 第 2-4 格：App 畫面
# ============================================================


# 表單送出按鈕套用跟一般按鈕一樣的藍色樣式
st.markdown("""
<style>
[data-testid="stFormSubmitButton"] button {
    width: 100%; border: none; border-radius: 15px; min-height: 46px;
    background: linear-gradient(135deg, #0878bd, #13b7e9); color: white; font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


# ---------- 畫面小零件 ----------

def fmt(v):
    return f"{v:,.2f}" if v is not None and np.isfinite(v) else "—"


# 卡片裡的一列：左邊名稱＋依據，右邊價位＋距現價
def row_html(label, level, price, sub=""):
    if isinstance(level, dict):
        value = level["price"]
        sub = sub or "依據：" + "、".join(sorted(level["sources"]))
    else:
        value = level if level is not None else np.nan
    return f"""
    <div class="row">
        <div><div class="label">{label}</div><div class="sub">{sub}</div></div>
        <div><div class="price">{fmt(value)}</div><div class="distance">{distance(price, value)}</div></div>
    </div>"""


def card_html(css, title_css, title, rows, note=""):
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f'<div class="card {css}"><div class="card-title {title_css}">{title}</div>{rows}{note_html}</div>'


def metric_html(title, value):
    return f'<div class="metric"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>'


# ---------- 頂部 ----------

st.markdown(f"""
<div class="hero">
    <div class="brand">STOCK RADAR PRO</div>
    <h1>📈 {APP_NAME}</h1>
    <p>輸入台股代碼，看支撐、突破與壓力位置</p>
</div>
""", unsafe_allow_html=True)


# ---------- 輸入 ----------

with st.form("query"):
    code_input = st.text_input(
        "台股代碼",
        value=st.session_state.get("code", ""),
        placeholder="例如 2330",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🔍 開始分析")

if submitted:
    st.session_state["code"] = clean_code(code_input)

code = st.session_state.get("code", "")
if not code:
    st.stop()


# ---------- 抓資料 ----------

with st.spinner("抓取資料中…"):
    hist = get_history(code)
    rt = realtime(code)

if hist is None and rt is None:
    st.error(f"找不到「{code}」的資料，請確認代碼是否正確。")
    st.stop()

if hist is None or len(hist) < 60:
    st.warning("歷史資料不足 60 天，無法計算支撐與壓力。")
    st.stop()

if rt:
    price, name = rt["price"], rt["name"]
    time_text = f'{rt["market"]}｜{rt["time"]}｜{rt["source"]}'
else:
    price, name = float(hist["Close"].iloc[-1]), code
    time_text = "即時報價取得失敗，改用最近收盤價"


# ---------- 分析 ----------

a = analyze(hist, price)

# 只有按下按鈕時才存一筆，避免畫面每次重整都重複存
if submitted:
    save_analysis(code, a)


# ---------- 📈 股價 ----------

st.markdown(f"""
<div class="quote">
    <div><div class="quote-name">{name}</div><div class="quote-code">{code}</div></div>
    <div><div class="quote-price">{fmt(price)}</div><div class="quote-time">{time_text}</div></div>
</div>
""", unsafe_allow_html=True)


# ---------- 小指標 ----------

chg = a["change_pct"]
chg_text = f"{chg:+.2f}%" if np.isfinite(chg) else "—"
vol_text = f'{a["vol_ratio"]:.2f} 倍' if np.isfinite(a["vol_ratio"]) else "—"

c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_html("漲跌幅", chg_text), unsafe_allow_html=True)
c2.markdown(metric_html("📊 趨勢", a["trend"]), unsafe_allow_html=True)
c3.markdown(metric_html("📊 量比（對 20 日均量）", vol_text), unsafe_allow_html=True)
c4.markdown(metric_html("參考評分", f'{a["score"]} / 100'), unsafe_allow_html=True)


# ---------- 🛡️ 支撐觀察 ----------

st.markdown(card_html(
    "support", "green-title", "🛡️ 支撐觀察",
    row_html("🟢 近端支撐", a["near_support"], price)
    + row_html("🟢 第二近端支撐", a["second_support"], price),
    "跌破近端支撐，下一個觀察點是第二支撐。"
), unsafe_allow_html=True)


# ---------- ⚡ 關鍵突破 ----------

bo = a["breakout"]
bo_note = "現價已站上前 20 日高點。" if np.isfinite(bo) and price > bo else "站上這個價位，代表突破近一個月的高點。"
st.markdown(card_html(
    "breakout", "blue-title", "⚡ 關鍵突破",
    row_html("前 20 日最高價", bo, price, sub="近一個月的高點"),
    bo_note
), unsafe_allow_html=True)


# ---------- 🔵 壓力 ----------

st.markdown(card_html(
    "pressure", "orange-title", "🔵 壓力觀察",
    row_html("🔵 壓力共振區", a["pressure"], price)
    + row_html("🔥 第二壓力", a["second_pressure"], price),
    "依據越多種（轉折、均線、量能），代表這個價位越多人關注。"
), unsafe_allow_html=True)


# ---------- 走勢圖 ----------

st.markdown("#### 📈 近一年走勢")
chart = pd.DataFrame({"收盤價": hist["Close"]})
chart["MA20"] = hist["Close"].rolling(20).mean()
chart["MA60"] = hist["Close"].rolling(60).mean()
st.line_chart(chart.tail(250))

st.markdown("#### 📊 近半年成交量")
st.bar_chart(hist["Volume"].tail(120))


# ---------- 免責聲明 ----------

st.markdown("""
<div class="warning">
本工具僅用公開價量資料做技術面整理，價位與評分都是公式算出的參考值，不構成任何投資建議。
即時報價可能有延遲，請以券商報價為準。
</div>
""", unsafe_allow_html=True)
# ============================================================
# 第 2-5 格：🧠 學習模式
# 回頭檢查過去的分析：之後 5 個交易日內，支撐有沒有守住、壓力有沒有碰到
# ============================================================


def load_analyses(code):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM analyses WHERE code = ? ORDER BY created_at", con, params=(code,)
    )
    con.close()
    if df.empty:
        return df
    # 同一天按很多次只留最後一筆
    df["date"] = pd.to_datetime(df["created_at"]).dt.date
    return df.drop_duplicates(subset="date", keep="last")


def evaluate(records, hist, days=5):
    results = []
    hist_dates = pd.Series(hist.index.date, index=hist.index)

    for _, r in records.iterrows():
        future = hist[hist_dates > r["date"]].head(days)
        if len(future) < days:
            results.append({"日期": r["date"], "分析價": r["price"], "狀態": "⏳ 等待中"})
            continue

        low, high = future["Low"].min(), future["High"].max()
        ret = (future["Close"].iloc[-1] / r["price"] - 1) * 100

        held = None if pd.isna(r["near_support"]) else bool(low >= r["near_support"] * 0.99)
        hit = None if pd.isna(r["pressure"]) else bool(high >= r["pressure"])

        results.append({
            "日期": r["date"],
            "分析價": r["price"],
            "趨勢": r["trend"],
            "評分": r["score"],
            "支撐守住": {True: "✅", False: "❌", None: "—"}[held],
            "碰到壓力": {True: "✅", False: "❌", None: "—"}[hit],
            f"{days}日後報酬%": round(ret, 2),
            "狀態": "已檢查",
        })

    return pd.DataFrame(results)


with st.expander("🧠 學習模式", expanded=False):
    records = load_analyses(code)

    if records.empty:
        st.info("這檔股票還沒有分析紀錄。每按一次「開始分析」就會存一筆，5 個交易日後可以回來檢查。")
    else:
        res = evaluate(records, hist)
        done = res[res["狀態"] == "已檢查"]

        if not done.empty:
            held_rate = (done["支撐守住"] == "✅").sum() / max((done["支撐守住"] != "—").sum(), 1) * 100
            hit_rate = (done["碰到壓力"] == "✅").sum() / max((done["碰到壓力"] != "—").sum(), 1) * 100
            avg_ret = done["5日後報酬%"].mean()

            m1, m2, m3 = st.columns(3)
            m1.markdown(metric_html("支撐守住率", f"{held_rate:.0f}%"), unsafe_allow_html=True)
            m2.markdown(metric_html("壓力觸及率", f"{hit_rate:.0f}%"), unsafe_allow_html=True)
            m3.markdown(metric_html("平均 5 日報酬", f"{avg_ret:+.2f}%"), unsafe_allow_html=True)
            st.caption(f"共檢查 {len(done)} 筆；樣本越多越有參考價值。")

        st.dataframe(res.iloc[::-1], hide_index=True, use_container_width=True)

    # 手動回填：自己記錄這次實際的操作結果
    st.markdown("**📝 記錄實際結果**")
    fb = st.number_input("這次實際報酬（%）", value=0.0, step=0.5, format="%.1f")
    if st.button("儲存紀錄"):
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO feedback (code, result, created_at) VALUES (?, ?, ?)",
            (code, fb, datetime.now(TZ).isoformat(timespec="seconds"))
        )
        con.commit()
        con.close()
        st.success("已儲存 ✅")

    con = sqlite3.connect(DB_PATH)
    fb_df = pd.read_sql_query("SELECT result FROM feedback WHERE code = ?", con, params=(code,))
    con.close()
    if not fb_df.empty:
        st.caption(f"已記錄 {len(fb_df)} 筆，平均實際報酬 {fb_df['result'].mean():+.2f}%")
