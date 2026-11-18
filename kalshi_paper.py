#!/usr/bin/env python3
"""Paper-trading logger for the thin-market longshot idea (no orders are ever placed).

Each run:
  1. SCAN  - finds open daily/weekly markets closing in 6-18 hours with a YES mid of 2-98c and a spread of 10c or less,
             and logs them to paper_log.csv. Markets in the 10-50c main rule also get a simulated sale of QTY
             contracts into the real order book.
  2. SCORE - fills in results for logged markets that have settled and prints a summary.
Only public Kalshi endpoints are used, so no API key is needed.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
LOG = "paper_log.csv"
QTY = 5
HOURS_MIN, HOURS_MAX = 6, 18
FAILS = {}


def fee(p):
    return 0.07 * p * (1 - p)  # approximate; replace with Kalshi's exact formula


def get_json(path, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(BASE + path, params=params or {}, timeout=25)
            if r.status_code == 200:
                return r.json()
            FAILS[r.status_code] = FAILS.get(r.status_code, 0) + 1
            time.sleep(2 * (i + 1))
        except requests.exceptions.RequestException:
            FAILS["exc"] = FAILS.get("exc", 0) + 1
            time.sleep(3)
    return None


def load_series():
    allser = (get_json("/series") or {}).get("series", [])
    cand = [s["ticker"] for s in allser
            if s.get("frequency") in ("daily", "weekly") and not s["ticker"].startswith("KXMVE")]
    meta = {s["ticker"]: s.get("category") for s in allser}
    return cand, meta


def scan(cand, meta, logged):
    now = time.time()
    new = []
    for ser in cand:
        j = get_json("/markets", {"series_ticker": ser, "status": "open", "limit": 200})
        for mk in (j or {}).get("markets", []):
            hrs = (pd.Timestamp(mk["close_time"]).timestamp() - now) / 3600
            if not HOURS_MIN <= hrs <= HOURS_MAX or mk["ticker"] in logged:
                continue
            try:
                b, a = float(mk["yes_bid_dollars"]), float(mk["yes_ask_dollars"])
            except (KeyError, ValueError, TypeError):
                continue
            mid = (a + b) / 2
            if b <= 0 or a <= 0 or not (0.02 < mid < 0.98 and a - b <= 0.10):
                continue
            in_rule = 0.10 < mid <= 0.50
            got, fp = np.nan, np.nan
            if in_rule:
                ob = (get_json(f"/markets/{mk['ticker']}/orderbook") or {}).get("orderbook_fp", {})
                book = sorted([(float(p), float(q)) for p, q in ob.get("yes_dollars", [])], reverse=True)
                need, cost, got = QTY, 0.0, 0.0
                for p, q in book:
                    take = min(need, q)
                    cost, got, need = cost + p * take, got + take, need - take
                    if need <= 0:
                        break
                fp = cost / got if got else np.nan
            new.append({"logged_at": pd.Timestamp.now(tz="UTC").isoformat(), "series": ser, "ticker": mk["ticker"],
                        "category": meta.get(ser), "close_time": mk["close_time"], "hrs": round(hrs, 1),
                        "bid": b, "ask": a, "mid": mid, "spread": a - b,
                        "volume": float(mk.get("volume_fp", 0) or 0),
                        "bid_size": float(mk.get("yes_bid_size_fp", 0) or 0),
                        "ask_size": float(mk.get("yes_ask_size_fp", 0) or 0),
                        "in_rule": int(in_rule), "filled": got, "fill_price": fp, "result": np.nan})
        time.sleep(0.1)
    return new


def score(pl):
    pending = pl[pl["result"].isna() & (pd.to_datetime(pl["close_time"], utc=True) < pd.Timestamp.now(tz="UTC"))]
    for i, r in pending.iterrows():
        mk = (get_json(f"/markets/{r.ticker}") or {}).get("market", {}) or {}
        if mk.get("result") in ("yes", "no"):
            pl.loc[i, "result"] = mk["result"]
        time.sleep(0.1)
    return pl


def summarize(pl):
    d = pl[pl["result"].isin(["yes", "no"])].copy()
    print(f"\n{len(d)} settled of {len(pl)} logged")
    if d.empty:
        return
    d["in_rule"] = d["in_rule"].fillna(1)
    d["won"] = (d["result"] == "yes").astype(int)
    d["px"] = d["fill_price"].fillna(d["bid"])
    d["sell_yes"] = d["px"] - d["won"] - fee(d["px"])
    d["buy_yes"] = d["won"] - d["ask"] - fee(d["ask"])
    d["qty"] = d["filled"].fillna(QTY)
    d["grp"] = pd.cut(d["volume"], [-1, 700, 3500, 1e12], labels=["thin <700", "middle", "busy >3500"])

    r_ = d[(d["in_rule"] == 1) & (d["qty"] > 0)]
    print("\nMAIN RULE (YES mid 10-50c, simulated sale at the book):")
    print(r_.groupby("grp", observed=True).agg(
        n=("won", "size"), win=("won", "mean"), price=("mid", "mean"), pnl_per_contract=("sell_yes", "mean"),
        total_dollars=("sell_yes", lambda s: (s * r_.loc[s.index, "qty"]).sum())).round(3).to_string())

    t_ = d[d["grp"] == "thin <700"].copy()
    t_["bucket"] = pd.cut(t_["mid"], [0.02, 0.10, 0.25, 0.50, 0.75, 0.98])
    print("\nTHIN MARKETS BY PRICE (sell at bid / buy at ask, per contract):")
    print(t_.groupby("bucket", observed=True).agg(
        n=("won", "size"), win=("won", "mean"), price=("mid", "mean"),
        sell_yes=("sell_yes", "mean"), buy_yes=("buy_yes", "mean")).round(3).to_string())


def main():
    cand, meta = load_series()
    print(len(cand), "daily/weekly series")
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    logged = set(old["ticker"]) if len(old) else set()
    new = scan(cand, meta, logged)
    pl = pd.concat([old, pd.DataFrame(new)], ignore_index=True) if len(old) else pd.DataFrame(new)
    print(len(new), "new markets logged |", int(sum(r["in_rule"] for r in new)), "in the 10-50c rule")
    if len(pl):
        pl = score(pl)
        pl.to_csv(LOG, index=False)
        summarize(pl)
    print("request failures:", FAILS)


if __name__ == "__main__":
    main()
