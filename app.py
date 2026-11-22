"""Headwater prototype app (interactive).

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from common import (BACKTEST_CSV, FORECAST_CSV, GEOJSON, OUT, PANEL_CSV,  # noqa: E402
                    PROGRAMS_CSV, RESULTS_JSON, clean_zip)

st.set_page_config(page_title="Headwater", page_icon="💧", layout="wide")

GRAY, FAINT = "#dcdcdc", "#f3f3f3"
OUTREACH = {  # label -> color, in legend order
    "Emerging: newly flagged": "#d95f02",
    "Persistent hot spot": "#7570b3",
    "Last year's hot spot, expected to cool": "#c9c6e6",
    "Not flagged": GRAY,
}
BEFORE = {
    "Headwater pick, NOT on last year's list": "#d95f02",
    "Headwater pick, also on last year's list": "#7570b3",
    "Not picked": GRAY,
}
AFTER = {
    "★ New hot spot Headwater caught": "#e6ab02",
    "Hot spot Headwater caught": "#1b9e77",
    "Hot spot Headwater missed": "#d62728",
    "Pick that didn't become a hot spot": "#fdd0a2",
    "Not a hot spot, not picked": GRAY,
}
RATE_LABELS = {"predicted_per_1k_customers": "per 1,000 residential customers",
               "predicted_per_1k_households": "per 1,000 households"}


# ---------------------------------------------------------------- data loading
def simplify(ring, tol=0.0006):
    """Douglas-Peucker line simplification (tolerance in degrees, ~60 m)."""
    pts = np.asarray(ring, dtype=float)
    if len(pts) < 8:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        rel = pts[a + 1:b] - pts[a]
        norm = np.hypot(*seg)
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm if norm else np.hypot(rel[:, 0], rel[:, 1])
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


@st.cache_data
def load():
    if not FORECAST_CSV.exists():
        return None
    fc = pd.read_csv(FORECAST_CSV, dtype={"zip": str})
    fc["zip"] = clean_zip(fc["zip"])
    if "city" not in fc:
        fc["city"] = ""
    fc["city"] = fc["city"].fillna("").astype(str).str.title()
    d = {"fc": fc,
         "results": json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {},
         "panel": pd.read_csv(PANEL_CSV, dtype={"zip": str}) if PANEL_CSV.exists() else None,
         "bt": pd.read_csv(BACKTEST_CSV, dtype={"zip": str}) if BACKTEST_CSV.exists() else None,
         "expl": (pd.read_csv(OUT / "explanations.csv", dtype={"zip": str})
                  if (OUT / "explanations.csv").exists() else None),
         "shapes": {}}
    if GEOJSON.exists():
        wanted = set(fc["zip"])
        for f in json.loads(GEOJSON.read_text())["features"]:
            z = f["properties"]["ZCTA5CE10"]
            if z not in wanted:
                continue
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            xs, ys = [], []
            for poly in polys:
                ring = simplify(poly[0])
                xs += ring[:, 0].round(4).tolist() + [None]
                ys += ring[:, 1].round(4).tolist() + [None]
            d["shapes"][z] = (xs, ys)
    return d


# ---------------------------------------------------------------- map drawing
def draw_map(shapes, df, legend=None, highlight=None, colorbar=None, height=620):
    """df needs columns: zip, color, group, label. Each ZIP is drawn as its own
    filled outline, so no internet basemap is needed."""
    fig = go.Figure()
    for name, color in (legend or {}).items():  # legend entries, in our order
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, legendgroup=name,
                                 marker=dict(color=color, size=12, symbol="square",
                                             line=dict(color="#888", width=0.5))))
    for _, r in df.iterrows():
        if r["zip"] not in shapes:
            continue
        xs, ys = shapes[r["zip"]]
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", mode="lines", fillcolor=r["color"],
                                 line=dict(width=0.4, color="white"), hoveron="fills",
                                 text=r["label"], hoverinfo="text", legendgroup=r["group"],
                                 showlegend=False))
    if highlight and highlight in shapes:
        xs, ys = shapes[highlight]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=3, color="#111"),
                                 hoverinfo="skip", showlegend=False))
    if colorbar:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale=colorbar["scale"], cmin=colorbar["min"],
                                             cmax=colorbar["max"], color=[colorbar["min"]],
                                             colorbar=dict(title=colorbar["title"], thickness=14,
                                                           tickvals=colorbar.get("tickvals"),
                                                           ticktext=colorbar.get("ticktext")))))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1.3)  # NJ's shape at ~40°N
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.0,
                                  bgcolor="rgba(255,255,255,0.85)"),
                      hoverlabel=dict(bgcolor="white"))
    return fig


def zip_name(fc, z):
    c = fc.loc[fc["zip"] == z, "city"]
    return f"{z} · {c.iloc[0]}" if len(c) and c.iloc[0] else z


def rate_column(fc):
    for c in RATE_LABELS:
        if c in fc and fc[c].notna().any():
            return c
    return None


# ---------------------------------------------------------------- page
data = load()
st.title("Headwater")
st.caption("Predicting where water shutoffs are rising in New Jersey, so assistance arrives first.")
if data is None:
    st.error("No forecast yet. Run `python scripts/build_panel.py` then `python scripts/backtest.py`.")
    st.stop()

fc, results, panel, bt, expl, shapes = (data[k] for k in ["fc", "results", "panel", "bt", "expl", "shapes"])
if results.get("synthetic"):
    st.warning("**Stand-in data.** These numbers are made up for testing. "
               "Replace data/shutoffs.csv with the real NRDC data before any demo.")
if not shapes:
    st.error("Map boundaries missing. Run `python scripts/fetch_geo.py`.")
    st.stop()

latest_col = [c for c in fc.columns if c.startswith("shutoffs_")][0]
latest_year = int(latest_col.split("_")[1])
next_year = results.get("forecast_year", latest_year + 1)
test_year = results.get("test_year", latest_year)

# Sidebar: outreach settings
st.sidebar.header("Outreach settings")
capacity = st.sidebar.slider("How many ZIP codes can your team reach?", 5, 80,
                             int(results.get("flag_count", 33)), key="capacity")
utilities = sorted(fc["utility"].unique())
chosen_utils = st.sidebar.multiselect("Water utilities", utilities, default=utilities, key="utils")
st.sidebar.caption("Headwater uses neighborhood-level public data only and is built for "
                   "assistance outreach, never debt collection.")

tab_map, tab_test, tab_help, tab_how = st.tabs(
    ["Risk map", f"Test it yourself ({test_year})", "Find help", "How it works"])

# ================================================================ RISK MAP
with tab_map:
    f = fc[fc["utility"].isin(chosen_utils)].copy()
    hw = set(f.nlargest(capacity, "predicted_next")["zip"])
    last = set(f.nlargest(capacity, latest_col)["zip"])
    f["status"] = np.select(
        [f["zip"].isin(hw) & ~f["zip"].isin(last), f["zip"].isin(hw) & f["zip"].isin(last),
         ~f["zip"].isin(hw) & f["zip"].isin(last)],
        list(OUTREACH)[:3], default="Not flagged")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"ZIP codes flagged for {next_year}", len(hw))
    c2.metric("Emerging (not on last year's list)", int((f["status"] == list(OUTREACH)[0]).sum()),
              help="Flagged by Headwater but not among last year's worst ZIP codes. "
                   "A 'repeat last year' approach would miss these.")
    share = f.loc[f["zip"].isin(hw), latest_col].sum() / max(f[latest_col].sum(), 1)
    c3.metric(f"These ZIPs' share of {latest_year} shutoffs", f"{share:.0%}",
              help="How much of last year's problem these ZIP codes held. This is not an accuracy "
                   "score; see the Test it yourself tab for that.")

    view = st.segmented_control("Map view", ["Outreach priority", "Predicted shutoff rate",
                                             "Shutoffs over time"], default="Outreach priority",
                                key="view") or "Outreach priority"

    # ZIP selection shared by the table, the search box and the map highlight
    zip_options = list(fc.sort_values("predicted_next", ascending=False)["zip"])
    if "zip_pick" not in st.session_state:
        st.session_state["zip_pick"] = zip_options[0]
    picked = st.session_state["zip_pick"]

    rc = rate_column(fc)
    rate_txt = (lambda r: f"<br>Predicted rate: {r[rc]:,.1f} {RATE_LABELS[rc]}"
                if rc and pd.notna(r.get(rc)) else "")
    label = lambda r: (f"<b>ZIP {r['zip']}{' · ' + r['city'] if r['city'] else ''}</b><br>{r['utility']}"
                       f"<br>Shutoffs {latest_year}: {r[latest_col]:,.0f}"
                       f"<br>Predicted {next_year}: {r['predicted_next']:,.0f}{rate_txt(r)}")
    others = fc[~fc["utility"].isin(chosen_utils)].assign(color=FAINT, group="Not selected",
                                                          label="Utility filtered out")
    if view == "Outreach priority":
        m = f.assign(color=f["status"].map(OUTREACH), group=f["status"])
        m["label"] = m.apply(lambda r: label(r) + f"<br><b>{r['status']}</b>", axis=1)
        fig = draw_map(shapes, pd.concat([others, m]), legend=OUTREACH, highlight=picked)
    elif view == "Predicted shutoff rate":
        col = rc or "predicted_next"
        pct = f[col].rank(pct=True)
        cols = sample_colorscale("Blues", (0.1 + 0.9 * pct.fillna(0)).tolist())
        m = f.assign(color=[GRAY if pd.isna(p) else c for p, c in zip(pct, cols)], group="rate")
        m["label"] = m.apply(label, axis=1)
        legend = {"No customer count (Veolia)": GRAY} if f[col].isna().any() else None
        fig = draw_map(shapes, pd.concat([others, m]), legend=legend, highlight=picked,
                       colorbar=dict(scale="Blues", min=0, max=100, title="Rate<br>percentile"))
    else:
        years = sorted(panel["year"].unique().tolist()) + [f"{next_year} (predicted)"]
        yr = st.select_slider("Year", options=years, value=latest_year, key="year_slider")
        if isinstance(yr, str):
            vals = f.set_index("zip")["predicted_next"]
        else:
            vals = panel[panel["year"] == yr].set_index("zip")["shutoffs"]
        vmax = np.log1p(max(panel["shutoffs"].max(), fc["predicted_next"].max()))
        v = f["zip"].map(vals)
        cols = sample_colorscale("OrRd", (np.log1p(v.fillna(0)) / vmax).clip(0, 1).tolist())
        m = f.assign(color=[FAINT if pd.isna(x) else c for x, c in zip(v, cols)], group="year")
        m["label"] = [f"<b>ZIP {z}{' · ' + c if c else ''}</b><br>{yr}: "
                      + ("no report" if pd.isna(x) else f"{x:,.0f} shutoffs")
                      for z, c, x in zip(m["zip"], m["city"], v)]
        ticks = [0, 10, 50, 100, 250, 500, 1000]
        fig = draw_map(shapes, pd.concat([others, m]), highlight=picked,
                       colorbar=dict(scale="OrRd", min=0, max=vmax, title="Shutoffs",
                                     tickvals=[np.log1p(t) for t in ticks], ticktext=[str(t) for t in ticks]))
        if yr == 2019:
            st.caption("2019 is incomplete: Aqua and Veolia did not report. 2020–2021: statewide shutoff moratorium.")
        elif isinstance(yr, str):
            st.caption(f"Predicted {next_year} shutoffs, assuming statewide totals stay at {latest_year} levels.")

    map_col, card_col = st.columns([3, 2], gap="large")
    with map_col:
        st.plotly_chart(fig, width="stretch", key="main_map")

    # ---- outreach list (selectable) ----
    table = f[f["zip"].isin(hw)].sort_values("predicted_next", ascending=False)
    st.session_state["table_zips"] = table["zip"].tolist()

    def on_table_select():
        rows = st.session_state["outreach_table"].selection.rows
        if rows:
            st.session_state["zip_pick"] = st.session_state["table_zips"][rows[0]]

    # ---- ZIP detail card ----
    with card_col:
        st.selectbox("Look up a ZIP code", zip_options, key="zip_pick",
                     format_func=lambda z: zip_name(fc, z))
        r = fc[fc["zip"] == st.session_state["zip_pick"]].iloc[0]
        status = f.loc[f["zip"] == r["zip"], "status"]
        status = status.iloc[0] if len(status) else "Utility filtered out"
        st.markdown(f"### ZIP {r['zip']}{' · ' + r['city'] if r['city'] else ''}")
        st.markdown(f"{r['utility']} &nbsp; <span style='background:{OUTREACH.get(status, GRAY)};"
                    f"color:{'white' if status in list(OUTREACH)[:2] else '#333'};padding:2px 8px;"
                    f"border-radius:10px;font-size:0.85em'>{status}</span>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{latest_year} shutoffs", f"{r[latest_col]:,.0f}")
        chg = (r["predicted_next"] / r[latest_col] - 1) if r[latest_col] else np.nan
        k2.metric(f"Predicted {next_year}", f"{r['predicted_next']:,.0f}",
                  delta=None if pd.isna(chg) else f"{chg:+.0%}", delta_color="inverse")
        k3.metric("Per 1k customers", f"{r[rc]:,.1f}" if rc and pd.notna(r.get(rc)) else "n/a",
                  help="Predicted shutoffs per 1,000 residential customers. n/a where the utility "
                       "reported no customer count (Veolia).")

        if panel is not None:
            h = panel[panel["zip"] == r["zip"]].sort_values("year")
            hist = go.Figure()
            hist.add_vrect(x0=2019.5, x1=2021.5, fillcolor="#eeeeee", line_width=0,
                           annotation_text="moratorium", annotation_position="top left",
                           annotation_font_size=10)
            hist.add_trace(go.Scatter(x=h["year"], y=h["shutoffs"], mode="lines+markers", name="Actual",
                                      line=dict(color="#1b6f8a", width=2)))
            if len(h):
                hist.add_trace(go.Scatter(x=[h["year"].iloc[-1], next_year],
                                          y=[h["shutoffs"].iloc[-1], r["predicted_next"]],
                                          mode="lines+markers", name="Predicted",
                                          line=dict(color="#d95f02", dash="dash", width=2)))
            hist.update_layout(height=210, margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
                               legend=dict(orientation="h", y=1.15, x=0),
                               yaxis=dict(title="Shutoffs", rangemode="tozero"),
                               xaxis=dict(tickmode="array", tickvals=list(range(2019, next_year + 1))))
            st.plotly_chart(hist, width="stretch", key="hist")

        if expl is not None:
            e = expl[expl["zip"] == r["zip"]].copy()
            names = {"Shutoffs this year": f"Size of {latest_year} count",
                     "Shutoffs in the previous reported year": f"{latest_year - 1} count",
                     "Change since previous reported year": f"Trend {latest_year - 1}→{latest_year}",
                     "Water utility": "Water utility"}
            e["factor"] = e["factor"].replace(names)
            e["pct"] = (np.exp(e["effect"]) - 1) * 100
            e = e.reindex(e["pct"].abs().sort_values(ascending=False).index).head(5).iloc[::-1]
            bar = go.Figure(go.Bar(x=e["pct"], y=e["factor"], orientation="h",
                                   marker_color=["#d62728" if v > 0 else "#1b9e77" for v in e["pct"]],
                                   text=[f"{v:+.0f}%" for v in e["pct"]], textposition="outside",
                                   hoverinfo="skip"))
            lim = max(10, e["pct"].abs().max() * 1.35)
            bar.update_layout(height=220, margin=dict(l=0, r=10, t=30, b=0), xaxis=dict(range=[-lim, lim],
                              ticksuffix="%", zeroline=True, nticks=5, tickangle=0), title=dict(
                              text="What moved this prediction (vs. a typical ZIP)", font_size=13, x=0))
            st.plotly_chart(bar, width="stretch", key="factors")
            st.caption("Red pushes predicted shutoffs up, green pushes them down. Approximate: each "
                       "factor is compared with its typical value while the others stay the same.")
        facts = []
        if pd.notna(r.get("median_income")):
            facts.append(f"Median income ${r['median_income']:,.0f}")
        if pd.notna(r.get("pct_people_of_color")):
            facts.append(f"People of color {r['pct_people_of_color']:.0%}")
        if pd.notna(r.get("customers")):
            facts.append(f"{r['customers']:,.0f} residential customers")
        if facts:
            st.caption(" · ".join(facts))

    st.subheader(f"Outreach list for {next_year} ({len(table)} ZIP codes)")
    st.caption("Tick the box at the left of a row to show that ZIP's details above.")
    show = table.assign(Place=table["city"]).rename(columns={
        "zip": "ZIP", "utility": "Utility", latest_col: f"Shutoffs {latest_year}",
        "predicted_next": f"Predicted {next_year}", "status": "Status"})
    cols = ["ZIP", "Place", "Utility", f"Shutoffs {latest_year}", f"Predicted {next_year}", "Status"]
    st.dataframe(show[cols].round(0), hide_index=True, width="stretch", key="outreach_table",
                 on_select=on_table_select, selection_mode="single-row")
    st.download_button("Download outreach list (CSV)", show[cols].to_csv(index=False),
                       file_name=f"headwater_outreach_{next_year}.csv")

# ================================================================ TEST IT YOURSELF
with tab_test:
    if bt is None:
        st.info("Run `python scripts/backtest.py` to create the test.")
    else:
        b = bt.copy()
        b["zip"] = clean_zip(b["zip"])
        b["city"] = b["city"].fillna("").astype(str).str.title() if "city" in b else ""
        from_year = int(b["from_year"].iloc[0])
        st.markdown(f"#### Rewind to the end of {from_year}.")
        st.markdown(f"Headwater has never seen any {test_year} data. Which ZIP codes would it tell "
                    f"outreach teams to focus on, and how does that compare with simply reusing "
                    f"{from_year}'s worst ZIP codes?")
        k = st.slider("ZIP codes to flag", 5, 80, int(results.get("main", {}).get("k", 31)), key="test_k")
        reveal = st.toggle(f"Reveal what actually happened in {test_year}", key="reveal")

        pred_top = set(b.nlargest(k, "predicted_next")["zip"])
        prev_top = set(b.nlargest(k, "shutoffs_from_year")["zip"])
        real_top = set(b.nlargest(k, "shutoffs_test_year")["zip"])
        new_hot = real_top - prev_top
        z = b["zip"]
        if not reveal:
            group = np.select([z.isin(pred_top) & ~z.isin(prev_top), z.isin(pred_top)],
                              list(BEFORE)[:2], default="Not picked")
            legend = BEFORE
        else:
            group = np.select(
                [z.isin(pred_top) & z.isin(new_hot), z.isin(pred_top) & z.isin(real_top),
                 z.isin(real_top), z.isin(pred_top)], list(AFTER)[:4], default=list(AFTER)[4])
            legend = AFTER
        b["group"] = group
        b["color"] = b["group"].map(legend)
        b["label"] = [f"<b>ZIP {zz}{' · ' + c if c else ''}</b><br>{from_year}: {a:,.0f} shutoffs"
                      + (f"<br>{test_year} actual: {t:,.0f}" if reveal else "")
                      + f"<br><b>{g}</b>"
                      for zz, c, a, t, g in zip(b["zip"], b["city"], b["shutoffs_from_year"],
                                                b["shutoffs_test_year"], b["group"])]
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.plotly_chart(draw_map(shapes, b, legend=legend, height=600), width="stretch", key="test_map")
        with right:
            if not reveal:
                st.metric("Headwater's picks", k)
                st.metric(f"Picks that weren't on {from_year}'s list", len(pred_top - prev_top),
                          help="These are Headwater's bets on neighborhoods that are heating up.")
                st.info(f"Flip the switch to see what really happened in {test_year}.")
            else:
                total = b["shutoffs_test_year"].sum()
                cap = lambda s: b.loc[b["zip"].isin(s), "shutoffs_test_year"].sum() / total
                st.markdown(f"**Share of {test_year}'s shutoffs in the {k} flagged ZIP codes**")
                m1, m2 = st.columns(2)
                m1.metric("Headwater's picks", f"{cap(pred_top):.1%}",
                          delta=f"{(cap(pred_top) - cap(prev_top)) * 100:+.1f} pts vs. last year's list")
                m2.metric(f"{from_year}'s worst ZIPs", f"{cap(prev_top):.1%}")
                m3, m4 = st.columns(2)
                m3.metric("New hot spots caught", f"{len(new_hot & pred_top)} of {len(new_hot)}",
                          help=f"ZIPs in {test_year}'s worst {k} that weren't in {from_year}'s worst {k}. "
                               "Last year's list can never catch these.")
                m4.metric("Best possible", f"{cap(real_top):.1%}",
                          help=f"If you had known {test_year}'s worst {k} ZIPs in advance.")
                ks = list(range(5, 101, 1))
                curve = pd.DataFrame({
                    "k": ks,
                    "Headwater": [b.nlargest(x, "predicted_next")["shutoffs_test_year"].sum() / total for x in ks],
                    f"Last year's list": [b.nlargest(x, "shutoffs_from_year")["shutoffs_test_year"].sum() / total
                                          for x in ks],
                    "Best possible": [b.nlargest(x, "shutoffs_test_year")["shutoffs_test_year"].sum() / total
                                      for x in ks]})
                cf = go.Figure()
                for name, color, dash in [("Best possible", "#bbbbbb", "dot"), ("Last year's list", "#555555", None),
                                          ("Headwater", "#d95f02", None)]:
                    cf.add_trace(go.Scatter(x=curve["k"], y=curve[name], name=name, mode="lines",
                                            line=dict(color=color, dash=dash, width=2.5)))
                cf.add_vline(x=k, line_width=1, line_dash="dash", line_color="#888")
                cf.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                                 title=dict(text=f"Share of {test_year} shutoffs captured", font_size=13, x=0),
                                 xaxis_title="ZIP codes flagged", yaxis=dict(tickformat=".0%"),
                                 legend=dict(orientation="h", y=-0.3))
                st.plotly_chart(cf, width="stretch", key="curve")
                caught = b[b["zip"].isin(new_hot & pred_top)]
                if len(caught):
                    st.markdown("**New hot spots Headwater saw coming:** " + ", ".join(
                        f"{zz}{' ' + c if c else ''} ({a:,.0f} → {t:,.0f})" for zz, c, a, t in
                        zip(caught["zip"], caught["city"], caught["shutoffs_from_year"],
                            caught["shutoffs_test_year"])))

# ================================================================ FIND HELP
T = {
    "English": {
        "title": "Behind on your water bill? Find help.", "saved": "Nothing you enter here is saved.",
        "zip": "Your ZIP code", "util": "Who is your water company?", "unsure": "Not sure",
        "behind": "Are you behind on your water bill?", "notice": "Have you received a shutoff notice?",
        "yes": "Yes", "no": "No", "submit": "Show my options",
        "urgent": "**You have a shutoff notice. Act now:** call your water company today to ask for a "
                  "payment plan or a hold, and call **211** for help finding assistance.",
        "risk": "ZIP {z} is in an area where Headwater predicts many shutoffs next year. "
                "Applying for help early is a good idea.",
        "programs": "**Programs that may help you:**", "apply": "How to apply",
        "verify": "Prototype: program details are being verified. Confirm eligibility with each program."},
    "Español": {
        "title": "¿Atrasado en su factura de agua? Encuentre ayuda.",
        "saved": "Nada de lo que escriba aquí se guarda.",
        "zip": "Su código postal", "util": "¿Cuál es su compañía de agua?", "unsure": "No estoy seguro/a",
        "behind": "¿Está atrasado/a en el pago de su factura de agua?",
        "notice": "¿Ha recibido un aviso de corte del servicio de agua?",
        "yes": "Sí", "no": "No", "submit": "Ver mis opciones",
        "urgent": "**Tiene un aviso de corte. Actúe ahora:** llame hoy a su compañía de agua para pedir un "
                  "plan de pagos o que detengan el corte, y llame al **211** para encontrar ayuda.",
        "risk": "El código postal {z} está en una zona donde Headwater predice muchos cortes el próximo año. "
                "Es buena idea solicitar ayuda pronto.",
        "programs": "**Programas que pueden ayudarle:**", "apply": "Cómo solicitar",
        "verify": "Prototipo: estamos verificando los detalles de los programas. Confirme los requisitos "
                  "con cada programa."},
}
with tab_help:
    lang = st.radio("Language / Idioma", list(T), horizontal=True, key="lang")
    t = T[lang]
    st.subheader(t["title"])
    st.caption(t["saved"])
    programs = pd.read_csv(PROGRAMS_CSV) if PROGRAMS_CSV.exists() else pd.DataFrame()
    utils = sorted(u for u in programs.get("utility", pd.Series(dtype=str)).unique() if u != "Any")
    with st.form("help_form"):
        zip_in = st.text_input(t["zip"], max_chars=5, key="q_zip")
        util_in = st.selectbox(t["util"], list(range(len(utils) + 1)), key="q_util",
                               format_func=lambda i: utils[i] if i < len(utils) else t["unsure"])
        st.radio(t["behind"], [0, 1, 2], horizontal=True, key="q_behind",
                 format_func=lambda i: [t["yes"], t["no"], t["unsure"]][i])
        notice = st.radio(t["notice"], [0, 1], horizontal=True, key="q_notice",
                          format_func=lambda i: [t["yes"], t["no"]][i])
        submitted = st.form_submit_button(t["submit"])
    if submitted:
        if notice == 0:
            st.error(t["urgent"])
        zz = zip_in.strip().zfill(5) if zip_in.strip() else ""
        top = set(fc.nlargest(int(results.get("flag_count", 33)), "predicted_next")["zip"])
        if zz in top:
            st.info(t["risk"].format(z=zz))
        if not programs.empty:
            util_name = utils[util_in] if util_in < len(utils) else None
            match = programs[(programs["utility"] == "Any") | (programs["utility"] == util_name)]
            match = match[~match["name"].str.startswith("TODO")]
            st.markdown(t["programs"])
            es = lang == "Español"
            for _, p in match.iterrows():
                who = p.get("who_it_helps_es") if es and pd.notna(p.get("who_it_helps_es")) else p["who_it_helps"]
                how = p.get("how_to_apply_es") if es and pd.notna(p.get("how_to_apply_es")) else p["how_to_apply"]
                st.markdown(f"- **{p['name']}**: {who}. *{t['apply']}:* {how}")
            if (match["verified"].astype(str).str.lower() != "yes").any():
                st.caption(t["verify"])

# ================================================================ HOW IT WORKS
with tab_how:
    r0 = results.get("main", {})
    st.markdown(f"""
**Data.** Residential water shutoffs by ZIP code for New Jersey's four largest investor-owned utilities
(New Jersey American Water, Aqua, Middlesex, Veolia), 2019 and 2022–{latest_year}, reported under New
Jersey's 2021 water affordability law and compiled by NRDC, plus each ZIP's residential customers,
income and population.

**Model.** Gradient-boosted decision trees (scikit-learn). Each ZIP starts from last year's count, and the
model learns which ZIPs tend to rise faster or slower **than the statewide trend**. Removing the trend
matters because {test_year - 1} was the rebound after the 2020–21 shutoff moratorium, when almost
every ZIP jumped. The serving utility is an input, because collection policy drives shutoffs as much as
hardship does.

**Honest test.** Trained only on data through {test_year - 1}, then asked to predict {test_year}. Compared
with the simplest alternative, reusing last year's worst ZIP codes. Flagging the top {r0.get('k', '')} ZIP codes,
Headwater's picks held **{r0.get('capture_headwater', 0):.1%}** of {test_year}'s shutoffs and last year's list held
**{r0.get('capture_persistence', 0):.1%}**. Headwater caught **{r0.get('newly_hot_caught_by_headwater', 0)} of
{r0.get('newly_hot_zips', 0)}** neighborhoods that newly became hot spots, which last year's list can never catch.

**Limits.** Annual data with only one training year; ZIP-level, not households; four utilities that report
(publicly owned systems don't); "shutoff risk" reflects utility policy as well as household need.

**Next steps.** Monthly filings from the NJ Board of Public Utilities (far more data points), a second full
test year, water bills for cost burden, and a pilot with a New Jersey assistance program.
""")
