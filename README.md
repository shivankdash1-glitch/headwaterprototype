# Headwater prototype: 3-day build guide

What you'll have at the end: a working app with three parts.

1. **Risk map**: New Jersey ZIP codes colored by predicted shutoff risk, with "emerging" ZIPs (newly flagged this year) highlighted.
2. **How accurate is it?**: an honest test. The model predicts a year it never saw, and is compared against the simplest alternative (reuse last year's worst ZIPs).
3. **Find help**: a short questionnaire that points residents to assistance programs.

Plus a 2-minute demo video for your Samsung application.

**Who does what**

- **Sumedh**: setup, running the scripts, the model, app fixes.
- **Shivank**: getting the spreadsheet into the right shape (Step 4), Census and bill data, checking numbers against NRDC's totals.
- **Cameron**: contacting NRDC if the download isn't available, checking every assistance program (Step 9), writing and narrating the demo.

---

## DAY 1: Get set up and get the real data

### Step 1. Install (Sumedh, 20 min)

1. Install Python 3.10 or newer (python.org). On Windows, tick "Add Python to PATH."
2. Unzip this folder somewhere easy, like your Desktop.
3. Open a terminal in the folder. On a Mac: Terminal, then `cd ~/Desktop/headwater-prototype`. On Windows: open the folder, type `cmd` in the address bar, and press Enter.
4. Run:
   ```
   pip install -r requirements.txt
   ```

### Step 2. Prove everything runs, using stand-in data (Sumedh, 10 min)

```
python scripts/make_fake_data.py
python scripts/build_panel.py
python scripts/backtest.py
streamlit run app.py
```

A browser tab opens with the app. You'll see a yellow **"Stand-in data"** warning. That's correct: these numbers are fake and only prove the code works. Never show or report them.

Then clear out the stand-in data:
```
python scripts/clean_fake_data.py
```

### Step 3. Download the real NRDC spreadsheet (Shivank or Cameron, 30 min)

1. Open NRDC's dashboard: https://experience.arcgis.com/experience/4d793a7f228047eab52663d83b4b2086
2. NRDC's article says the dashboard lets you "download a spreadsheet with all of the raw data on shutoffs and demographics." Look for a download, data, or table button. Article: https://www.nrdc.org/resources/water-shutoffs-new-nrdc-map-shows-hot-spots-across-new-jersey
3. **If you can't find it within 30 minutes, don't keep hunting.** Email NRDC right away. The data authors are Susan Lee (data analysis) and Larry Levine (water program). Use https://www.nrdc.org/contact-us, and mention your Food & Water Watch contact, who may be able to introduce you. Last resort: the raw utility filings on the BPU docket (https://publicaccess.bpu.state.nj.us/CaseSummary.aspx?case_id=2109305). That's hundreds of Excel files, so it's too slow for this deadline.

**Step 3b. If the dashboard has no export button:** capture the data your browser loads. Use Chrome or Edge.

1. Open the dashboard, then press **F12** (Mac: **Cmd+Option+I**) and click the **Network** tab.
2. In the Network filter box, type **`query`**.
3. Reload the page (**Ctrl+R**, Mac: **Cmd+R**), open the table, and click the **2024** tab.
4. Requests appear with names starting `query?...`. Click one, then under **Headers** copy the **Request URL**. It contains `/FeatureServer/` or `/MapServer/`.
5. In your terminal, run (keep the quotes):
   ```
   python scripts/fetch_nrdc.py "PASTE-THE-URL-HERE"
   ```
   Every table behind the dashboard is saved to `data/raw/`, and the script prints each file's column names.

What the data covers, per NRDC: 346 ZIP codes, four utilities (New Jersey American Water, Aqua, Middlesex, Veolia), and the years 2019, 2022, 2023 and 2024.

### Step 4 (DONE for the NRDC file). Convert it automatically

If you downloaded the NRDC file with `fetch_nrdc.py`, one command does all of Step 4:
```
python scripts/convert_nrdc.py
```
It writes `data/shutoffs.csv` (with residential customer counts) and `data/nrdc_demographics.csv`, and checks its totals against NRDC's own. **This has already been run on your file**, so `data/` and `outputs/` contain real results. Run `streamlit run app.py` to see them. The manual steps below are only needed for some other spreadsheet.

### Step 4 (manual version). Turn a spreadsheet into `data/shutoffs.csv`

The code needs **exactly these four columns**, with one row per ZIP code, per year, per utility:

| zip | year | utility | shutoffs |
|---|---|---|---|
| 07302 | 2023 | New Jersey American Water | 412 |
| 07302 | 2024 | New Jersey American Water | 530 |

How to get there:

- **If the spreadsheet has one column per year** (like "2022 shutoffs", "2023 shutoffs"), you need to "unpivot" it. In Excel, use Data > From Table/Range > select the year columns > Transform > **Unpivot Columns**. Or paste this into Python:
  ```python
  import pandas as pd
  df = pd.read_excel("nrdc.xlsx")                 # your downloaded file
  long = df.melt(id_vars=["ZIP", "Utility"],       # change to the real column names
                 value_vars=["2019", "2022", "2023", "2024"],
                 var_name="year", value_name="shutoffs")
  long.columns = ["zip", "utility", "year", "shutoffs"]
  long.to_csv("data/shutoffs.csv", index=False)
  ```
- **If it has monthly rows**, add up the months for each ZIP, year and utility before saving.
- **Use the same name for each utility every time.** Write "New Jersey American Water" the same way on every row, because the bill file matches on that exact name.
- **Leading zeros:** Excel turns 07302 into 7302. The code adds the zero back, so don't worry about it.
- **Known gaps (per NRDC):** Aqua's and Veolia's 2019 data is incomplete, and Veolia didn't file the last three months of 2024. For Veolia's 2024 rows, multiply by 12/9 to estimate a full year, and write that down. You'll mention it in the demo.

### Step 5. Real Census data and boundaries (Sumedh, 5 min)

```
python scripts/fetch_acs.py
python scripts/fetch_geo.py
```
The first gets households, income, poverty rate and race for each ZIP from the U.S. Census. The second gets ZIP outlines for the map. It's already included in this folder, so you only need it if the file goes missing.

### Step 6. Water bills, for the cost burden figure (Shivank, 30 min, optional but recommended)

Open `data/utility_bills.csv`. For each utility, enter a **typical residential annual water bill** from that utility's current rate sheet or bill calculator, and paste the source link in the `source` column. Headwater divides this by the ZIP's median income to estimate how much of a typical household's income goes to water. Your Samsung answer mentions cost burden, so this keeps that sentence true.

---

## DAY 2: Build and test the model

### Step 7. Build the table and check it (Sumedh and Shivank, 30 min)

```
python scripts/build_panel.py
```
Read what it prints, and **compare the total shutoffs per year with NRDC's published totals.** If they're far off, a reshaping step went wrong. Fix it before going further.

### Step 8. Run the honest test (Sumedh, 30 min plus discussion)

```
python scripts/backtest.py
```
Results go to `outputs/results.md`. Here's what the test does:

- It trains on 2022 → 2023 and predicts 2024, a year the model never saw.
- It flags the top 10% of ZIPs, about 35, and asks what share of 2024's shutoffs landed in them. Headwater and the baseline (last year's worst 35 ZIPs) each get a score.
- It counts the **new hot spots**: ZIPs that became top-35 in 2024 without being top-35 in 2023. The baseline can never catch these. How many did Headwater catch?

**Rules for staying honest.** Judges respect this, and it protects you in Q&A.

- **Decide your settings before you look at results:** test year 2024, top 10%. Don't keep re-running with different settings until the numbers look good.
- **Report whatever comes out.** With only one training year, matching the baseline overall while catching a few new hot spots is a real result. Losing to the baseline is also possible. If that happens, say so, and say why: one training year, annual totals, and the post-moratorium rebound. Then explain how monthly data, which the law requires utilities to file, fixes this. That's your next step.
- The model learns each ZIP's change **relative to the statewide trend**. It does this because 2022 → 2023 was the rebound after the shutoff moratorium, when almost every ZIP jumped. Be ready to explain that in one sentence.

**Optional upgrade (if you have time).** NRDC's 2025 supplement for NJ American Water is here: https://www.nrdc.org/sites/default/files/2026-05/njaw-data-2025-supplement-to-new-jersey-water-shutoff.pdf. If it lists shutoffs by ZIP code, add those 2025 rows to `shutoffs.csv` and run:
```
python scripts/backtest.py --test-year 2025
```
That trains on two years instead of one, and tests on 2025, NJ American Water ZIPs only. It's a stronger test.

---

## DAY 3: Polish, record, submit

### Step 9. Check the assistance programs (Cameron, 1 hour)

Open `data/programs.csv`. For every program:
- confirm it still exists and who qualifies, from the program's own website,
- fix the description,
- set `verified` to `yes`,
- replace each `TODO` row with the real program for Aqua, Middlesex and Veolia, or delete the row.

**Don't show a program in the demo that you haven't checked.**

### Step 10. Rehearse and record the demo (all, 2 hours)

```
streamlit run app.py
```
Record your screen. Mac: Cmd+Shift+5. Windows: Win+Alt+R, or OBS. Make the browser full screen, and zoom to 90% if the map and the ZIP card don't fit side by side. Aim for about 2.5 minutes.

1. **(15s) Problem.** "Thousands of New Jersey families lose water service every year. Help exists, but it usually arrives after the shutoff."
2. **(35s) Risk map.** Start on *Outreach priority*. Drag the **"How many ZIP codes can your team reach?"** slider from 33 to 15 and then to 50, and let the map and counts update. Point at an **orange** ZIP: "Last year's list would miss this one. Headwater flags it as emerging."
3. **(25s) One ZIP up close.** Use **Look up a ZIP code** to pick one of the orange (emerging) ZIPs. Show its history chart, the dashed 2025 prediction, and **"What moved this prediction."** Avoid Camden (08105): Headwater predicted a drop, but NRDC's 2025 report shows it rose to 852.
4. **(20s) Shutoffs over time.** Switch to *Shutoffs over time* and drag the **Year** slider from 2019 to "2025 (predicted)."
5. **(40s) Test it yourself.** "We rewound to the end of 2023, so the model never saw 2024." Show the picks, then flip **Reveal what actually happened in 2024**. Read the real numbers honestly: "Headwater's picks held 46%, last year's list 49%. But Headwater caught 2 of the 8 neighborhoods that newly became hot spots, like Riverside and Pennsauken, which last year's list can never catch." Point at the gold ZIPs and the curve.
6. **(15s) Find help.** Switch to **Español**, enter a ZIP, choose "Sí" for a shutoff notice, and show the urgent message and the programs.
7. **(10s) Next steps.** Monthly data, a pilot with a NJ nonprofit, assistance outreach only.

Before recording, rerun `python scripts/backtest.py` with your final settings and check that every number you say matches the app.

Upload to YouTube as **Unlisted**, test the link in a private window, and paste it into Samsung Q2.

### Step 11. Final checks

- [ ] No yellow "Stand-in data" banner anywhere in the video.
- [ ] Every number you say matches `outputs/results.md`.
- [ ] Your Samsung answers match the prototype. If Q2 still says "monthly" ZIP reports, change it to "ZIP-code shutoff reports," because the prototype uses annual totals. Monthly data is the next step.

---

## What each file does

| File | What it does |
|---|---|
| `scripts/make_fake_data.py` | Stand-in data for testing (Step 2 only) |
| `scripts/clean_fake_data.py` | Deletes the stand-in data and its outputs |
| `scripts/fetch_acs.py` | Downloads Census data for each ZIP |
| `scripts/fetch_geo.py` | Downloads ZIP outlines for the map |
| `scripts/build_panel.py` | Combines shutoffs, Census data and bills into one table |
| `scripts/backtest.py` | Trains the model, runs the honest test, makes the forecast |
| `app.py` | The app (map, accuracy test, find help) |
| `data/utility_bills.csv` | You fill in: typical annual bill per utility |
| `data/programs.csv` | You verify: assistance programs |

## If something breaks

- **"shutoffs.csv is missing column(s)"**: the headers must be exactly `zip,year,utility,shutoffs`.
- **"X% of rows have no Census match"**: the ZIP column is probably wrong (text, merged cells, or ZIP+4). Check a few rows.
- **"Need at least two pairs of back-to-back years"**: you need 2022, 2023 and 2024 all present.
- **The map is blank**: run `python scripts/fetch_geo.py`.
- **Anything else**: copy the full error message and ask for help with that exact text.
