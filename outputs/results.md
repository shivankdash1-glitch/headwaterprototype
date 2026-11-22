# Headwater backtest results


Trained on years predicting [2023]; tested on 2024 (313 ZIP codes), a year the model never saw.

Flagging the top 31 ZIP codes (10%):
- Headwater's flagged ZIPs held **46.4%** of 2024's shutoffs.
- The persistence baseline (last year's worst ZIPs) held **49.0%**.
- The best possible choice of 31 ZIPs would hold 53.9%.
- 8 ZIPs became hot spots in 2024 without being one the year before. Headwater flagged **2** of them; the baseline, by design, flagged 0.
- Rank correlation with actual shutoffs: Headwater 0.87, baseline 0.90.

| Share of ZIPs flagged | Headwater capture | Baseline capture | New hot spots caught |
|---|---|---|---|
| top 16 | 27.0% | 36.0% | 0 of 3 |
| top 31 | 46.4% | 49.0% | 2 of 8 |
| top 63 | 67.3% | 70.2% | 2 of 13 |

What the model relies on most (on the test year):
- Shutoffs this year: 0.230
- Residential water customers: 0.074
- Median household income: 0.017
- utility: Veolia New Jersey: 0.001
- Number of utilities serving the ZIP: 0.000
- utility: Aqua New Jersey: -0.001