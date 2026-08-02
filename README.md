# Anti-Money Laundering: Behavioral Account Profiling

Account-level behavioral analysis on the PKDD'99 "Berka" Czech bank
dataset: 4,500 accounts, 682 loans, and just over a million individual
transactions (1993-1998). There's no fraud/AML label anywhere in this
data (it's ordinary retail banking activity, not a labeled benchmark),
so this project treats that honestly: unsupervised anomaly detection to
flag unusual account behavior with no fabricated label, and a genuinely
supervised model on the one real outcome this data does have, whether a
loan was repaid.

**Live report:** https://nik8x.github.io/Anti_money_Laundering/

## Data

The 8 Berka tables (account, client, disposition, district, card, loan,
order, transaction), sourced from the [jlacko/berka-dataset](https://github.com/jlacko/berka-dataset)
mirror of the original PKDD'99 Discovery Challenge dataset. Not committed
to this repo, `00_data_setup_eda.ipynb` downloads all 8 tables
automatically on first run (largest is the ~66MB transaction log).

## Notebooks

| Notebook | What it covers |
|---|---|
| `00_data_setup_eda.ipynb` | Downloads all 8 tables, proper `left`-joined account view (every account stays in the picture, not just the small subset with a loan *and* a card *and* transactions), loan/card/district overview. |
| `01_feature_engineering.ipynb` | One row per account: transaction counts/amounts/balance behavior, tenure, recency, standing orders, all vectorized `groupby` aggregation. |
| `02_anomaly_detection.ipynb` | Isolation Forest and Local Outlier Factor, run independently, flagging accounts with unusual behavior, no fabricated label, cross-checked against loan outcomes only as a partial, honest signal. |
| `03_loan_risk_model.ipynb` | Logistic Regression vs Random Forest predicting bad loan outcomes (not repaid / in debt) from account behavior, a real, independent target, evaluated with PR-AUC/ROC-AUC/recall rather than accuracy. |

Run them in order (00 to 03), each stage loads the previous stage's saved
features/results from `outputs/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Key findings

- **Two independent anomaly-detection methods agree far more than chance
  would predict.** Isolation Forest and Local Outlier Factor each flag
  225 accounts (5%); 37 accounts get flagged by *both*, about 3.3x more
  overlap than the ~11 expected if the two methods were flagging randomly.
- **The anomaly flag has a partial, honest link to bad loan outcomes.**
  Among the 682 accounts that actually have a loan, those flagged as
  anomalous by both methods have a 16.7% bad-loan rate vs 11.1% overall,
  a real but modest signal, and only checkable for the minority of
  accounts that have a loan at all.
- **Predicting bad loans from account behavior works well, honestly.**
  A class-weighted Logistic Regression reaches **test PR-AUC 0.847**
  (ROC-AUC 0.962) against an 11.1% baseline positive rate, average
  account balance and its volatility are the strongest signals, well
  ahead of raw transaction counts.
- **Loan risk and general behavioral anomaly are different signals, not
  proxies for each other.** Accounts in the top 15% of predicted loan
  risk get flagged as behaviorally anomalous at almost exactly the same
  rate as everyone else (23.1% vs 22.8%), a good reminder that "unusual"
  and "risky" aren't the same thing, and one honest null result is more
  useful than forcing a headline finding that isn't really there.

## Future work

- Extend the anomaly-detection stage with an autoencoder reconstruction-error
  score, to compare against Isolation Forest/LOF with a third, differently-shaped method.
- Build a proper time-series view per account (month-over-month balance
  and transaction-count trends) rather than static aggregate features,
  AML detection in practice leans heavily on *changes* in behavior, not
  just its average shape.
- Bring in richer per-table EDA on transaction counterparty bank
  diversity as an additional feature source.

Full detail, code, and every number above are in the notebooks and the
[live report](https://nik8x.github.io/Anti_money_Laundering/).
