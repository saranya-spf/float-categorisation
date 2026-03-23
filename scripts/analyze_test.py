"""Quick analysis of test data patterns for misclassified classes."""

import pandas as pd

test = pd.read_csv("data/test_data.csv")
test.columns = test.columns.str.strip().str.casefold()
print("Test shape:", test.shape)

targets = {
    "5784": "5784",
    "5785": "5785",
    "5410": "5410",
    "cwb": "cwb",
    "5700": "5700",
    "5772": "5772",
    "5773": "5773",
    "account payable": "account payable",
}

for name, pattern in targets.items():
    rows = test[test["gl code"].str.strip().str.lower().str.contains(pattern, na=False)]
    if len(rows) > 0:
        print(f"\n=== GL containing '{pattern}' ({len(rows)} rows) ===")
        for _, r in rows.head(8).iterrows():
            tt = str(r.get("transaction type", "")).strip()[:25]
            payee = str(r.get("payee", "")).strip()[:40]
            detail = str(r.get("transaction detail", "")).strip()[:55]
            print(f"  type={tt:25s} payee={payee:40s} detail={detail}")

# Also check training data for "cash cwb" and "5410"
print("\n\n=== TRAINING DATA checks ===")
train = pd.read_csv("data/Copy of Float Sample Data - Sheet5.csv")
train.columns = train.columns.str.strip().str.casefold()

for pattern in ["cwb", "5410"]:
    rows = train[
        train["gl code"].str.strip().str.lower().str.contains(pattern, na=False)
    ]
    print(f"\nTrain rows with '{pattern}' in gl_code: {len(rows)}")
    if len(rows) > 0:
        for _, r in rows.head(5).iterrows():
            tt = str(r.get("transaction type", "")).strip()[:25]
            payee = str(r.get("payee", "")).strip()[:40]
            detail = str(r.get("transaction detail", "")).strip()[:55]
            print(f"  type={tt:25s} payee={payee:40s} detail={detail}")

# Check transaction type distributions for key classes
print("\n\n=== TRANSACTION TYPE by GL CODE (test data) ===")
for pattern in ["5784", "5785", "5410", "5700", "5773"]:
    rows = test[test["gl code"].str.strip().str.lower().str.contains(pattern, na=False)]
    if len(rows) > 0:
        print(
            f"\n{pattern}: transaction types = {rows['transaction type'].value_counts().to_dict()}"
        )
