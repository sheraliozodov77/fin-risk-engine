# Data Contract — Financial Transactions Dataset

**Purpose:** Authoritative schema, join keys, and data dictionary for the Fraud Early-Warning pipeline.  
**Sources:** CaixaBank Tech 2024 AI Hackathon (5 files).

---

## 1. File Inventory

| File | Format | Rows (approx.) | Size (approx.) |
|------|--------|-----------------|----------------|
| transactions_data.csv | CSV | ~13.3M | Large |
| cards_data.csv | CSV | ~6,146 | ~510 KB |
| users_data.csv | CSV | ~1,999 | ~165 KB |
| mcc_codes.json | JSON | 109 keys | ~5 KB |
| train_fraud_labels.json | JSON | ~8.9M keys | ~159 MB |

---

## 2. Transactions (Fact Table)

**File:** `transactions_data.csv`  
**Primary key:** `id` (transaction ID; integer in CSV; use string for label join).

| Column | Type | Description | Notes |
|--------|------|-------------|--------|
| id | int | Transaction ID | Join key for fraud labels (as string) |
| date | datetime | Transaction timestamp | Normalize to UTC; sort by this |
| client_id | int | Customer ID | → users.id |
| card_id | int | Card ID | → cards.id |
| amount | str | Amount with $ (e.g. $-77.00, $14.57) | Parse to float |
| use_chip | str | e.g. "Swipe Transaction" | Categorical |
| merchant_id | int | Merchant identifier | |
| merchant_city | str | City | |
| merchant_state | str | State | |
| zip | float | ZIP code | |
| mcc | int/str | Merchant category code | → mcc_codes key (string) |
| errors | str | Error field | May be empty |

---

## 3. Cards (Dimension)

**File:** `cards_data.csv`  
**Primary key:** `id` (card_id).

| Column | Type | Description | Notes |
|--------|------|-------------|--------|
| id | int | Card ID | Join from transactions.card_id |
| client_id | int | Customer ID | → users.id |
| card_brand | str | Visa, Mastercard, Other | |
| card_type | str | Debit, Credit, Other | |
| card_number | str | Masked number | Do not use as feature (PII) |
| expires | str | e.g. 12/2022 | |
| cvv | int | CVV | Do not use as feature (PII) |
| has_chip | str | YES/NO | Normalize to bool |
| num_cards_issued | int | | |
| credit_limit | str | e.g. $24295 | Parse to float |
| acct_open_date | str | e.g. 09/2002 | Parse to date |
| year_pin_last_changed | int | | |
| card_on_dark_web | str | No/Yes | High-risk signal; normalize to bool |

---

## 4. Users (Dimension)

**File:** `users_data.csv`  
**Primary key:** `id` (client_id).

| Column | Type | Description | Notes |
|--------|------|-------------|--------|
| id | int | Customer ID | Join from transactions.client_id, cards.client_id |
| current_age | int | | |
| retirement_age | int | | |
| birth_year | int | | |
| birth_month | int | 1–12 | |
| gender | str | Female, Male | |
| address | str | | PII; optional for geo features only |
| latitude | float | | |
| longitude | float | | |
| per_capita_income | str | e.g. $29278 | Parse to float |
| yearly_income | str | e.g. $59696 | Parse to float |
| total_debt | str | e.g. $127613 | Parse to float |
| credit_score | int | | |
| num_credit_cards | int | | |

---

## 5. MCC Codes (Dimension)

**File:** `mcc_codes.json`  
**Structure:** Flat object `{ "mcc_code_string": "Description", ... }`  
**Key:** MCC code as string (e.g. "5499"). Transaction `mcc` must be normalized to string for lookup.

---

## 6. Fraud Labels

**File:** `train_fraud_labels.json`  
**Structure:** `{ "target": { "transaction_id_str": "Yes" | "No", ... } }`  
**Keys:** Transaction `id` as string.  
**Values:** "Yes" = fraud (1), "No" = non-fraud (0).  
**Coverage:** Subset of all transactions (~8.9M labeled); unlabeled must not be treated as non-fraud.

**Observed stats (as of audit):**
- Label count: ~8,914,963  
- Fraud ("Yes"): ~13,332  
- Fraud rate (among labeled): ~0.15%

---

## 7. Join Diagram

```
transactions (id, card_id, client_id, mcc, ...)
    │
    ├── card_id  ──→  cards.id
    │                     └── client_id  ──→  users.id
    ├── client_id  ──→  users.id
    ├── mcc  (as string)  ──→  mcc_codes key
    └── id  (as string)  ──→  train_fraud_labels["target"] key
```

---

## 8. Data Types (Parsing Rules)

| Source | Field | Raw | Parsed |
|--------|--------|-----|--------|
| transactions | date | str | datetime (UTC) |
| transactions | amount | e.g. "$-77.00" | float |
| cards | credit_limit | e.g. "$24295" | float |
| cards | acct_open_date | e.g. "09/2002" | date |
| cards | has_chip | YES/NO | bool |
| cards | card_on_dark_web | No/Yes | bool |
| users | per_capita_income, yearly_income, total_debt | e.g. "$29278" | float |
| labels | target value | "Yes"/"No" | 1/0 |

---

## 9. Known Issues / Assumptions

- **Partial labeling:** Only a subset of transactions have fraud labels; training uses only labeled rows.
- **Transaction id type:** Labels use string keys; join `transactions.id.astype(str)` to label key.
- **MCC in transactions:** May be int; normalize to string for mcc_codes lookup.
- **Negative amounts:** Present (e.g. $-77.00); do not drop without business rule; flag for EDA.

---

*Last updated: Kick-off.*
