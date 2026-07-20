-- ---------------------------------------------------------------------------
-- Stage: raw -> clean
--
-- Portability contract: this statement must execute unchanged on BigQuery
-- standard SQL and on DuckDB. That rules out SAFE_DIVIDE, FARM_FINGERPRINT,
-- backtick quoting and FLOAT64/DOUBLE casts. Division is guarded with the
-- portable NULLIF form and floats are forced with `* 1.0`.
--
-- Cleaning decisions are driven by the published codebook (Yeh & Lien, 2009)
-- versus what the file actually contains:
--
--   EDUCATION  codebook defines 1=graduate school, 2=university, 3=high school,
--              4=others. The data also contains 0, 5 and 6 (345 rows) with no
--              documented meaning. Collapsed into 4 ("other") rather than
--              dropped: the rows are otherwise valid, and inventing three
--              undocumented categories invites the model to fit noise.
--
--   MARRIAGE   codebook defines 1=married, 2=single, 3=others. The data also
--              contains 0 (54 rows). Collapsed into 3.
--
--   PAY_*      codebook documents -1 = paid duly and 1..9 = months of delay, but
--              the data also holds -2 and 0. The accepted reading is
--              -2 = no consumption that month, 0 = revolving credit (paid at
--              least the minimum, carried a balance). Preserved as distinct
--              values because "used no credit" and "revolved a balance" are very
--              different risk signals and must not be merged.
--
-- Negative bill amounts (overpayment / credit balance) and balances above the
-- credit limit are left untouched: both are real account states, not errors, and
-- both are predictive.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE {clean_table} AS
WITH deduplicated AS (
    -- The source has no duplicate ids today. The guard is here so that a
    -- re-delivered or partially replayed load cannot silently double-count
    -- clients downstream.
    SELECT *
    FROM (
        SELECT
            r.*,
            ROW_NUMBER() OVER (PARTITION BY client_id ORDER BY ingested_at DESC) AS row_rank
        FROM {raw_table} AS r
    ) AS ranked
    WHERE row_rank = 1
)
SELECT
    client_id,
    limit_bal,

    -- Protected attribute: retained so the fairness audit can slice on it, but
    -- excluded from the model feature set by config.
    sex,

    CASE WHEN education IN (1, 2, 3) THEN education ELSE 4 END AS education,
    CASE WHEN marriage IN (1, 2, 3) THEN marriage ELSE 3 END AS marriage,
    age,

    pay_status_1, pay_status_2, pay_status_3,
    pay_status_4, pay_status_5, pay_status_6,

    bill_amt_1, bill_amt_2, bill_amt_3,
    bill_amt_4, bill_amt_5, bill_amt_6,

    pay_amt_1, pay_amt_2, pay_amt_3,
    pay_amt_4, pay_amt_5, pay_amt_6,

    default_next_month,
    ingested_at
FROM deduplicated
WHERE limit_bal > 0        -- a zero/negative limit makes every utilisation ratio meaningless
  AND age BETWEEN 18 AND 100;
