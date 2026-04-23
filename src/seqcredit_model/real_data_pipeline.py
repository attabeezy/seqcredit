"""
Spark-based pipeline for real Telecel Ghana MoMo data.

Reads the raw transaction table from Databricks, derives credit risk labels from
loan lifecycle signals, engineers user-level aggregate features, and writes
user_features.csv + user_labels.csv that are directly compatible with
CreditRiskDataLoader in credit_model.py.

Temporal design
---------------
For each borrower, the *index loan* is their most recent loan disbursement.
Labels are derived from repayment/penalty events that occur AFTER the index
loan date.  Features are computed from all transactions BEFORE the index loan
date.  This eliminates data leakage: repayment behaviour on the target loan
cannot be used as a predictor of that same loan's outcome.

Usage (Databricks notebook):
    from seqcredit_model.real_data_pipeline import build_pipeline
    df = spark.sql("SELECT * FROM melodatabricks616.default.yara_dump_table")
    build_pipeline(df)
"""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── Constants ────────────────────────────────────────────────────────────────

LENDER_ID = "E7C89F8C4A27F173"

LOAN_DISBURSEMENT_TYPE      = "Loan Payment Via API"
LOAN_PRINCIPAL_TYPE         = "Loan Principal Collection via API"
LOAN_INTEREST_TYPE          = "Loan Interest via API"
LOAN_PENALTY_TYPE           = "Loan Penalty via API"

CUSTOMER_ACCOUNT_TYPE       = "M-Pesa Account For Customer"

# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_timestamp(df):
    """Parse Oracle-style 'DD-MON-YY HH.MI.SS.FFFFFFFFF' → Spark timestamp.

    Handles variable-length seconds (1 or 2 digits) with a coalesce over two
    format patterns so that all rows are parsed correctly.
    """
    return df.withColumn(
        "ts",
        F.expr("""
            coalesce(
                try_to_timestamp(substring(TRANSACTION_TIMESTAMP, 1, 18), 'dd-MMM-yy HH.mm.ss'),
                try_to_timestamp(substring(TRANSACTION_TIMESTAMP, 1, 17), 'dd-MMM-yy HH.mm.s')
            )
        """),
    )


def _derive_loan_cutoffs(df):
    """Return a Spark DataFrame (user_id, last_loan_ts) for every borrower.

    ``last_loan_ts`` is the timestamp of the borrower's most recent loan
    disbursement — the *index loan*.  Features are computed from transactions
    strictly before this date; labels are derived from events after it.

    Parameters
    ----------
    df : Spark DataFrame
        Raw transaction table with the ``ts`` column already added by
        ``_parse_timestamp``.
    """
    return (
        df.filter(
            (F.col("TRANSACTION_TYPE") == LOAN_DISBURSEMENT_TYPE)
            & (F.col("DEBIT_PARTY_ID") == LENDER_ID)
            & (F.col("CREDIT_PARTY_TYPE") == "Customer")
        )
        .withColumnRenamed("CREDIT_PARTY_ID", "user_id")
        .groupBy("user_id")
        .agg(F.max("ts").alias("last_loan_ts"))
    )


def _categorize_txtype(df):
    """Add txtype_cat column: map 243 raw transaction types → 9 semantic categories."""
    t = F.col("TRANSACTION_TYPE")
    return df.withColumn(
        "txtype_cat",
        F.when(t == LOAN_DISBURSEMENT_TYPE,   "loan_disbursement")
         .when(t == LOAN_PRINCIPAL_TYPE,       "loan_repayment_principal")
         .when(t == LOAN_INTEREST_TYPE,        "loan_repayment_interest")
         .when(t == LOAN_PENALTY_TYPE,         "loan_penalty")
         .when(t.like("%Withdrawal%"),         "cash_out")
         .when(t.like("%Deposit at Agent%"),   "cash_in")
         .when(
             t.like("%Airtime%") | t.like("%Data Purchase%") | t.like("%EVD Top%"),
             "airtime_data",
         )
         .when(
             t.like("%Ghipss%") | t.like("%GHIPSS%") | t.like("%GHiPSS%")
             | t.like("%Transfer%") | t.like("%P2P%"),
             "transfer",
         )
         .when(
             t.like("%Pay Bill%") | t.like("%Online Payment%") | t.like("%Direct Debit%"),
             "payment",
         )
         .when(t.like("%Betting%"),            "betting")
         .when(t.like("%FSI%"),                "fsi")
         .otherwise("other"),
    )


# ── Public API ───────────────────────────────────────────────────────────────

def derive_labels(df, cutoffs_spark):
    """Derive a three-class credit risk label for every borrower.

    Labels are based on repayment / penalty events that occur STRICTLY AFTER
    each borrower's index loan (their most recent disbursement).  This prevents
    the repayment signals used to construct the label from also appearing as
    input features.

    Label scheme (mirrors the synthetic data convention):
        0 — Good    : made at least one repayment after index loan, no penalties
        1 — Risky   : has at least one penalty after index loan but did repay
        2 — Default : never made a principal or interest repayment after index loan

    Parameters
    ----------
    df : Spark DataFrame
        Raw transaction table with ``ts`` column added by ``_parse_timestamp``.
    cutoffs_spark : Spark DataFrame
        (user_id, last_loan_ts) — output of ``_derive_loan_cutoffs``.

    Returns
    -------
    pandas.DataFrame with columns: user_id, credit_risk_label
    """
    # Penalised AFTER their index loan
    penalised = (
        df.filter(
            (F.col("TRANSACTION_TYPE") == LOAN_PENALTY_TYPE)
            & (F.col("DEBIT_PARTY_TYPE") == "Customer")
        )
        .withColumnRenamed("DEBIT_PARTY_ID", "user_id")
        .join(cutoffs_spark, on="user_id", how="inner")
        .filter(F.col("ts") > F.col("last_loan_ts"))
        .select("user_id")
        .distinct()
        .withColumn("has_penalty", F.lit(1))
    )

    # Repayers AFTER their index loan
    repayers = (
        df.filter(
            F.col("TRANSACTION_TYPE").isin([LOAN_PRINCIPAL_TYPE, LOAN_INTEREST_TYPE])
            & (F.col("DEBIT_PARTY_TYPE") == "Customer")
        )
        .withColumnRenamed("DEBIT_PARTY_ID", "user_id")
        .join(cutoffs_spark, on="user_id", how="inner")
        .filter(F.col("ts") > F.col("last_loan_ts"))
        .select("user_id")
        .distinct()
        .withColumn("has_repayment", F.lit(1))
    )

    labels_spark = (
        cutoffs_spark
        .join(penalised, on="user_id", how="left")
        .join(repayers,  on="user_id", how="left")
        .withColumn(
            "credit_risk_label",
            F.when(F.col("has_repayment").isNull(), F.lit(2))   # never repaid → default
             .when(F.col("has_penalty") == 1,        F.lit(1))  # penalised but repaid → risky
             .otherwise(F.lit(0)),                              # clean repayment → good
        )
        .select("user_id", "credit_risk_label")
    )

    return labels_spark.toPandas()


def engineer_features(df, borrower_ids_spark, cutoffs_spark):
    """Compute user-level aggregate features from the raw transaction table.

    Only transactions that occurred STRICTLY BEFORE each borrower's index loan
    (``last_loan_ts`` from ``cutoffs_spark``) are included.  This ensures no
    post-loan repayment or penalty signals leak into the feature set.

    For repeat borrowers (n_loans_received > 0 in the pre-loan window), past
    repayment features such as ``n_principal_repayments`` and
    ``loan_repayment_ratio`` reflect behaviour on *previous* loans — a
    legitimate credit signal.  For first-time borrowers these will be zero.

    Features are split by transaction direction:
      - Outgoing (customer is DEBIT party): spending behaviour, temporal patterns,
        prior repayment activity, recipient diversity.
      - Incoming (customer is CREDIT party): inflow amounts, prior loan
        disbursements received, deposit activity.

    Combined derived features: net flow, loan repayment ratio, account age, etc.

    Parameters
    ----------
    df : Spark DataFrame
        Raw transaction table with ``ts`` and ``txtype_cat`` columns already
        added (call ``_parse_timestamp`` and ``_categorize_txtype`` first).
    borrower_ids_spark : Spark DataFrame with a single column 'user_id'
    cutoffs_spark : Spark DataFrame (user_id, last_loan_ts)

    Returns
    -------
    pandas.DataFrame — one row per borrower, user_id + all feature columns
    """
    if "ts" not in df.columns:
        df = _parse_timestamp(df)
    if "txtype_cat" not in df.columns:
        df = _categorize_txtype(df)

    # Join borrower list with cutoffs to get (user_id, last_loan_ts)
    borrower_cutoffs = borrower_ids_spark.join(cutoffs_spark, on="user_id", how="inner")

    # ── Outgoing transactions (customer initiates) BEFORE index loan ──────────
    out_df = (
        df.filter(
            (F.col("DEBIT_PARTY_TYPE") == "Customer")
            & (F.col("DEBIT_ACCOUNT_TYPE") == CUSTOMER_ACCOUNT_TYPE)
        )
        .withColumnRenamed("DEBIT_PARTY_ID", "user_id")
        .join(borrower_cutoffs, on="user_id", how="inner")
        .filter(F.col("ts") < F.col("last_loan_ts"))
    )

    # Recipient concentration requires a sub-aggregation:
    # max single-recipient transaction count / total outgoing transactions
    max_recip = (
        out_df.groupBy("user_id", "CREDIT_PARTY_ID")
        .count()
        .groupBy("user_id")
        .agg(F.max("count").alias("max_recip_count"))
    )

    out_agg = (
        out_df.groupBy("user_id").agg(
            # Volume and amount distribution
            F.count("*").alias("obs_txn_count"),
            F.sum("TRANSACTION_AMOUNT").alias("total_volume"),
            F.avg("TRANSACTION_AMOUNT").alias("avg_transaction_amount"),
            F.expr("percentile_approx(TRANSACTION_AMOUNT, 0.5)").alias("median_transaction_amount"),
            F.stddev("TRANSACTION_AMOUNT").alias("std_transaction_amount"),
            F.max("TRANSACTION_AMOUNT").alias("max_transaction_amount"),
            F.min("TRANSACTION_AMOUNT").alias("min_transaction_amount"),
            # Transaction type mix (as fraction of outgoing transactions)
            F.avg(F.when(F.col("txtype_cat") == "transfer",   1.0).otherwise(0.0)).alias("pct_transfers"),
            F.avg(F.when(F.col("txtype_cat") == "cash_out",   1.0).otherwise(0.0)).alias("pct_cashouts"),
            F.avg(F.when(F.col("txtype_cat") == "payment",    1.0).otherwise(0.0)).alias("pct_payments"),
            F.avg(F.when(F.col("txtype_cat") == "airtime_data", 1.0).otherwise(0.0)).alias("pct_airtime"),
            F.avg(F.when(F.col("txtype_cat") == "betting",    1.0).otherwise(0.0)).alias("pct_betting"),
            F.avg(F.when(F.col("txtype_cat") == "fsi",        1.0).otherwise(0.0)).alias("pct_fsi"),
            # Temporal patterns
            F.avg(
                F.when((F.hour("ts") >= 22) | (F.hour("ts") < 6), 1.0).otherwise(0.0)
            ).alias("pct_night_txns"),
            F.avg(F.when(F.hour("ts") < 6, 1.0).otherwise(0.0)).alias("pct_early_morning_txns"),
            F.avg(
                F.when(F.dayofweek("ts").isin([1, 7]), 1.0).otherwise(0.0)
            ).alias("pct_weekend_txns"),
            # Loan repayment behaviour (outgoing = customer repaying)
            F.sum(F.when(F.col("txtype_cat") == "loan_repayment_principal", 1).otherwise(0)).alias("n_principal_repayments"),
            F.sum(F.when(F.col("txtype_cat") == "loan_repayment_interest",  1).otherwise(0)).alias("n_interest_repayments"),
            F.sum(F.when(F.col("txtype_cat") == "loan_penalty",             1).otherwise(0)).alias("n_penalty_txns"),
            F.sum(
                F.when(F.col("txtype_cat") == "loan_repayment_principal", F.col("TRANSACTION_AMOUNT")).otherwise(0.0)
            ).alias("total_principal_repaid"),
            F.sum(
                F.when(F.col("txtype_cat") == "loan_repayment_interest", F.col("TRANSACTION_AMOUNT")).otherwise(0.0)
            ).alias("total_interest_repaid"),
            F.sum(
                F.when(F.col("txtype_cat") == "loan_penalty", F.col("TRANSACTION_AMOUNT")).otherwise(0.0)
            ).alias("total_penalties_paid"),
            # Behavioural diversity — top ablation signal in synthetic experiments
            F.countDistinct("CREDIT_PARTY_ID").alias("unique_recipients"),
            F.countDistinct("txtype_cat").alias("unique_txn_types"),
            # Timing anchors for account age
            F.min("ts").alias("first_out_ts"),
            F.max("ts").alias("last_out_ts"),
        )
        .join(max_recip, on="user_id", how="left")
    )

    # ── Incoming transactions (customer receives) BEFORE index loan ───────────
    in_df = (
        df.filter(
            (F.col("CREDIT_PARTY_TYPE") == "Customer")
            & (F.col("CREDIT_ACCOUNT_TYPE") == CUSTOMER_ACCOUNT_TYPE)
        )
        .withColumnRenamed("CREDIT_PARTY_ID", "user_id")
        .join(borrower_cutoffs, on="user_id", how="inner")
        .filter(F.col("ts") < F.col("last_loan_ts"))
    )

    in_agg = in_df.groupBy("user_id").agg(
        F.sum("TRANSACTION_AMOUNT").alias("total_inflow"),
        F.avg("TRANSACTION_AMOUNT").alias("avg_inflow_amount"),
        F.sum(
            F.when(F.col("txtype_cat") == "cash_in", F.col("TRANSACTION_AMOUNT")).otherwise(0.0)
        ).alias("total_cash_in"),
        # Loan disbursements received — named to match _engineer_loan_features output
        F.sum(F.when(F.col("txtype_cat") == "loan_disbursement", 1).otherwise(0)).alias("n_loans_received"),
        F.sum(
            F.when(F.col("txtype_cat") == "loan_disbursement", F.col("TRANSACTION_AMOUNT")).otherwise(0.0)
        ).alias("total_loan_volume"),
        F.avg(
            F.when(F.col("txtype_cat") == "loan_disbursement", F.col("TRANSACTION_AMOUNT"))
        ).alias("avg_loan_amount"),
        F.max(
            F.when(F.col("txtype_cat") == "loan_disbursement", F.col("TRANSACTION_AMOUNT"))
        ).alias("max_loan_amount"),
        F.countDistinct("DEBIT_PARTY_ID").alias("unique_senders"),
        F.min("ts").alias("first_in_ts"),
        F.max("ts").alias("last_in_ts"),
    )

    # ── Combine and derive final features ────────────────────────────────────
    features = out_agg.join(in_agg, on="user_id", how="outer")

    # Fill numeric nulls (users with no outgoing or no incoming activity)
    fill_zero = [
        c for c in features.columns
        if c not in ("user_id", "first_out_ts", "last_out_ts", "first_in_ts", "last_in_ts")
    ]
    features = features.fillna(0.0, subset=fill_zero)

    first_ts = F.least(F.col("first_out_ts"), F.col("first_in_ts"))
    last_ts  = F.greatest(F.col("last_out_ts"), F.col("last_in_ts"))

    features = (
        features
        # Temporal span and pace
        .withColumn("account_age_days", F.datediff(last_ts, first_ts).cast("double"))
        .withColumn("transactions_per_day",
            F.col("obs_txn_count") / (F.col("account_age_days") + 1))
        .withColumn("avg_hours_between_txns",
            (F.col("account_age_days") * 24.0) / (F.col("obs_txn_count") + 1))
        # Coefficient of variation for transaction amounts
        .withColumn("cv_transaction_amount",
            F.col("std_transaction_amount") / (F.col("avg_transaction_amount") + 1))
        # Recipient concentration: fraction of txns to the single most-used recipient
        .withColumn("recipient_concentration",
            F.col("max_recip_count") / (F.col("obs_txn_count") + 1))
        # Cash flow balance
        .withColumn("net_flow", F.col("total_inflow") - F.col("total_volume"))
        .withColumn("inflow_outflow_ratio",
            F.col("total_inflow") / (F.col("total_volume") + 1))
        # Loan repayment summary
        .withColumn("total_repaid",
            F.col("total_principal_repaid") + F.col("total_interest_repaid"))
        .withColumn("loan_repayment_ratio",
            F.col("total_repaid") / (F.col("total_loan_volume") + 1))
        # Loan-specific columns expected by _engineer_loan_features interface
        .withColumn("loan_to_total_volume_ratio",
            F.col("total_loan_volume") / (F.col("total_volume") + F.col("total_inflow") + 1))
        .withColumn("pct_credit_transactions",
            F.col("n_loans_received") / (F.col("obs_txn_count") + 1))
        # Balance-at-loan fields: not available in real data — set to 0 so the
        # existing model interface is satisfied without error
        .withColumn("loan_timing_in_sequence",           F.lit(0.0))
        .withColumn("avg_balance_at_loan",               F.lit(0.0))
        .withColumn("min_balance_at_loan",               F.lit(0.0))
        .withColumn("balance_to_loan_ratio_at_disbursement", F.lit(0.0))
        # Drop timestamp helpers
        .drop("first_out_ts", "last_out_ts", "first_in_ts", "last_in_ts", "max_recip_count")
    )

    return features.toPandas()


def build_pipeline(df, output_dir=None):
    """Run the full real-data pipeline and save outputs to CSV.

    Derives labels → engineers features → writes user_features.csv and
    user_labels.csv to output_dir (defaults to the project data/ directory).

    Parameters
    ----------
    df : Spark DataFrame
        Raw transaction table, e.g.
        ``spark.sql("SELECT * FROM melodatabricks616.default.yara_dump_table")``
    output_dir : str or Path, optional
        Directory to write output CSVs. Defaults to the project's data/ dir
        as resolved by config.py. In Databricks you may prefer a /dbfs/ path,
        e.g. ``output_dir='/dbfs/tmp/seqcredit/'``.
    """
    if output_dir is None:
        try:
            from seqcredit_model.config import DATA_DIR
        except ModuleNotFoundError:
            from config import DATA_DIR
        out_path = Path(DATA_DIR)
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Real Data Pipeline — Telecel Ghana MoMo")
    print("=" * 60)

    spark = SparkSession.builder.getOrCreate()

    # Parse timestamps once; pass pre-parsed df to all downstream functions
    # so Spark can cache/reuse the parsed column across multiple actions.
    df_ts = _parse_timestamp(df)
    df_ts = _categorize_txtype(df_ts)

    # ── Step 1: Loan cutoffs (index loan date per borrower) ───────────────────
    print("\n[1/4] Computing index-loan cutoff dates per borrower...")
    cutoffs_spark = _derive_loan_cutoffs(df_ts)
    n_borrowers = cutoffs_spark.count()
    print(f"  Borrowers (with ≥1 loan): {n_borrowers:,}")

    # ── Step 2: Labels (post-cutoff repayment / penalty signals) ─────────────
    print("\n[2/4] Deriving labels from post-index-loan signals...")
    labels_pd = derive_labels(df_ts, cutoffs_spark)
    total     = len(labels_pd)
    n_good    = (labels_pd["credit_risk_label"] == 0).sum()
    n_risky   = (labels_pd["credit_risk_label"] == 1).sum()
    n_default = (labels_pd["credit_risk_label"] == 2).sum()
    print(f"  Total borrowers : {total:,}")
    print(f"  Good      (0)   : {n_good:,}  ({n_good / total:.1%})")
    print(f"  Risky     (1)   : {n_risky:,}  ({n_risky / total:.1%})")
    print(f"  Default   (2)   : {n_default:,}  ({n_default / total:.1%})")

    # ── Step 3: Features (pre-cutoff transactions only) ───────────────────────
    print("\n[3/4] Engineering user-level features (pre-index-loan window)...")
    borrower_ids_spark = spark.createDataFrame(
        labels_pd[["user_id"]], schema="user_id string"
    )
    features_pd = engineer_features(df_ts, borrower_ids_spark, cutoffs_spark)
    print(f"  Shape   : {features_pd.shape}")
    print(f"  Columns : {list(features_pd.columns)}")

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    print("\n[4/4] Saving CSVs...")
    features_path = out_path / "user_features.csv"
    labels_path   = out_path / "user_labels.csv"
    features_pd.to_csv(features_path, index=False)
    labels_pd.to_csv(labels_path, index=False)
    print(f"  {features_path}")
    print(f"  {labels_path}")

    print("\nDone. Next: run src/seqcredit_model/run_cv_benchmark.py")
    return features_pd, labels_pd
