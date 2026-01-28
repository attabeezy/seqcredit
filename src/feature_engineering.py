"""
Temporal Feature Engineering for Real Mobile Money Transaction Data

Location: src/feature_engineering.py

Import path:
    from src.feature_engineering import TemporalTransactionFeatureEngineer

Or from project root:
    import sys
    sys.path.append('src')
    from feature_engineering import TemporalTransactionFeatureEngineer

This demonstrates the COMPLETE feature engineering pipeline on actual transaction data
with timestamps, balance tracking, and counterparty information.

This is what you need for Papers A & B!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TemporalTransactionFeatureEngineer:
    """
    Extract temporal features from sequential transaction data.

    Implements all 7 feature categories from the framework:
    1. Transaction-level static features
    2. Categorical encodings
    3. Risk indicators (derived)
    4. Interaction features
    5. User-level aggregates
    6. Temporal/Sequential features
    7. Balance dynamics
    """

    def __init__(self):
        self.feature_names = []

    def extract_all_features(self, df, windows=[3, 7, 14, 30]):
        """
        Extract complete feature set from transaction sequence.

        Args:
            df: DataFrame with columns [TRANSACTION DATE, AMOUNT, TRANS. TYPE,
                BAL BEFORE, BAL AFTER, etc.]
            windows: List of rolling window sizes (in days)

        Returns:
            DataFrame with engineered features
        """

        df = df.copy()
        df['TRANSACTION DATE'] = pd.to_datetime(df['TRANSACTION DATE'])
        df = df.sort_values('TRANSACTION DATE').reset_index(drop=True)

        # ============================================
        # CATEGORY 1: TRANSACTION-LEVEL STATIC FEATURES
        # ============================================

        # Amount transformations
        df['log_amount'] = np.log1p(df['AMOUNT'])
        df['sqrt_amount'] = np.sqrt(df['AMOUNT'])
        df['amount_squared'] = df['AMOUNT'] ** 2

        # Amount categories
        df['is_micro_txn'] = (df['AMOUNT'] < 10).astype(int)
        df['is_small_txn'] = ((df['AMOUNT'] >= 10) & (df['AMOUNT'] < 50)).astype(int)
        df['is_medium_txn'] = ((df['AMOUNT'] >= 50) & (df['AMOUNT'] < 200)).astype(int)
        df['is_large_txn'] = (df['AMOUNT'] >= 200).astype(int)

        # Fee indicators
        df['has_fees'] = (df['FEES'] > 0).astype(int)
        df['has_elevy'] = (df['E-LEVY'] > 0).astype(int)
        df['total_cost'] = df['AMOUNT'] + df['FEES'] + df['E-LEVY']
        df['fee_to_amount_ratio'] = df['FEES'] / (df['AMOUNT'] + 1)
        df['elevy_to_amount_ratio'] = df['E-LEVY'] / (df['AMOUNT'] + 1)

        # ============================================
        # CATEGORY 2: CATEGORICAL ENCODINGS
        # ============================================

        # Transaction type one-hot
        df['is_transfer'] = (df['TRANS. TYPE'] == 'TRANSFER').astype(int)
        df['is_debit'] = (df['TRANS. TYPE'] == 'DEBIT').astype(int)
        df['is_payment'] = (df['TRANS. TYPE'] == 'PAYMENT').astype(int)
        df['is_payment_send'] = (df['TRANS. TYPE'] == 'PAYMENT_SEND').astype(int)
        df['is_cash_out'] = (df['TRANS. TYPE'] == 'CASH_OUT').astype(int)
        df['is_cash_in'] = (df['TRANS. TYPE'] == 'CASH_IN').astype(int)
        df['is_adjustment'] = (df['TRANS. TYPE'] == 'ADJUSTMENT').astype(int)

        # ============================================
        # CATEGORY 3: TEMPORAL EXTRACTION
        # ============================================

        # Time components
        df['hour'] = df['TRANSACTION DATE'].dt.hour
        df['day_of_week'] = df['TRANSACTION DATE'].dt.dayofweek  # 0=Monday
        df['day_of_month'] = df['TRANSACTION DATE'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_weekday'] = (df['day_of_week'] < 5).astype(int)

        # Time of day categories
        df['is_early_morning'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['is_night'] = (df['hour'] >= 22).astype(int)

        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # ============================================
        # CATEGORY 4: BALANCE DYNAMICS
        # ============================================

        # Balance levels
        df['log_balance_before'] = np.log1p(df['BAL BEFORE'])
        df['log_balance_after'] = np.log1p(df['BAL AFTER'])
        df['balance_change'] = df['BAL AFTER'] - df['BAL BEFORE']
        df['balance_pct_change'] = df['balance_change'] / (df['BAL BEFORE'] + 1)

        # Balance status indicators
        df['is_low_balance'] = (df['BAL BEFORE'] < 20).astype(int)
        df['is_zero_balance'] = (df['BAL BEFORE'] == 0).astype(int)
        df['is_high_balance'] = (df['BAL BEFORE'] > 200).astype(int)

        # Transaction size relative to balance
        df['amount_to_balance_ratio'] = df['AMOUNT'] / (df['BAL BEFORE'] + 1)
        df['is_large_relative_to_balance'] = (df['amount_to_balance_ratio'] > 0.5).astype(int)
        df['will_deplete_balance'] = ((df['BAL AFTER'] < 10) & (df['BAL BEFORE'] > 10)).astype(int)

        # ============================================
        # CATEGORY 5: SEQUENCE FEATURES (LOOKBACK)
        # ============================================

        # Inter-transaction time
        df['time_since_last_txn_hours'] = df['TRANSACTION DATE'].diff().dt.total_seconds() / 3600
        df['time_since_last_txn_hours'] = df['time_since_last_txn_hours'].fillna(0)

        # Transaction number
        df['txn_number'] = range(1, len(df) + 1)
        df['reverse_txn_number'] = len(df) - df['txn_number']

        # Cumulative statistics
        df['cumulative_volume'] = df['AMOUNT'].cumsum()
        df['cumulative_txn_count'] = df['txn_number']
        df['cumulative_fees_paid'] = (df['FEES'] + df['E-LEVY']).cumsum()

        # Recent transaction patterns (last N transactions)
        for n in [3, 5, 10]:
            df[f'last_{n}_avg_amount'] = df['AMOUNT'].rolling(n, min_periods=1).mean()
            df[f'last_{n}_std_amount'] = df['AMOUNT'].rolling(n, min_periods=1).std().fillna(0)
            df[f'last_{n}_max_amount'] = df['AMOUNT'].rolling(n, min_periods=1).max()
            df[f'last_{n}_min_amount'] = df['AMOUNT'].rolling(n, min_periods=1).min()

            # Ratio of current to recent average
            df[f'amount_vs_last_{n}_avg'] = df['AMOUNT'] / (df[f'last_{n}_avg_amount'] + 1)

            # Transaction type counts in last N
            df[f'last_{n}_transfer_count'] = df['is_transfer'].rolling(n, min_periods=1).sum()
            df[f'last_{n}_debit_count'] = df['is_debit'].rolling(n, min_periods=1).sum()
            df[f'last_{n}_cashout_count'] = df['is_cash_out'].rolling(n, min_periods=1).sum()

        # ============================================
        # CATEGORY 6: ROLLING WINDOW STATISTICS (TIME-BASED)
        # ============================================

        for window_days in windows:
            window_key = f'{window_days}d'

            # Set datetime index for time-based rolling
            df_temp = df.set_index('TRANSACTION DATE')

            # Amount statistics
            df[f'rolling_{window_key}_count'] = df_temp['AMOUNT'].rolling(f'{window_days}D').count().values
            df[f'rolling_{window_key}_sum'] = df_temp['AMOUNT'].rolling(f'{window_days}D').sum().values
            df[f'rolling_{window_key}_mean'] = df_temp['AMOUNT'].rolling(f'{window_days}D').mean().fillna(0).values
            df[f'rolling_{window_key}_std'] = df_temp['AMOUNT'].rolling(f'{window_days}D').std().fillna(0).values

            # Balance statistics
            df[f'rolling_{window_key}_min_balance'] = df_temp['BAL BEFORE'].rolling(f'{window_days}D').min().values
            df[f'rolling_{window_key}_max_balance'] = df_temp['BAL BEFORE'].rolling(f'{window_days}D').max().values
            df[f'rolling_{window_key}_balance_volatility'] = df_temp['BAL BEFORE'].rolling(f'{window_days}D').std().fillna(0).values

        # ============================================
        # CATEGORY 7: BEHAVIORAL PATTERNS
        # ============================================

        # Counterparty diversity (expanding window) - fixed to handle strings
        recipient_counts = []
        seen_recipients = set()
        for recipient in df['TO NAME']:
            seen_recipients.add(recipient)
            recipient_counts.append(len(seen_recipients))
        df['unique_recipients_so_far'] = recipient_counts

        # Transaction type diversity - fixed to handle strings
        txn_type_diversity = []
        for i in range(len(df)):
            start_idx = max(0, i - 9)
            window_types = df['TRANS. TYPE'].iloc[start_idx:i+1]
            txn_type_diversity.append(window_types.nunique())
        df['unique_txn_types_last_10'] = txn_type_diversity

        # Repeated recipient indicator
        df['is_repeated_recipient'] = df.duplicated(subset=['TO NAME'], keep=False).astype(int)

        # Self-transfer (circular transactions)
        df['is_self_transfer'] = (df['TO NAME'] == df['FROM NAME']).astype(int)

        # ============================================
        # CATEGORY 8: DERIVED RISK INDICATORS
        # ============================================

        # Unusual timing
        df['unusual_hour'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        df['rapid_transaction'] = (df['time_since_last_txn_hours'] < 0.5).astype(int)

        # Unusual amounts
        df['unusual_amount_high'] = (df['amount_vs_last_10_avg'] > 3).astype(int)
        df['unusual_amount_low'] = (df['amount_vs_last_10_avg'] < 0.3).astype(int)

        # Depletion patterns
        df['rapid_balance_drop'] = ((df['BAL BEFORE'] > 100) & (df['BAL AFTER'] < 20)).astype(int)

        # Consecutive withdrawals (simplified)
        is_withdrawal = (df['is_cash_out'] + df['is_transfer'] + df['is_debit']) > 0
        prev_is_withdrawal = is_withdrawal.shift(1).fillna(False)
        df['consecutive_withdrawals'] = (is_withdrawal & prev_is_withdrawal).astype(int)

        # Frequency anomalies
        df['high_frequency_period'] = (df['time_since_last_txn_hours'] < 1).astype(int)

        # Composite risk score
        df['risk_score'] = (
            df['unusual_hour'] +
            df['rapid_transaction'] +
            df['unusual_amount_high'] +
            df['rapid_balance_drop'] +
            df['is_large_relative_to_balance']
        )

        return df

    def create_user_level_summary(self, df):
        """Create aggregate features at user level (for multi-user datasets)"""

        # Ensure datetime
        df = df.copy()
        df['TRANSACTION DATE'] = pd.to_datetime(df['TRANSACTION DATE'])

        user_features = {
            # Volume statistics
            'total_transactions': len(df),
            'total_volume': df['AMOUNT'].sum(),
            'avg_transaction_amount': df['AMOUNT'].mean(),
            'median_transaction_amount': df['AMOUNT'].median(),
            'std_transaction_amount': df['AMOUNT'].std(),
            'max_transaction_amount': df['AMOUNT'].max(),
            'min_transaction_amount': df['AMOUNT'].min(),
            'cv_transaction_amount': df['AMOUNT'].std() / df['AMOUNT'].mean() if df['AMOUNT'].mean() > 0 else 0,

            # Transaction type distribution
            'pct_transfers': (df['TRANS. TYPE'] == 'TRANSFER').mean(),
            'pct_debits': (df['TRANS. TYPE'] == 'DEBIT').mean(),
            'pct_cashouts': (df['TRANS. TYPE'] == 'CASH_OUT').mean(),
            'pct_payments': ((df['TRANS. TYPE'] == 'PAYMENT') |
                           (df['TRANS. TYPE'] == 'PAYMENT_SEND')).mean(),

            # Temporal patterns
            'avg_hours_between_txns': df['TRANSACTION DATE'].diff().dt.total_seconds().mean() / 3600 if len(df) > 1 else 0,
            'pct_weekend_txns': (df['TRANSACTION DATE'].dt.dayofweek >= 5).mean(),
            'pct_night_txns': (df['TRANSACTION DATE'].dt.hour >= 22).mean(),
            'pct_early_morning_txns': (df['TRANSACTION DATE'].dt.hour < 6).mean(),

            # Balance dynamics
            'avg_balance': df['BAL BEFORE'].mean(),
            'min_balance': df['BAL BEFORE'].min(),
            'max_balance': df['BAL BEFORE'].max(),
            'balance_volatility': df['BAL BEFORE'].std(),
            'pct_low_balance_txns': (df['BAL BEFORE'] < 20).mean(),

            # Cost statistics
            'total_fees_paid': (df['FEES'] + df['E-LEVY']).sum(),
            'avg_fees_per_txn': (df['FEES'] + df['E-LEVY']).mean(),
            'pct_txns_with_fees': (df['FEES'] > 0).mean(),

            # Behavioral
            'unique_recipients': df['TO NAME'].nunique(),
            'recipient_concentration': df['TO NAME'].value_counts().iloc[0] / len(df) if len(df) > 0 else 0,
            'unique_txn_types': df['TRANS. TYPE'].nunique(),

            # Temporal coverage
            'account_age_days': (df['TRANSACTION DATE'].max() - df['TRANSACTION DATE'].min()).days,
            'transactions_per_day': len(df) / max((df['TRANSACTION DATE'].max() -
                                                   df['TRANSACTION DATE'].min()).days, 1)
        }

        return pd.Series(user_features)


def demonstrate_temporal_features():
    """Demonstrate temporal feature engineering on real data"""

    print("=" * 80)
    print("TEMPORAL FEATURE ENGINEERING - REAL MOBILE MONEY DATA")
    print("=" * 80)

    # Load data
    print("\n1. Loading real transaction data...")
    df = pd.read_csv('data/transactions.xlsx - Table 5.csv')
    print(f"   Loaded {len(df)} transactions")
    print(f"   Date range: {pd.to_datetime(df['TRANSACTION DATE']).min()} to {pd.to_datetime(df['TRANSACTION DATE']).max()}")

    # Filter to primary user
    primary_user = df['FROM ACCT'].value_counts().index[0]
    df_user = df[df['FROM ACCT'] == primary_user].copy()
    print(f"   Filtered to primary user (Account {primary_user}): {len(df_user)} transactions")

    # Extract features
    print("\n2. Extracting temporal features...")
    engineer = TemporalTransactionFeatureEngineer()
    df_features = engineer.extract_all_features(df_user)

    # Count feature categories
    feature_cols = [col for col in df_features.columns if col not in df.columns]
    print(f"   Created {len(feature_cols)} engineered features")

    # Categorize features
    categories = {
        'Transaction-level static': [c for c in feature_cols if any(x in c for x in ['log_amount', 'sqrt_amount', 'is_micro', 'is_small', 'is_medium', 'is_large', 'fee_to', 'elevy_to', 'total_cost'])],
        'Categorical encodings': [c for c in feature_cols if c.startswith('is_') and any(x in c for x in ['transfer', 'debit', 'payment', 'cash', 'adjustment'])],
        'Temporal extraction': [c for c in feature_cols if any(x in c for x in ['hour', 'day_', 'weekend', 'weekday', 'morning', 'afternoon', 'evening', 'night', '_sin', '_cos'])],
        'Balance dynamics': [c for c in feature_cols if any(x in c for x in ['balance', 'amount_to_balance', 'deplete'])],
        'Sequence features': [c for c in feature_cols if any(x in c for x in ['last_', 'cumulative', 'time_since', 'txn_number', 'reverse'])],
        'Rolling windows': [c for c in feature_cols if 'rolling_' in c],
        'Behavioral patterns': [c for c in feature_cols if any(x in c for x in ['unique_', 'repeated', 'self_transfer', 'diversity'])],
        'Risk indicators': [c for c in feature_cols if any(x in c for x in ['unusual', 'rapid', 'risk_score', 'consecutive', 'frequency'])]
    }

    print("\n3. Feature breakdown by category:")
    for category, features in categories.items():
        print(f"   - {category}: {len(features)} features")

    # Show sample features
    print("\n4. Sample feature values (last 5 transactions):")
    sample_features = [
        'AMOUNT', 'BAL BEFORE', 'TRANS. TYPE',
        'log_amount', 'is_transfer', 'hour', 'is_evening',
        'balance_change', 'time_since_last_txn_hours',
        'last_5_avg_amount', 'rolling_7d_count', 'risk_score'
    ]
    print(df_features[sample_features].tail().to_string(index=False))

    # Temporal trends
    print("\n5. Temporal feature analysis:")
    print(f"   - Average time between transactions: {df_features['time_since_last_txn_hours'].mean():.2f} hours")
    print(f"   - Transactions per day: {len(df_user) / 200:.2f}")
    print(f"   - Weekend transaction rate: {df_features['is_weekend'].mean():.1%}")
    print(f"   - Night transaction rate: {df_features['is_night'].mean():.1%}")
    print(f"   - Average risk score: {df_features['risk_score'].mean():.2f}")

    # User-level summary
    print("\n6. User-level aggregated features:")
    user_summary = engineer.create_user_level_summary(df_user)
    print(f"   - Total volume: GHS {user_summary['total_volume']:.2f}")
    print(f"   - Average transaction: GHS {user_summary['avg_transaction_amount']:.2f}")
    print(f"   - Transaction frequency: {user_summary['transactions_per_day']:.2f} per day")
    print(f"   - Balance volatility: {user_summary['balance_volatility']:.2f}")
    print(f"   - Unique recipients: {int(user_summary['unique_recipients'])}")

    # Save engineered dataset
    output_path = 'data/engineered_features_real_data.csv'
    df_features.to_csv(output_path, index=False)
    print(f"\nSaved engineered features to: {output_path}")

    # Save user summary
    summary_path = 'data/user_level_summary.csv'
    user_summary.to_csv(summary_path, header=['value'])
    print(f"Saved user-level summary to: {summary_path}")

    print("\n" + "=" * 80)
    print("NEXT STEPS FOR PAPERS A & B")
    print("=" * 80)
    print("""
For Paper A (Feature Engineering Framework):
1. You have temporal feature extraction working on real data
2. Generate multiple synthetic users with varying behavior profiles
3. Test framework robustness across different user types
4. Compare against baseline feature sets (raw, simple stats, automated extraction)
5. Evaluate: discriminative power, redundancy, stability, efficiency

For Paper B (Static vs Sequential Comparison):
1. You have sequential features (rolling windows, lookbacks)
2. Create static aggregates from these features (user-level summary)
3. Build LSTM model using sequence of last N transactions
4. Compare static (Logistic Regression) vs sequential (LSTM) performance
5. Analyze when sequential modeling provides meaningful gains

Key Advantage of This Real Data:
- Demonstrates framework works on actual mobile money transactions
- Can validate synthetic data generator against real patterns
- Shows practical feature engineering for deployment
""")

    return df_features


if __name__ == "__main__":
    df_features = demonstrate_temporal_features()
