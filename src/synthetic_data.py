"""
Calibrated Synthetic Mobile Money Data Generator

Location: src/synthetic_data.py

Import path:
    from src.synthetic_data import CalibratedMoMoDataGenerator

Or from project root:
    import sys
    sys.path.append('src')
    from synthetic_data import CalibratedMoMoDataGenerator

This generator creates realistic multi-user transaction data calibrated against
actual mobile money patterns from Ghana (482 real transactions analyzed).

Calibration Parameters (from real data):
- Transaction frequency: 2.41 per day (every 10 hours)
- Amount distribution: Lognormal(mu=2.84, sigma=1.00)
- Balance: mean=GHS 305.88, std=GHS 302.55
- Transaction types: 52.9% transfers, 27.2% debits, 10.8% payments
- Temporal: 32.2% weekend, 8.1% night, 36.1% afternoon
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class CalibratedMoMoDataGenerator:
    """
    Generate realistic mobile money transaction data calibrated to real patterns.
    """

    def __init__(self,
                 n_users=2000,
                 avg_transactions_per_user=15,
                 fraud_rate=0.05,
                 start_date='2024-01-01',
                 duration_days=180,
                 calibration_file='data/real_data_calibration.json'):
        """
        Initialize generator with calibration parameters from real data.

        Args:
            n_users: Number of users to generate
            avg_transactions_per_user: Average transactions per user
            fraud_rate: Proportion of fraudulent users
            start_date: Start date of observation period
            duration_days: Length of observation period in days
            calibration_file: Path to JSON file with real data parameters
        """
        self.n_users = n_users
        self.avg_transactions_per_user = avg_transactions_per_user
        self.fraud_rate = fraud_rate
        self.start_date = pd.to_datetime(start_date)
        self.duration_days = duration_days
        self.end_date = self.start_date + timedelta(days=duration_days)

        # Load calibration parameters
        with open(calibration_file, 'r') as f:
            self.calibration = json.load(f)

        print(f"Loaded calibration from real data:")
        print(f"  - Amount: mu={self.calibration['amount_lognormal_mu']:.2f}, sigma={self.calibration['amount_lognormal_sigma']:.2f}")
        print(f"  - Frequency: {self.calibration['transaction_frequency_hours']:.2f} hours between transactions")
        print(f"  - Balance: mu={self.calibration['balance_mean']:.2f}, sigma={self.calibration['balance_std']:.2f}")

    def generate_users(self):
        """Generate user profiles with variation around real data patterns"""

        n_fraud_users = int(self.n_users * self.fraud_rate)
        n_legit_users = self.n_users - n_fraud_users

        users = []

        # Generate legitimate users
        for i in range(n_legit_users):
            # Vary parameters around real data with +/-30% deviation
            profile = {
                'user_id': f'USER_{i:06d}',
                'phone_number': f'233{np.random.randint(200000000, 600000000)}',
                'is_fraudster': False,
                'fraud_type': 'legitimate',

                # Initial balance: centered on real mean with variation
                'initial_balance': max(50, np.random.normal(
                    self.calibration['balance_mean'],
                    self.calibration['balance_std'] * 0.5
                )),

                # Transaction amount: use real lognormal parameters with slight variation
                'amount_mu': np.random.normal(
                    self.calibration['amount_lognormal_mu'],
                    0.2  # Small variation between users
                ),
                'amount_sigma': np.random.normal(
                    self.calibration['amount_lognormal_sigma'],
                    0.1
                ),

                # Transaction frequency: vary around real average (10 hours)
                # Use absolute value since real data was sorted reverse
                'hours_between_txns': np.random.gamma(
                    shape=2,
                    scale=abs(self.calibration['transaction_frequency_hours']) / 2
                ),

                # Transaction type preferences (vary around real distribution)
                'pref_transfer': max(0.3, min(0.8, np.random.normal(0.529, 0.1))),
                'pref_debit': max(0.1, min(0.4, np.random.normal(0.272, 0.05))),
                'pref_payment': max(0.05, min(0.25, np.random.normal(0.149, 0.05))),

                # Temporal preferences
                'pref_weekend': np.random.beta(2, 5),  # Slight weekend preference
                'pref_night': np.random.beta(1, 10),   # Low night activity
                'pref_hour': np.random.choice([10, 12, 14, 16, 18, 20]),  # Preferred hour

                # Balance management
                'min_balance_threshold': np.random.uniform(10, 100),
                'max_balance_target': np.random.uniform(500, 2000),

                # Fee tolerance
                'accepts_fees': np.random.random() < self.calibration['fee_rate'],

                # Social network size
                'typical_recipients': int(np.random.gamma(shape=3, scale=10))  # avg ~30 recipients
            }

            users.append(profile)

        # Generate fraudulent users (different patterns)
        fraud_types = ['account_takeover', 'social_engineering', 'sim_swap']

        for i in range(n_fraud_users):
            fraud_type = np.random.choice(fraud_types)

            profile = {
                'user_id': f'USER_{n_legit_users + i:06d}',
                'phone_number': f'233{np.random.randint(200000000, 600000000)}',
                'is_fraudster': True,
                'fraud_type': fraud_type,

                # Start with similar balance to legitimate
                'initial_balance': max(100, np.random.normal(
                    self.calibration['balance_mean'] * 0.8,
                    self.calibration['balance_std'] * 0.4
                )),

                # Fraudsters make larger transactions
                'amount_mu': self.calibration['amount_lognormal_mu'] + 0.5,
                'amount_sigma': self.calibration['amount_lognormal_sigma'] + 0.3,

                # More frequent transactions
                'hours_between_txns': abs(self.calibration['transaction_frequency_hours']) * 0.3,

                # Prefer cash-outs and transfers
                'pref_transfer': 0.4,
                'pref_debit': 0.1,
                'pref_payment': 0.1,
                'pref_cashout': 0.4,  # High cash-out rate

                # Unusual timing
                'pref_weekend': 0.4,
                'pref_night': 0.3,  # Higher night activity
                'pref_hour': np.random.choice([1, 3, 22, 23]),

                # Drain account
                'min_balance_threshold': 0,
                'max_balance_target': 500,

                # Avoid fees if possible
                'accepts_fees': False,

                # Fewer repeated recipients
                'typical_recipients': int(np.random.gamma(shape=2, scale=5)),

                # When fraud starts (earlier in observation period)
                'fraud_start_day': np.random.randint(10, min(60, self.duration_days - 30))
            }

            users.append(profile)

        return users

    def generate_transaction_amount(self, user_profile, txn_type, is_fraud=False):
        """Generate transaction amount based on type and user profile"""

        # Sample from lognormal distribution
        base_amount = np.random.lognormal(
            user_profile['amount_mu'],
            user_profile['amount_sigma']
        )

        # Adjust by transaction type (from real data patterns)
        type_multipliers = {
            'TRANSFER': 1.0,      # baseline (mean: 38.26)
            'DEBIT': 0.55,        # smaller (mean: 21.31)
            'PAYMENT': 0.44,      # smaller (mean: 16.90)
            'PAYMENT_SEND': 0.63, # smaller (mean: 24.22)
            'CASH_OUT': 1.81,     # larger (mean: 69.38)
            'CASH_IN': 1.2
        }

        amount = base_amount * type_multipliers.get(txn_type, 1.0)

        # Fraud modifier
        if is_fraud:
            amount *= np.random.uniform(1.5, 3.0)  # Larger amounts

        # Minimum viable transaction
        amount = max(0.5, amount)

        # Round to 2 decimals
        return round(amount, 2)

    def calculate_fees(self, amount, txn_type, user_accepts_fees):
        """Calculate transaction fees based on real patterns"""

        fees = 0.0
        elevy = 0.0

        if not user_accepts_fees:
            return fees, elevy

        # Fee rates from real data: 24.3% have fees, 12.7% have e-levy
        if txn_type in ['CASH_OUT', 'PAYMENT_SEND'] and np.random.random() < 0.5:
            # Fee is typically 1% with min/max
            fees = max(0.25, min(amount * 0.01, 5.0))
            fees = round(fees, 2)

        if txn_type in ['TRANSFER', 'PAYMENT_SEND', 'CASH_OUT'] and amount > 100:
            # E-levy on large transactions
            if np.random.random() < 0.3:
                elevy = round(amount * 0.015, 2)  # 1.5% e-levy

        return fees, elevy

    def generate_transactions_for_user(self, user_profile, recipient_pool):
        """Generate transaction sequence for a single user"""

        transactions = []
        current_date = self.start_date + timedelta(
            days=np.random.randint(0, 7)  # Stagger user starts
        )
        current_balance = user_profile['initial_balance']

        # Determine number of transactions
        n_transactions = int(np.random.poisson(self.avg_transactions_per_user))
        n_transactions = max(5, min(50, n_transactions))  # Constrain to 5-50

        # Track recipients for realism
        user_recipients = []

        for txn_idx in range(n_transactions):
            # Stop if beyond observation period
            if current_date > self.end_date:
                break

            # Check if this transaction is fraudulent
            is_fraud = False
            if user_profile['is_fraudster']:
                days_active = (current_date - self.start_date).days
                if days_active >= user_profile.get('fraud_start_day', 0):
                    is_fraud = True

            # Determine transaction type
            if is_fraud and user_profile.get('pref_cashout', 0) > 0:
                # Fraudsters prefer cash-outs
                txn_type = np.random.choice(
                    ['TRANSFER', 'CASH_OUT', 'DEBIT'],
                    p=[0.3, 0.6, 0.1]
                )
            else:
                # Normal users follow calibrated distribution
                types = ['TRANSFER', 'DEBIT', 'PAYMENT', 'PAYMENT_SEND', 'CASH_OUT']
                probs = [
                    user_profile['pref_transfer'],
                    user_profile['pref_debit'],
                    user_profile['pref_payment'] * 0.5,
                    user_profile['pref_payment'] * 0.5,
                    0.05
                ]
                probs = np.array(probs) / sum(probs)  # Normalize
                txn_type = np.random.choice(types, p=probs)

            # Generate amount
            amount = self.generate_transaction_amount(user_profile, txn_type, is_fraud)

            # For outgoing transactions, ensure sufficient balance
            if txn_type in ['TRANSFER', 'DEBIT', 'PAYMENT', 'PAYMENT_SEND', 'CASH_OUT']:
                available = current_balance - user_profile['min_balance_threshold']
                if available <= 0:
                    # Skip this transaction or make it a CASH_IN
                    txn_type = 'CASH_IN'
                    amount = np.random.uniform(50, 200)
                else:
                    amount = min(amount, available)

            # Calculate fees
            fees, elevy = self.calculate_fees(
                amount, txn_type, user_profile['accepts_fees']
            )

            # Determine time of day based on preferences
            if np.random.random() < user_profile['pref_night']:
                hour = np.random.choice([22, 23, 0, 1, 2, 3])
            else:
                # Center around preferred hour with noise
                hour = int(np.clip(
                    np.random.normal(user_profile['pref_hour'], 3),
                    0, 23
                ))

            # Construct timestamp
            txn_datetime = current_date.replace(
                hour=hour,
                minute=np.random.randint(0, 60),
                second=np.random.randint(0, 60)
            )

            # Select recipient
            if txn_type in ['TRANSFER', 'PAYMENT_SEND', 'CASH_OUT']:
                # Reuse recipients or select new ones
                if len(user_recipients) > 0 and np.random.random() < 0.6:
                    recipient = np.random.choice(user_recipients)
                else:
                    recipient = np.random.choice(recipient_pool)
                    user_recipients.append(recipient)
                    # Limit recipient list size
                    if len(user_recipients) > user_profile['typical_recipients']:
                        user_recipients.pop(0)
            else:
                # Service providers for debits/payments
                recipient = np.random.choice([
                    'one4all.sp', 'MTN', 'Vodafone', 'AirtelTigo',
                    'hubtel.sp', 'jumia.sp', 'uber.sp'
                ])

            # Update balance
            balance_before = current_balance
            if txn_type == 'CASH_IN':
                current_balance += amount
            else:
                current_balance -= (amount + fees + elevy)

            balance_after = current_balance

            # Create transaction record
            recipient_name = recipient if isinstance(recipient, str) else recipient.get('name', 'Unknown')
            recipient_phone = '0' if isinstance(recipient, str) else recipient.get('phone', '0')

            transaction = {
                'TRANSACTION DATE': txn_datetime,
                'FROM ACCT': user_profile['user_id'],
                'FROM NAME': f"User {user_profile['user_id'][-6:]}",
                'FROM NO.': user_profile['phone_number'],
                'TRANS. TYPE': txn_type,
                'AMOUNT': amount,
                'FEES': fees,
                'E-LEVY': elevy,
                'BAL BEFORE': round(balance_before, 2),
                'BAL AFTER': round(balance_after, 2),
                'TO NO.': recipient_phone,
                'TO NAME': recipient_name,
                'TO ACCT': f"ACCT_{abs(hash(str(recipient))) % 100000000}",
                'is_fraud': 1 if is_fraud else 0,
                'fraud_type': user_profile['fraud_type']
            }

            transactions.append(transaction)

            # Next transaction time
            hours_gap = np.random.exponential(user_profile['hours_between_txns'])

            # Add weekend effect
            if current_date.weekday() >= 5:  # Weekend
                if np.random.random() < user_profile['pref_weekend']:
                    hours_gap *= 0.7  # More frequent on weekends

            current_date += timedelta(hours=max(0.5, hours_gap))

        return transactions

    def generate_dataset(self):
        """Generate complete multi-user dataset"""

        print(f"\nGenerating {self.n_users} users...")
        users = self.generate_users()

        # Create recipient pool for realistic social network
        print("Creating recipient pool...")
        recipient_pool = [
            {'phone': f'233{np.random.randint(200000000, 600000000)}',
             'name': f'Recipient_{i:05d}'}
            for i in range(500)  # Pool of 500 potential recipients
        ]

        # Add common service providers
        service_providers = [
            'one4all.sp', 'MTN', 'Vodafone', 'AirtelTigo', 'hubtel.sp',
            'jumia.sp', 'uber.sp', 'bolt.sp', 'glovo.sp'
        ]
        recipient_pool.extend(service_providers)

        print(f"Generating transactions for {len(users)} users...")
        all_transactions = []

        for i, user_profile in enumerate(users):
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{len(users)} users...")

            user_transactions = self.generate_transactions_for_user(
                user_profile, recipient_pool
            )
            all_transactions.extend(user_transactions)

        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)

        # Sort by timestamp
        df = df.sort_values('TRANSACTION DATE').reset_index(drop=True)

        print(f"\n{'='*80}")
        print("GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"  Total users: {len(users)}")
        print(f"  Legitimate users: {sum(1 for u in users if not u['is_fraudster'])}")
        print(f"  Fraudulent users: {sum(1 for u in users if u['is_fraudster'])}")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
        print(f"  Date range: {df['TRANSACTION DATE'].min()} to {df['TRANSACTION DATE'].max()}")
        print(f"  Average transactions per user: {len(df) / len(users):.1f}")

        # Validation against real data
        print(f"\n{'='*80}")
        print("VALIDATION AGAINST REAL DATA")
        print(f"{'='*80}")

        print(f"\nAmount statistics:")
        print(f"  Real data - Mean: GHS {self.calibration['amount_mean']:.2f}, Median: GHS {self.calibration['amount_median']:.2f}")
        print(f"  Synthetic  - Mean: GHS {df['AMOUNT'].mean():.2f}, Median: GHS {df['AMOUNT'].median():.2f}")

        print(f"\nTransaction type distribution:")
        synthetic_types = df['TRANS. TYPE'].value_counts(normalize=True)
        for txn_type in ['TRANSFER', 'DEBIT', 'PAYMENT_SEND', 'CASH_OUT', 'PAYMENT']:
            real_pct = self.calibration['type_distribution'].get(txn_type, 0)
            synthetic_pct = synthetic_types.get(txn_type, 0)
            print(f"  {txn_type}: Real={real_pct:.1%}, Synthetic={synthetic_pct:.1%}")

        print(f"\nTemporal patterns:")
        df['hour'] = pd.to_datetime(df['TRANSACTION DATE']).dt.hour
        df['is_weekend'] = pd.to_datetime(df['TRANSACTION DATE']).dt.dayofweek >= 5
        print(f"  Weekend rate: Real={self.calibration['weekend_rate']:.1%}, Synthetic={df['is_weekend'].mean():.1%}")
        print(f"  Night rate: Real={self.calibration['night_rate']:.1%}, Synthetic={(df['hour'] >= 22).mean():.1%}")

        return df, users


def main():
    """Generate calibrated synthetic dataset"""

    print("=" * 80)
    print("CALIBRATED SYNTHETIC MOBILE MONEY DATA GENERATION")
    print("=" * 80)

    # Initialize generator
    generator = CalibratedMoMoDataGenerator(
        n_users=2000,
        avg_transactions_per_user=15,
        fraud_rate=0.05,
        start_date='2024-01-01',
        duration_days=180  # 6 months
    )

    # Generate dataset
    df, users = generator.generate_dataset()

    # Save dataset
    output_path = 'data/synthetic_momo_calibrated.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved synthetic dataset to: {output_path}")

    # Save user profiles for reference
    user_df = pd.DataFrame(users)
    user_path = 'data/synthetic_user_profiles.csv'
    user_df.to_csv(user_path, index=False)
    print(f"Saved user profiles to: {user_path}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"\nTransaction volume by user (first 10 users):")
    user_txn_counts = df.groupby('FROM ACCT').size().sort_values(ascending=False)
    print(user_txn_counts.head(10))

    print(f"\nFraud distribution:")
    fraud_types = df['fraud_type'].value_counts()
    print(fraud_types)

    return df, users


if __name__ == "__main__":
    df, users = main()
