"""
Calibrated Synthetic Mobile Money Data Generator v2.0

Location: src/seqcredit_model/synthetic_data.py

This generator creates realistic per-user transaction datasets calibrated against
actual mobile money patterns from Ghana (482 real transactions analyzed).

Key Features:
- Generates 10,000 individual user transaction files
- Includes CREDIT (loan disbursement) and LOAN_REPAYMENT transactions
- Models credit risk behavior with multiple user archetypes
- Calibrated to real Ghanaian mobile money patterns

Loan System (based on MTN QwikLoan Ghana):
- Loan amounts: GHS 25 - GHS 1,000
- Interest rate: 6.9% (30-day term)
- Penalty rate: 12.5% (on default)
- Providers: QWIKLOAN, XPRESSLOAN, XTRACASH, FIDO, CEDISPAY

Credit Risk Labels:
- 0: Good (repaid in full within term)
- 1: Late (repaid but after 30 days)
- 2: Default (failed to repay within 60 days)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path


class CalibratedMoMoDataGenerator:
    """
    Generate realistic mobile money transaction data with credit/loan functionality.
    Outputs individual CSV files per user.
    """

    # Loan providers in Ghana
    LOAN_PROVIDERS = ['QWIKLOAN', 'XPRESSLOAN', 'XTRACASH', 'FIDO', 'CEDISPAY']

    # Service providers for debits/payments
    SERVICE_PROVIDERS = [
        'one4all.sp', 'MTN', 'Telecel', 'AirtelTigo', 'hubtel.sp',
        'jumia.sp', 'uber.sp', 'bolt.sp', 'glovo.sp', 'ECG', 'GWCL'
    ]

    # Credit behavior archetypes
    CREDIT_ARCHETYPES = {
        'non_borrower': {
            'weight': 0.40,
            'takes_loans': False,
            'description': 'Never takes loans'
        },
        'responsible_borrower': {
            'weight': 0.35,
            'takes_loans': True,
            'loan_frequency': 'regular',
            'repayment_behavior': 'on_time',
            'credit_limit_growth': True,
            'default_probability': 0.0,
            'description': 'Takes loans, always repays on time'
        },
        'occasional_borrower': {
            'weight': 0.15,
            'takes_loans': True,
            'loan_frequency': 'occasional',
            'repayment_behavior': 'variable',
            'credit_limit_growth': True,
            'default_probability': 0.02,
            'description': 'Infrequent loans, variable repayment timing'
        },
        'risky_borrower': {
            'weight': 0.08,
            'takes_loans': True,
            'loan_frequency': 'frequent',
            'repayment_behavior': 'often_late',
            'credit_limit_growth': False,
            'default_probability': 0.15,
            'description': 'Takes loans near limit, sometimes late'
        },
        'defaulter': {
            'weight': 0.02,
            'takes_loans': True,
            'loan_frequency': 'aggressive',
            'repayment_behavior': 'default',
            'credit_limit_growth': False,
            'default_probability': 1.0,
            'description': 'Takes loans, fails to repay'
        }
    }

    def __init__(self,
                 n_users=10000,
                 avg_transactions_per_user=15,
                 start_date='2024-01-01',
                 duration_days=180,
                 output_dir='data/user_transactions',
                 calibration_file='data/real_data_calibration.json'):
        """
        Initialize generator with calibration parameters.

        Args:
            n_users: Number of unique users to generate
            avg_transactions_per_user: Average transactions per user
            start_date: Start date of observation period
            duration_days: Length of observation period in days
            output_dir: Directory to save individual user CSV files
            calibration_file: Path to JSON file with real data parameters
        """
        self.n_users = n_users
        self.avg_transactions_per_user = avg_transactions_per_user
        self.start_date = pd.to_datetime(start_date)
        self.duration_days = duration_days
        self.end_date = self.start_date + timedelta(days=duration_days)
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load calibration parameters
        with open(calibration_file, 'r') as f:
            self.calibration = json.load(f)

        # Loan parameters (calibrated to Ghana mobile money loans)
        self.loan_params = {
            'min_amount': 25,
            'max_amount': 1000,
            'interest_rate': 0.069,
            'penalty_rate': 0.125,
            'term_days': 30,
            'min_active_days': 90,
            'min_transactions': 15,
            'initial_limit': 50,
            'max_limit': 1000,
            'limit_growth_rate': 1.25
        }

        # Create recipient pool
        self._create_recipient_pool()

        print(f"Initialized generator for {n_users:,} users")
        print(f"Output directory: {self.output_dir}")
        print("Calibration loaded:")
        print(f"  - Amount: mu={self.calibration['amount_lognormal_mu']:.2f}, "
              f"sigma={self.calibration['amount_lognormal_sigma']:.2f}")
        print(f"  - Balance: mean={self.calibration['balance_mean']:.2f}")

    def _create_recipient_pool(self):
        """Create pool of potential transaction recipients."""
        self.recipient_pool = [
            {
                'phone': f'233{np.random.randint(200000000, 600000000)}',
                'name': f'Recipient_{i:05d}',
                'account': f'ACCT_{np.random.randint(10000000, 99999999)}'
            }
            for i in range(1000)
        ]

    def _assign_credit_archetype(self):
        """Randomly assign a credit behavior archetype to a user."""
        archetypes = list(self.CREDIT_ARCHETYPES.keys())
        weights = [self.CREDIT_ARCHETYPES[a]['weight'] for a in archetypes]
        return np.random.choice(archetypes, p=weights)

    def generate_user_profile(self, user_id):
        """Generate a single user profile with credit behavior."""

        credit_archetype = self._assign_credit_archetype()
        archetype_config = self.CREDIT_ARCHETYPES[credit_archetype]

        profile = {
            'user_id': f'USER_{user_id:06d}',
            'phone_number': f'233{np.random.randint(200000000, 600000000)}',
            'account_number': f'{np.random.randint(10000000, 99999999)}',

            # Credit behavior
            'credit_archetype': credit_archetype,
            'takes_loans': archetype_config['takes_loans'],

            # Initial balance
            'initial_balance': max(50, np.random.normal(
                self.calibration['balance_mean'],
                self.calibration['balance_std'] * 0.5
            )),

            # Transaction amount parameters
            'amount_mu': np.random.normal(
                self.calibration['amount_lognormal_mu'], 0.2
            ),
            'amount_sigma': max(0.5, np.random.normal(
                self.calibration['amount_lognormal_sigma'], 0.1
            )),

            # Transaction frequency (hours between transactions)
            'hours_between_txns': max(2, np.random.gamma(
                shape=2,
                scale=abs(self.calibration['transaction_frequency_hours']) / 2
            )),

            # Transaction type preferences
            'pref_transfer': max(0.3, min(0.8, np.random.normal(0.529, 0.1))),
            'pref_debit': max(0.1, min(0.4, np.random.normal(0.272, 0.05))),
            'pref_payment': max(0.05, min(0.25, np.random.normal(0.149, 0.05))),
            'pref_cashout': max(0.02, min(0.15, np.random.normal(0.05, 0.02))),

            # Temporal preferences
            'pref_weekend': np.random.beta(2, 5),
            'pref_night': np.random.beta(1, 10),
            'pref_hour': np.random.choice([9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]),

            # Balance management
            'min_balance_threshold': np.random.uniform(10, 100),

            # Fee tolerance
            'accepts_fees': np.random.random() < self.calibration['fee_rate'],

            # Social network
            'typical_recipients': int(np.random.gamma(shape=3, scale=10)),

            # Account start date (staggered)
            'account_start_day': np.random.randint(0, 14)
        }

        # Add loan-specific parameters if user takes loans
        if profile['takes_loans']:
            profile.update(self._generate_loan_profile(archetype_config))

        return profile

    def _generate_loan_profile(self, archetype_config):
        """Generate loan-specific parameters based on archetype."""

        loan_frequency_map = {
            'regular': np.random.randint(2, 5),      # 2-4 loans in period
            'occasional': np.random.randint(1, 3),   # 1-2 loans
            'frequent': np.random.randint(4, 7),     # 4-6 loans
            'aggressive': np.random.randint(3, 6)    # 3-5 loans (before default)
        }

        repayment_timing_map = {
            'on_time': (0.9, 0.05, 0.05),      # (on_time, late, very_late) probabilities
            'variable': (0.6, 0.3, 0.1),
            'often_late': (0.3, 0.4, 0.3),
            'default': (0.1, 0.2, 0.7)
        }

        freq = archetype_config.get('loan_frequency', 'occasional')
        repay_behavior = archetype_config.get('repayment_behavior', 'variable')

        return {
            'loan_count_target': loan_frequency_map.get(freq, 2),
            'repayment_timing': repayment_timing_map.get(repay_behavior, (0.6, 0.3, 0.1)),
            'default_probability': archetype_config.get('default_probability', 0.05),
            'credit_limit': self.loan_params['initial_limit'],
            'credit_limit_growth': archetype_config.get('credit_limit_growth', True),
            'preferred_loan_provider': np.random.choice(self.LOAN_PROVIDERS),
            'loan_amount_preference': np.random.uniform(0.5, 0.9)  # % of limit typically borrowed
        }

    def generate_transaction_amount(self, user_profile, txn_type):
        """Generate transaction amount based on type and user profile."""

        base_amount = np.random.lognormal(
            user_profile['amount_mu'],
            user_profile['amount_sigma']
        )

        # Type multipliers (from real data analysis)
        type_multipliers = {
            'TRANSFER': 1.0,
            'DEBIT': 0.55,
            'PAYMENT': 0.44,
            'PAYMENT_SEND': 0.63,
            'CASH_OUT': 1.81,
            'CASH_IN': 1.2
        }

        amount = base_amount * type_multipliers.get(txn_type, 1.0)
        return round(max(0.5, amount), 2)

    def calculate_fees(self, amount, txn_type, user_accepts_fees):
        """Calculate transaction fees based on real patterns."""

        fees = 0.0
        elevy = 0.0

        if not user_accepts_fees:
            return fees, elevy

        if txn_type in ['CASH_OUT', 'PAYMENT_SEND'] and np.random.random() < 0.5:
            fees = round(max(0.25, min(amount * 0.01, 5.0)), 2)

        if txn_type in ['TRANSFER', 'PAYMENT_SEND', 'CASH_OUT'] and amount > 100:
            if np.random.random() < 0.3:
                elevy = round(amount * 0.015, 2)

        return fees, elevy

    def _select_transaction_type(self, user_profile, current_balance):
        """Select transaction type based on user preferences and balance."""

        # If balance is low, might need cash in
        if current_balance < user_profile['min_balance_threshold']:
            if np.random.random() < 0.7:
                return 'CASH_IN'

        types = ['TRANSFER', 'DEBIT', 'PAYMENT', 'PAYMENT_SEND', 'CASH_OUT']
        probs = [
            user_profile['pref_transfer'],
            user_profile['pref_debit'],
            user_profile['pref_payment'] * 0.5,
            user_profile['pref_payment'] * 0.5,
            user_profile['pref_cashout']
        ]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(types, p=probs)

    def _generate_timestamp(self, current_date, user_profile):
        """Generate transaction timestamp based on user preferences."""

        if np.random.random() < user_profile['pref_night']:
            hour = np.random.choice([22, 23, 0, 1, 2, 3])
        else:
            hour = int(np.clip(
                np.random.normal(user_profile['pref_hour'], 3),
                6, 22
            ))

        return current_date.replace(
            hour=hour,
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )

    def _select_recipient(self, txn_type, user_recipients):
        """Select transaction recipient based on type."""

        if txn_type in ['TRANSFER', 'PAYMENT_SEND', 'CASH_OUT']:
            if len(user_recipients) > 0 and np.random.random() < 0.6:
                return np.random.choice(user_recipients)
            else:
                recipient = np.random.choice(self.recipient_pool)
                user_recipients.append(recipient)
                return recipient
        else:
            provider = np.random.choice(self.SERVICE_PROVIDERS)
            return {
                'phone': '0',
                'name': provider,
                'account': f'{provider}_ACCT'
            }

    def generate_loan_transaction(self, user_profile, current_date, current_balance,
                                   is_disbursement=True, loan_amount=None):
        """Generate a CREDIT or LOAN_REPAYMENT transaction."""

        provider = user_profile.get('preferred_loan_provider', 'QWIKLOAN')

        if is_disbursement:
            # CREDIT: Loan disbursement
            credit_limit = user_profile.get('credit_limit', self.loan_params['initial_limit'])
            pref = user_profile.get('loan_amount_preference', 0.7)

            # Calculate loan amount (percentage of available limit)
            max_loan = min(credit_limit, self.loan_params['max_amount'])
            min_loan = self.loan_params['min_amount']

            amount = round(np.random.uniform(
                min_loan,
                max_loan * pref
            ), 2)
            amount = max(min_loan, min(amount, max_loan))

            transaction = {
                'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
                'FROM ACCT': f'{provider}_LENDING',
                'FROM NAME': f'{provider} Loan Service',
                'FROM NO.': '0',
                'TRANS. TYPE': 'CREDIT',
                'AMOUNT': amount,
                'FEES': 0.0,
                'E-LEVY': 0.0,
                'BAL BEFORE': round(current_balance, 2),
                'BAL AFTER': round(current_balance + amount, 2),
                'TO NO.': user_profile['phone_number'],
                'TO NAME': f"User {user_profile['user_id'][-6:]}",
                'TO ACCT': user_profile['account_number'],
                'LOAN_PROVIDER': provider,
                'LOAN_PRINCIPAL': amount,
                'LOAN_INTEREST_RATE': self.loan_params['interest_rate'],
                'LOAN_DUE_DATE': (current_date + timedelta(days=self.loan_params['term_days'])).strftime('%Y-%m-%d')
            }

            return transaction, amount

        else:
            # LOAN_REPAYMENT: Repaying the loan
            if loan_amount is None:
                return None, 0

            # Calculate repayment amount with interest
            interest = loan_amount * self.loan_params['interest_rate']
            total_due = loan_amount + interest

            # Check if user can afford repayment
            repay_amount = min(total_due, current_balance - 5)  # Keep minimum balance
            repay_amount = max(0, round(repay_amount, 2))

            if repay_amount <= 0:
                return None, 0

            transaction = {
                'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
                'FROM ACCT': user_profile['account_number'],
                'FROM NAME': f"User {user_profile['user_id'][-6:]}",
                'FROM NO.': user_profile['phone_number'],
                'TRANS. TYPE': 'LOAN_REPAYMENT',
                'AMOUNT': repay_amount,
                'FEES': 0.0,
                'E-LEVY': 0.0,
                'BAL BEFORE': round(current_balance, 2),
                'BAL AFTER': round(current_balance - repay_amount, 2),
                'TO NO.': '0',
                'TO NAME': f'{provider} Loan Service',
                'TO ACCT': f'{provider}_LENDING',
                'LOAN_PROVIDER': provider,
                'LOAN_PRINCIPAL_PAID': round(repay_amount - (repay_amount * self.loan_params['interest_rate'] / (1 + self.loan_params['interest_rate'])), 2),
                'LOAN_INTEREST_PAID': round(repay_amount * self.loan_params['interest_rate'] / (1 + self.loan_params['interest_rate']), 2)
            }

            return transaction, repay_amount

    def _determine_repayment_timing(self, user_profile, loan_date):
        """Determine when the user will repay based on their archetype."""

        timing_probs = user_profile.get('repayment_timing', (0.6, 0.3, 0.1))
        default_prob = user_profile.get('default_probability', 0.05)

        # Check for default
        if np.random.random() < default_prob:
            return None, 2  # Default - no repayment, risk label 2

        timing_choice = np.random.choice(['on_time', 'late', 'very_late'], p=timing_probs)

        if timing_choice == 'on_time':
            days = np.random.randint(7, 30)
            risk_label = 0
        elif timing_choice == 'late':
            days = np.random.randint(31, 45)
            risk_label = 1
        else:  # very_late
            days = np.random.randint(45, 60)
            risk_label = 1

        repay_date = loan_date + timedelta(days=days)
        return repay_date, risk_label

    def generate_transactions_for_user(self, user_profile):
        """Generate complete transaction sequence for a single user."""

        transactions = []
        current_date = self.start_date + timedelta(days=user_profile['account_start_day'])
        current_balance = user_profile['initial_balance']

        # Tracking
        user_recipients = []
        n_transactions = max(5, min(50, int(np.random.poisson(self.avg_transactions_per_user))))

        # Loan tracking
        active_loan = None
        loans_taken = 0
        loan_target = user_profile.get('loan_count_target', 0) if user_profile['takes_loans'] else 0
        credit_risk_labels = []

        # Calculate loan timing based on transaction index (not dates)
        # This ensures loans happen within the user's transaction history
        loan_at_txn_indices = []
        if loan_target > 0:
            # Schedule loans at regular intervals within the transaction sequence
            # First loan after ~20% of transactions, then spread evenly
            total_expected = n_transactions
            first_loan_idx = max(3, int(total_expected * 0.15))  # After first 15% of txns
            loan_interval = max(3, (total_expected - first_loan_idx) // (loan_target + 1))

            for i in range(loan_target):
                loan_idx = first_loan_idx + (i * loan_interval) + np.random.randint(-1, 2)
                loan_idx = max(3, min(loan_idx, total_expected - 5))
                loan_at_txn_indices.append(loan_idx)

            loan_at_txn_indices.sort()

        txn_idx = 0
        regular_txn_count = 0
        while current_date < self.end_date and regular_txn_count < n_transactions:
            txn_idx += 1

            # Check if it's time for a loan (based on transaction count)
            if loan_at_txn_indices and regular_txn_count >= loan_at_txn_indices[0] and active_loan is None:
                loan_at_txn_indices.pop(0)

                # Generate loan disbursement (CREDIT)
                loan_txn, loan_amount = self.generate_loan_transaction(
                    user_profile, current_date, current_balance, is_disbursement=True
                )

                if loan_txn and loan_amount > 0:
                    transactions.append(loan_txn)
                    current_balance += loan_amount
                    loans_taken += 1

                    # Determine repayment timing (in transactions, not days)
                    # Repayment happens after 3-8 more transactions
                    repay_timing, risk_label = self._determine_repayment_timing(
                        user_profile, current_date
                    )

                    # Calculate repay transaction index
                    if repay_timing is None:
                        repay_at_txn = None  # Default - no repayment
                    else:
                        # Repayment happens after 2-6 more transactions
                        # This ensures repayment occurs within user's transaction history
                        remaining_txns = n_transactions - regular_txn_count
                        if remaining_txns > 3:
                            txns_to_repay = np.random.randint(2, min(7, remaining_txns))
                            repay_at_txn = regular_txn_count + txns_to_repay
                        else:
                            # Not enough transactions left, repay at end
                            repay_at_txn = regular_txn_count + max(1, remaining_txns - 1)

                    active_loan = {
                        'amount': loan_amount,
                        'disbursement_date': current_date,
                        'repay_at_txn': repay_at_txn,
                        'risk_label': risk_label
                    }
                    credit_risk_labels.append(risk_label)

                    # Grow credit limit if successful borrower
                    if user_profile.get('credit_limit_growth', False) and risk_label == 0:
                        current_limit = user_profile.get('credit_limit', self.loan_params['initial_limit'])
                        new_limit = min(
                            current_limit * self.loan_params['limit_growth_rate'],
                            self.loan_params['max_limit']
                        )
                        user_profile['credit_limit'] = new_limit

            # Check if it's time to repay loan (based on transaction count)
            if active_loan and active_loan.get('repay_at_txn') is not None:
                if regular_txn_count >= active_loan['repay_at_txn']:
                    repay_txn, repay_amount = self.generate_loan_transaction(
                        user_profile, current_date, current_balance,
                        is_disbursement=False, loan_amount=active_loan['amount']
                    )

                    if repay_txn and repay_amount > 0:
                        transactions.append(repay_txn)
                        current_balance -= repay_amount

                    active_loan = None

            # Generate regular transaction
            txn_type = self._select_transaction_type(user_profile, current_balance)
            amount = self.generate_transaction_amount(user_profile, txn_type)

            # Ensure sufficient balance for outgoing transactions
            if txn_type not in ['CASH_IN', 'CREDIT']:
                available = current_balance - user_profile['min_balance_threshold']
                if available <= 0:
                    txn_type = 'CASH_IN'
                    amount = np.random.uniform(50, 200)
                else:
                    amount = min(amount, available)

            fees, elevy = self.calculate_fees(amount, txn_type, user_profile['accepts_fees'])

            recipient = self._select_recipient(txn_type, user_recipients)
            if len(user_recipients) > user_profile['typical_recipients']:
                user_recipients.pop(0)

            # Update balance
            balance_before = current_balance
            if txn_type == 'CASH_IN':
                current_balance += amount
            else:
                current_balance -= (amount + fees + elevy)

            transaction = {
                'TRANSACTION DATE': self._generate_timestamp(current_date, user_profile),
                'FROM ACCT': user_profile['account_number'],
                'FROM NAME': f"User {user_profile['user_id'][-6:]}",
                'FROM NO.': user_profile['phone_number'],
                'TRANS. TYPE': txn_type,
                'AMOUNT': amount,
                'FEES': fees,
                'E-LEVY': elevy,
                'BAL BEFORE': round(balance_before, 2),
                'BAL AFTER': round(current_balance, 2),
                'TO NO.': recipient['phone'],
                'TO NAME': recipient['name'],
                'TO ACCT': recipient['account']
            }

            transactions.append(transaction)
            regular_txn_count += 1  # Increment regular transaction counter

            # Next transaction time
            hours_gap = np.random.exponential(user_profile['hours_between_txns'])
            if current_date.weekday() >= 5 and np.random.random() < user_profile['pref_weekend']:
                hours_gap *= 0.7

            current_date += timedelta(hours=max(0.5, hours_gap))

        # Handle defaulted loan (no repayment transaction)
        if active_loan and active_loan.get('repay_at_txn') is None:
            # Loan defaulted - risk label already recorded
            pass

        # Sort transactions by date
        transactions.sort(key=lambda x: x['TRANSACTION DATE'])

        # Determine final credit risk label for user
        if credit_risk_labels:
            final_risk_label = max(credit_risk_labels)  # Worst outcome
        else:
            final_risk_label = -1  # No loans taken

        return transactions, {
            'user_id': user_profile['user_id'],
            'credit_archetype': user_profile['credit_archetype'],
            'loans_taken': loans_taken,
            'credit_risk_label': final_risk_label,
            'final_credit_limit': user_profile.get('credit_limit', 0),
            'total_transactions': len(transactions)
        }

    def save_user_transactions(self, user_id, transactions):
        """Save transactions for a single user to CSV."""

        if not transactions:
            return None

        df = pd.DataFrame(transactions)
        filename = f"{user_id}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath

    def generate_dataset(self):
        """Generate complete dataset with individual files per user."""

        print(f"\n{'='*80}")
        print("GENERATING {0:,} INDIVIDUAL USER DATASETS".format(self.n_users))
        print(f"{'='*80}")

        user_summaries = []
        total_transactions = 0
        archetype_counts = {k: 0 for k in self.CREDIT_ARCHETYPES.keys()}
        risk_label_counts = {-1: 0, 0: 0, 1: 0, 2: 0}

        for i in range(self.n_users):
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i+1:,}/{self.n_users:,} users generated...")

            # Generate user profile
            user_profile = self.generate_user_profile(i)
            archetype_counts[user_profile['credit_archetype']] += 1

            # Generate transactions
            transactions, summary = self.generate_transactions_for_user(user_profile)

            # Save to individual file
            self.save_user_transactions(user_profile['user_id'], transactions)

            # Track statistics
            user_summaries.append(summary)
            total_transactions += len(transactions)
            risk_label_counts[summary['credit_risk_label']] += 1

        # Save user summaries
        summary_df = pd.DataFrame(user_summaries)
        summary_path = self.output_dir.parent / 'user_summaries.csv'
        summary_df.to_csv(summary_path, index=False)

        # Print statistics
        print(f"\n{'='*80}")
        print("GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nTotal users: {self.n_users:,}")
        print(f"Total transactions: {total_transactions:,}")
        print(f"Average transactions per user: {total_transactions / self.n_users:.1f}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"User summaries: {summary_path}")

        print(f"\n{'='*80}")
        print("CREDIT BEHAVIOR DISTRIBUTION")
        print(f"{'='*80}")
        for archetype, count in archetype_counts.items():
            pct = count / self.n_users * 100
            print(f"  {archetype}: {count:,} ({pct:.1f}%)")

        print(f"\n{'='*80}")
        print("CREDIT RISK LABEL DISTRIBUTION")
        print(f"{'='*80}")
        risk_labels = {
            -1: "No loans taken",
            0: "Good (repaid on time)",
            1: "Late (repaid after term)",
            2: "Default (failed to repay)"
        }
        for label, count in risk_label_counts.items():
            pct = count / self.n_users * 100
            print(f"  {label} - {risk_labels[label]}: {count:,} ({pct:.1f}%)")

        # Calculate borrower-specific default rate
        borrowers = self.n_users - risk_label_counts[-1]
        if borrowers > 0:
            defaults = risk_label_counts[2]
            default_rate = defaults / borrowers * 100
            print(f"\nDefault rate (among borrowers): {defaults:,}/{borrowers:,} ({default_rate:.2f}%)")

        return summary_df


def main():
    """Generate calibrated synthetic dataset with per-user files."""

    print("=" * 80)
    print("CALIBRATED MOBILE MONEY DATA GENERATOR v2.0")
    print("Per-User Transaction Datasets with Credit/Loan Functionality")
    print("=" * 80)

    # Initialize generator
    generator = CalibratedMoMoDataGenerator(
        n_users=10000,
        avg_transactions_per_user=15,
        start_date='2024-01-01',
        duration_days=180,
        output_dir='data/user_transactions'
    )

    # Generate dataset
    summary_df = generator.generate_dataset()

    print(f"\n{'='*80}")
    print("SAMPLE USER SUMMARIES")
    print(f"{'='*80}")
    print(summary_df.head(20).to_string())

    return summary_df


if __name__ == "__main__":
    summary_df = main()
