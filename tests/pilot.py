"""Exercise the full training budget on a temporary 5,000-borrower dataset."""

from workflow import check


if __name__ == "__main__":
    check(borrowers=5000, epochs=20, controls=True)
