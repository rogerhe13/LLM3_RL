import pandas as pd
df = pd.read_csv("cpa_data/cpa_clean.csv")

# Look for any columns that might contain "social", "advocate", "determinants", etc.
all_cols = df.columns.tolist()
potential_cols = [col for col in all_cols if any(word in col.lower() for word in ['social', 'advocate', 'determinant', 'health', 'ces'])]

print("Columns that might be the advocacy/social determinants score:")
for col in potential_cols:
    print(f"  {col}")

# Also check if there are any other communication-related columns
other_comm_cols = [col for col in all_cols if 'comm' in col.lower() and not col.startswith('comm_')]
print("\nOther potential communication columns:")
for col in other_comm_cols:
    print(f"  {col}")

# Show first few rows of data to see structure
print("\nFirst few rows of communication columns:")
comm_cols = [col for col in df.columns if col.startswith('comm_')]
print(df[comm_cols].head())