import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 1. Create a sample laptop dataset
data = {
    'Brand': ['Dell', 'Apple', 'HP', 'Apple', 'Dell'],
    'Model': ['XPS', 'MacBook', 'Spectre', 'MacBook', 'Inspiron'],
    'Performance_Tier': ['Mid', 'High', 'Low', 'High', 'Low'], # Ordinal
    'Price': [1200, 2500, 800, 2300, 600]
}

df = pd.DataFrame(data)
print("--- Original Dataset ---")
print(df)

# 2. Label Encoding (Best for Ordinal data where order matters)
# We will encode 'Performance_Tier'
le = LabelEncoder()
df['Performance_Tier_Encoded'] = le.fit_transform(df['Performance_Tier'])

# 3. One-Hot Encoding (Best for Nominal data where no order exists)
# We will encode 'Brand'
ohe = OneHotEncoder(sparse_output=False) # sparse=False returns an array
ohe_transformed = ohe.fit_transform(df[['Brand']])

# Create a DataFrame from the one-hot encoded columns
ohe_df = pd.DataFrame(ohe_transformed, columns=ohe.get_feature_names_out(['Brand']))

# Combine back with the original dataframe
final_df = pd.concat([df, ohe_df], axis=1).drop(columns=['Brand'])

print("\n--- Processed Dataset ---")
print(final_df)
