from get_dataframe import get_dfs

df, label_df = get_dfs(0.02, 0.01)
print(label_df.head())
print(label_df['created_date'].dt.year.unique())

