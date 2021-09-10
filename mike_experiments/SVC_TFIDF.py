import get_dataframe


df, label_df = get_dataframe.get_dfs(pct_of_df=0.2, pct_meshterms=0.05)

print(df.shape)
print(label_df.shape)
print(label_df)