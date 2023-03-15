def lagged_df(df):
    df = df.with_column(pl.col("aid").shift(periods=1).over("session")
                              #.cast(pl.Int32)
                              #.fill_null(pl.col("aid"))
                              .alias("prev_aid"))
    return df

def generate_Graph(df):
    Graph = torch.tensor(np.transpose(df[['prev_aid','aid']].to_numpy()),dtype=torch.long)
    return Graph