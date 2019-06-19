def multindex_iloc(df, index, level=0):
    label = df.index.levels[level][index]
    return df.iloc[df.index.get_loc(label)]
