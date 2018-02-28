# Predict CHAID generated tree
def predict(df,tree):
    rules = tree.classification_rules()
    lenrules  = len(rules)
    j=0
    df.index = range(0,df.shape[0])
    Response = np.repeat(0, df.shape[0])
    while(j <= lenrules-1):
        r1 = rules[j]
        ruleset = list(r1.items())[1][1]
        lenruleset = len(ruleset)
        k = 0
        df1 = df
        while(k <= lenruleset-1):
            r = ruleset[k]
            v = r.get('variable')
            d = r.get('data')
            ind = []
            for i in range(0,df1[v].shape[0]):
                if df1[v].iloc[i] in d:
                    ind.append(df1.index[i])
            df1 = df1.loc[ind]
            if k == lenruleset-1:
                xset = tree.tree_store[list(r1.items())[0][1]]._members
                perc0,perc1 = xset.get(0),xset.get(1)
                print("Node:",j,perc0,perc1)
                if perc0 >= perc1:
                    Response[ind] =0
                else:
                    Response[ind] = 1
            k = k+1
        j = j+1
    return Response
