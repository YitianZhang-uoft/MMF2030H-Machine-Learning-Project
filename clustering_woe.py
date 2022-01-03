import pandas as pd
from varclushi import VarClusHi


if __name__ == '__main__':
    data = pd.read_csv('training_woe.csv')

    iv = pd.read_csv('IV_R_2.csv')
    iv.Variable = 'WOE_' + iv.Variable

    exp_df_1 = data.drop(columns=['SK_ID_CURR', 'TARGET'])
    exp_df_1 = exp_df_1.sample(n=int(0.1 * len(exp_df_1)), replace=False, random_state=1)
    exp_df_1 = exp_df_1.loc[:, exp_df_1.apply(pd.Series.nunique) != 1]

    # Multivariate Screening
    # Variable Clustering
    vc = VarClusHi(exp_df_1)
    vc.varclus()
    vc_df = vc.rsquare
    vc_df = vc_df.merge(iv, on='Variable', how='left')
    exp_idx_2 = []
    for grp in range(vc_df['Cluster'].max()):
        grp_df = vc_df[vc_df['Cluster'] == grp]
        v_1_row = grp_df[grp_df['IV'] == grp_df['IV'].max()]
        if len(v_1_row) != 1:
            v_1_row = v_1_row[v_1_row['RS_Ratio'] == v_1_row['RS_Ratio'].min()]
            if len(v_1_row) != 1:
                v_1_row = v_1_row.reset_index(drop=True)
                v_1_row = v_1_row[:1]
        v_1 = v_1_row['Variable'].to_string(index=False)
        v_2_row = grp_df[grp_df['RS_Ratio'] == grp_df['RS_Ratio'].min()]
        if len(v_2_row) != 1:
            v_2_row = v_2_row[v_2_row['IV'] == v_2_row['IV'].max()]
            if len(v_2_row) != 1:
                v_2_row = v_2_row.reset_index(drop=True)
                v_2_row = v_2_row[:1]
        v_2 = v_2_row['Variable'].to_string(index=False)
        if v_1 == v_2:
            exp_idx_2.append(v_1)
        else:
            exp_idx_2.append(v_1)
            exp_idx_2.append(v_2)

    print(exp_idx_2)
