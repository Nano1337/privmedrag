"""
Utilities for processing EHR data for MCQ generation
"""

from typing import Dict, List, Any, Tuple, Set
def anonymize_data(df, level=0, k=5, l=2):
    """
    Anonymize EHR data based on privacy level:
    0 - No anonymization
    1 - Remove direct identifiers
    2 - Apply k-anonymity and l-diversity
    """
    if level == 0:
        return df

    df = remove_pii(df)

    if level >= 2:
        df = generalize_columns(df)
        df = apply_k_anonymity(df, k)
        df = apply_l_diversity(df, l)

    return df

def remove_pii(df):
    """Remove direct identifiers from the dataset"""
    pii_fields = ['name', 'address', 'phone_number', 'email', 'ssn']
    return df.drop(columns=[col for col in pii_fields if col in df.columns], errors='ignore')

def generalize_columns(df):
    """Generalize quasi-identifiers like age and zip"""
    import numpy as np
    if 'age' in df.columns:
        df['age'] = df['age'].apply(lambda x: f"{(x // 10) * 10}-{(x // 10) * 10 + 9}" if pd.notnull(x) else x)
    if 'zip' in df.columns:
        df['zip'] = df['zip'].astype(str).str[:3] + 'XX'
    return df

def apply_k_anonymity(df, k):
    """Ensure that each group defined by quasi-identifiers has at least k entries"""
    from collections import Counter

    qid_cols = ['age', 'gender', 'zip']
    qid_cols = [col for col in qid_cols if col in df.columns]
    if not qid_cols:
        return df

    group_sizes = df.groupby(qid_cols).size()
    valid_groups = group_sizes[group_sizes >= k].index

    df_k = df.set_index(qid_cols)
    df_k = df_k.loc[df_k.index.isin(valid_groups)].reset_index()
    return df_k

def apply_l_diversity(df, l):
    """Ensure sensitive attributes have at least l distinct values per group"""
    sensitive_attrs = ['conditions']
    qid_cols = ['age', 'gender', 'zip']
    qid_cols = [col for col in qid_cols if col in df.columns]

    def is_diverse(group):
        for attr in sensitive_attrs:
            if attr not in group.columns:
                continue
            unique_vals = group[attr].apply(lambda x: tuple(sorted(x)) if isinstance(x, list) else x).nunique()
            if unique_vals < l:
                return False
        return True

    if not qid_cols:
        return df

    grouped = df.groupby(qid_cols)
    diverse_groups = [group for _, group in grouped if is_diverse(group)]

    return pd.concat(diverse_groups, ignore_index=True) if diverse_groups else df.head(0)
