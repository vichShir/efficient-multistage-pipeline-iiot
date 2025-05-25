def get_protocols(df, targets_cols):
    modules = list(set(map(lambda x: x.split('.')[0], df.columns)))  # get protocols from columns
    modules = [m for m in modules if m not in targets_cols]  # remove target columns
    modules.remove('frame')
    return modules


def filter_module(columns, module):
    match_module = map(lambda x: x if module+'.' in x else None, columns)  # match module columns else None
    clean_module = filter(lambda x: x is not None, match_module)  # remove None values
    return list(clean_module)