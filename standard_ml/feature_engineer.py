import pandas as pd
def add_rolling_metrics(raw_pd, field_id_list=["value"], window_size_list=[5]):
    """ Expanding features by rolling metrics according to specified window sizes.

    :param raw_pd: An input Pandas data frame
    :param field_id_list: list of column names to expand from
    :param window_size_list: list of window sizes to expand from
    :return: expanded pandas data frame with number of columns increased by
        len(field_id_list) * len(window_size_list), and
        the first max(window_size_list) will be cut off.
    """

    prepared_pd = pd.DataFrame()
    for field_id in field_id_list:
        for window_size in window_size_list:
            field_name = "%s_EXPANDED_AVG_W%s" % (field_id, window_size)
            prepared_pd[field_name] = raw_pd.rolling(window=window_size).mean()[field_id]
            field_name = "%s_EXPANDED_STD_W%s" % (field_id, window_size)
            prepared_pd[field_name] = raw_pd.rolling(window=window_size).std()[field_id]
            field_name = "%s_EXPANDED_MIN_W%s" % (field_id, window_size)
            prepared_pd[field_name] = raw_pd.rolling(window=window_size).min()[field_id]

            field_name = "%s_EXPANDED_MAX_W%s" % (field_id, window_size)
            prepared_pd[field_name] = raw_pd.rolling(window=window_size).max()[field_id]

            field_name = "%s_EXPANDED_SKEWNESS_W%s" % (field_id, window_size)
            prepared_pd[field_name] = raw_pd.rolling(window=window_size).skew()[field_id]

    raw_pd = raw_pd.iloc[(max(window_size_list)):, :]
    prepared_pd = prepared_pd.iloc[(max(window_size_list)):, :]
    prepared_pd.index = raw_pd.index
    return pd.concat([raw_pd, prepared_pd], axis=1)


def add_diff_previous(raw_pd, field_id_list=["value"], shift_period_list=[1]):
    """ Expanding features by derivative values according to shifting periods.

        :param raw_pd: An input Pandas data frame
        :param field_id_list: list of column names to expand from
        :param shift_period_list: list of shift period to expand from
        :return: expanded pandas data frame with number of columns increased by
            len(field_id_list) * len(shift_period_list), and
            the first max(shift_period_list) will be cut off.
    """

    prepared_pd = pd.DataFrame()
    for field_id in field_id_list:
        for shift_period in shift_period_list:
            field_name = "%s_EXPANDED_SHIFT_P%s" % (field_id, shift_period)
            prepared_pd[field_name] = raw_pd[str(field_id)] - raw_pd[str(field_id)].shift(periods=shift_period)

    raw_pd = raw_pd.iloc[max(shift_period_list):, :]
    prepared_pd = prepared_pd.iloc[max(shift_period_list):, :]
    prepared_pd.index = raw_pd.index
    return pd.concat([raw_pd, prepared_pd], axis=1)