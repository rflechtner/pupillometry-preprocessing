import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
def max_diff_filter(tracker_data, T, columns=None, lower=-np.inf, upper=np.inf,
                    prefix=None):
    """
Filters data points that differ more than T from the previous or next data
point. Addtionally, lower and upper bounds can be defined for valid data.
Argument 'columns' allows to process only select columns. In this case, if a
prefix is given, the processed data will be added to the input data frame with
prefixed column names. Otherwise the original columns are overwritten in the
output.
    """
    if isinstance(columns, list):
        original_data = tracker_data.copy()
        tracker_data = tracker_data.loc[:, columns]

    tracker_data = tracker_data.where((tracker_data.diff().abs() < T) & (
        tracker_data.diff(-1).abs() < T))
    tracker_data = tracker_data.where((tracker_data >= lower) & (
        tracker_data <= upper))

    if isinstance(prefix, str):
        tracker_data.rename(columns=lambda x: prefix+x, inplace=True)

    if isinstance(columns, list):
        if isinstance(prefix, str):
            tracker_data = original_data.join(tracker_data)
        else:
            original_data.loc[:, columns] = tracker_data
            tracker_data = original_data

    return tracker_data


#%%
def hampel(x, win_size, thresh, thresh_type='fixed', upper='auto',
           lower='auto', max_fill=None, win_type=None, min_periods=0,
           return_rm=False, mask=False, **kwargs):
    """
This is a Hampel-like filter function which identifies outliers by comparing
each data point to a rolling (window) median. A data point is considered an
outlier, if the absolute of its difference from the rolling median exceeds a
certain fixed value. Note that the proper Hampel filter uses a dynamical
threshold that is inferred from the variance between data points over the same
window.

Input
-----
x : list, pandas Series or other one-dimensional sequential data.

Parameters
----------
win_size : integer.
    Size of the rolling median window. The median will be computed over the
    last win_size/2 and the next win_size/2 values.

thresh : numerical scalar.
    Maximum absolute difference allowed from rolling median. Each value that
    deviates by more than this value from the rolling median will be removed
    and replaced with the rolling median.

thresh_type : 'fixed', 'MAD' or 'SD', default : 'fixed'.
    Whether to use a fixed numerical threshold ('fixed') where threshold =
    thresh or whether to estimate the threshold from the data's variance,
    either using the standard deviation from the rolling median ('SD') or an
    unbiased estimate of the standard deviation assuming gaussian distribution
    ('MAD'). These two are calculated as follows:
    'SD' : thresh * sqrt( sum((x[k] - median(x[k-win_size/2 ;
    k+win_size/2])**2) / len(x)-1 )
    'MAD' : thresh * 1.4826 * median( abs((x[k] - median(x[k-win_size/2 ;
    k+win_size/2])) )

upper : 'auto' or numerical scalar. Default: 'auto'.
    Values above this threshold are ignored when computing the rolling median,
    and thus most likely will be classified outliers. If 'auto', will try to
    infer an upper limit as the maximum of a rolling median with window size
    100, plus 0.1 * upper-lower.

lower : 'auto' or numerical scalar. Default: 'auto'.
    Values below this threshold are ignored when computing the rolling median,
    and thus most likely will be classified outliers. If 'auto', will try to
    infer a lower limit as the minimum of a rolling median with window size
    100, minus 0.1 * upper-lower.

win_type : string or None.
    Alternative window types can be passed to pandas.DataFrame.rolling().
    See https://pandas.pydata.org/pandas-docs/stable/generated
    /pandas.DataFrame.rolling.html

min_periods : integer, default : 0.
    Minimum number of non-na data points in a window to compute the median
    from. If for any position this condition is not met, will return NaN at
    this position.

max_fill : integer or None, default : None.
    Limit of how many consecutive removed or NaN values to replace. Wherever a
    section of NaN values / removed values is longer than this threshold, the
    whole section will be returned as NaNs.

return_rm : Boolean, default : False.
    If set to true, additionally returns a pandas Series containing the rolling
    median.

Returns
-------
x, where values considered outliers are replaced with the rolling median.
or
(x, rm) if return_rm == True
    """
    assert isinstance(mask, bool)

    try:
        x = pd.Series(x)
    except:
        raise ValueError('Input must be one-dimensional')

    if upper == 'auto' and lower == 'auto':
        rm = x.where(x > 0).rolling(100, center=True, min_periods=30,
                                    win_type='triang').mean()
        upper = rm.max()
        lower = rm.min()
        upper += 0.1*(upper-lower)
        lower -= 0.1*(upper-lower)
        print('lower: %0.2f ; upper: %0.2f' % (lower, upper))
    elif upper == 'auto':
        rm = x.where(x >= lower).rolling(100, center=True, min_periods=30,
                                         win_type='triang').mean()
        upper = rm.max()+rm.max()*0.0
        print('upper: %0.2f' % (upper))
    elif lower == 'auto':
        rm = x.where(x <= upper).rolling(100, center=True, min_periods=30,
                                         win_type='triang').mean()
        lower = rm.min()-rm.min()*0.0
        print('lower: %0.2f' % (lower))
    elif not(isinstance(lower, (int, float)) and isinstance(upper,
                                                            (int, float))):
        raise ValueError('Boundaries must be float, int or "auto"')

    rm = x.where(x.between(lower, upper)).rolling(
            win_size, center=True, min_periods=min_periods,
            win_type=win_type, **kwargs
            ).median()

    error = (x - rm).where(x.between(lower, upper))

    if thresh_type == 'fixed':
        pass
    elif thresh_type == 'MAD':
        thresh = thresh * error.mad()
    elif thresh_type == 'SD':
        thresh = thresh * error.std()
    else:
        raise ValueError('threshold type "%s" not recognized' % thresh_type)

    print('threshold : %f' % thresh)

    if mask:
        return ~((rm-x).abs() < thresh)

    out = x.where(((rm-x).abs() < thresh))

    if isinstance(max_fill, int):
        col = out.isna()
        consecutives = col.groupby((col != col.shift()).cumsum()).transform(
            'count') <= max_fill
        consecutives |= ~col

    elif max_fill is not None:
        raise ValueError('max_fill must be integer or None.')
    else:
        # to return an object with shape of out where every value is True
        consecutives = out | True
    # never mind pep8 here, we're dealing with a df
    out = out.where(((rm-x).abs() < thresh) | (consecutives == False), rm)

    if (return_rm is True):
        return out, rm

    return out


#%%
def trend_line_deviations(data, n, win_size, win_type=None, method="max",
                          timestamps="index", mask=False, min_periods=0,
                          passes=2, interpolate=True,
                          reintroduce_all_for_threshold=True, **kwargs):
    """
xxxxxxx

Input
-----
data : list, pandas Series or other one-dimensional sequential data.

Parameters
----------
win_size : integer.
    Size of the rolling median window. The median will be computed over the
    last win_size/2 and the next win_size/2 values.

win_type : string or None.
    Alternative window types can be passed to pandas.DataFrame.rolling().
    See https://pandas.pydata.org/pandas-docs/stable/generated
    /pandas.DataFrame.rolling.html

min_periods : integer, default : 0.
    Minimum number of non-na data points in a window to compute the median
    from. If for any position this condition is not met, will return NaN at
    this position.


Returns
-------
data, where values considered outliers are removed.
    """
    for var in ["mask", "reintroduce_all_for_threshold"]:
        try:
            assert eval("isinstance(%s, bool)" % var)
        except:
            raise TypeError(
                    "Argument for '%s' is expected to be a boolean." % var)

    try:
        data = pd.Series(data)
    except:
        raise TypeError('Input must be one-dimensional')

    # create filter mask that's False everywhere to include all values when
    # using mask
    filter_mask = pd.Series(len(data) * [False])
    filter_mask.index = data.index

    # filter routine allowing for multiple passes
    for npass in range(passes):

        filtered = data.mask(filter_mask)

        trendline = filtered.rolling(
                win_size, center=True, min_periods=min_periods,
                win_type=win_type, **kwargs
                ).mean()

        if interpolate is True:
            trendline.interpolate(method="index", inplace=True)

        # // TO DO : decide whether to use absolute trend line deviations from
        # filtered values (from last pass) or from all values to determine the
        # cutoff for outliers //
        # i'll allow both ways for now

        if reintroduce_all_for_threshold:
            threshold = calculate_MAD_distance((data - trendline).abs(), n)
        else:
            threshold = calculate_MAD_distance((filtered - trendline).abs(), n)

        # now we reintroduce all values
        absolute_dev = (data - trendline).abs()
        filter_mask = absolute_dev > threshold

        # return and pass again

    if mask is True:
        return filter_mask

    return data.mask(filter_mask)


def calculate_MAD_distance(x, n):
    MAD = x.mad(skipna=True)
    threshold = x.median(skipna=True) + n * MAD
    return threshold


# %%
def dilation_speed_outliers(data, n=1, method="max", timestamps="index",
                            mask=False):
    """
To detect dilation speed outliers, the median absolute deviation (MAD), which
is a robust and outlier resilient data dispersion metric (Leys, Ley, Klein,
Bernard, & Licata, 2013), is calculated from the dilation speed series,
multiplied by a constant (n),and summed with the median dilation speed.
Samples with dilation speeds above the threshold are marked as outliers and
rejected.

    """
    assert isinstance(mask, bool)

    try:
        data = pd.Series(data)
        assert data.apply(np.isreal).all()
    except:
        raise TypeError(
            "data must be of type pandas.Series or compatible and can " +
            "only contain numerical or missing values.")

    if timestamps == "index":
        # timestamps in index as a series
        ts = data.index.to_series()
    else:
        try:
            ts = pd.Series(timestamps)
            assert len(ts) == len(data)
            assert ts.apply(np.isreal).all()
            assert not ts.isna().any()
        except:
            raise ValueError(
                "timestamps must be of type pandas.Series or compatible" +
                " and can only contain numerical values.")

    # velocity profile could be computed either by comparing the current and
    # the preceeding sample, or by comparing the current and the next sample.
    # we'll implement both options and additionally allow to calculate the max
    # or mean from both

    try:
        assert method in ["previous", "following", "max", "mean"]
    except AssertionError:
        raise ValueError(
            'method must be one of ["previous","following","max","mean"]')
    # calculate velocity
    derivatives = pd.DataFrame()
    derivatives["previous"] = (data.diff(1) / ts.diff(1)).abs()
    derivatives["following"] = (data.diff(-1) / ts.diff(-1)).abs()

    if method == "max":
        dilation_speed = derivatives.max(axis=1, skipna=True)
    elif method == "mean":
        dilation_speed = derivatives.mean(axis=1, skipna=True)
    elif method == "previous":
        dilation_speed = derivatives.previous
    else:
        dilation_speed = derivatives.following

    # finding dilation speed outliers is done by setting a threshold of
    # median + n * MAD from the dilation speed series

    threshold = calculate_MAD_distance(dilation_speed, n)

    filter_mask = dilation_speed > threshold

    if mask:
        return filter_mask

    return data.mask(filter_mask)


#%%
def sparsity_filter(data, min_gap_size, min_chunk_size, null_threshold=0,
                    mask=False):
    """
Filter to remove smaller, isolated chunks of data between larger gaps (missing
data).

Parameters:

data : pandas Series or compatible iterable
    Data to be filtered

min_gap_size : integer.
    How many consecutive missing values constitute a "gap" in the data.

min_chunk_size : integer.
    Chunks of data between "gaps" will be removed if not at least consisting of
    this many values.

null_threshold : scalar, default 0.
    Values <= this value, in addition to NaN values, will be considered
    missing.

mask : boolean, default False.
    If True, returns a boolean filter mask instead of the filtered data.

Returns:

pandas.Series where chunks of length < min_chunk_size are removed.
    Or: boolean filter mask that is True where chunks are shorter than
    min_chunk_size
    """
    try:
        assert isinstance(min_gap_size, int)
        assert isinstance(min_chunk_size, int)
    except AssertionError:
        raise TypeError(
                "min_gap_size & min_chunk_size need to be of type integer.")

    # we first need to identify larger chunks of invalids
    # these lines count changes from valid to invalid, returning the number
    # of invalids / valids in a row
    valid = data > null_threshold
    consecutive_invalids = valid.groupby((valid != valid.shift(
            )).cumsum()).transform('count')
    # in the following step, we remove all valids
    consecutive_invalids = consecutive_invalids.mask(valid)
    # this now gives us a series where each invalid data point is represented
    # by an integer indicating the number of invalid values around it (itself
    # included)

    # this can be used to identify large enough gaps
    gaps = consecutive_invalids >= min_gap_size

    # now we repeat the above to count data points between gaps
    between_gaps = gaps.groupby((gaps != gaps.shift(
            )).cumsum()).transform('count')
    # in the following step, we remove all invalids / gap values
    between_gaps = between_gaps.mask(gaps)
    # this leaves us with a series containing a number indicating the size of
    # the chunk of data between gaps at every location where there is no gap.
    # We can now create a filter mask to remove chunks of insufficient size.
    filter_mask = between_gaps < min_chunk_size

    if mask:
        return filter_mask

    return data.mask(filter_mask)


# %%
def plot_filter(data, mask, size=4):
    plt.scatter(data[~mask].index, data[~mask].values, color="black", s=size)
    plt.scatter(data[mask].index, data[mask].values, color="red", s=size)
