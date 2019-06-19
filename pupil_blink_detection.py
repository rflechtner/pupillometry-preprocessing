import pandas as pd
import numpy as np


def velocity_blinks(data, timestamps_column='index', thresh=40, smoothing=5,
                    min_length=0, max_length=1, last_upstroke_in_range=False,
                    ignore_zeros=False, policy='all', between_blinks=0,
                    offset_thr=2, win_type='hamming', margin=0, debug=False):

    if (last_upstroke_in_range is True) and not isinstance(
            max_length, (int, float)):
        raise ValueError('max_length must be given if last_upstroke_in_range'+
                         ' is set to True')

    data = pd.DataFrame(data, copy=True)

    if (isinstance(timestamps_column, str)) and timestamps_column == 'index':
        timestamps_column = data.index.to_series()
    elif timestamps_column in data.columns:
        timestamps_column = data.loc[:, timestamps_column]
    else:
        raise ValueError('%s not in columns' % timestamps_column)

    if isinstance(timestamps_column.iloc[0], pd.datetime):
        timestamps_diff = timestamps_column.diff().dt.total_seconds()
    else:
        timestamps_diff = timestamps_column.diff()

    if ignore_zeros is True:
        data = data.where(data != 0.0)

    derivatives = pd.DataFrame()

    for column in data.columns:
        derivatives[column] = (data[column].rolling(
            smoothing, center=True, win_type=win_type)
            .mean().diff() / timestamps_diff)

    if policy == 'all':
        downstroke = (derivatives < -thresh).all(axis=1)
        upstroke = (derivatives > thresh).all(axis=1)
        offset = (derivatives.abs() < offset_thr).all(axis=1)
    elif policy == 'any':
        downstroke = (derivatives < -thresh).any(axis=1)
        upstroke = (derivatives > thresh).any(axis=1)
        offset = (derivatives.abs() < offset_thr).any(axis=1)

    downstroke = downstroke.astype(int).diff() == 1
    upstroke = upstroke.astype(int).diff(-1) == 1

    downstroke_times = timestamps_column[downstroke.values]
    upstroke_times = timestamps_column[upstroke.values]
    offset_times = timestamps_column[offset.values]

    frame = pd.DataFrame(downstroke_times.values, columns=['downstroke'])

    if last_upstroke_in_range is True:
        frame['upstroke'] = frame.downstroke.apply(
            lambda x: upstroke_times[
                    upstroke_times.between(x, x+max_length)].max())
        frame['offset'] = frame.upstroke.apply(
            lambda x: offset_times[(offset_times - x) > 0].min())
        frame.drop_duplicates('upstroke', keep='first', inplace=True)

    else:
        frame['upstroke'] = frame.downstroke.apply(
            lambda x: upstroke_times[(upstroke_times - x) > 0].min())
        frame['offset'] = frame.upstroke.apply(
            lambda x: offset_times[(offset_times - x) > 0].min())
        frame.drop_duplicates('upstroke', keep='last', inplace=True)

    frame.drop_duplicates('offset', keep='first', inplace=True)

    frame.dropna(inplace=True)
    frame['length'] = frame.upstroke - frame.downstroke
    frame['to_next'] = frame.downstroke.shift(-1) - frame.upstroke
    frame['to_last'] = frame.to_next.shift()
    frame = frame[frame.length.between(min_length, max_length)]
    frame = frame[(frame.loc[:, ['to_next', 'to_last']]
                   > between_blinks).all(axis=1)]

    frame['since_last_offset'] = frame.downstroke - frame.offset.shift()

    frame.loc[:, 'offset'] = frame.loc[:, 'offset'] + margin
    frame.loc[:, 'downstroke'] = frame.downstroke - margin

    if debug is True:
        return frame.to_dict('records'), derivatives, data.rolling(
                smoothing, center=True, win_type=win_type).mean()
        
    return frame.loc[:, ['downstroke', 'upstroke', 'offset', 'length',
                             'since_last_offset']].to_dict('records')


def noise_blinks(data, null_threshold=0, min_valid_block_size=0,
                 smoothing=False, win_type='hamming',
                 include_cutoff_blinks=False, min_length=0,
                 max_length=0.5):
    """
Eye blinks in pupillometry data are characterized by episodes of missing /
invalid data (such as 0 or negative dilations) flanked by a sudden drop from
and recovery to baseline. This function identifies episodes of missing data and
the characteristic drops by determining where the data stops increasing
monotonically when moving away from the onsetor offset. Returns a data frame
with rows containing indices for the last/first valid data point before and
after an episode of missing/invalid data and indices of data points where onset
and offset of anomalous data due to the eye blink are thought to be. Based on
an algorythm described in *Hershman, Ronen; Henik, Avishai; Cohen, Noga (2018):
A novel blink detection method based on pupillometry noise. In Behavior
research methods 50 (1), pp. 107–114. DOI: 10.3758/s13428-017-1008-1.*

Arguments:
data : pandas.Series or convertible.

null_threshold: float, default = 0.
    Data <= null_threshold will be considered missing. Missing data (NaN) will
    always be considered missing.

min_valid_block_size: int, default = 0.
    Can be used to consider closeby segments of invalid/missing data as one.
    Argument indicates how many valid need to seperate two missing values for
    them to be considered independent blinks. Values <=1 will disable
    this function.

smoothing: int or False, default = False.
    Allows to apply smoothing before identification of the point where values
    stop to increase monotonically. Argument indicates window size of the
    rolling mean. If <= 1 or False, data will remain unsmoothed. Useful if
    tracker has sufficiently high sample rate and noise characteristics that
    allow for single samples breaking the stereotypical pattern of
    monotonically increasing values, even though the general trend is still
    rising. Window size of 0.01 * sampling rate is proposed to yield a
    smoothing window of 10ms (Hershman, Henik, Cohen 2018).

win_type: str, default = 'hamming'.
    Window type to be used for smoothing. Will be ignored if *smoothed* <= 1 or
    **False**. If ``win_type=None`` all points are evenly weighted. To learn
    more about different window types see `scipy.signal window functions
    <https://docs.scipy.org/doc/scipy/reference/signal.html#window-functions>`.

include_cutoff_blinks: boolean, default = False.
    Whether or not to include episodes of missing/invalid data at the beginning
    or end of the series. If True, episodes of missing data at the beginning of
    the series will have their onset / last valid on the first data point, and
    respectively episodes at the end will have their offset/first valid on the
    last data point. If ´False´, only episodes surrounded by valid data will be
    considered.

´´´Returns´´´:
    pandas DataFrame where each row is a blink, described by indices of four
    data points:
    1. onset : last value before data monotonously drops until invalid / missing.
    2. last_valid : last valid value before episode of invalid / missing data.
    3. first_valid : first valid value after episode of invalid / missing data.
    4. offset : first value where data has stopped increasing monotonously after
    episode of invalid data.

    """
    assert isinstance(min_valid_block_size, int)
    try:
        data = pd.Series(data, copy=True)
    except Exception:
        raise TypeError(
            'Input must be of type pandas.Series or convertible to such.')
    # this line counts how many consecutive valid (or invalid)
    # data points we have.
    consecutive_valids = data.groupby((data.le(null_threshold) != data.shift(
            ).le(null_threshold)).cumsum()).transform('count')
    # in the following step, we replace all invalids with 0
    consecutive_valids = consecutive_valids.mask(data.le(null_threshold), 0)
    # this now gives us a series where each invalid data point is represented
    # by a 0, and each valid one by an integer indicating the number of valid
    # values around it (itself included)

    # which now allows us to replace all values, where we only have a few
    # valid values surrounded by invalids, which we deem not trustworthy
    data = data.mask(consecutive_valids < min_valid_block_size, null_threshold)

    # now we identify all data points directly before and after episodes of
    # invalid data, which are onsets & offsets for onset: find data points that
    # are the last valid values before invalids
    onsets = (data <= 0).astype(float).diff(-1)
    onsets = onsets[onsets == -1]

    # for offset: find data points that are the first valid values
    # after invalids
    offsets = (data <= 0).astype(float).diff(1)
    offsets = offsets[offsets == -1]

    # the next step is to determine where the data stops increasing
    # monotonically when moving away from the onset/offset

    # at this point, Hershman, Henik & Cohen suggest smoothing the data with a
    # 10ms window. Here, we'll just use number of samples as measure of window
    # width. If < 2, no smoothing takes place.
    if isinstance(smoothing, int) and (smoothing >= 2):
        smoothed = data.rolling(smoothing, center=True,
                                win_type=win_type).mean()
    elif isinstance(smoothing, int) or smoothing is False:
        smoothed = data
    else:
        raise ValueError('"smoothing must be int or False"')

    # compute a boolean data frame that tells us whether the next and the last
    # value are larger / smaller
    frame = pd.concat([smoothed.diff(1) < 0, smoothed.diff(-1) < 0], axis=1)
    frame.columns = ["last_one_bigger", "next_one_bigger"]

    onsets = onsets.index.to_series().apply(
        lambda x: frame.index[
                frame.next_one_bigger & (x-frame.index >= 0)].max())

    offsets = offsets.index.to_series().apply(
        lambda x: frame.index[
                frame.last_one_bigger & (x-frame.index <= 0)].min())

    # EDGE CASES
    # usually, we'd have as many onsets as offsets, as each onset is followed
    # by an offset. However, there are two edge cases we need to take care of:
    # blinks starting before the beginning of the time series (no onset) and
    # blinks being cut of at the end of the time series (no offset)

    # first: quick check whether all data conforms to our expectations about
    # the standard case
    try:
        # for each onset, there is an offset -> same number of onsets & offsets
        assert len(onsets) == len(offsets)
        # all onsets are earlier than their corresponding offset
        assert (onsets.index < offsets.index).all()
        # there is no offset that is earlier than the next onset
        assert not (onsets.index.to_series().shift(-1)
                < offsets.index).any()
        # there is no onset, for which the difference to its offset is larger
        # than its difference to the next onset
        assert not (offsets.index-onsets.index <
                onsets.index.to_series().diff(-1)).any()

    except AssertionError:
        print('Exceptional Data!')

    # FIX EDGE CASES

    # if there is no matching onset for the first offset, set first onset to
    # first data point:
    if not (onsets.index < offsets.index[0]).any():
        print('first onset missing')
        if data.iloc[0] <= null_threshold:
            print('reason: data begins with invalid / blink value')
            if include_cutoff_blinks:
                onsets = onsets.append(
                    pd.Series(index=[
                            data.index.min()], data=[data.index.min()]))
                onsets.sort_index(inplace=True)
                print('fixed by setting first data point as onset')
            else:
                offsets = offsets.iloc[1:]
                print('fixed by dropping first offset')

    # if there is no matching offset for the last onset:
    if not (offsets.index > onsets.index[-1]).any():
        print('last offset missing')
        if data.iloc[-1] <= null_threshold:
            print('reason: data ends on invalid / blink value')
            if include_cutoff_blinks:
                offsets = offsets.append(
                    pd.Series(index=[
                            data.index.max()], data=[data.index.max()]))
                offsets.sort_index(inplace=True)
                print('fixed by setting last data point as offset')
            else:
                onsets = onsets.iloc[:-1]
                print('fixed by dropping last onset')

    # re-check whether all data conforms to our expectations about the
    # standard case
    try:
        # for each onset, there is an offset -> same number of onsets & offsets
        assert len(onsets) == len(offsets)
        # all onsets are earlier than their corresponding offset
        assert (onsets.index < offsets.index).all()
        # there is no offset that is earlier than the next onset
        assert not (onsets.index.to_series().shift(-1)
                < offsets.index).any()
        # there is no onset, for which the difference to its offset is larger
        # than its difference to the next onset
        assert not (offsets.index-onsets.index <
                onsets.index.to_series().diff(-1)).any()

        print('all good')
    except AssertionError:
        print('Data remains exceptional !')

    blinks = pd.DataFrame([onsets.values, onsets.index, offsets.index,
                           offsets.values],
                          index=['onset', 'last_valid', 'first_valid',
                                 'offset']).T

    blinks['length'] = blinks.first_valid - blinks.last_valid

    blinks = blinks[blinks.length.le(
        max_length) & blinks.length.ge(min_length)]

    return blinks


def interpolate_blinks(tracker_data, blink_times, order=2, lower=None,
                       upper=None, max_median_distance=None,
                       max_median_replace=0, remove_non_interpolatable=True,
                       margin=None):
    """
Uses onset & offset times supplied in blink_times to interpolate these episodes
in 'data'. Applies cubic spline interpolation by default, which can be
overwritten by setting the order. Spline interpolation is applied on 4 points
around the episode defined by onset & offset times. These points can be
validated through upper and lower bounds and/or a maximum distance threshold
from the median of all four points. Optionally, points can be allowed to be
replaced with the median if they fail validation by increasing
max_median_replace. If the number of invalid points exceeds this threshold, the
episode in question will not be interpolated, but will be removed instead.
    """

    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf

    # make sure tracker data is pandas DataFrame
    tracker_data = pd.DataFrame(tracker_data).copy()

    # make sure format is DataFrame and columns are named correctly
    blink_times = pd.DataFrame(blink_times)
    blink_times.rename(columns=str.lower, inplace=True)
    blink_times.rename(columns={'downstroke': 'onset'}, inplace=True)

    if not pd.Series(['onset', 'offset']).isin(blink_times.columns).all():
        raise ValueError(
                'blink_times must be pandas DataFrame containing columns ' +
                '"onset" & "offset" or convertible to such object.')

    if isinstance(margin, (int, float)):
        blink_times['onset'] = blink_times['onset'] - margin
        blink_times['offset'] = blink_times['offset'] + margin

    for blink_nr, this_blink in blink_times.iterrows():
        # find nearest index to onset & offset times
        onset = (tracker_data.index.to_series() -
                 this_blink.onset).abs().idxmin()
        offset = (tracker_data.index.to_series() -
                  this_blink.offset).abs().idxmin()

        # calculate length of interpolated interval
        length = offset - onset

        # select points one length before onset & one length after offset
        first = (tracker_data.index.to_series() -
                 (onset-length)).abs().idxmin()
        last = (tracker_data.index.to_series() -
                (offset+length)).abs().idxmin()

        # select segment from first to last point
        segment = tracker_data.loc[first:last, :].copy()

        # only select the 4 points we use for interpolation
        points = segment.loc[[first, onset, offset, last], :]

        if (lower != -np.inf) or (
                upper != np.inf) or (max_median_distance is not None):
            # check whether points are in range:
            in_range = (points >= lower) & (points <= upper)

            if (max_median_distance is not None):
                in_range = points.apply(lambda x: (
                    x-x.median()).abs() < max_median_distance) & in_range

            # replace points that are deemed invalid with median
            points = points.where(in_range).fillna(points.median())

            # remove columns where we have less than 4-max_median_replace
            # valid values
            points.loc[:, 4-in_range.sum() > max_median_replace] = np.nan

        # clear all values from segment
        segment.loc[:, :] = np.nan
        # put points values in empty df
        segment.update(points)

        # run interpolation on df where only points are available
        inter = segment.interpolate(
            method='spline', limit_area='inside', order=order)

        # write interpolated data back to original df
        if not remove_non_interpolatable:
            tracker_data.update(inter.loc[onset:offset])
        elif remove_non_interpolatable is True:
            tracker_data.loc[onset:offset, :] = inter.loc[onset:offset]

    return tracker_data


def remove_blinks(data, blink_times, timestamps_column='index',
                  out_window=0.100, in_window=0.200):
    if (isinstance(timestamps_column, str)) and timestamps_column == 'index':
        timestamps_column = data.index.to_series()

    if isinstance(timestamps_column.iloc[0], pd.datetime):
        out_window = pd.Timedelta(out_window, 's')
        in_window = pd.Timedelta(in_window, 's')

    drop1 = timestamps_column.isin(blink_times['downstroke'])
    blink_times['downstroke'] = timestamps_column[(drop1.astype(
        int).diff(-1) + drop1.astype(int).diff()) > 0].tolist()

    drop2 = timestamps_column.isin(blink_times['upstroke'])
    blink_times['upstroke'] = timestamps_column[(drop2.astype(
        int).diff(-1) + drop2.astype(int).diff()) > 0].tolist()

    drop = (drop1 | drop2)

#    return data.where(~drop), drop

    assert isinstance(drop, pd.Series)

    for time in blink_times['downstroke']:
        drop[timestamps_column.between(time-out_window, time+in_window)] = True

    for time in blink_times['upstroke']:
        drop[timestamps_column.between(time-in_window, time+out_window)] = True

    assert isinstance(drop, pd.Series)

    return data.where(~drop), drop


def zero_blinks(data, timestamps_column='index', consecutive=1,
                na_handling='zerofill'):

    if na_handling == 'zerofill':
        data = data.fillna(0)

    if (isinstance(timestamps_column, str)) and timestamps_column == 'index':
        timestamps_column = data.index.to_series()

    kill = {}

    if consecutive == 1:
        kill['downstroke'] = timestamps_column.where(
            data.sum(axis=1) == 0).dropna().tolist()
        kill['upstroke'] = []

    else:
        kill['upstroke'] = timestamps_column.where(data.rolling(
            consecutive, min_periods=1).sum().sum(axis=1) == 0
            ).dropna().tolist()
        kill['downstroke'] = timestamps_column.where(
            data[::-1].rolling(
                    consecutive, min_periods=1
                    ).sum()[::-1].sum(axis=1) == 0).dropna().tolist()

    return kill
