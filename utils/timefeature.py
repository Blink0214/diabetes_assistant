import logging
from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# 时间特征工程
class TimeFeature:
    '''定义时间特征'''
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 接受一个pd.DatetimeIndex对象作为输入，并返回一个NumPy数组，表示该时间索引对应的特征值。
        pass

    def __repr__(self):
        # 用于返回类的字符串表示形式。
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    # 返回分钟内的秒数，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    # 返回小时内的分钟数，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    # 返回一天中的小时数，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    # 返回一周中的星期几，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    # 返回一个月中的日期，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    # 返回一年中的日期，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    # 返回一年中的月份，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    # 返回一年中的周数，范围在[-0.5, 0.5]之间。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """
    '''
    一个工厂函数，根据给定的时间频率字符串生成对应的时间特征列表。
    它根据频率字符串来选择使用哪些时间特征类，然后返回一个时间特征类的列表。
    如果给定的频率字符串不受支持，它将引发RuntimeError异常。
    '''
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        ME   - monthly
        W   - weekly
        D   - daily
        B   - business days
        h   - hourly
        min   - minutely
        s   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    """
    > all encoded between [-0.5 and 0.5]:
    > * ME - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * h - [Hour of day, day of week, day of month, day of year]
    > * min - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * s - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    '''
    一个主要的接口函数，用于将日期数据转换为时间特征向量。它接受两个参数：dates是日期数据，可以是任何包含日期时间信息的数据结构，freq是频率字符串，默认为小时。
    它首先将日期数据转换为pd.DatetimeIndex对象，然后根据指定的频率字符串调用time_features_from_frequency_str函数获取对应的时间特征列表，最后返回这些特征值的堆叠数组。
    '''
    # dates = pd.to_datetime(dates.date.values)
    dates = pd.to_datetime(dates.date.values, format='mixed', dayfirst=True)
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1, 0)
