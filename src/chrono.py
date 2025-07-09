#!/usr/bin/env python3

from datetime import datetime, timedelta
from typing import Union, Self # Any


class Chrono(object):
    def __init__(self, t: datetime = None):
        if t is None:
            self.at = datetime.now().astimezone()
        else:
            self.at = t.astimezone()

    def __repr__(self):
        return f"{self.at.isoformat('T')}"

    def __str__(self):
        return f"{self.at.isoformat('T')}"

    def __add__(self, delta: timedelta) -> Self:
        return Chrono(self.at + delta)

    def __sub__(self, t: Union[Self, timedelta, datetime]) -> Union[Self, timedelta]:
        if isinstance(t, timedelta):
            return Self(self.at - t)
        elif isinstance(t, Chrono):
            return self.at - t.at
        elif isinstance(t, datetime):
            if t.tzinfo is None:
                return self.at - t.astimezone()
            else:
                return self.at - t
        else:
            raise ValueError("t must be a timedelta or Chrono")

    def elapsed(self) -> str:
        t1 = Chrono()
        return f"{t1.at - self.at}"
