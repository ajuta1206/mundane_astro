"""Planetary positions and astronomical events via the Swiss Ephemeris.

Uses the built-in Moshier analytical ephemeris (FLG_MOSEPH), which is accurate
to well under an arcsecond for the planets over 3000 BC - 3000 AD -- far finer
than any astrological orb. No external ephemeris files are needed.

All positions are geocentric ecliptic longitudes of date (tropical zodiac) by
default; set sidereal=True for the Lahiri ayanamsa used in Vedic mundane work.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import swisseph as swe

PLANETS = {
    "sun": swe.SUN,
    "moon": swe.MOON,
    "mercury": swe.MERCURY,
    "venus": swe.VENUS,
    "mars": swe.MARS,
    "jupiter": swe.JUPITER,
    "saturn": swe.SATURN,
    "uranus": swe.URANUS,
    "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
    "node": swe.MEAN_NODE,  # Rahu (north lunar node)
}

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
ELEMENTS = ["fire", "earth", "air", "water"] * 3

# Hour of day (UT) at which positions are sampled: 20:00 UT ~ US market close.
SAMPLE_HOUR_UT = 20.0


def julday(d: date | datetime, hour: float = SAMPLE_HOUR_UT) -> float:
    if isinstance(d, datetime):
        hour = d.hour + d.minute / 60 + d.second / 3600
    return swe.julday(d.year, d.month, d.day, hour)


def jd_to_datetime(jd: float) -> datetime:
    y, m, d, h = swe.revjul(jd)
    return datetime(y, m, d, tzinfo=timezone.utc) + timedelta(hours=h)


def _flags(sidereal: bool) -> int:
    f = swe.FLG_MOSEPH | swe.FLG_SPEED
    if sidereal:
        f |= swe.FLG_SIDEREAL
    return f


def positions(jd: float, sidereal: bool = False) -> dict[str, tuple[float, float]]:
    """Return {planet: (longitude_deg, speed_deg_per_day)} at a Julian day."""
    if sidereal:
        swe.set_sid_mode(swe.SIDM_LAHIRI)
    fl = _flags(sidereal)
    out = {}
    for name, pid in PLANETS.items():
        xx, _ = swe.calc_ut(jd, pid, fl)
        out[name] = (xx[0] % 360.0, xx[3])
    return out


def daily_ephemeris(start: str | date, end: str | date, sidereal: bool = False) -> pd.DataFrame:
    """Daily table with <planet>_lon and <planet>_spd columns, indexed by date."""
    idx = pd.date_range(start, end, freq="D")
    rows = []
    for ts in idx:
        p = positions(julday(ts.date()), sidereal)
        row = {}
        for k, (lon, spd) in p.items():
            row[f"{k}_lon"] = lon
            row[f"{k}_spd"] = spd
        rows.append(row)
    df = pd.DataFrame(rows, index=idx)
    df.index.name = "date"
    return df


def angle_diff(a, b):
    """Signed smallest angle a-b in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def sign_of(lon) -> np.ndarray:
    return (np.asarray(lon) // 30).astype(int) % 12


# ---------------------------------------------------------------- events ----

def eclipses(start: str | date, end: str | date) -> pd.DataFrame:
    """All solar and lunar eclipses (global) between start and end."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    jd_end = julday(e.date(), 0)
    out = []
    for kind, fn in (("solar", swe.sol_eclipse_when_glob), ("lunar", swe.lun_eclipse_when)):
        jd = julday(s.date(), 0)
        while True:
            res, tret = fn(jd, swe.FLG_MOSEPH, 0, False)
            t = tret[0]
            if t > jd_end or t <= 0:
                break
            if res & swe.ECL_TOTAL:
                sub = "total"
            elif res & swe.ECL_ANNULAR:
                sub = "annular"
            elif res & (swe.ECL_ANNULAR_TOTAL if hasattr(swe, "ECL_ANNULAR_TOTAL") else 0):
                sub = "hybrid"
            elif res & swe.ECL_PARTIAL:
                sub = "partial"
            else:
                sub = "penumbral"
            out.append({"time": jd_to_datetime(t), "kind": kind, "type": sub})
            jd = t + 20
    df = pd.DataFrame(out).sort_values("time").reset_index(drop=True)
    if len(df):
        df["date"] = pd.to_datetime(df["time"].dt.date)
    return df


def lunations(start: str | date, end: str | date) -> pd.DataFrame:
    """New and full moons found by root-finding Sun-Moon elongation."""
    idx = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=2), end, freq="6h")
    jds = np.array([swe.julday(t.year, t.month, t.day, t.hour) for t in idx])
    el = np.array([
        (swe.calc_ut(j, swe.MOON, swe.FLG_MOSEPH)[0][0] - swe.calc_ut(j, swe.SUN, swe.FLG_MOSEPH)[0][0]) % 360
        for j in jds
    ])
    out = []
    for target, name in ((0.0, "new"), (180.0, "full")):
        d = angle_diff(el, target)
        for i in range(len(d) - 1):
            if d[i] < 0 <= d[i + 1] and abs(d[i]) < 30:
                # linear interpolation is accurate to ~minutes at 6h steps
                frac = -d[i] / (d[i + 1] - d[i])
                jd = jds[i] + frac * (jds[i + 1] - jds[i])
                out.append({"time": jd_to_datetime(jd), "phase": name})
    df = pd.DataFrame(out).sort_values("time").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["time"].dt.date)
    return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
