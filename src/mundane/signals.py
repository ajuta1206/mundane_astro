"""Mundane-astrology factors as daily time series.

Every factor is defined *mechanically* from the ephemeris so it can be tested
without hindsight. Definitions follow the mainstream financial-astrology
literature (e.g. Bradley 1948 "Stock Market Prediction"; McWhirter 1938;
Gann-school planetary cycles; lunar-cycle work by Dichev & Janes 2003 and
Yuan, Zheng & Zhu 2006, J. Empirical Finance).
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
import swisseph as swe

from .ephemeris import (ELEMENTS, SIGNS, angle_diff, daily_ephemeris, eclipses,
                        julday, lunations, sign_of)

SLOW = ["jupiter", "saturn", "uranus", "neptune", "pluto"]
ASPECTS = {"conj": 0, "sext": 60, "square": 90, "trine": 120, "opp": 180}
HARD = ("conj", "square", "opp")
SOFT = ("sext", "trine")

# Orbs (degrees) for slow-planet aspects; tighter for the fast Mars.
SLOW_ORB = 5.0
MARS_ORB = 3.0

FAMOUS = {  # human-readable labels used in reports
    "jupiter_saturn_conj": "Great Conjunction (Jupiter-Saturn)",
    "saturn_uranus_hard": "Saturn-Uranus hard aspect",
    "saturn_pluto_hard": "Saturn-Pluto hard aspect",
    "jupiter_uranus_hard": "Jupiter-Uranus hard aspect",
    "saturn_neptune_hard": "Saturn-Neptune hard aspect",
    "mercury_rx": "Mercury retrograde",
}


def aspect_orb(lon_a, lon_b, angle):
    """Distance (deg) of the a-b separation from an exact aspect angle."""
    sep = np.abs(angle_diff(lon_a, lon_b))  # 0..180
    return np.abs(sep - angle)


def _window(dates: pd.DatetimeIndex, events: pd.Series, before: int, after: int) -> np.ndarray:
    """Boolean mask: True within [event-before, event+after] days."""
    mask = np.zeros(len(dates), dtype=bool)
    pos = dates.searchsorted(pd.DatetimeIndex(events))
    for p in pos:
        lo, hi = max(0, p - before), min(len(dates), p + after + 1)
        mask[lo:hi] = True
    return mask


def _stations(spd: pd.Series) -> pd.DatetimeIndex:
    s = np.sign(spd.values)
    flips = np.where(s[1:] != s[:-1])[0] + 1
    return spd.index[flips]


def declination(df_index: pd.DatetimeIndex, planet: int) -> np.ndarray:
    fl = swe.FLG_MOSEPH | swe.FLG_EQUATORIAL
    return np.array([swe.calc_ut(julday(t.date()), planet, fl)[0][1] for t in df_index])


# --------------------------------------------------------------- Bradley ----

# Bradley (1948) valences: sextile & trine positive, square & opposition negative.
# Conjunction sign depends on the pair (benefic/malefic); we follow the
# convention used in public reconstructions of the Siderograph.
_BRADLEY_VAL = {"conj": None, "sext": 6, "square": -10, "trine": 10, "opp": -10}
_BENEFIC = {"venus", "jupiter"}
_MALEFIC = {"mars", "saturn"}
_BRADLEY_ORB = 15.0


def _conj_valence(a: str, b: str) -> float:
    s = {a, b}
    if s & _BENEFIC and not s & _MALEFIC:
        return 10.0
    if s & _MALEFIC and not s & _BENEFIC:
        return -10.0
    return 0.0  # mixed or neutral pairs


def _bradley_component(eph: pd.DataFrame, planets: list[str]) -> np.ndarray:
    total = np.zeros(len(eph))
    for a, b in itertools.combinations(planets, 2):
        for name, ang in ASPECTS.items():
            v = _BRADLEY_VAL[name] if name != "conj" else _conj_valence(a, b)
            if not v:
                continue
            orb = aspect_orb(eph[f"{a}_lon"], eph[f"{b}_lon"], ang)
            w = np.clip(1 - orb / _BRADLEY_ORB, 0, None)  # linear taper to 0 at 15 deg
            total += v * w
    return total


def bradley_siderograph(eph: pd.DataFrame) -> pd.DataFrame:
    """Bradley Siderograph = 4*long-term + mid-term + declination term."""
    long_term = _bradley_component(eph, SLOW)
    mid = _bradley_component(eph, ["mercury", "venus", "mars"] + SLOW)
    dec = declination(eph.index, swe.VENUS) + declination(eph.index, swe.MARS)
    sidero = 4 * long_term + mid + dec
    out = pd.DataFrame({"bradley_long": long_term, "bradley_mid": mid,
                        "bradley_decl": dec, "bradley": sidero}, index=eph.index)
    return out


def bradley_turns(sidero: pd.Series, half_window: int = 10) -> pd.DatetimeIndex:
    """Local extrema of the siderograph (classic 'turning point' dates)."""
    s = sidero.values
    turns = []
    for i in range(half_window, len(s) - half_window):
        seg = s[i - half_window:i + half_window + 1]
        if s[i] == seg.max() or s[i] == seg.min():
            turns.append(sidero.index[i])
    return pd.DatetimeIndex(turns)


# ----------------------------------------------------------- main table ----

@lru_cache(maxsize=4)
def build_signals(start: str = "1870-01-01", end: str = "2028-12-31",
                  sidereal: bool = False) -> pd.DataFrame:
    """Daily (calendar) DataFrame of boolean / numeric astrological factors."""
    eph = daily_ephemeris(start, end, sidereal)
    idx = eph.index
    sig = pd.DataFrame(index=idx)

    # --- retrogrades & stations
    for p in ["mercury", "venus", "mars", "jupiter", "saturn"]:
        sig[f"{p}_rx"] = eph[f"{p}_spd"] < 0
    sig["mercury_station"] = _window(idx, _stations(eph["mercury_spd"]), 3, 3)
    sig["any_inner_rx"] = sig[["mercury_rx", "venus_rx", "mars_rx"]].any(axis=1)

    # --- lunar cycle
    elong = (eph["moon_lon"] - eph["sun_lon"]) % 360
    sig["lunar_phase_deg"] = elong
    sig["waxing_moon"] = elong < 180
    lun = lunations(start, end)
    sig["new_moon_window"] = _window(idx, lun.loc[lun.phase == "new", "date"], 3, 3)
    sig["full_moon_window"] = _window(idx, lun.loc[lun.phase == "full", "date"], 3, 3)
    moon_sign = sign_of(eph["moon_lon"])
    for el in ("fire", "earth", "air", "water"):
        sig[f"moon_in_{el}"] = np.array([ELEMENTS[s] == el for s in moon_sign])

    # --- eclipses
    ecl = eclipses(start, end)
    sig["eclipse_window"] = _window(idx, ecl["date"], 7, 7)
    sig["solar_eclipse_window"] = _window(idx, ecl.loc[ecl.kind == "solar", "date"], 7, 7)
    sig["lunar_eclipse_window"] = _window(idx, ecl.loc[ecl.kind == "lunar", "date"], 7, 7)
    sig["eclipse_season"] = _window(idx, ecl["date"], 17, 17)  # ~ +/- half an eclipse season

    # --- slow-planet aspects
    hard_count = np.zeros(len(idx), dtype=int)
    soft_count = np.zeros(len(idx), dtype=int)
    for a, b in itertools.combinations(SLOW, 2):
        hard = np.zeros(len(idx), dtype=bool)
        soft = np.zeros(len(idx), dtype=bool)
        for name, ang in ASPECTS.items():
            on = aspect_orb(eph[f"{a}_lon"], eph[f"{b}_lon"], ang) <= SLOW_ORB
            if name == "conj":
                sig[f"{a}_{b}_conj"] = on
            if name in HARD:
                hard |= on
            else:
                soft |= on
        sig[f"{a}_{b}_hard"] = hard
        sig[f"{a}_{b}_soft"] = soft
        hard_count += hard
        soft_count += soft
    sig["slow_hard_count"] = hard_count
    sig["slow_soft_count"] = soft_count
    sig["slow_stress"] = hard_count >= 2           # >= 2 concurrent hard outer aspects
    sig["slow_harmony"] = (soft_count >= 2) & (hard_count == 0)

    # --- Mars triggers to slow planets
    mars_hard_any = np.zeros(len(idx), dtype=bool)
    for b in SLOW:
        on = np.zeros(len(idx), dtype=bool)
        for name in HARD:
            on |= aspect_orb(eph["mars_lon"], eph[f"{b}_lon"], ASPECTS[name]) <= MARS_ORB
        sig[f"mars_{b}_hard"] = on
        mars_hard_any |= on
    sig["mars_hard_slow_any"] = mars_hard_any

    # --- Jupiter & Saturn by element; Sun sign (note: confounded with seasonality)
    for p in ("jupiter", "saturn"):
        s = sign_of(eph[f"{p}_lon"])
        for el in ("fire", "earth", "air", "water"):
            sig[f"{p}_in_{el}"] = np.array([ELEMENTS[x] == el for x in s])
    sig["sun_sign"] = [SIGNS[x] for x in sign_of(eph["sun_lon"])]

    # --- Bradley
    sig = sig.join(bradley_siderograph(eph))
    sig["bradley_rising"] = sig["bradley"].diff() > 0
    sig["bradley_turn_window"] = _window(idx, bradley_turns(sig["bradley"]), 2, 2)

    from .factors_ext import extend
    sig = extend(sig, eph, start, end)
    sig = sig.copy()  # defragment

    sig.attrs["eclipses"] = ecl
    sig.attrs["lunations"] = lun
    return sig


def boolean_signals(sig: pd.DataFrame) -> list[str]:
    return [c for c in sig.columns if sig[c].dtype == bool]


VEDIC_LABELS = {
    "guru_chandal_yoga": "Guru-Chandal yoga (Jupiter with Rahu)",
    "kaal_sarp_yoga": "Kaal Sarp yoga (all grahas on one side of Rahu-Ketu)",
    "angarak_yoga": "Angarak yoga (Mars with Rahu)",
    "saturn_with_rahu": "Shani-Rahu (Shrapit) conjunction",
    "sade_sati_india": "Sade Sati of India chart",
    "sade_sati_usa": "Sade Sati of USA chart",
    "jupiter_saturn_samasaptaka": "Jupiter-Saturn samasaptaka (mutual 7th)",
    "rahu_ketu_ingress_window": "Rahu/Ketu sign change (+/-15d)",
    "jupiter_ingress_window": "Jupiter rashi change (+/-10d)",
    "tithi_rikta": "Rikta tithi (4th/9th/14th)",
    "tithi_purna": "Purna tithi (5th/10th/15th)",
}
_RASHI = ("mesha", "vrishabha", "mithuna", "karka", "simha", "kanya", "tula", "vrischika",
          "dhanu", "makara", "kumbha", "meena")
_OPP = dict(zip(_RASHI, _RASHI[6:] + _RASHI[:6]))


def is_vedic(name: str) -> bool:
    return (name in VEDIC_LABELS or name.startswith(("sid_", "moon_nakshatra", "rahu_in_"))
            or any(name.endswith("_in_" + r) for r in _RASHI) or "ketu" in name
            or name.startswith(("jupiter_transit_moon", "jupiter_exalted_sid", "jupiter_debilitated_sid",
                                "saturn_exalted_sid", "jupiter_aspects_rahu", "saturn_aspects_jupiter",
                                "saturn_mars_same_rashi")))


def describe_signal(name: str) -> str:
    if name in FAMOUS:
        return FAMOUS[name]
    if name in VEDIC_LABELS:
        return VEDIC_LABELS[name]
    if name.startswith("rahu_in_"):
        r = name[8:]
        return f"Rahu in {r.capitalize()} / Ketu in {_OPP[r].capitalize()} (sidereal)"
    for p in ("jupiter", "saturn"):
        if name.startswith(f"{p}_in_") and name[len(p) + 4:] in _RASHI:
            return f"{p.capitalize()} in {name[len(p) + 4:].capitalize()} (sidereal)"
    if name.startswith("moon_nakshatra_lord_"):
        return f"Moon in a {name[20:].capitalize()}-ruled nakshatra"
    if name.startswith("usa_") or name.startswith("nyse_"):
        chart, rest = name.split("_", 1)
        chart = "US Sibly chart" if chart == "usa" else "NYSE chart"
        return f"{chart}: " + rest.replace("_", " ").replace("natal", "natal").replace("hard", "hard aspect to")
    return name.replace("_", " ").replace(" rx", " retrograde").replace("hard", "hard aspect") \
        .replace("soft", "soft aspect").replace("conj", "conjunction").capitalize()
