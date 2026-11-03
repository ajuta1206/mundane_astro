"""Extended mundane factors: national/exchange natal charts, dignities, combustion,
out-of-bounds, stations, node axis, supermoons, sidereal placements, Vedic
(Rahu/Ketu/Jupiter/Saturn rashi transits, yogas, Sade Sati, nakshatras, tithis),
and continuous cycle-phase features for machine learning.

Natal charts (the most widely used versions in mundane astrology):
* USA "Sibly" chart: 4 Jul 1776, 17:10 LMT, Philadelphia (39N57, 75W10)
  -> 22:10:40 UT. Source: Ebenezer Sibly (1787); see Campion, *Book of World Horoscopes*.
* NYSE: 17 May 1792, 08:52 LMT, New York (40N42, 74W00) -> 13:48 UT
  (Buttonwood Agreement; the rectified time used by most financial astrologers).
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import requests
import swisseph as swe

from .ephemeris import ELEMENTS, PLANETS, angle_diff, daily_ephemeris, julday, sign_of

log = logging.getLogger(__name__)

NATAL = {
    "usa": dict(y=1776, m=7, d=4, ut=22 + 10 / 60 + 40 / 3600, lat=39.95, lon=-75.1667),
    "nyse": dict(y=1792, m=5, d=17, ut=13.8, lat=40.7, lon=-74.0),
}
NATAL_POINTS = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "asc", "mc"]
TRANSITERS = ["jupiter", "saturn", "uranus", "neptune", "pluto"]
TRANSIT_ORB = 2.0  # deg; tight, since slow planets linger

# Traditional dignities (domicile + exaltation) and debilities (detriment + fall), sign index 0=Aries
DIGNITY = {"jupiter": {8, 11, 3}, "saturn": {9, 10, 6}, "mars": {0, 7, 9}, "venus": {1, 6, 11}}
DEBILITY = {"jupiter": {2, 5, 9}, "saturn": {3, 4, 0}, "mars": {6, 1, 3}, "venus": {7, 0, 5}}

SYNODIC_PAIRS = list(itertools.combinations(["jupiter", "saturn", "uranus", "neptune", "pluto"], 2)) + \
    [("sun", "moon"), ("mars", "jupiter"), ("mars", "saturn"), ("venus", "jupiter")]



def natal_chart(name: str) -> dict[str, float]:
    n = NATAL[name]
    jd = swe.julday(n["y"], n["m"], n["d"], n["ut"])
    out = {p: swe.calc_ut(jd, PLANETS[p], swe.FLG_MOSEPH)[0][0] for p in NATAL_POINTS[:7]}
    cusps, ascmc = swe.houses(jd, n["lat"], n["lon"], b"P")
    out["asc"], out["mc"] = ascmc[0], ascmc[1]
    return out


def _orb(a, b, ang):
    return np.abs(np.abs(angle_diff(a, b)) - ang)


def _window(index, dates, before, after):
    m = np.zeros(len(index), bool)
    for p in index.searchsorted(pd.DatetimeIndex(dates)):
        m[max(0, p - before):p + after + 1] = True
    return m


def _stations(spd: pd.Series) -> pd.DatetimeIndex:
    s = np.sign(spd.values)
    return spd.index[np.where(s[1:] != s[:-1])[0] + 1]


def _decl(index, pid):
    fl = swe.FLG_MOSEPH | swe.FLG_EQUATORIAL
    return np.array([swe.calc_ut(julday(t.date()), pid, fl)[0][1] for t in index])




RASHI = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
         "Tula", "Vrischika", "Dhanu", "Makara", "Kumbha", "Meena"]
# Nakshatra lords in Vimshottari order, repeating 3x over the 27 nakshatras
NAK_LORDS = ["ketu", "venus", "sun", "moon", "mars", "rahu", "jupiter", "saturn", "mercury"]
# Natal Moon rashi (sidereal Lahiri), computed rather than hard-coded:
#   India independence: 15 Aug 1947 00:00 IST (= 14 Aug 18:30 UT), Delhi
#   USA Sibly chart (UT as in NATAL["usa"])


def _sid_moon_rashi(y, m, d, ut) -> int:
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    lon = swe.calc_ut(swe.julday(y, m, d, ut), swe.MOON, swe.FLG_MOSEPH | swe.FLG_SIDEREAL)[0][0]
    return int(lon // 30)


INDIA_MOON_RASHI = _sid_moon_rashi(1947, 8, 14, 18.5)
USA_MOON_RASHI = _sid_moon_rashi(1776, 7, 4, NATAL["usa"]["ut"])


def vedic_factors(sid: pd.DataFrame, elong: np.ndarray) -> dict[str, np.ndarray]:
    """Classic Jyotish mundane factors, all from the sidereal (Lahiri) ephemeris."""
    out: dict[str, np.ndarray] = {}
    rahu = sid["node_lon"].values % 360
    ketu = (rahu + 180) % 360
    rs = sign_of(rahu)
    ks = sign_of(ketu)
    js = sign_of(sid["jupiter_lon"].values)
    ss = sign_of(sid["saturn_lon"].values)
    ms = sign_of(sid["mars_lon"].values)

    # Rahu-Ketu axis by sign (6 axes; Rahu in X <=> Ketu in X+6)
    for i in range(12):
        out[f"rahu_in_{RASHI[i].lower()}"] = rs == i
    # Jupiter (Guru) and Saturn (Shani) rashi transits
    for i in range(12):
        out[f"jupiter_in_{RASHI[i].lower()}"] = js == i
        out[f"saturn_in_{RASHI[i].lower()}"] = ss == i

    # Guru-Chandal yoga: Jupiter with Rahu (same rashi); Jupiter with Ketu
    out["guru_chandal_yoga"] = js == rs
    out["jupiter_with_ketu"] = js == ks
    # Shani-Rahu (Shrapit) and Shani-Ketu; Angarak yoga (Mars-Rahu); Mars-Ketu
    out["saturn_with_rahu"] = ss == rs
    out["saturn_with_ketu"] = ss == ks
    out["angarak_yoga"] = ms == rs
    out["mars_with_ketu"] = ms == ks
    out["saturn_mars_same_rashi"] = ss == ms
    # Jupiter aspects Rahu (Vedic Jupiter drishti: 5th, 7th, 9th from itself)
    rel = (rs - js) % 12
    out["jupiter_aspects_rahu"] = np.isin(rel, [4, 6, 8])
    # Saturn drishti (3rd, 7th, 10th) on Jupiter
    out["saturn_aspects_jupiter"] = np.isin((js - ss) % 12, [2, 6, 9])
    out["jupiter_saturn_samasaptaka"] = ((js - ss) % 12) == 6  # mutual 7th (opposition)

    # Rahu/Ketu sign-change windows (+/- 15 days of node ingress)
    ch = np.flatnonzero(rs[1:] != rs[:-1]) + 1
    w = np.zeros(len(rs), bool)
    for c in ch:
        w[max(0, c - 15):c + 16] = True
    out["rahu_ketu_ingress_window"] = w
    # Jupiter rashi-change windows
    ch = np.flatnonzero(js[1:] != js[:-1]) + 1
    w = np.zeros(len(js), bool)
    for c in ch:
        w[max(0, c - 10):c + 11] = True
    out["jupiter_ingress_window"] = w

    # Kaal Sarp Yoga: Sun..Saturn all on one side of the Rahu-Ketu axis
    rel_side = []
    for p in ("sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"):
        rel_side.append(((sid[f"{p}_lon"].values - rahu) % 360) < 180)
    rel_side = np.array(rel_side)
    out["kaal_sarp_yoga"] = rel_side.all(axis=0) | (~rel_side).all(axis=0)

    # Sade Sati (Saturn in 12th, 1st, 2nd from natal Moon) for India and USA charts
    for name, moon in (("india", INDIA_MOON_RASHI), ("usa", USA_MOON_RASHI)):
        out[f"sade_sati_{name}"] = np.isin((ss - moon) % 12, [11, 0, 1])
        out[f"jupiter_transit_moon_{name}"] = js == moon  # Guru over natal Moon (benefic)

    # Moon nakshatra lord (9 groups) -- the Vimshottari ruler of the day
    nak = (sid["moon_lon"].values % 360 // (360 / 27)).astype(int)
    lord = np.array(NAK_LORDS * 3)[nak]
    for l in NAK_LORDS:
        out[f"moon_nakshatra_lord_{l}"] = lord == l

    # Exaltation / debilitation (sidereal) of Jupiter and Saturn
    out["jupiter_exalted_sid"] = js == 3
    out["jupiter_debilitated_sid"] = js == 9
    out["saturn_exalted_sid"] = ss == 6
    # Amavasya/Purnima are covered by the lunar factors; Tithi groups (Nanda..Purna)
    tithi = (elong % 360 // 12).astype(int) % 15
    out["tithi_rikta"] = np.isin(tithi, [3, 8, 13])   # 4th, 9th, 14th tithi: traditionally inauspicious
    out["tithi_purna"] = np.isin(tithi, [4, 9, 14])   # 5th, 10th, 15th: "full"
    return out


def extend(sig: pd.DataFrame, eph: pd.DataFrame, start, end) -> pd.DataFrame:
    idx = sig.index
    new = {}

    # --- transits to national / exchange charts
    for chart in NATAL:
        nat = natal_chart(chart)
        hard_any = np.zeros(len(idx), bool)
        for t in TRANSITERS:
            on_t = np.zeros(len(idx), bool)
            for pt in NATAL_POINTS:
                hard = np.zeros(len(idx), bool)
                for ang in (0, 90, 180):
                    hard |= _orb(eph[f"{t}_lon"].values, nat[pt], ang) <= TRANSIT_ORB
                if t in ("saturn", "uranus", "pluto") and pt in ("sun", "moon", "asc", "mc"):
                    new[f"{chart}_{t}_hard_natal_{pt}"] = hard
                on_t |= hard
            new[f"{chart}_{t}_hard_any_natal"] = on_t
            hard_any |= on_t if t != "jupiter" else False
        new[f"{chart}_malefic_transit_any"] = hard_any
        jup = np.zeros(len(idx), bool)
        for pt in NATAL_POINTS:
            for ang in (0, 120):
                jup |= _orb(eph["jupiter_lon"].values, nat[pt], ang) <= TRANSIT_ORB
        new[f"{chart}_jupiter_benefic_natal"] = jup

    # --- dignity / debility of Jupiter, Saturn, Mars, Venus
    for p in DIGNITY:
        s = sign_of(eph[f"{p}_lon"])
        new[f"{p}_dignified"] = np.isin(s, list(DIGNITY[p]))
        new[f"{p}_debilitated"] = np.isin(s, list(DEBILITY[p]))

    # --- combustion (within 8.5 deg of the Sun) & cazimi
    for p in ("mercury", "venus", "mars"):
        o = np.abs(angle_diff(eph[f"{p}_lon"], eph["sun_lon"]))
        new[f"{p}_combust"] = o <= 8.5

    # --- out-of-bounds declination (beyond the Sun's max ~23.44 deg)
    for p in ("moon", "mercury", "venus", "mars"):
        d = _decl(idx, PLANETS[p])
        new[f"{p}_out_of_bounds"] = np.abs(d) > 23.44
        sig[f"x_{p}_decl"] = d

    # --- stations of slow planets (+/- 5 days) and Venus/Mars stations
    for p in ("venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"):
        new[f"{p}_station"] = _window(idx, _stations(eph[f"{p}_spd"]), 5, 5)
    new["any_outer_station"] = np.any([new[f"{p}_station"] for p in
                                       ("jupiter", "saturn", "uranus", "neptune", "pluto")], axis=0)

    # --- lunar nodes: axis element, Sun conjunct node (eclipse proxy is separate)
    ns = sign_of(eph["node_lon"])
    for el in ("fire", "earth", "air", "water"):
        new[f"node_in_{el}"] = np.array([ELEMENTS[x] == el for x in ns])
    for p in ("saturn", "jupiter", "mars"):
        new[f"{p}_conj_node_axis"] = (_orb(eph[f"{p}_lon"], eph["node_lon"], 0) <= 3) | \
                                     (_orb(eph[f"{p}_lon"], eph["node_lon"], 180) <= 3)

    # --- inner-planet conjunctions with benefic/malefic signatures
    new["venus_jupiter_conj"] = _orb(eph["venus_lon"], eph["jupiter_lon"], 0) <= 3
    new["mars_saturn_conj"] = _orb(eph["mars_lon"], eph["saturn_lon"], 0) <= 3
    new["sun_saturn_hard"] = np.any([_orb(eph["sun_lon"], eph["saturn_lon"], a) <= 3 for a in (0, 90, 180)], axis=0)
    new["sun_jupiter_conj"] = _orb(eph["sun_lon"], eph["jupiter_lon"], 0) <= 3

    # --- supermoon / micromoon (lunar distance)
    dist = np.array([swe.calc_ut(julday(t.date()), swe.MOON, swe.FLG_MOSEPH)[0][2] for t in idx])
    sig["x_moon_distance"] = dist
    elong = sig["lunar_phase_deg"].values
    near_syzygy = (np.abs(angle_diff(elong, 0)) < 20) | (np.abs(angle_diff(elong, 180)) < 20)
    new["supermoon_window"] = near_syzygy & (dist < np.quantile(dist, 0.15))

    # --- Mercury retrograde shadow (2 weeks before & after)
    rx = sig["mercury_rx"].values
    sh = np.zeros(len(idx), bool)
    for st in _stations(eph["mercury_spd"]):
        p = idx.searchsorted(st)
        sh[max(0, p - 14):p + 15] = True
    new["mercury_shadow"] = sh & ~rx

    # --- sidereal (Lahiri / Vedic) placements
    sid = daily_ephemeris(start, end, sidereal=True)
    for p in ("jupiter", "saturn", "node"):
        s = sign_of(sid[f"{p}_lon"])
        for el in ("fire", "earth", "air", "water"):
            new[f"sid_{p}_in_{el}"] = np.array([ELEMENTS[x] == el for x in s])
    # Vedic: Saturn in Sade-Sati-like transit over Moon sign of USA chart is too chart-specific;
    # instead: Jupiter (Guru) in own/exalted sidereal sign and Saturn debilitated (Aries).
    new["sid_jupiter_strong"] = np.isin(sign_of(sid["jupiter_lon"]), [8, 11, 3])
    new["sid_saturn_debilitated"] = sign_of(sid["saturn_lon"]) == 0

    # --- Vedic / Jyotish layer (sidereal Lahiri): Rahu, Ketu, Jupiter, Saturn transits & yogas
    new.update(vedic_factors(sid, sig["lunar_phase_deg"].values))
    for k in ("rahu", "ketu", "jupiter", "saturn"):
        lon = sid["node_lon"] if k == "rahu" else (sid["node_lon"] + 180) % 360 if k == "ketu" else sid[f"{k}_lon"]
        a = np.deg2rad(lon)
        sig[f"x_sid_{k}_sin"], sig[f"x_sid_{k}_cos"] = np.sin(a), np.cos(a)

    # --- continuous cycle phases (for ML): sin/cos of synodic angle
    for a, b in SYNODIC_PAIRS:
        ang = np.deg2rad((eph[f"{a}_lon"] - eph[f"{b}_lon"]) % 360)
        sig[f"x_{a}_{b}_sin"] = np.sin(ang)
        sig[f"x_{a}_{b}_cos"] = np.cos(ang)
    for p in ("jupiter", "saturn", "node"):
        ang = np.deg2rad(eph[f"{p}_lon"])
        sig[f"x_{p}_lon_sin"] = np.sin(ang)
        sig[f"x_{p}_lon_cos"] = np.cos(ang)
    sig["x_lunar_sin"] = np.sin(np.deg2rad(elong))
    sig["x_lunar_cos"] = np.cos(np.deg2rad(elong))

    for k, v in new.items():
        sig[k] = np.asarray(v, dtype=bool)
    return sig
