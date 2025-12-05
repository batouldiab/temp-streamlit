#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from pydeck.settings import settings as pdk_settings  # noqa

# ==========================
# Page & chrome
# ==========================
st.set_page_config(
    page_title="Job Posts & Skills Map",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] { display: none !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Config
# ==========================
UNKNOWN_SHEET_NAME = "Unknown_Country"

# Visual bubble defaults (meters) — used to scale the red cores
DEFAULT_COUNTRY_MIN_RADIUS = 400
DEFAULT_COUNTRY_MAX_RADIUS = 10000

# When Country = "All" (bigger bubbles)
DEFAULT_GLOBAL_MIN_RADIUS_ALL = 4000
DEFAULT_GLOBAL_MAX_RADIUS_ALL = 300000

# Skills aggregation fallback radius (km) if needed
DEFAULT_FALLBACK_RADIUS_KM = 10

# Static always-label list if desired
ALWAYS_LABEL_CITIES_STATIC: Set[Tuple[str, str]] = set()

# ==========================
# HARDCODED MAJOR CITIES LIST
# ==========================
# Rename mappings applied: Nineveh -> Mosul, South Darfur -> Nyala
MAJOR_CITIES_DATA = [
    {"country": "Algeria", "Center_City": "Algiers", "Radius_km_max": 62.9007654},
    {"country": "Algeria", "Center_City": "Oran", "Radius_km_max": 30.42232798},
    {"country": "Bahrain", "Center_City": "Muharraq", "Radius_km_max": 6.663181933},
    {"country": "Bahrain", "Center_City": "Manama", "Radius_km_max": 6.663181933},
    {"country": "Djibouti", "Center_City": "Djibouti City", "Radius_km_max": 42.11333798},
    {"country": "Djibouti", "Center_City": "Danan", "Radius_km_max": None},
    {"country": "Egypt", "Center_City": "Cairo", "Radius_km_max": 6.839630151},
    {"country": "Egypt", "Center_City": "Giza", "Radius_km_max": 0},
    {"country": "Egypt", "Center_City": "Ash Sharqiyah", "Radius_km_max": 56.24925234},
    {"country": "Egypt", "Center_City": "Alexandria", "Radius_km_max": 55.79403918},
    {"country": "Iraq", "Center_City": "Baghdad", "Radius_km_max": 28.54811737},
    {"country": "Iraq", "Center_City": "Erbil", "Radius_km_max": 27.35519252},
    {"country": "Iraq", "Center_City": "Basrah", "Radius_km_max": 72.90978213},
    {"country": "Iraq", "Center_City": "Mosul", "Radius_km_max": 24.25085482},  # Renamed from Nineveh
    {"country": "Jordan", "Center_City": "Amman", "Radius_km_max": 4.314975651},
    {"country": "Jordan", "Center_City": "Irbid", "Radius_km_max": 2.009650729},
    {"country": "Jordan", "Center_City": "Zarqa", "Radius_km_max": 6.088137817},
    {"country": "Kuwait", "Center_City": "Kuwait City", "Radius_km_max": 1.988171029},
    {"country": "Kuwait", "Center_City": "Al Ahmadi", "Radius_km_max": 4.550297455},
    {"country": "Lebanon", "Center_City": "Beirut", "Radius_km_max": 11.13333155},
    {"country": "Lebanon", "Center_City": "Jounieh", "Radius_km_max": 5.565229353},
    {"country": "Lebanon", "Center_City": "Tripoli", "Radius_km_max": 20.08922109},
    {"country": "Libya", "Center_City": "Tripoli", "Radius_km_max": 17.95324943},
    {"country": "Libya", "Center_City": "Benghazi", "Radius_km_max": 36.64613091},
    {"country": "Libya", "Center_City": "Misratah", "Radius_km_max": 13.96515014},
    {"country": "Mauritania", "Center_City": "Nouakchott", "Radius_km_max": 141.696767},
    {"country": "Mauritania", "Center_City": "Nouadhibou", "Radius_km_max": None},
    {"country": "Morocco", "Center_City": "Casablanca", "Radius_km_max": 61.14344935},
    {"country": "Morocco", "Center_City": "Rabat", "Radius_km_max": 11.62120549},
    {"country": "Oman", "Center_City": "Muscat", "Radius_km_max": 20.25350108},
    {"country": "Oman", "Center_City": "Sur", "Radius_km_max": 111.0932931},
    {"country": "Oman", "Center_City": "Salalah", "Radius_km_max": 151.4463597},
    {"country": "Palestine", "Center_City": "Ramallah", "Radius_km_max": 7.254707425},
    {"country": "Palestine", "Center_City": "Al Khalil", "Radius_km_max": 2.656222299},
    {"country": "Palestine", "Center_City": "Gaza", "Radius_km_max": 7.046621886},
    {"country": "Palestine", "Center_City": "Bethlehem", "Radius_km_max": 6.097981882},
    {"country": "Palestine", "Center_City": "Nablus", "Radius_km_max": 0},
    {"country": "Palestine", "Center_City": "Jerusalem", "Radius_km_max": 1.628588034},
    {"country": "Qatar", "Center_City": "Doha", "Radius_km_max": 0.162125118},
    {"country": "Saudi_Arabia", "Center_City": "Riyadh", "Radius_km_max": 101.795957},
    {"country": "Saudi_Arabia", "Center_City": "Ad Dammam", "Radius_km_max": 271.4993303},
    {"country": "Saudi_Arabia", "Center_City": "Jeddah", "Radius_km_max": 31.05731798},
    {"country": "Somalia", "Center_City": "Mogadishu", "Radius_km_max": 12.08129607},
    {"country": "Somalia", "Center_City": "Somaliland", "Radius_km_max": 0},
    {"country": "Somalia", "Center_City": "Puntland", "Radius_km_max": 92.66219621},
    {"country": "Somalia", "Center_City": "Hargeisa", "Radius_km_max": 68.690555},
    {"country": "Sudan", "Center_City": "Khartoum", "Radius_km_max": 71.60161658},
    {"country": "Sudan", "Center_City": "Jimiliaa", "Radius_km_max": 0},
    {"country": "Sudan", "Center_City": "Red Sea", "Radius_km_max": 196.4902192},
    {"country": "Sudan", "Center_City": "Al Qadarif State", "Radius_km_max": 3.977463549},
    {"country": "Sudan", "Center_City": "West Darfur", "Radius_km_max": 24.06510667},
    {"country": "Sudan", "Center_City": "Kassala", "Radius_km_max": 86.96361072},
    {"country": "Sudan", "Center_City": "South Kordofan", "Radius_km_max": 39.20008199},
    {"country": "Sudan", "Center_City": "North Darfur", "Radius_km_max": 0},
    {"country": "Sudan", "Center_City": "River Nile", "Radius_km_max": 60.5549495},
    {"country": "Sudan", "Center_City": "Qaysan", "Radius_km_max": 37.77116759},
    {"country": "Sudan", "Center_City": "White Nile", "Radius_km_max": 51.10850598},
    {"country": "Sudan", "Center_City": "Central Darfur State", "Radius_km_max": 62.8731533},
    {"country": "Sudan", "Center_City": "Kadugli", "Radius_km_max": 58.38896535},
    {"country": "Sudan", "Center_City": "Kosti", "Radius_km_max": 11.64821278},
    {"country": "Sudan", "Center_City": "Nyala", "Radius_km_max": 128.3357922},  # Renamed from South Darfur
    {"country": "Syria", "Center_City": "Damascus", "Radius_km_max": 32.45831863},
    {"country": "Syria", "Center_City": "Daraa", "Radius_km_max": 59.60760587},
    {"country": "Syria", "Center_City": "Aleppo", "Radius_km_max": 36.61524044},
    {"country": "Tunisia", "Center_City": "Tunis", "Radius_km_max": 47.73523264},
    {"country": "Tunisia", "Center_City": "Sousse", "Radius_km_max": 13.17458131},
    {"country": "Tunisia", "Center_City": "Sfax", "Radius_km_max": 38.98456853},
    {"country": "United_Arab_Emirates", "Center_City": "Dubai", "Radius_km_max": 0},
    {"country": "United_Arab_Emirates", "Center_City": "Abu Dhabi", "Radius_km_max": 9.950761851},
    {"country": "United_Arab_Emirates", "Center_City": "Sharjah", "Radius_km_max": 4.982468654},
    {"country": "Yemen", "Center_City": "Sanaa", "Radius_km_max": 45.8632008},
    {"country": "Yemen", "Center_City": "Adan", "Radius_km_max": 38.76712573},
]

# Build lookup structures from hardcoded data
MAJOR_CITIES_DF = pd.DataFrame(MAJOR_CITIES_DATA)
MAJOR_CITIES_DF["__City_norm__"] = MAJOR_CITIES_DF["Center_City"].apply(
    lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold() if x else ""
)
MAJOR_CITIES_DF["__Country_norm__"] = MAJOR_CITIES_DF["country"].apply(
    lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold() if x else ""
)

# Mapping from original names in Excel to renamed display names
CITY_RENAME_MAP = {
    "nineveh": "Mosul",
    "south darfur": "Nyala",
}

def _apply_city_rename(name: str) -> str:
    """Apply rename mapping (Nineveh -> Mosul, South Darfur -> Nyala)"""
    if not name:
        return name
    norm = re.sub(r"\s+", " ", str(name)).strip().casefold()
    return CITY_RENAME_MAP.get(norm, name)

# Set of valid (country_norm, city_norm) pairs from major_cities
VALID_MAJOR_CITIES = set(
    zip(MAJOR_CITIES_DF["__Country_norm__"], MAJOR_CITIES_DF["__City_norm__"])
)

# Lookup for hardcoded radius by (country_norm, city_norm)
HARDCODED_RADIUS_LOOKUP = {
    (row["__Country_norm__"], row["__City_norm__"]): row["Radius_km_max"]
    for _, row in MAJOR_CITIES_DF.iterrows()
}

# ==========================
# Helpers
# ==========================
EXPECTED_SKILLS_COLS = ["Skill", "SkillType", "City", "Latitude", "Longitude", "count"]
COL_ALIASES = {
    "skill": "Skill",
    "skill type": "SkillType",
    "skilltype": "SkillType",
    "type": "SkillType",
    "city": "City",
    "latitude": "Latitude",
    "lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "long": "Longitude",
    "count": "count",
    "n": "count",
    "freq": "count",
    "frequency": "count",
}

def color_for_sheet(name: str) -> List[int]:
    h = hashlib.md5(name.encode("utf-8")).digest()
    return [int(h[0]), int(h[1]), int(h[2]), 200]

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = re.sub(r"\s+", " ", str(c)).strip().lower()
        mapping[c] = COL_ALIASES.get(key, c)
    return df.rename(columns=mapping)

def _parse_int_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(s):
        return s
    if pd.api.types.is_float_dtype(s):
        return s.fillna(0).astype(int)
    s = s.astype(str).str.replace(r"[^\d]", "", regex=True)
    s = s.replace("", np.nan)
    return s.fillna(0).astype(int)

def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _strip_weird_whitespace(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def _clean_punctuation(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"^[\"':,\s]+", "", regex=True)
        .str.replace(r"[\"':,\s]+$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def _normcase(x: Optional[str]) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip().casefold()

def _scale_radius_by_global_sum(counts: pd.Series, global_sum: float, min_r: int, max_r: int) -> np.ndarray:
    denom = global_sum if global_sum > 0 else 1.0
    proportions = counts.astype(float).clip(lower=0) / denom
    return (min_r + proportions * (max_r - min_r)).to_numpy()

def _haversine_km_vec(lat_arr, lon_arr, lat0, lon0) -> np.ndarray:
    R = 6371.0088
    lat1 = np.radians(lat_arr.astype(float))
    lon1 = np.radians(lon_arr.astype(float))
    lat2 = np.radians(float(lat0))
    lon2 = np.radians(float(lon0))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def _best_excel_engine() -> str:
    try:
        import calamine  # noqa: F401
        return "calamine"
    except Exception:
        return "openpyxl"

def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0

# ==========================
# Paths
# ==========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SKILLS_XLSX_PATH = DATA_DIR / "cities_skills_with_desc_updated.xlsx"
AGGREGATED_XLSX_PATH = DATA_DIR / "all_OJAs_aggregated.xlsx"

# ==========================
# Aggregated data: read once, cache
# ==========================
@st.cache_data(show_spinner=False)
def load_aggregated_all_sheets(path: Path, mtime: float) -> Dict[str, pd.DataFrame]:
    """
    Load all_OJAs_aggregated.xlsx which has sheets per country with columns:
    City, Latitude, Longitude, Count
    """
    if not path.exists():
        return {}
    engine = _best_excel_engine()
    sheets = pd.read_excel(path, sheet_name=None, engine=engine)
    out: Dict[str, pd.DataFrame] = {}
    for country, df in (sheets or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # Normalize column names
        rename_map = {}
        for col in list(df.columns):
            cl = str(col).strip().lower()
            if cl == "city":
                rename_map[col] = "City"
            elif cl == "latitude":
                rename_map[col] = "Latitude"
            elif cl == "longitude":
                rename_map[col] = "Longitude"
            elif cl == "count":
                rename_map[col] = "Count"
        df = df.rename(columns=rename_map)

        needed = ["City", "Latitude", "Longitude", "Count"]
        if any(k not in df.columns for k in needed):
            continue

        # Clean city names and filter out empty ones
        df["City"] = df["City"].astype(str).str.strip()
        df = df[df["City"].notna() & (df["City"] != "") & (df["City"] != "nan")].copy()
        
        if df.empty:
            continue

        # Apply city rename (Nineveh -> Mosul, South Darfur -> Nyala)
        df["City"] = df["City"].apply(_apply_city_rename)

        cdf = pd.DataFrame(
            {
                "Sheet": country,
                "City": df["City"],
                "__City_norm__": df["City"].map(_normcase),
                "lat": pd.to_numeric(df["Latitude"], errors="coerce"),
                "lon": pd.to_numeric(df["Longitude"], errors="coerce"),
                "Count": pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int),
            }
        ).dropna(subset=["lat", "lon"])

        cdf = cdf[cdf["__City_norm__"].isin(set(cdf["__City_norm__"]) - {"", "nan"})].copy()
        out[country] = cdf

    return out

@st.cache_data(show_spinner=False)
def build_major_centers_with_aggregated_posts(
    aggregated_by_country: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Build center data for major cities only, using coordinates from aggregated data
    and radius from hardcoded MAJOR_CITIES_DATA.
    
    For each major city:
    - Own_posts: the posts count directly associated with the major city
    - Total_posts: sum of own posts + posts from all locations within Radius_km_max
    - Included_locations: count of locations included in the radius
    """
    out: Dict[str, pd.DataFrame] = {}
    
    for country, df in aggregated_by_country.items():
        if df is None or df.empty:
            continue
        
        country_norm = _normcase(country)
        
        # Get major cities for this country from hardcoded list
        major_cities_for_country = MAJOR_CITIES_DF[
            MAJOR_CITIES_DF["__Country_norm__"] == country_norm
        ].copy()
        
        if major_cities_for_country.empty:
            continue
        
        records = []
        
        for _, major_row in major_cities_for_country.iterrows():
            city_norm = major_row["__City_norm__"]
            city_name = major_row["Center_City"]
            radius_km = major_row["Radius_km_max"]
            
            # Handle None/NaN radius
            if radius_km is None or pd.isna(radius_km):
                radius_km = DEFAULT_FALLBACK_RADIUS_KM
            
            # Find the major city's own data in aggregated
            major_city_data = df[df["__City_norm__"] == city_norm]
            
            if major_city_data.empty:
                # Major city not found in aggregated data, skip
                continue
            
            # Get the major city's coordinates and own posts
            lat0 = float(major_city_data["lat"].iloc[0])
            lon0 = float(major_city_data["lon"].iloc[0])
            own_posts = int(major_city_data["Count"].iloc[0])
            
            # Calculate distances from this major city to ALL locations in the country
            distances = _haversine_km_vec(
                df["lat"].to_numpy(),
                df["lon"].to_numpy(),
                lat0,
                lon0
            )
            
            # Find all locations within radius (excluding the major city itself)
            within_radius_mask = (distances <= float(radius_km)) & (df["__City_norm__"] != city_norm)
            locations_within = df[within_radius_mask]
            
            # Sum posts from included locations
            included_posts = int(locations_within["Count"].sum())
            total_posts = own_posts + included_posts
            included_count = len(locations_within)
            
            records.append({
                "Sheet": country,
                "Center_City": city_name,
                "__City_norm__": city_norm,
                "lat": lat0,
                "lon": lon0,
                "Radius_km_max": radius_km,
                "Own_posts": own_posts,
                "Total_posts": total_posts,
                "Included_locations": included_count,
                "Members_count": included_count,  # For compatibility
            })
        
        if records:
            out[country] = pd.DataFrame.from_records(records)
    
    return out

@st.cache_data(show_spinner=False)
def build_blue_dots_from_aggregated(
    aggregated_by_country: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Build blue dots dataframe from all cities in aggregated data
    that are NOT in the major cities list.
    """
    all_records = []
    
    for country, df in aggregated_by_country.items():
        if df is None or df.empty:
            continue
        
        country_norm = _normcase(country)
        
        # Get major city norms for this country
        major_city_norms = {
            row["__City_norm__"] 
            for _, row in MAJOR_CITIES_DF.iterrows() 
            if row["__Country_norm__"] == country_norm
        }
        
        # Filter to non-major cities only (these become blue dots)
        non_major_df = df[~df["__City_norm__"].isin(major_city_norms)].copy()
        
        for _, row in non_major_df.iterrows():
            all_records.append({
                "Sheet": country,
                "Member_City": row["City"],
                "Total_posts": row["Count"],
                "lat": row["lat"],
                "lon": row["lon"],
            })
    
    if not all_records:
        return pd.DataFrame(columns=["Sheet", "Member_City", "Total_posts", "lat", "lon"])
    return pd.DataFrame.from_records(all_records)

@st.cache_data(show_spinner=False)
def compute_global_total_posts(centers_by_country: Dict[str, pd.DataFrame]) -> int:
    pieces = [
        df["Total_posts"]
        for df in centers_by_country.values()
        if isinstance(df, pd.DataFrame) and not df.empty
    ]
    return int(pd.concat(pieces, ignore_index=True).sum()) if pieces else 0

@st.cache_data(show_spinner=False)
def compute_total_posts_by_country(centers_by_country: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for country, df in centers_by_country.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            out[country] = int(df["Total_posts"].sum())
    return out

@st.cache_data(show_spinner=False)
def compute_global_total_from_aggregated(aggregated_by_country: Dict[str, pd.DataFrame]) -> int:
    """Compute total posts from all aggregated data (for scaling)"""
    pieces = [
        df["Count"]
        for df in aggregated_by_country.values()
        if isinstance(df, pd.DataFrame) and not df.empty
    ]
    return int(pd.concat(pieces, ignore_index=True).sum()) if pieces else 0

# ==========================
# Skills workbook: LAZY per-sheet
# ==========================
@st.cache_resource(show_spinner=False)
def _skills_excel_handle(path: Path, mtime: float):
    engine = _best_excel_engine()
    return pd.ExcelFile(path, engine=engine)

@st.cache_data(show_spinner=False)
def skills_sheet_names(path: Path, mtime: float) -> List[str]:
    xf = _skills_excel_handle(path, mtime)
    return list(xf.sheet_names or [])

@st.cache_data(show_spinner=False)
def load_country_skills(path: Path, mtime: float, country: str) -> pd.DataFrame:
    xf = _skills_excel_handle(path, mtime)
    if country not in xf.sheet_names:
        return pd.DataFrame(columns=EXPECTED_SKILLS_COLS + ["__City_norm__"])

    usecols = EXPECTED_SKILLS_COLS
    df = xf.parse(sheet_name=country, usecols=usecols)

    df = _canonicalize_columns(df)
    if any(c not in df.columns for c in EXPECTED_SKILLS_COLS):
        return pd.DataFrame(columns=EXPECTED_SKILLS_COLS + ["__City_norm__"])

    df["Skill"] = _clean_punctuation(_strip_weird_whitespace(df["Skill"]))
    df["SkillType"] = _clean_punctuation(
        _strip_weird_whitespace(df.get("SkillType", ""))
    ).fillna("")
    df["City"] = _clean_punctuation(_strip_weird_whitespace(df["City"])).fillna("")
    df["Latitude"] = _safe_to_numeric(df["Latitude"])
    df["Longitude"] = _safe_to_numeric(df["Longitude"])
    df["count"] = _parse_int_series(df["count"])

    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["__City_norm__"] = df["City"].apply(_normcase)
    df = df[~df["__City_norm__"].isin({"", "nan"})].copy()
    return df

@st.cache_data(show_spinner=False)
def build_city_geo_for_country(path: Path, mtime: float, country: str) -> pd.DataFrame:
    df = load_country_skills(path, mtime, country)
    if df.empty:
        return pd.DataFrame(columns=["Sheet", "City", "__City_norm__", "lat", "lon", "count"])

    w = df["count"].astype(float).clip(lower=0)
    df["_wlat"] = df["Latitude"] * w
    df["_wlon"] = df["Longitude"] * w

    agg = (
        df.groupby("City", as_index=False)
        .agg(count=("count", "sum"), _wlat=("_wlat", "sum"), _wlon=("_wlon", "sum"))
    )
    agg["lat"] = (agg["_wlat"] / agg["count"].replace(0, np.nan)).astype(float)
    agg["lon"] = (agg["_wlon"] / agg["count"].replace(0, np.nan)).astype(float)

    if agg["lat"].isna().any() or agg["lon"].isna().any():
        means = (
            df.groupby("City", as_index=False)[["Latitude", "Longitude"]]
            .mean()
            .rename(columns={"Latitude": "lat_mean", "Longitude": "lon_mean"})
        )
        agg = agg.merge(means, on="City", how="left")
        agg["lat"] = agg["lat"].fillna(agg["lat_mean"])
        agg["lon"] = agg["lon"].fillna(agg["lon_mean"])
        agg = agg.drop(columns=["lat_mean", "lon_mean"])

    agg["Sheet"] = country
    agg["__City_norm__"] = agg["City"].apply(_normcase)
    agg = agg.drop(columns=["_wlat", "_wlon"]).dropna(subset=["lat", "lon"])
    return agg[["Sheet", "City", "__City_norm__", "lat", "lon", "count"]].copy()

# ==========================
# Load aggregated data & set up globals
# ==========================
if not AGGREGATED_XLSX_PATH.exists():
    st.error(f"Missing aggregated Excel file: {AGGREGATED_XLSX_PATH}")
    st.stop()
if not SKILLS_XLSX_PATH.exists():
    st.error(f"Missing skills Excel file: {SKILLS_XLSX_PATH}")
    st.stop()

# Load all aggregated data (all cities from all countries)
aggregated_by_country = load_aggregated_all_sheets(
    AGGREGATED_XLSX_PATH, _file_mtime(AGGREGATED_XLSX_PATH)
)
FILTERED_AGGREGATED = {k: v for k, v in aggregated_by_country.items() if k != UNKNOWN_SHEET_NAME}
if not FILTERED_AGGREGATED:
    st.error(f"No valid sheets found in {AGGREGATED_XLSX_PATH.name}.")
    st.stop()

# Build major centers with aggregated posts (includes Own_posts and Total_posts)
MAJOR_CENTERS = build_major_centers_with_aggregated_posts(FILTERED_AGGREGATED)

# Build blue dots (all non-major cities)
ALL_MEMBERS_DF = build_blue_dots_from_aggregated(FILTERED_AGGREGATED)

# Compute totals
GLOBAL_TOTAL_POSTS = compute_global_total_from_aggregated(FILTERED_AGGREGATED)
TOTAL_POSTS_BY_COUNTRY = compute_total_posts_by_country(MAJOR_CENTERS)
SKILLS_MTIME = _file_mtime(SKILLS_XLSX_PATH)

def _total_posts_for_scope(country: str) -> int:
    """Global sum for 'All', or per-country sum otherwise."""
    if not country or country == "All":
        return GLOBAL_TOTAL_POSTS
    return TOTAL_POSTS_BY_COUNTRY.get(country, GLOBAL_TOTAL_POSTS)

# ==========================
# Center-city options & labels (only from major_cities)
# ==========================
@st.cache_data(show_spinner=False)
def get_center_cities_per_country_major(centers: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for ctry, df in centers.items():
        if df is None or df.empty:
            m[ctry] = []
            continue
        m[ctry] = df.sort_values("Total_posts", ascending=False)["Center_City"].tolist()
    return m

def get_top_center_norm_pairs(centers: Dict[str, pd.DataFrame]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for sheet, df in centers.items():
        if df is None or df.empty:
            continue
        row = df.sort_values("Total_posts", ascending=False).head(1)
        if row.empty:
            continue
        cn = _normcase(sheet)
        cty = _normcase(str(row["Center_City"].iloc[0]).strip())
        if cty not in {"", "nan"}:
            out.add((cn, cty))
    return out

CENTER_CITIES_BY_COUNTRY = get_center_cities_per_country_major(MAJOR_CENTERS)
ALWAYS_LABEL_CITIES = {
    (_normcase(cn), _normcase(cty)) for (cn, cty) in ALWAYS_LABEL_CITIES_STATIC
} | get_top_center_norm_pairs(MAJOR_CENTERS)

# ==========================
# Pending selection (must run BEFORE widgets)
# ==========================
def _ingest_pending_selection() -> None:
    pc = st.session_state.pop("__pending_country", None)
    pcity = st.session_state.pop("__pending_city", None)
    if pc:
        st.session_state["v3_country"] = pc
        st.session_state["__pending_city_after_country__"] = pcity

# ==========================
# UI
# ==========================
available_sheets = sorted(MAJOR_CENTERS.keys())
country_options = ["All"] + available_sheets

_ingest_pending_selection()

st.session_state.setdefault("v3_country", "All")
st.session_state.setdefault("v3_city", "All")

c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
with c1:
    sel_country = st.selectbox(
        "Country",
        options=country_options,
        index=country_options.index(st.session_state["v3_country"])
        if st.session_state["v3_country"] in country_options
        else 0,
        key="v3_country",
    )

if sel_country != "All":
    city_choices = ["All"] + CENTER_CITIES_BY_COUNTRY.get(sel_country, [])
else:
    city_choices = ["All"]

_pending_city = st.session_state.pop("__pending_city_after_country__", None)
if _pending_city:
    st.session_state["v3_city"] = _pending_city if _pending_city in city_choices else "All"

with c2:
    current_city = st.session_state.get("v3_city", "All")
    if current_city not in city_choices:
        current_city = "All"
        st.session_state["v3_city"] = "All"
    sel_city = st.selectbox(
        "City (Center)",
        options=city_choices,
        index=city_choices.index(current_city),
        key="v3_city",
    )

with c3:
    bubble_mode = st.selectbox(
        "Bubbles to show",
        options=["Coverage only (green)", "Posts only (red)", "Both"],
        index=2,
        key="bubble_mode",
    )
    st.markdown(
        "**Bubble sizing**: \n\n"
        "- **Red core**: scaled by Total_posts (aggregated) **within the selected country** (or globally when Country = 'All'), independent of the green.\n\n"
        "- **Green halo**: coverage area (Radius_km_max from hardcoded list).\n\n"
        "- **Blue dots**: ALL member locations from all clusters (always visible, hoverable).\n\n"
        "- **Tooltip**: Shows Own posts (city only) and Total posts (including nearby locations within radius)."
    )

# Visual range (for red scaling)
if sel_country == "All":
    rmin = DEFAULT_GLOBAL_MIN_RADIUS_ALL
    rmax = DEFAULT_GLOBAL_MAX_RADIUS_ALL
else:
    rmin = DEFAULT_COUNTRY_MIN_RADIUS
    rmax = DEFAULT_COUNTRY_MAX_RADIUS
if rmax <= rmin:
    rmax = rmin + 1

st.caption(f"Visual core radius range (red): {rmin}m - {rmax}m")

# ==========================
# Centers dataframe for map (only major cities)
# ==========================
def _centers_concat(
    centers: Dict[str, pd.DataFrame], restrict_country: Optional[str] = None
) -> pd.DataFrame:
    """
    Concatenate cluster centers, ensuring ONE row per Center_City.
    Only the geographic center cities are plotted (no member cities).
    """
    pieces = []
    for ctry, df in centers.items():
        if restrict_country and ctry != restrict_country:
            continue
        if df is None or df.empty:
            continue
        tmp = df[
            [
                "Sheet",
                "Center_City",
                "__City_norm__",
                "lat",
                "lon",
                "Radius_km_max",
                "Members_count",
                "Own_posts",
                "Total_posts",
                "Included_locations",
            ]
        ].copy()
        tmp = tmp.drop_duplicates(subset=["Center_City"])
        pieces.append(tmp)
    if pieces:
        return pd.concat(pieces, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "Sheet",
            "Center_City",
            "__City_norm__",
            "lat",
            "lon",
            "Radius_km_max",
            "Members_count",
            "Own_posts",
            "Total_posts",
            "Included_locations",
        ]
    )

if sel_country == "All":
    centers_df = _centers_concat(MAJOR_CENTERS)
else:
    centers_df = _centers_concat(MAJOR_CENTERS, restrict_country=sel_country)

if centers_df.empty:
    st.info("No center-city rows to plot.")
    st.stop()

centers_df = centers_df.copy()
centers_df["radius"] = _scale_radius_by_global_sum(
    centers_df["Total_posts"], float(GLOBAL_TOTAL_POSTS), rmin, rmax
)
centers_df["color"] = centers_df["Sheet"].apply(color_for_sheet)

# ==========================
# Dual-radius & color helpers
# ==========================
def _add_dual_radii(
    df: pd.DataFrame, rmin_visual: int, rmax_visual: int, total_posts_sum: int
) -> pd.DataFrame:
    """
    Add green (geo) and red (relative) radii in meters to the dataframe.

    - Green halo: proportional to Radius_km_max (coverage) - now using hardcoded values.
    - Red core: proportional to Total_posts in the chosen scope (country/global),
      independent from the green radius.
    """
    out = df.copy()
    out["green_radius_m"] = (
        pd.to_numeric(out.get("Radius_km_max", np.nan), errors="coerce")
        .fillna(DEFAULT_FALLBACK_RADIUS_KM)
        .clip(lower=0)
        * 1000.0
    )
    out["green_radius_m"] = out["green_radius_m"].clip(lower=float(rmin_visual))

    red_scaled = _scale_radius_by_global_sum(
        out["Total_posts"].fillna(0),
        float(total_posts_sum) if total_posts_sum > 0 else 1.0,
        rmin_visual,
        rmax_visual,
    )
    out["red_radius_m"] = np.maximum(red_scaled, 1.0)
    return out

def _apply_selection_colors(df: pd.DataFrame, selected_city_norm: Optional[str]) -> pd.DataFrame:
    """
    Add RGBA components for halos and cores, darkening the selected cluster
    and lightening all others (if a selection is provided).
    """
    out = df.copy()
    if selected_city_norm:
        sel_mask = out["__City_norm__"] == selected_city_norm

        # Light colors for non-selected clusters
        out["green_r"] = 160
        out["green_g"] = 220
        out["green_b"] = 160
        out["green_a"] = 70

        out["red_r"] = 245
        out["red_g"] = 160
        out["red_b"] = 160
        out["red_a"] = 110

        # Darker / stronger colors for the selected cluster
        out.loc[sel_mask, ["green_r", "green_g", "green_b", "green_a"]] = [0, 140, 0, 140]
        out.loc[sel_mask, ["red_r", "red_g", "red_b", "red_a"]] = [200, 0, 0, 220]
    else:
        # Default colors when nothing is selected
        out["green_r"] = 0
        out["green_g"] = 200
        out["green_b"] = 0
        out["green_a"] = 70

        out["red_r"] = 235
        out["red_g"] = 40
        out["red_b"] = 40
        out["red_a"] = 150
    return out

def _dual_layers(df: pd.DataFrame, tooltip_text: str, bubble_mode: str, pick_red: bool = True):
    """
    ScatterplotLayers (optionally filtered by bubble_mode):
      - green halo = coverage (Radius_km_max)
      - red core  = relative volume (scaled by Total_posts)

    Clicking logic:
      - If bubble_mode == "Coverage only (green)", green circles are clickable
        and use id="city-points" (so clicks select cities).
      - Otherwise, red cores are clickable with id="city-points".
    """
    if bubble_mode == "Coverage only (green)":
        green_id = "city-points"       # main clickable layer in this mode
        green_pickable = True
        green_auto_highlight = True
        red_id = "city-cores"
        red_pickable = False
        red_auto_highlight = False
    else:
        green_id = "city-halos"
        green_pickable = False
        green_auto_highlight = False
        red_id = "city-points" if pick_red else "city-cores"
        red_pickable = True
        red_auto_highlight = True

    green = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        id=green_id,
        get_position="[lon, lat]",
        get_radius="green_radius_m",
        get_fill_color="[green_r, green_g, green_b, green_a]",
        pickable=green_pickable,
        auto_highlight=green_auto_highlight,
        opacity=0.6,
        stroked=False,
    )
    red = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        id=red_id,
        get_position="[lon, lat]",
        get_radius="red_radius_m",
        get_fill_color="[red_r, red_g, red_b, red_a]",
        pickable=red_pickable,
        opacity=0.6,
        auto_highlight=red_auto_highlight,
        stroked=False,
    )

    layers: List[pdk.Layer] = []
    if bubble_mode in ("Coverage only (green)", "Both"):
        layers.append(green)
    if bubble_mode in ("Posts only (red)", "Both"):
        layers.append(red)

    return layers, {"text": tooltip_text}

# ==========================
# All members layer (blue dots - always visible)
# ==========================
def all_members_layer(members_df: pd.DataFrame) -> Optional[pdk.Layer]:
    """
    Build a small blue dot layer for ALL cluster members.
    Blue dots are hoverable but NOT clickable.
    """
    if members_df is None or members_df.empty:
        return None

    BLUE_RGBA = [0, 80, 255, 190]
    return pdk.Layer(
        "ScatterplotLayer",
        data=members_df,
        id="all-members",
        get_position="[lon, lat]",
        get_radius=1200,   # smaller blue dots
        get_fill_color=BLUE_RGBA,
        pickable=True,     # Hoverable for tooltip
        opacity=0.9,
        stroked=False,
    )

# ==========================
# Skills panel (lazy)
# ==========================
def render_top_skills_panel(
    country: str,
    city: str,
    include_cities_norm: Optional[Set[str]] = None,
    radius_km: Optional[float] = None,
) -> None:
    if not country or not city:
        return

    df_country = load_country_skills(SKILLS_XLSX_PATH, SKILLS_MTIME, country)
    if df_country is None or df_country.empty:
        st.info(f"No skills sheet found for **{country}**.")
        return

    if include_cities_norm and len(include_cities_norm) > 0:
        city_filter = include_cities_norm
        scope_label = (
            f"Aggregated within {int(radius_km)} km"
            if radius_km
            else "Aggregated (nearby cities)"
        )
        scope_note = f"({len(city_filter)} cities)"
    else:
        city_filter = {_normcase(city)}
        scope_label = "City only"
        scope_note = ""

    df_city_scope = df_country[df_country["__City_norm__"].isin(city_filter)].copy()
    if df_city_scope.empty:
        st.info(
            f"No skills found in scope **{scope_label} {scope_note}** for **{country}**."
        )
        return

    st.markdown(f"#### Skills — **{country}** • **{scope_label} {scope_note}**")

    def _render_top10(df_in: pd.DataFrame, title: str) -> None:
        top10 = (
            df_in.groupby("Skill", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(10)
        )
        st.markdown(f"**{title}**")
        st.dataframe(top10, use_container_width=True, hide_index=True)

    if "SkillType" not in df_city_scope.columns or df_city_scope["SkillType"].eq("").all():
        _render_top10(df_city_scope, "Top 10 skills (no SkillType)")
        return

    c1_, c2_ = st.columns(2)
    with c1_:
        _render_top10(
            df_city_scope[df_city_scope["SkillType"] == "ST1"], "Top 10 — Hard Skills"
        )
    with c2_:
        _render_top10(
            df_city_scope[df_city_scope["SkillType"] == "ST2"], "Top 10 — Soft Skills"
        )

# ==========================
# Click utilities
# ==========================
def _extract_clicked_center(event_obj: Any) -> Tuple[Optional[str], Optional[str]]:
    if event_obj is None:
        return None, None
    sel = getattr(event_obj, "selection", None)
    if isinstance(sel, dict):
        objs = sel.get("objects") or {}
        picked = objs.get("city-points") or []
        if picked:
            obj = picked[0]
            city = (
                str(obj.get("Center_City", "")).strip()
                if isinstance(obj, dict)
                else None
            )
            country = (
                str(obj.get("Sheet", "")).strip() if isinstance(obj, dict) else None
            )
            return (city or None), (country or None)
    return None, None

def resolve_center_radius_and_centroid(
    country: str, center_city: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Use hardcoded radius from major_cities list and coordinates from MAJOR_CENTERS"""
    country_norm = _normcase(country)
    city_norm = _normcase(center_city)
    
    # Get hardcoded radius
    radius = HARDCODED_RADIUS_LOOKUP.get((country_norm, city_norm))
    if radius is None or pd.isna(radius):
        radius = DEFAULT_FALLBACK_RADIUS_KM
    
    # Get lat/lon from major centers data
    df = MAJOR_CENTERS.get(country)
    if df is None or df.empty:
        return None, None, None
    r = df[df["__City_norm__"] == city_norm]
    if r.empty:
        return None, None, None
    return float(radius), float(r["lat"].iloc[0]), float(r["lon"].iloc[0])

def _set_dropdown_selection(country: str, city: str) -> None:
    st.session_state["__pending_country"] = country
    st.session_state["__pending_city"] = city

def _apply_click_selection(event: Any) -> bool:
    clicked_city, clicked_country = _extract_clicked_center(event)
    if clicked_city and clicked_country:
        _set_dropdown_selection(clicked_country, clicked_city)
        st.rerun()
        return True
    return False

# ==========================
# Map rendering
# ==========================
MAP_STYLE = "https://basemaps.cartocdn.com/gl/voyager-nolabels-gl-style/style.json"

# Prepare blue dots layer (always visible)
blue_layer = all_members_layer(ALL_MEMBERS_DF)

if sel_country != "All" and sel_city != "All":
    R_km, lat0, lon0 = resolve_center_radius_and_centroid(sel_country, sel_city)
    country_total_posts = _total_posts_for_scope(sel_country)

    if (lat0 is None) or (lon0 is None) or (R_km is None):
        st.warning(
            f'No radius/centroid found for "{sel_city}" — showing city-only skills.'
        )

        df_country_geo = build_city_geo_for_country(
            SKILLS_XLSX_PATH, SKILLS_MTIME, sel_country
        )
        if not df_country_geo.empty:
            row = df_country_geo[
                df_country_geo["__City_norm__"] == _normcase(sel_city)
            ]
            if not row.empty:
                lat0 = float(row["lat"].iloc[0])
                lon0 = float(row["lon"].iloc[0])

        one = centers_df[
            (centers_df["Sheet"] == sel_country)
            & (centers_df["__City_norm__"] == _normcase(sel_city))
        ].copy()
        if one.empty and (lat0 is not None) and (lon0 is not None):
            one = pd.DataFrame(
                [
                    {
                        "Sheet": sel_country,
                        "Center_City": sel_city,
                        "__City_norm__": _normcase(sel_city),
                        "lat": lat0,
                        "lon": lon0,
                        "Own_posts": 0,
                        "Total_posts": 0,
                        "Included_locations": 0,
                        "Radius_km_max": DEFAULT_FALLBACK_RADIUS_KM,
                    }
                ]
            )

        one = _add_dual_radii(one, rmin, rmax, country_total_posts)
        one = _apply_selection_colors(one, _normcase(sel_city))

        label_df = one.copy()
        label_df["__country_norm__"] = label_df["Sheet"].apply(_normcase)
        label_df["__city_norm__"] = label_df["Center_City"].apply(_normcase)
        label_df["__should_label__"] = label_df.apply(
            lambda row: (row["__country_norm__"], row["__city_norm__"])
            in ALWAYS_LABEL_CITIES,
            axis=1,
        )
        text_layer = pdk.Layer(
            "TextLayer",
            data=label_df[label_df["__should_label__"]].copy(),
            id="city-labels",
            get_position="[lon, lat]",
            get_text="Center_City",
            get_size=14,
            get_color=[0, 0, 0, 255],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
            background=False,
            get_background_color=[255, 255, 255, 200],
            background_padding=[4, 2, 4, 2],
            pickable=False,
        )

        vlat = float(one["lat"].iloc[0])
        vlon = float(one["lon"].iloc[0])
        layers, tooltip = _dual_layers(one, "{Center_City}\nOwn: {Own_posts} | Total: {Total_posts} posts", bubble_mode)

        # Add blue layer and member tooltip
        all_layers = ([blue_layer] if blue_layer is not None else []) + layers + [text_layer]
        
        # Combined tooltip for both center cities and members
        tooltip = {
            "html": "<b>{Center_City}</b><br/>Own Posts: {Own_posts}<br/>Total Posts: {Total_posts}<br/>Included Locations: {Included_locations}<br/>Radius: {Radius_km_max} km",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        event = st.pydeck_chart(
            pdk.Deck(
                layers=all_layers,
                initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlon, zoom=6),
                tooltip=tooltip,
                height=520,
                map_style=MAP_STYLE,
            ),
            use_container_width=True,
            key="v3_map_fallback",
            on_select="rerun",
            selection_mode="single-object",
        )

        _apply_click_selection(event)

        st.markdown("---")
        render_top_skills_panel(sel_country, sel_city)
        st.caption(
            "(Fallback) No centroid/radius in centers file — showing city-only skills."
        )

    else:
        # We have a valid radius & centroid: show ALL clusters in this country,
        # but darken the selected one and lighten the others.
        scope_df = centers_df.copy()
        scope_df = _add_dual_radii(scope_df, rmin, rmax, country_total_posts)
        scope_df = _apply_selection_colors(scope_df, _normcase(sel_city))

        # Zoom level based on the selected cluster's radius
        if R_km <= 10:
            zoom = 10
        elif R_km <= 25:
            zoom = 9
        elif R_km <= 50:
            zoom = 8
        elif R_km <= 100:
            zoom = 7
        elif R_km <= 200:
            zoom = 6
        elif R_km <= 350:
            zoom = 5
        else:
            zoom = 4

        label_df = scope_df.copy()
        label_df["__country_norm__"] = label_df["Sheet"].apply(_normcase)
        label_df["__city_norm__"] = label_df["Center_City"].apply(_normcase)
        label_df["__should_label__"] = label_df.apply(
            lambda row: (row["__country_norm__"], row["__city_norm__"])
            in ALWAYS_LABEL_CITIES,
            axis=1,
        )
        text_layer = pdk.Layer(
            "TextLayer",
            data=label_df[label_df["__should_label__"]].copy(),
            id="city-labels",
            get_position="[lon, lat]",
            get_text="Center_City",
            get_size=14,
            get_color=[0, 0, 0, 255],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
            background=False,
            get_background_color=[255, 255, 255, 200],
            background_padding=[4, 2, 4, 2],
            pickable=False,
        )

        layers, _ = _dual_layers(
            scope_df,
            "{Center_City}\nOwn: {Own_posts} | Total: {Total_posts} posts\nIncluded: {Included_locations} locations\nRadius: {Radius_km_max} km",
            bubble_mode,
        )

        # Add blue layer (always visible)
        all_layers = ([blue_layer] if blue_layer is not None else []) + layers + [text_layer]
        
        # Combined tooltip showing both own posts and total posts
        tooltip = {
            "html": "<b>{Center_City}</b><br/>Own Posts: {Own_posts}<br/>Total Posts: {Total_posts}<br/>Included Locations: {Included_locations}<br/>Radius: {Radius_km_max} km",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        event = st.pydeck_chart(
            pdk.Deck(
                layers=all_layers,
                initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom),
                tooltip=tooltip,
                height=520,
                map_style=MAP_STYLE,
            ),
            use_container_width=True,
            key="v3_map_radius",
            on_select="rerun",
            selection_mode="single-object",
        )

        _apply_click_selection(event)

        st.markdown("---")
        df_country_geo = build_city_geo_for_country(
            SKILLS_XLSX_PATH, SKILLS_MTIME, sel_country
        )
        include_norms: Set[str] = set()
        if not df_country_geo.empty:
            d = _haversine_km_vec(
                df_country_geo["lat"].to_numpy(),
                df_country_geo["lon"].to_numpy(),
                lat0,
                lon0,
            )
            df_near = df_country_geo.assign(__dist_km__=d)
            df_near = df_near[df_near["__dist_km__"] <= float(R_km)].copy()
            include_norms = {
                _normcase(c)
                for c in df_near["City"].dropna().astype(str).tolist()
                if _normcase(c) not in {"", "nan"}
            }
        if include_norms:
            render_top_skills_panel(
                sel_country,
                sel_city,
                include_cities_norm=include_norms,
                radius_km=float(R_km),
            )
        else:
            render_top_skills_panel(sel_country, sel_city)
            st.caption(
                "(Note) No nearby cities found in skills data — showing city-only skills."
            )

else:
    # Non-radius view: show ONLY center cities (All OR Country & City=All)
    map_df = centers_df.copy()
    vlat = float(map_df["lat"].mean())
    vlon = float(map_df["lon"].mean())
    zoom = 3 if sel_country == "All" else 4

    scope_total_posts = _total_posts_for_scope(sel_country)
    map_df = _add_dual_radii(map_df, rmin, rmax, scope_total_posts)
    map_df = _apply_selection_colors(map_df, None)

    label_df = map_df.copy()
    label_df["__country_norm__"] = label_df["Sheet"].apply(_normcase)
    label_df["__city_norm__"] = label_df["Center_City"].apply(_normcase)
    label_df["__should_label__"] = label_df.apply(
        lambda row: (row["__country_norm__"], row["__city_norm__"])
        in ALWAYS_LABEL_CITIES,
        axis=1,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=label_df[label_df["__should_label__"]].copy(),
        id="city-labels",
        get_position="[lon, lat]",
        get_text="Center_City",
        get_size=14,
        get_color=[0, 0, 0, 255],
        get_text_anchor="'middle'",
        get_alignment_baseline="'bottom'",
        background=False,
        get_background_color=[255, 255, 255, 200],
        background_padding=[4, 2, 4, 2],
        pickable=False,
    )

    layers, _ = _dual_layers(
        map_df,
        "{Center_City}\nOwn: {Own_posts} | Total: {Total_posts} posts",
        bubble_mode,
    )

    # Add blue layer (always visible)
    all_layers = ([blue_layer] if blue_layer is not None else []) + layers + [text_layer]
    
    # Combined tooltip showing both own posts and total posts
    tooltip = {
        "html": "<b>{Center_City}</b><br/>Own Posts: {Own_posts}<br/>Total Posts: {Total_posts}<br/>Included Locations: {Included_locations}<br/>Radius: {Radius_km_max} km",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    event = st.pydeck_chart(
        pdk.Deck(
            layers=all_layers,
            initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlon, zoom=zoom),
            tooltip=tooltip,
            height=520,
            map_style=MAP_STYLE,
        ),
        use_container_width=True,
        key="v3_map_all",
        on_select="rerun",
        selection_mode="single-object",
    )

    if _apply_click_selection(event):
        st.stop()