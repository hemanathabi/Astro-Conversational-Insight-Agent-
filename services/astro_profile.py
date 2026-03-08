from datetime import datetime, date

# Sun sign date boundaries (month, day) -> sign
ZODIAC_DATES = [
    ((1, 20), (2, 18), "Aquarius"),
    ((2, 19), (3, 20), "Pisces"),
    ((3, 21), (4, 19), "Aries"),
    ((4, 20), (5, 20), "Taurus"),
    ((5, 21), (6, 20), "Gemini"),
    ((6, 21), (7, 22), "Cancer"),
    ((7, 23), (8, 22), "Leo"),
    ((8, 23), (9, 22), "Virgo"),
    ((9, 23), (10, 22), "Libra"),
    ((10, 23), (11, 21), "Scorpio"),
    ((11, 22), (12, 21), "Sagittarius"),
    ((12, 22), (1, 19), "Capricorn"),
]

# Approximate moon sign mapping based on birth month
# (Simplified — real calculation requires ephemeris data)
MOON_SIGN_APPROX = {
    1: "Cancer",
    2: "Leo",
    3: "Virgo",
    4: "Libra",
    5: "Scorpio",
    6: "Sagittarius",
    7: "Capricorn",
    8: "Aquarius",
    9: "Pisces",
    10: "Aries",
    11: "Taurus",
    12: "Gemini",
}

# Nakshatra mapping per zodiac sign (simplified — 2-3 nakshatras per sign)
SIGN_TO_NAKSHATRAS = {
    "Aries": ["Ashwini", "Bharani", "Krittika"],
    "Taurus": ["Krittika", "Rohini", "Mrigashira"],
    "Gemini": ["Mrigashira", "Ardra", "Punarvasu"],
    "Cancer": ["Punarvasu", "Pushya", "Ashlesha"],
    "Leo": ["Magha", "Purva Phalguni", "Uttara Phalguni"],
    "Virgo": ["Uttara Phalguni", "Hasta", "Chitra"],
    "Libra": ["Chitra", "Swati", "Vishakha"],
    "Scorpio": ["Vishakha", "Anuradha", "Jyeshtha"],
    "Sagittarius": ["Moola", "Purva Ashadha", "Uttara Ashadha"],
    "Capricorn": ["Uttara Ashadha", "Shravana", "Dhanishtha"],
    "Aquarius": ["Dhanishtha", "Shatabhisha", "Purva Bhadrapada"],
    "Pisces": ["Purva Bhadrapada", "Uttara Bhadrapada", "Revati"],
}


def get_sun_sign(birth_date_str):
    """Determine sun sign from birth date string (YYYY-MM-DD)."""
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except ValueError:
        return "Unknown"

    month, day = birth_date.month, birth_date.day

    for start, end, sign in ZODIAC_DATES:
        if sign == "Capricorn":
            # Capricorn spans Dec-Jan
            if (month == 12 and day >= 22) or (month == 1 and day <= 19):
                return sign
        else:
            start_month, start_day = start
            end_month, end_day = end
            if month == start_month and day >= start_day:
                return sign
            if month == end_month and day <= end_day:
                return sign

    return "Unknown"


def get_moon_sign(birth_date_str, birth_time_str=None):
    """Approximate moon sign from birth date.

    Note: True Vedic moon sign requires ephemeris calculations (Swiss Ephemeris).
    This is a simplified approximation for demonstration purposes.
    """
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except ValueError:
        return "Unknown"

    # Use birth month as primary factor, adjust by day for variation
    base_sign_index = (birth_date.month - 1)
    day_offset = birth_date.day // 10  # 0, 1, or 2
    adjusted_index = (base_sign_index + day_offset) % 12

    signs = list(MOON_SIGN_APPROX.values())
    return signs[adjusted_index]


def get_nakshatra(moon_sign):
    """Get approximate nakshatra based on moon sign.

    Returns the primary nakshatra for the given moon sign.
    """
    nakshatras = SIGN_TO_NAKSHATRAS.get(moon_sign, [])
    return nakshatras[0] if nakshatras else "Unknown"


def get_age(birth_date_str):
    """Calculate age from birth date string."""
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age
    except ValueError:
        return None


def build_profile(user_profile):
    """Build enriched astro profile from user input.

    Args:
        user_profile: dict with name, birth_date, birth_time, birth_place, preferred_language

    Returns:
        dict with all original fields plus zodiac, moon_sign, nakshatra, age
    """
    birth_date = user_profile.get("birth_date", "")
    birth_time = user_profile.get("birth_time", "")

    sun_sign = get_sun_sign(birth_date)
    moon_sign = get_moon_sign(birth_date, birth_time)
    nakshatra = get_nakshatra(moon_sign)
    age = get_age(birth_date)

    enriched = dict(user_profile)
    enriched.update({
        "zodiac": sun_sign,
        "moon_sign": moon_sign,
        "nakshatra": nakshatra,
        "age": age,
    })
    return enriched
