"""
data/make_calendar.py
----------------------
Builds the UAE retail calendar and the event lift table used by the synthetic data generator,
the seasonality node and (later) Prophet's `holidays` frame.  ONE source of truth for seasonality.

Outputs (in data/synthetic/):
  uae_retail_calendar.csv   one row per event occurrence (start/end, lead-in, post-event dip)
  event_lifts.csv           demand multipliers per event x category (optionally x name keyword)

Dev-only dependencies (NOT needed at runtime):  pip install holidays hijridate

Date provenance is recorded per row in `date_basis`:
  computed_umm_al_qura   Islamic dates from the Umm al-Qura calendar. The UAE follows moon
                         sighting, so real dates can differ by +/- 1 day. Treat as approximate.
  computed_holidays_lib  Hindu festival dates from the `holidays` package (India).
  fixed                  fixed Gregorian date (National Day, Christmas, Valentine's, ...).
  rule                   derived by rule (White Friday = day after the 4th Thursday of November).
  estimate_verify        an organiser/school-calendar window that is not a fixed rule. Check it.

The LIFT MULTIPLIERS are modelling assumptions, not measurements. They are synthetic by design.
Not yet included because their dates could not be verified: Onam, Raksha Bandhan, Ganesh
Chaturthi, Easter. Add rows (and lifts) when verified.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import holidays
from hijridate import Hijri

OUT = Path(__file__).resolve().parent / "synthetic"
YEARS = range(2024, 2028)
ISLAMIC_YEARS = range(1445, 1450)

CAL_COLUMNS = ["event", "event_type", "community", "start_date", "end_date", "lead_in_days",
               "dip_days", "dip_factor", "hero_flag", "date_basis", "notes"]
rows: list[dict] = []


def add(event, etype, community, start, end, lead, dip, dip_factor, hero, basis, notes=""):
    rows.append(dict(event=event, event_type=etype, community=community, start_date=start.isoformat(),
                     end_date=end.isoformat(), lead_in_days=lead, dip_days=dip, dip_factor=dip_factor,
                     hero_flag=hero, date_basis=basis, notes=notes))


def d(y, m, day):
    return date(y, m, day)


# ── Islamic (moving) ──────────────────────────────────────────────────────────
for hy in ISLAMIC_YEARS:
    ramadan = Hijri(hy, 9, 1).to_gregorian()
    fitr = Hijri(hy, 10, 1).to_gregorian()
    adha = Hijri(hy, 12, 10).to_gregorian()
    if ramadan.year in YEARS:
        add("Ramadan", "religious", "Muslim", ramadan, fitr - timedelta(days=1), 14, 0, 1.0,
            "is_ramadan_hero", "computed_umm_al_qura", "start of Ramadan; shopping builds ~2 weeks before")
        add("Eid al-Fitr", "religious", "Muslim", fitr - timedelta(days=4), fitr + timedelta(days=2), 7, 14, 0.80,
            "", "computed_umm_al_qura", "last days of Ramadan to Eid+2; post-Eid slump")
    if adha.year in YEARS:
        add("Eid al-Adha", "religious", "Muslim", adha - timedelta(days=3), adha + timedelta(days=3), 7, 7, 0.90,
            "", "computed_umm_al_qura", "Arafat Day to Eid+3")

# ── Hindu (moving) ────────────────────────────────────────────────────────────
india = holidays.India(years=list(YEARS))
for dt, name in sorted(india.items()):
    if "Diwali" in name:
        add("Diwali", "religious", "Hindu / Indian", dt - timedelta(days=10), dt + timedelta(days=2), 14, 7, 0.90,
            "", "computed_holidays_lib", f"Diwali day {dt}; window covers Dhanteras gifting to Diwali+2")
    elif "Holi" in name:
        add("Holi", "religious", "Hindu / Indian", dt - timedelta(days=5), dt, 5, 3, 0.95,
            "", "computed_holidays_lib", f"Holi day {dt}")
    elif "Dussehra" in name:
        add("Navratri-Dussehra", "religious", "Hindu / Indian", dt - timedelta(days=10), dt, 3, 3, 0.95,
            "", "computed_holidays_lib", f"Dussehra day {dt}; Navratri is the 9 nights before")

# ── Fixed / rule-based / organiser windows ────────────────────────────────────
DSF = {2024: (d(2024, 12, 6), d(2025, 1, 12)), 2025: (d(2025, 12, 5), d(2026, 1, 11)),
       2026: (d(2026, 12, 4), d(2027, 1, 10))}
for y in YEARS:
    add("UAE National Day", "national", "All residents", d(y, 11, 27), d(y, 12, 3), 7, 5, 0.92, "", "fixed",
        "National Day is 2 Dec; Eid Al Etihad holiday 2-3 Dec")
    add("Christmas", "religious", "Christian / expat", d(y, 12, 15), d(y, 12, 25), 10, 0, 1.0, "", "fixed")
    add("New Year / Year-End Sale", "shopping", "All residents", d(y, 12, 26), d(y + 1, 1, 3), 0, 7, 0.90, "", "fixed",
        "follows Christmas; year-end sales")
    add("Valentine's Day", "gifting", "All residents", d(y, 2, 7), d(y, 2, 14), 7, 3, 0.95, "", "fixed")
    add("Mother's Day", "gifting", "All residents", d(y, 3, 14), d(y, 3, 21), 7, 3, 0.95, "", "fixed",
        "21 March in the Arab world")
    add("Back to School", "school", "Families", d(y, 8, 10), d(y, 9, 7), 14, 7, 0.90, "", "estimate_verify",
        "UAE school year starts late August; check the KHDA / MoE calendar")
    add("Summer Season", "season", "All residents", d(y, 6, 1), d(y, 9, 15), 14, 0, 1.0, "", "fixed",
        "heat season: indoor, travel, cooling, swimwear")
    add("Winter Season", "season", "All residents", d(y, 11, 15), d(y + 1, 3, 15), 14, 0, 1.0, "", "fixed",
        "cooler season: outdoor, camping, light layers")
    add("Singles Day 11.11", "shopping", "All residents", d(y, 11, 9), d(y, 11, 11), 5, 3, 0.95, "", "fixed")
    thanksgiving = [d(y, 11, k) for k in range(1, 31) if d(y, 11, k).weekday() == 3][3]
    fri = thanksgiving + timedelta(days=1)
    add("White Friday", "shopping", "All residents", fri - timedelta(days=4), fri + timedelta(days=3), 7, 7, 0.90,
        "", "rule", f"Friday {fri}: day after the 4th Thursday of November, through Cyber Monday")
    if y in DSF:
        s, e = DSF[y]
        add("Dubai Shopping Festival", "shopping", "All residents", s, e, 7, 5, 0.92, "is_dsf_hero",
            "estimate_verify", "official dates change yearly; 2026-27 is an estimate")

rows.sort(key=lambda r: (r["start_date"], r["event"]))
with open(OUT / "uae_retail_calendar.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=CAL_COLUMNS)
    w.writeheader()
    w.writerows(rows)

# ── Lift table: (event, category, name_regex, multiplier) ─────────────────────
# A SKU's multiplier for an event = the MAX over matching rows (a keyword row is absolute, it does
# not stack on the category row). Matching is case-insensitive against name_en + tags_en.
# Unlisted category = 1.0.
L = []


def lift(event, **cats):
    for cat, spec in cats.items():
        specs = spec if isinstance(spec, list) else [spec]
        for s in specs:
            if isinstance(s, tuple):
                L.append((event, cat, s[0], s[1]))
            else:
                L.append((event, cat, "", s))


lift("Ramadan", food=[2.2, ("date|saffron|nut|spice|rose|tamarind|jallab", 3.0)],
     home=[1.6, ("lantern|ramadan|prayer|incense|bakhoor", 3.5)],
     fashion=[1.2, ("abaya|kandura|thobe|modest|prayer", 1.9)],
     beauty=[1.3, ("oud|perfume|attar|bakhoor", 1.8)], electronics=0.95)
lift("Eid al-Fitr", fashion=[2.2, ("abaya|kandura|thobe|dress|modest", 2.8)],
     beauty=[2.0, ("oud|perfume|attar|gift", 2.5)], food=[1.4, ("date|sweet|chocolate|nut", 1.9)],
     electronics=1.2, home=1.2)
lift("Eid al-Adha", fashion=1.6, beauty=1.4, food=[1.5, ("spice|saffron|rice|ghee", 1.9)], home=1.1, electronics=1.1)
lift("UAE National Day", fashion=1.3, home=[1.4, ("flag|uae|decor|light", 2.2)], electronics=1.15, food=1.1, beauty=1.1)
lift("Dubai Shopping Festival", fashion=1.7, electronics=1.6, beauty=1.5, home=1.3, food=1.1)
lift("White Friday", electronics=2.8, fashion=2.0, beauty=1.8, home=1.7, food=1.1)
lift("Singles Day 11.11", electronics=1.6, fashion=1.3, beauty=1.3, home=1.1)
lift("Diwali", food=[1.9, ("date|nut|saffron|sweet|dry fruit|cardamom|ghee", 2.6)],
     home=[1.8, ("candle|lantern|lamp|light|decor|diya", 2.6)],
     fashion=[1.5, ("kurta|saree|lehenga|ethnic|jewel", 2.5)],
     beauty=[1.4, ("gift|perfume|set", 1.9)], electronics=1.4)
lift("Navratri-Dussehra", fashion=1.2, home=1.15, food=1.15, beauty=1.1)
lift("Holi", beauty=[1.25, ("hair|skin|moistur|oil", 1.5)], fashion=1.15, food=1.2)
lift("Christmas", home=[1.9, ("christmas|tree|ornament|candle|light|decor", 2.8)],
     food=[1.5, ("chocolate|nut|date|gift", 1.9)], fashion=1.5, beauty=[1.6, ("gift|set|perfume", 2.0)], electronics=1.6)
lift("New Year / Year-End Sale", fashion=1.6, electronics=1.5, beauty=1.4, home=1.3, food=1.15)
lift("Valentine's Day", beauty=[1.9, ("perfume|oud|gift|set", 2.4)], fashion=1.3,
     food=[1.4, ("chocolate|date|gift", 1.9)], home=[1.2, ("candle|rose|decor", 1.8)])
lift("Mother's Day", beauty=1.8, home=1.4, fashion=1.3, electronics=1.2, food=1.2)
lift("Back to School", electronics=1.7, fashion=[1.5, ("bag|backpack|shoe|sneaker|uniform", 1.9)], home=1.2)
lift("Summer Season", fashion=[0.95, ("swim|summer|sun|sandal|linen|short|kaftan|dress", 1.6)],
     beauty=[("sun|spf|after|body mist|deodorant", 1.5)], home=[("fan|cool|ice|picnic|beach", 1.5)],
     food=[("juice|water|ice|smoothie|salad", 1.4)], electronics=[("fan|cool|portable|speaker|power bank", 1.15)])
lift("Winter Season", fashion=[("jacket|coat|hoodie|sweater|scarf|boot|winter", 2.0)],
     home=[("camping|heater|blanket|throw|tent|grill", 1.6)], beauty=[("moistur|lip|butter|cream", 1.4)],
     electronics=[("camera|power bank|portable", 1.15)])

with open(OUT / "event_lifts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["event", "category", "name_regex", "multiplier"])
    w.writerows(L)

print(f"calendar rows: {len(rows)}  lift rows: {len(L)}  -> {OUT}")