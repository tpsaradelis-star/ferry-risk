import re
import math
import datetime as dt

import requests
import streamlit as st

# ------------------------------------------------------------
# 1. Core risk model
# ------------------------------------------------------------

def predict_ferry_run(wvht_ft, wspd_kt, gust_kt, dpd_s, tod_hour=None):
    """
    Rough risk model for Hy-Line high-speed HY <-> NA.
    Inputs:
      wvht_ft : significant wave height (feet)
      wspd_kt : sustained wind speed (knots)
      gust_kt : wind gust (knots)
      dpd_s   : dominant wave period (seconds)
      tod_hour: time of day in hours (0-24, local)
    Returns:
      (prob_run, prob_cancel, risk_band)
    """

    # Start optimistic: in benign conditions they almost always run
    score = 90.0  # interpreted as % chance to run before penalties

    # Wave height penalties
    if wvht_ft > 4:
        score -= (wvht_ft - 4) * 6.0
    if wvht_ft > 6:
        score -= (wvht_ft - 6) * 8.0

    # Wind speed penalties
    if wspd_kt > 22:
        score -= (wspd_kt - 22) * 1.5
    if gust_kt > 30:
        score -= (gust_kt - 30) * 1.0

    # Wave period penalties: short, steep seas are worse
    if dpd_s < 6:
        score -= (6 - dpd_s) * 5.0
    elif dpd_s < 7:
        score -= (7 - dpd_s) * 2.0

    # Time-of-day penalty: mornings / evenings often see rougher seas in gale patterns
    if tod_hour is not None:
        if 5 <= tod_hour <= 8 or 18 <= tod_hour <= 22:
            score -= 3.0

    # Clamp
    score = max(1.0, min(99.0, score))

    prob_run = score / 100.0
    prob_cancel = 1.0 - prob_run

    # Risk band for humans
    if prob_run >= 0.9:
        band = "LOW"
    elif prob_run >= 0.7:
        band = "MODERATE"
    elif prob_run >= 0.4:
        band = "HIGH"
    else:
        band = "VERY HIGH"

    return prob_run, prob_cancel, band


# ------------------------------------------------------------
# 2. Fetch & clean ANZ232 forecast text (KBOX)
# ------------------------------------------------------------

KBOX_MARINE_URL = "https://www.ndbc.noaa.gov/data/Forecasts/FZUS51.KBOX.html"

def _clean_html_to_text(raw):
    """Strip simple HTML tags and entities from the NDBC KBOX product."""
    text = re.sub(r"<[^>]+>", "\n", raw)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text


def fetch_anz232_block():
    """
    Fetch the KBOX marine forecast and return the Nantucket Sound (ANZ232) block as plain text.
    This uses 'Nantucket Sound' as an anchor and stops before the next '* Sound' block, if present.
    """
    resp = requests.get(KBOX_MARINE_URL, timeout=10)
    resp.raise_for_status()
    cleaned = _clean_html_to_text(resp.text)

    m = re.search(r"Nantucket Sound[\s\S]*", cleaned)
    if not m:
        raise RuntimeError("Could not find 'Nantucket Sound' in KBOX marine product.")
    sub = cleaned[m.start():]

    m2 = re.search(r"\n[A-Z][A-Za-z ]+Sound\n", sub[1:])
    if m2:
        block = sub[:m2.start()+1]
    else:
        block = sub

    return block.strip()


# ------------------------------------------------------------
# 3. Parse forecast periods within the Nantucket Sound block
# ------------------------------------------------------------

def parse_periods_from_block(block_text):
    """
    Parse the Nantucket Sound forecast block into periods like:
      ('THIS AFTERNOON', 'W winds 20 to 25 kt... Seas 4 to 6 ft...'),
      ('TONIGHT', 'W winds 30 to 35 kt...'), etc.
    """

    lines = [ln.rstrip() for ln in block_text.splitlines()]

    periods = []
    current_label = None
    current_body_lines = []

    def flush_current():
        nonlocal current_label, current_body_lines, periods
        if current_label and current_body_lines:
            body = " ".join(line.strip() for line in current_body_lines).strip()
            periods.append((current_label, body))
        current_label = None
        current_body_lines = []

    started = False
    for ln in lines:
        if not started:
            if "Nantucket Sound" in ln:
                started = True
            continue

        if not ln.strip():
            continue

        if re.search(r"\d{3,4}\s+AM|PM", ln):
            continue
        if "WARNING" in ln or "WATCH" in ln:
            continue

        if ln.strip().isupper() and len(ln.strip()) <= 20:
            flush_current()
            current_label = ln.strip()
            current_body_lines = []
        else:
            if current_label is not None:
                current_body_lines.append(ln)

    flush_current()

    if not periods:
        raise RuntimeError("No forecast periods parsed from Nantucket Sound block.")

    return periods


# ------------------------------------------------------------
# 4. Extract numbers from a period body
# ------------------------------------------------------------

def extract_wind_seas_period(body_text):
    """
    Given period text like:
      'W winds 25 to 35 kt. Seas 5 to 7 ft. Wave Detail: W 6 ft at 6 seconds...'
    Extract:
      seas_ft (midpoint if a range),
      wspd_kt (max of the range or single),
      gust_kt (if phrase 'gusts up to XX kt' exists, else wspd+5),
      dpd_s (from 'at XX seconds' if present, else 6.0).
    """

    text = body_text

    wspd = None
    m = re.search(r"(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*kt", text, re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        wspd = max(a, b)
    else:
        m = re.search(r"(\d{1,2})\s*kt", text, re.IGNORECASE)
        if m:
            wspd = int(m.group(1))

    gust = None
    mg = re.search(r"gusts?\s+(?:up to\s*)?(\d{1,2})\s*kt", text, re.IGNORECASE)
    if mg:
        gust = int(mg.group(1))

    seas = None
    ms = re.search(r"seas?\s+(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*ft", text, re.IGNORECASE)
    if ms:
        a, b = int(ms.group(1)), int(ms.group(2))
        seas = (a + b) / 2.0
    else:
        ms = re.search(r"seas?\s+(\d{1,2})\s*ft", text, re.IGNORECASE)
        if ms:
            seas = float(ms.group(1))

    dpd = None
    mp = re.search(r"at\s+(\d{1,2})\s*seconds", text, re.IGNORECASE)
    if mp:
        dpd = float(mp.group(1))

    if wspd is None:
        wspd = 20.0
    if gust is None:
        gust = wspd + 5.0
    if seas is None:
        seas = 4.0
    if dpd is None:
        dpd = 6.0

    return seas, wspd, gust, dpd


# ------------------------------------------------------------
# 5. Map date -> forecast period
# ------------------------------------------------------------

def choose_period_for_date(periods, departure_date):
    """
    Choose which forecast period to use for a given local date.
    Simple heuristic:
      - Use the period whose label starts with the departure weekday abbreviation,
        e.g. 'MON' for Monday, 'TUE' for Tuesday, etc.
      - If none match, fall back to 'TONIGHT' or 'THIS AFTERNOON' if present.
      - Otherwise, use the first period.
    """

    weekday_abbrev = departure_date.strftime("%a").upper()[:3]

    for label, body in periods:
        if label.startswith(weekday_abbrev):
            return label, body

    priority_labels = ["TONIGHT", "THIS AFTERNOON", "TODAY"]
    for pref in priority_labels:
        for label, body in periods:
            if label.startswith(pref):
                return label, body

    return periods[0]


# ------------------------------------------------------------
# 6. Public function: risk_for_date
# ------------------------------------------------------------

def risk_for_date(date_obj, time_obj):
    """
    date_obj: datetime.date
    time_obj: datetime.time (local)
    """
    departure_date = date_obj
    depart_hour = time_obj.hour
    depart_minute = time_obj.minute
    tod_hour = depart_hour + depart_minute / 60.0

    block = fetch_anz232_block()
    periods = parse_periods_from_block(block)
    chosen_label, chosen_body = choose_period_for_date(periods, departure_date)
    seas_ft, wspd_kt, gust_kt, dpd_s = extract_wind_seas_period(chosen_body)

    prob_run, prob_cancel, band = predict_ferry_run(
        wvht_ft=seas_ft,
        wspd_kt=wspd_kt,
        gust_kt=gust_kt,
        dpd_s=dpd_s,
        tod_hour=tod_hour,
    )

    return {
        "date": departure_date.isoformat(),
        "depart_local_time": time_obj.strftime("%H:%M"),
        "forecast_label_used": chosen_label,
        "forecast_body_used": chosen_body,
        "seas_ft": seas_ft,
        "wspd_kt": wspd_kt,
        "gust_kt": gust_kt,
        "dpd_s": dpd_s,
        "prob_run": prob_run,
        "prob_cancel": prob_cancel,
        "risk_band": band,
    }


# ------------------------------------------------------------
# 7. Streamlit UI
# ------------------------------------------------------------

def main():
    st.title("Hy-Line Fast Ferry Risk Checker – Nantucket Sound (ANZ232)")

    # Inputs
    today = dt.date.today()
    default_time = dt.time(hour=6, minute=10)

    col1, col2 = st.columns(2)
    with col1:
        date_input = st.date_input("Departure date", value=today)
    with col2:
        time_input = st.time_input("Departure time (local)", value=default_time)

    # Fetch forecast periods up front
    try:
        block = fetch_anz232_block()
        periods = parse_periods_from_block(block)
    except Exception as e:
        st.error(f"Error fetching or parsing marine forecast: {e}")
        return

    # Build a label -> body mapping for easy selection
    label_to_body = {label: body for (label, body) in periods}
    period_labels = list(label_to_body.keys())

    st.markdown("### Forecast period from NWS marine forecast (ANZ232)")
    selected_label = st.selectbox(
        "Choose the period that best matches your departure time:",
        period_labels,
    )

    if st.button("Run risk assessment"):
        body = label_to_body[selected_label]

        # Extract seas, winds, gusts, and period from the selected forecast text
        seas_ft, wspd_kt, gust_kt, dpd_s = extract_wind_seas_period(body)

        # Time-of-day in hours for the model
        tod_hour = time_input.hour + time_input.minute / 60.0

        prob_run, prob_cancel, band = predict_ferry_run(
            wvht_ft=seas_ft,
            wspd_kt=wspd_kt,
            gust_kt=gust_kt,
            dpd_s=dpd_s,
            tod_hour=tod_hour,
        )

        st.subheader("Ferry Risk Estimate")
        st.write(f"**Date (local):** {date_input.isoformat()}")
        st.write(f"**Departure (local):** {time_input.strftime('%H:%M')}")
        st.write(f"**Forecast period used:** {selected_label}")
        st.write(f"**Forecast text:** {body}")

        st.markdown("---")
        st.write(f"**Seas (ft):** {seas_ft:.2f}")
        st.write(f"**Sustained wind (kt):** {wspd_kt:.1f}")
        st.write(f"**Gust (kt):** {gust_kt:.1f}")
        st.write(f"**Dominant period (s):** {dpd_s:.1f}")

        st.markdown("### Risk")
        st.write(f"**RUN probability:** {prob_run:.2f}")
        st.write(f"**CANCEL probability:** {prob_cancel:.2f}")
        st.write(f"**Risk band:** {band}")

        if band in ["HIGH", "VERY HIGH"]:
            st.warning("Conditions are in the high-risk zone for cancellations.")
        elif band == "MODERATE":
            st.info("Moderate risk: conditions could go either way.")
        else:
            st.success("Low risk: conditions look favorable for running.")



if __name__ == "__main__":
    main()

