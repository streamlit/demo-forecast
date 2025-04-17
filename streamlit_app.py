from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import streamlit as st
import pandas as pd
import altair as alt
from prophet import Prophet

import pandas as pd
from prophet import Prophet
import streamlit as st

st.set_page_config(
    page_title="Sales units forecast",
    page_icon=":money_mouth_face:",
)

# Calculate reference dates
TODAY = pd.to_datetime(datetime.now().date()) - timedelta(days=1)
TOMORROW = TODAY + timedelta(days=1)
ONE_MONTH_AGO = TODAY - relativedelta(months=1)
THREE_MONTHS_AGO = TODAY - relativedelta(months=3)
SIX_MONTHS_AGO = TODAY - relativedelta(months=6)
ONE_YEAR_AGO = TODAY - relativedelta(years=1)
TWO_YEARS_AGO = TODAY - relativedelta(years=2)

DATE_COL = "Date"
VALUE_COL = "Units sold"
FORECAST_COL = "Forecast"
UPPER_COL = "Lower bound"
LOWER_COL = "Upper bound"


def get_quarter_start(date):
    year = date.year
    if date.month in [1, 2, 3]:
        return pd.to_datetime(datetime(year, 2, 1))
    elif date.month in [4, 5, 6]:
        return pd.to_datetime(datetime(year, 5, 1))
    elif date.month in [7, 8, 9]:
        return pd.to_datetime(datetime(year, 8, 1))
    else:
        return pd.to_datetime(datetime(year, 11, 1))


CURRENT_QUARTER_START = get_quarter_start(TODAY)
LAST_QUARTER_START = CURRENT_QUARTER_START - relativedelta(months=3)

START_DATES = {
    "Tomorrow": TOMORROW,
    "1 month ago": ONE_MONTH_AGO,
    "3 months ago": THREE_MONTHS_AGO,
    "Beginning of this quarter": CURRENT_QUARTER_START,
    "6 months ago": SIX_MONTHS_AGO,
    "Beginning of last quarter": LAST_QUARTER_START,
    "1 year ago": ONE_YEAR_AGO,
    "2 years ago": TWO_YEARS_AGO,
}

# Define forecast periods mapping
FORECAST_LENGTHS = {
    "1 month": 30,
    "3 months": 30 * 3,
    "6 months": 30 * 6,
    "1 year": 365,
    "2 years": 730,
}


################################################################
# Functions


@st.cache_data
def get_data():
    df = pd.read_csv("data.csv")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


@st.cache_data
def run_forecast_model(
    df,
    forecast_start,
    forecast_days,
    **prophet_kwargs,
):
    # Filter data up to forecast start
    prophet_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(df[df[DATE_COL] <= forecast_start][DATE_COL]),
            "y": df[df[DATE_COL] <= forecast_start][VALUE_COL].astype(float),
        }
    )

    model = Prophet(**prophet_kwargs)

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Prepare visualization data
    fore_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={
            "ds": DATE_COL,
            "yhat": FORECAST_COL,
            "yhat_lower": LOWER_COL,
            "yhat_upper": UPPER_COL,
        }
    )

    # Only keep the actual forecast, not the model of the past.
    # fore_df = fore_df[fore_df["TS"] >= forecast_start]

    return df, fore_df


################################################################
# Application logic

"""
# :money_mouth_face: Sales units forecast

Streamlit example with fake data demonstrating how to build an awesome forecast app.
"""

""

df = get_data()

cols = st.columns(2)
selected_timeframe = cols[0].selectbox(
    "Forecast start date",
    options=list(START_DATES.keys()),
)
forecast_start = START_DATES[selected_timeframe]

forecast_period = cols[1].selectbox(
    "Forecast length",
    options=list(FORECAST_LENGTHS.keys()),
)
forecast_days = FORECAST_LENGTHS[forecast_period]

""
""

hist_df, fore_df = run_forecast_model(
    df,
    forecast_start,
    forecast_days,
    #
    # Pass Prophet parameters here.
    #
    # If you change the data and do not do some tweaking, the forecast will
    # likely be completely off!
    #
    # Documentation:
    # https://facebook.github.io/prophet/docs/diagnostics.html
    #
    # Example parameters:
    changepoint_prior_scale=0.05,  # Default=0.05
    seasonality_prior_scale=10.0,  # Default=10.0
    holidays_prior_scale=10.0,  # Default=10.0
    # Changed below to better match end of historical trend,
    # using a value proposed by the docs as reasonable when
    # there's a large historical series.
    changepoint_range=0.95,  # Default=0.8
    # Changed below to better match ground truth.
    seasonality_mode="multiplicative",  # Default="additive"
)


# Create visualization
base = (
    alt.Chart(hist_df)
    .mark_line(color="blue", tooltip=True)
    .encode(
        alt.X(DATE_COL, type="temporal", title=None),
        alt.Y(VALUE_COL, type="quantitative", title=None),
    )
    .interactive()
)

forecast_line = (
    alt.Chart(fore_df)
    .mark_line(color="red", strokeWidth=1, tooltip=True)
    .encode(
        alt.X(DATE_COL, type="temporal", title=None),
        alt.Y(FORECAST_COL, type="quantitative", title="Forecast"),
    )
)

confidence_area = (
    alt.Chart(fore_df)
    .mark_area(opacity=0.2, color="red", tooltip=True)
    .encode(
        alt.X(DATE_COL, type="temporal", title=None),
        alt.Y(LOWER_COL, type="quantitative", title="Forecast lower bound"),
        alt.Y2(UPPER_COL, title="Forecast upper bound"),
    )
)

chart = base + confidence_area + forecast_line
st.altair_chart(chart, use_container_width=True)

with st.expander("Raw data"):
    df
