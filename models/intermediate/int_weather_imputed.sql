{{ config(materialized='table') }}

-- Weather with the gaps filled, and a flag saying which rows were filled.
--
-- The missingness is NOT random: the station drops out far more often on hot and
-- cold extremes. That makes naive imputation actively harmful -- it pulls exactly
-- the observations carrying the temperature signal back toward the middle and
-- biases the weather coefficient toward zero.
--
-- Two things follow. Gaps are filled from the store's own climatology for that
-- calendar month rather than from a single global mean, which preserves at least
-- the seasonal part of the signal. And `temperature_is_imputed` is carried all
-- the way to the feature mart, so the size of the remaining bias can be measured
-- instead of assumed -- ml/run.py does exactly that.

with weather as (

    select * from {{ ref('stg_jaffle_shop__weather') }}

),

climatology as (

    select
        store_id,
        cast(extract(month from weather_date) as integer) as calendar_month,
        avg(temperature_c)                                as climatology_c
    from weather
    where temperature_c is not null
    group by 1, 2

)

select
    weather.store_id,
    weather.weather_date,
    coalesce(weather.temperature_c, climatology.climatology_c) as temperature_c,
    weather.temperature_c is null                              as temperature_is_imputed

from weather
inner join climatology
    on  climatology.store_id       = weather.store_id
    and climatology.calendar_month = cast(extract(month from weather.weather_date) as integer)
