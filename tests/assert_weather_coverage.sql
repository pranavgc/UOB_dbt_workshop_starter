{{ config(severity='warn') }}

-- Weather readings go missing, and they go missing more often on extreme days.
-- This is a WARN, not an error: the gaps are a fact about the source, and
-- int_weather_imputed handles them. What would be dangerous is the rate drifting
-- upward unnoticed until imputation is carrying most of the signal, so the test
-- fires when any store-month is more than a quarter imputed.

select
    store_id,
    cast(date_trunc('month', order_date) as date)                       as order_month,
    avg(case when temperature_is_imputed then 1.0 else 0.0 end)         as imputed_share
from {{ ref('mart_ml_features') }}
group by 1, 2
having avg(case when temperature_is_imputed then 1.0 else 0.0 end) > 0.25
