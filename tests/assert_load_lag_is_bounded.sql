{{ config(severity='warn') }}

-- Orders normally land in the warehouse the day after they happen; some arrive
-- later. A backtest that builds features "as of" the order date silently uses
-- rows that had not landed yet, which is information production would not have
-- had.
--
-- WARN rather than error: late arrival is a property of the source, not a bug to
-- fix here. The point is that the lag is visible and bounded, so a change in the
-- upstream loader shows up as a test result rather than as a mysteriously good
-- backtest.

select
    store_id,
    max(load_lag_days) as max_lag_days,
    count(*)           as orders
from {{ ref('int_orders_repaired') }}
group by 1
having max(load_lag_days) > 5
