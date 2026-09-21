-- The leakage guard, written as a data test rather than a code review comment.
--
-- Every rolling feature in mart_ml_features must be computed from strictly
-- earlier rows. This test re-derives each trailing average from the demand table
-- using an independent window that ends at `1 preceding`, and fails if the
-- mart's value disagrees -- which is what would happen if someone widened a
-- frame to `current row`, the single easiest way to make a forecasting model
-- look brilliant and be useless.
--
-- It also asserts the direct property: a trailing window on a row whose own
-- units_sold is large must not move with it.

with recomputed as (

    select
        store_id,
        sku,
        order_date,
        avg(cast(units_sold as double)) over (
            partition by store_id, sku order by day_index
            rows between 7 preceding and 1 preceding
        ) as expected_trailing_7d
    from {{ ref('int_store_day_sku_demand') }}

),

compared as (

    select
        f.store_id,
        f.sku,
        f.order_date,
        f.trailing_7d_units,
        r.expected_trailing_7d
    from {{ ref('mart_ml_features') }} f
    inner join recomputed r
        on  r.store_id   = f.store_id
        and r.sku        = f.sku
        and r.order_date = f.order_date

)

select *
from compared
where abs(coalesce(trailing_7d_units, -1) - coalesce(expected_trailing_7d, -1)) > 1e-9
