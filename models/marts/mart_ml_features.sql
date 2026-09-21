{{ config(materialized='table') }}

-- The modelling surface. One row per store x sku x day; `units_sold` is the target.
--
-- POINT-IN-TIME CORRECTNESS
-- Every rolling feature below uses a window frame that ends at `1 preceding`.
-- A frame ending at `current row` would include the target in its own predictor
-- and produce an impossibly good score. tests/assert_no_feature_leakage.sql
-- asserts the property as a data test rather than trusting the SQL to stay right.
--
-- The rolling windows are ROWS-framed, which is only correct because
-- int_demand_spine guarantees a gapless daily series per partition.

with demand as (

    select * from {{ ref('int_store_day_sku_demand') }}

),

type_demand as (

    select * from {{ ref('int_store_day_type_demand') }}

),

stores as (

    select * from {{ ref('dim_store') }}

),

sku_windows as (

    select
        store_id,
        sku,
        order_date,

        avg(cast(units_sold as double)) over (
            partition by store_id, sku order by day_index
            rows between 7 preceding and 1 preceding
        )                                                       as trailing_7d_units,

        avg(cast(units_sold as double)) over (
            partition by store_id, sku order by day_index
            rows between 28 preceding and 1 preceding
        )                                                       as trailing_28d_units,

        max(case when is_featured then day_index end) over (
            partition by store_id, sku order by day_index
            rows between unbounded preceding and 1 preceding
        )                                                       as last_featured_day_index

    from demand

),

type_windows as (

    select
        store_id,
        product_type,
        order_date,

        avg(cast(type_units_sold as double)) over (
            partition by store_id, product_type order by day_index
            rows between 7 preceding and 1 preceding
        )                                                       as type_trailing_7d_units,

        avg(cast(type_units_sold as double)) over (
            partition by store_id, product_type order by day_index
            rows between 28 preceding and 1 preceding
        )                                                       as type_trailing_28d_units

    from type_demand

),

assembled as (

    select
        demand.store_id,
        stores.store_name,
        demand.sku,
        demand.product_type,
        demand.order_date,
        demand.day_index,

        -- target
        demand.units_sold,

        -- price
        demand.list_price,
        demand.base_price,
        ln(demand.list_price / nullif(demand.base_price, 0))    as log_price_rel,

        -- treatment
        demand.is_featured,

        -- weather, entered both linearly and as the deviation-squared term the
        -- U-shaped response actually needs
        demand.temperature_c,
        demand.temperature_is_imputed,
        (demand.temperature_c - 20.0) / 10.0                     as temp_dev,
        power((demand.temperature_c - 20.0) / 10.0, 2)           as temp_dev_sq,

        -- calendar
        {{ day_of_week_monday_zero('demand.order_date') }}       as day_of_week,
        cast(extract(dayofyear from demand.order_date) as integer) as day_of_year,

        -- store lifecycle
        least(
            cast(demand.day_index - stores_open.open_day_index as double) / 90.0,
            1.0
        )                                                        as ramp,

        -- lagged demand history: the confounder controls
        sku_windows.trailing_7d_units,
        sku_windows.trailing_28d_units,
        type_windows.type_trailing_7d_units,
        type_windows.type_trailing_28d_units,
        case
            when type_windows.type_trailing_28d_units > 0
            then type_windows.type_trailing_7d_units / type_windows.type_trailing_28d_units
        end                                                      as type_demand_ratio,
        demand.day_index - sku_windows.last_featured_day_index    as days_since_last_featured

    from demand
    inner join stores
        on stores.store_id = demand.store_id
    inner join (
        select store_id, min(day_index) as open_day_index
        from demand
        group by 1
    ) as stores_open
        on stores_open.store_id = demand.store_id
    left join sku_windows
        on  sku_windows.store_id   = demand.store_id
        and sku_windows.sku        = demand.sku
        and sku_windows.order_date = demand.order_date
    left join type_windows
        on  type_windows.store_id     = demand.store_id
        and type_windows.product_type = demand.product_type
        and type_windows.order_date   = demand.order_date

)

select
    assembled.*,
    sin(2 * cast(3.141592653589793 as double) * cast(day_of_year as double) / 365.25)    as fourier_sin1,
    cos(2 * cast(3.141592653589793 as double) * cast(day_of_year as double) / 365.25)    as fourier_cos1,
    sin(4 * cast(3.141592653589793 as double) * cast(day_of_year as double) / 365.25)    as fourier_sin2,
    cos(4 * cast(3.141592653589793 as double) * cast(day_of_year as double) / 365.25)    as fourier_cos2

from assembled
