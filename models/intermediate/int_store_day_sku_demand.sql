{{ config(materialized='table') }}

-- Grain: one row per store x sku x day the store was open. Zero-sales days are
-- present with units_sold = 0, which is what makes the ROWS-framed rolling
-- windows in mart_ml_features mean what they say.

with spine as (

    select * from {{ ref('int_demand_spine') }}

),

sold as (

    select
        store_id,
        sku,
        order_date,
        count(*)            as units_sold,
        sum(line_revenue)   as gross_revenue,
        sum(line_cost)      as supply_cost,
        sum(line_margin)    as gross_margin

    from {{ ref('int_order_lines_enriched') }}
    group by 1, 2, 3

),

featured as (

    select distinct store_id, sku, placement_date, campaign_id
    from {{ ref('stg_jaffle_shop__featured_placements') }}

),

weather as (

    select * from {{ ref('int_weather_imputed') }}

)

select
    spine.store_id,
    spine.sku,
    spine.order_date,
    spine.day_index,
    spine.product_type,
    spine.list_price,
    spine.base_price,
    spine.opened_at,
    weather.temperature_c,
    coalesce(weather.temperature_is_imputed, false) as temperature_is_imputed,

    coalesce(sold.units_sold, 0)        as units_sold,
    coalesce(sold.gross_revenue, 0)     as gross_revenue,
    coalesce(sold.supply_cost, 0)       as supply_cost,
    coalesce(sold.gross_margin, 0)      as gross_margin,

    case when featured.store_id is not null then true else false end as is_featured,
    featured.campaign_id

from spine
left join sold
    on  sold.store_id   = spine.store_id
    and sold.sku        = spine.sku
    and sold.order_date = spine.order_date
left join featured
    on  featured.store_id       = spine.store_id
    and featured.sku            = spine.sku
    and featured.placement_date = spine.order_date
left join weather
    on  weather.store_id     = spine.store_id
    and weather.weather_date = spine.order_date
