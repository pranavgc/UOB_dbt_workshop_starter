-- Replaces the original store_margin model.
--
-- Three things changed. Revenue is the transacted price from the daily price
-- snapshot rather than the catalogue list price, so it reconciles to the source
-- order totals. The grain is store x sku x day rather than store, so there is
-- something to analyse over time. And zero-sales days are present rather than
-- dropped.

with demand as (

    select * from {{ ref('int_store_day_sku_demand') }}

),

stores as (

    select * from {{ ref('dim_store') }}

)

select
    demand.store_id,
    stores.store_name,
    demand.sku,
    demand.product_type,
    demand.order_date,

    demand.units_sold,
    demand.gross_revenue,
    demand.supply_cost,
    demand.gross_margin,
    round(
        100.0 * demand.gross_margin / nullif(demand.gross_revenue, 0)
    , 2)                                as margin_pct,

    demand.list_price,
    demand.is_featured,
    demand.temperature_c,
    demand.temperature_is_imputed

from demand
inner join stores
    on stores.store_id = demand.store_id
