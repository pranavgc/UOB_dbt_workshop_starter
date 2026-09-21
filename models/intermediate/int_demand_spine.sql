{{ config(materialized='table') }}

-- The dense store x sku x day universe.
--
-- Built from the daily price snapshots, which exist for every store, sku and day
-- regardless of whether anything sold, then clipped to days on or after the
-- store opened. Aggregating sales alone would silently drop zero-sales days and
-- leave gaps in the series; every rolling window downstream is defined by ROWS,
-- so a gap would quietly change what "trailing 7 days" means.

with prices as (

    select * from {{ ref('stg_jaffle_shop__price_snapshots') }}

),

openings as (

    select * from {{ ref('stg_jaffle_shop__store_openings') }}

),

products as (

    select sku, product_type, product_price from {{ ref('stg_jaffle_shop__products') }}

),

spine as (

    select
        prices.store_id,
        prices.sku,
        prices.price_date as order_date,
        prices.list_price,
        prices.list_price_cents,
        products.product_type,
        products.product_price as base_price,
        openings.opened_at

    from prices
    inner join openings
        on openings.store_id = prices.store_id
    inner join products
        on products.sku = prices.sku
    where prices.price_date >= openings.opened_at

)

select
    spine.*,
    dense_rank() over (order by order_date) as day_index

from spine
