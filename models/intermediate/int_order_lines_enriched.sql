{{ config(materialized='view') }}

-- One row per unit sold, priced.
--
-- The price is joined from the daily price snapshot for that store, sku and
-- order date -- not from the product catalogue's list price. That distinction is
-- the fix for the original store_margin model, which valued every line at
-- catalogue price and therefore could never reconcile to the order totals the
-- source system recorded. tests/assert_order_revenue_reconciles.sql now asserts
-- that it does.

with items as (

    select * from {{ ref('stg_jaffle_shop__items') }}

),

orders as (

    -- The repaired orders, not the staging model: deduplicated, unit slip
    -- rescaled, refunds routed out. Joining items to the raw orders would fan
    -- out on the duplicate rows and inflate revenue.
    select * from {{ ref('int_orders_repaired') }}

),

products as (

    select * from {{ ref('stg_jaffle_shop__products') }}

),

prices as (

    select * from {{ ref('stg_jaffle_shop__price_snapshots') }}

),

unit_cost as (

    select * from {{ ref('int_product_unit_cost') }}

)

select
    items.order_item_id,
    items.order_id,
    items.sku,
    orders.store_id,
    orders.customer_id,
    orders.ordered_at,
    orders.order_date,
    products.product_name,
    products.product_type,
    prices.list_price                       as line_revenue,
    prices.list_price_cents                 as line_revenue_cents,
    orders.was_unit_slip_repaired,
    unit_cost.unit_cost                     as line_cost,
    prices.list_price - unit_cost.unit_cost as line_margin

from items
inner join orders
    on items.order_id = orders.order_id
inner join products
    on items.sku = products.sku
inner join prices
    on  prices.store_id  = orders.store_id
    and prices.sku       = items.sku
    and prices.price_date = orders.order_date
left join unit_cost
    on unit_cost.sku = items.sku
