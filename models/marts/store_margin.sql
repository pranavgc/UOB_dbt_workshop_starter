with orders as (
    select * from {{ ref('stg_jaffle_shop__orders') }}
),

items as (
    select * from {{ ref('stg_jaffle_shop__items') }}
),

products as (
    select * from {{ ref('stg_jaffle_shop__products') }}
),

supplies as (
    select * from {{ ref('stg_jaffle_shop__supplies') }}
),

stores as (
    select * from {{ ref('stg_jaffle_shop__stores') }}
),

-- Calculate revenue per order item
order_item_revenue as (
    select 
        items.order_id,
        orders.store_id,
        items.sku as product_id,
        -- using currency macro conversion if available, otherwise raw price
        products.product_price as item_revenue
    from items
    left join orders on items.order_id = orders.order_id
    left join products on items.sku = products.sku
),

-- Calculate supply costs associated with products
product_supply_costs as (
    select 
        sku,
        sum(supply_cost) as total_supply_cost
    from supplies
    group by 1
),

-- Combine metrics at the store level
store_performance as (
    select
        rev.store_id,
        stores.store_name,
        count(distinct rev.order_id) as total_orders,
        sum(rev.item_revenue) as gross_revenue,
        coalesce(sum(cost.total_supply_cost), 0) as gross_supply_cost,
        (sum(rev.item_revenue) - coalesce(sum(cost.total_supply_cost), 0)) as net_profit
    from order_item_revenue rev
    left join stores on rev.store_id = stores.store_id
    left join product_supply_costs cost on rev.product_id = cost.sku
    group by 1, 2
)

select 
    *,
    case 
        when gross_revenue = 0 then 0
        else round((net_profit / gross_revenue) * 100, 2)
    end as profit_margin_percentage
from store_performance