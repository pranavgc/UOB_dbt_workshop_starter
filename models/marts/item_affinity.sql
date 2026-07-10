with items as (
    select * from {{ ref('stg_jaffle_shop__items') }}
),
products as (
    select * from {{ ref('stg_jaffle_shop__products') }}
),

-- Self join the items table to find products in the same order
item_pairs as (
    select 
        i1.order_id,
        i1.sku as product_a_id,
        i2.sku as product_b_id
    from items i1
    -- The > condition prevents duplicating pairs (e.g., A-B and B-A) and prevents pairing a product with itself
    join items i2 on i1.order_id = i2.order_id and i1.sku > i2.sku 
),

pair_counts as (
    select 
        product_a_id,
        product_b_id,
        count(order_id) as times_bought_together
    from item_pairs
    group by 1, 2
)

select 
    p1.product_name as product_a_name,
    p2.product_name as product_b_name,
    pair_counts.times_bought_together
from pair_counts
join products p1 on pair_counts.product_a_id = p1.sku
join products p2 on pair_counts.product_b_id = p2.sku
order by times_bought_together desc