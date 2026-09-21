with products as (

    select * from {{ ref('stg_jaffle_shop__products') }}

),

cost as (

    select * from {{ ref('int_product_unit_cost') }}

)

select
    products.sku,
    products.product_name,
    products.product_type,
    products.product_price                                  as catalogue_price,
    cost.unit_cost,
    cost.perishable_cost,
    cost.has_perishable_inputs,
    products.product_price - cost.unit_cost                 as catalogue_margin,
    round(
        100.0 * (products.product_price - cost.unit_cost)
        / nullif(products.product_price, 0)
    , 2)                                                    as catalogue_margin_pct

from products
left join cost
    on cost.sku = products.sku
