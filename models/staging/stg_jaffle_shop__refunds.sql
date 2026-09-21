-- Refunds arrive in the same source table as sales, with negative values and no
-- line items. They are routed here rather than dropped: the money is real and
-- someone will eventually ask about it, but leaving them in the demand series
-- would net real units away against returns that happened days later.

with orders as (

    select * from {{ ref('stg_jaffle_shop__orders') }}

)

select
    order_id            as refund_id,
    customer_id,
    store_id,
    ordered_at          as refunded_at,
    order_date          as refund_date,
    -order_subtotal     as refund_subtotal,
    -order_total        as refund_total

from orders
where order_subtotal_cents < 0
