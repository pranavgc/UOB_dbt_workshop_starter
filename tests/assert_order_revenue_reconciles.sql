-- The test the original store_margin model could not have passed, and the one
-- that independently proves the unit-slip repair worked: line items are an
-- entirely separate source from the order header, and the slip never touched
-- them.
--
-- Every order's priced line items must sum to the subtotal the source system
-- recorded. If a price join ever goes wrong -- a missing snapshot, a fan-out, a
-- silently dropped line -- this fails instead of producing a mart that looks
-- plausible and is quietly wrong.
--
-- Tolerance is 1 cent per order to absorb rounding in the cents-to-dollars cast.

with per_order as (

    select
        order_id,
        sum(line_revenue_cents) as line_total_cents
    from {{ ref('int_order_lines_enriched') }}
    group by 1

),

orders as (

    -- The repaired orders: deduplicated, unit slip rescaled, refunds routed out.
    -- Comparing against the raw staging model would fail on all three by design.
    select
        order_id,
        order_subtotal_cents
    from {{ ref('int_orders_repaired') }}

)

select
    orders.order_id,
    orders.order_subtotal_cents,
    per_order.line_total_cents,
    abs(orders.order_subtotal_cents - per_order.line_total_cents) as discrepancy_cents

from orders
left join per_order
    on per_order.order_id = orders.order_id
where per_order.line_total_cents is null
   or abs(orders.order_subtotal_cents - per_order.line_total_cents) > 1
