-- Tax must be computed with the rate in force on the day of the order, not the
-- store's current rate. One store changes rate partway through the window, so a
-- pipeline that joins the static catalogue rate passes every key test and is
-- still wrong for every order after the change date.
--
-- Tolerance of 2 cents absorbs per-order rounding.

with orders as (

    select * from {{ ref('int_orders_repaired') }}

),

rates as (

    select * from {{ ref('stg_jaffle_shop__tax_rate_changes') }}

),

applicable as (

    select
        orders.order_id,
        orders.order_subtotal_cents,
        orders.order_tax_paid_cents,
        rates.tax_rate
    from orders
    inner join rates
        on  rates.store_id = orders.store_id
        and orders.order_date >= rates.effective_date
        and (rates.next_effective_date is null
             or orders.order_date < rates.next_effective_date)

)

select
    *,
    abs(order_tax_paid_cents - (order_subtotal_cents * tax_rate)) as discrepancy_cents
from applicable
where abs(order_tax_paid_cents - (order_subtotal_cents * tax_rate)) > 2
