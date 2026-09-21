-- After repair, no store-month should have a mean order value wildly out of line
-- with the chain. Before repair this fails on exactly one store-month; the fact
-- that it now passes is the evidence the rescale worked, and the fact that it
-- still exists is what stops a future change from quietly reintroducing the bug.

with monthly as (

    select
        store_id,
        cast(date_trunc('month', order_date) as date) as order_month,
        avg(order_subtotal_cents)                     as month_mean_cents
    from {{ ref('int_orders_repaired') }}
    group by 1, 2

),

chain as (

    select avg(order_subtotal_cents) as chain_mean_cents
    from {{ ref('int_orders_repaired') }}

)

select
    monthly.*,
    chain.chain_mean_cents,
    monthly.month_mean_cents / chain.chain_mean_cents as ratio
from monthly
cross join chain
where monthly.month_mean_cents < chain.chain_mean_cents / 4.0
   or monthly.month_mean_cents > chain.chain_mean_cents * 4.0
