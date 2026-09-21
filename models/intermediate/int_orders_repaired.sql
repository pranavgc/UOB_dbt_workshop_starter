{{ config(materialized='table') }}

-- Sales orders, with the three known source defects repaired. Each repair is
-- narrow, documented, and guarded by a test -- a silent "clean-up" that nobody
-- can point at is how a pipeline ends up lying to people.
--
--   1. Duplicate order rows from a till double-submit. Line items are NOT
--      duplicated, so joining items to orders before deduplicating fans out the
--      join and inflates revenue by exactly the duplicate rate -- plausibly
--      enough that nothing looks wrong. Deduplicated here.
--      Guard: source `unique` on raw_orders.id (warn) + `unique` on this model (error).
--
--   2. One store-month with the cents conversion applied twice, leaving its
--      header values 100x too large. Invisible to a row-level range check,
--      because the values are still positive and numeric. Detected by comparing
--      each store-month's mean order value against the chain mean, and rescaled.
--
--      A mean is not a robust statistic and a subtler slip would hide from it.
--      That is acceptable here only because the repair does not rely on the
--      heuristic being right: assert_order_revenue_reconciles.sql checks every
--      order header against the sum of its own line items, which is an
--      independent source the slip never touched. The heuristic proposes; the
--      reconciliation test disposes.
--      Guard: assert_unit_slip_repaired.sql and assert_order_revenue_reconciles.sql.
--
--   3. Refunds carried as negative totals with no line items. Routed out to
--      stg_jaffle_shop__refunds rather than deleted.
--      Guard: accepted_range on order_subtotal below.

with orders as (

    select * from {{ ref('stg_jaffle_shop__orders') }}
    where order_subtotal_cents > 0          -- refunds leave here, see stg refunds

),

deduplicated as (

    select *
    from orders
    -- The duplicate rows are byte-identical, so any tie-break keeps the same
    -- record; order_id is named explicitly to keep the result deterministic.
    qualify row_number() over (partition by order_id order by order_id) = 1

),

-- A store-month whose mean order value is orders of magnitude above the chain's
-- is not a good month. It is a unit error.
scale_check as (

    select
        store_id,
        cast(date_trunc('month', order_date) as date)   as order_month,
        avg(order_subtotal_cents)                       as month_mean_cents
    from deduplicated
    group by 1, 2

),

chain_mean as (

    select avg(order_subtotal_cents) as chain_mean_cents from deduplicated

),

flagged as (

    select
        scale_check.store_id,
        scale_check.order_month,
        scale_check.month_mean_cents > (chain_mean.chain_mean_cents * 20.0) as is_unit_slip
    from scale_check
    cross join chain_mean

)

select
    deduplicated.order_id,
    deduplicated.customer_id,
    deduplicated.store_id,
    deduplicated.ordered_at,
    deduplicated.order_date,
    deduplicated.loaded_date,
    deduplicated.loaded_date - deduplicated.order_date as load_lag_days,

    case when flagged.is_unit_slip then deduplicated.order_subtotal_cents / 100
         else deduplicated.order_subtotal_cents end    as order_subtotal_cents,
    case when flagged.is_unit_slip then deduplicated.order_tax_paid_cents / 100
         else deduplicated.order_tax_paid_cents end    as order_tax_paid_cents,
    {{ cents_to_dollars(
        'case when flagged.is_unit_slip then deduplicated.order_subtotal_cents / 100
              else deduplicated.order_subtotal_cents end') }} as order_subtotal,
    flagged.is_unit_slip                               as was_unit_slip_repaired

from deduplicated
inner join flagged
    on  flagged.store_id    = deduplicated.store_id
    and flagged.order_month = cast(date_trunc('month', deduplicated.order_date) as date)
