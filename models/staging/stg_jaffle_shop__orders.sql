-- Cross-dialect note: the original workshop model used BigQuery's timestamp()
-- and date() functions. Both are replaced with standard CAST so the project
-- builds on DuckDB and BigQuery from the same source.
--
-- This model stays a pure rename/cast. It does NOT deduplicate, rescale the
-- known unit slip, or filter refunds -- those are repairs, and repairs belong in
-- the intermediate layer where they can be explained and tested. See
-- int_orders_repaired.

with source as (

    select * from {{ source('jaffle_sim', 'raw_orders') }}

),

renamed as (

    select
        id                                      as order_id,
        customer                                as customer_id,
        store_id,
        cast(ordered_at as timestamp)           as ordered_at,
        cast(ordered_at as date)                as order_date,
        cast(loaded_at as date)                 as loaded_date,
        {{ cents_to_dollars('subtotal') }}      as order_subtotal,
        {{ cents_to_dollars('tax_paid') }}      as order_tax_paid,
        {{ cents_to_dollars('order_total') }}   as order_total,
        subtotal                                as order_subtotal_cents,
        tax_paid                                as order_tax_paid_cents

    from source

)

select * from renamed
