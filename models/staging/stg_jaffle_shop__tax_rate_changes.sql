-- Tax rate history. The seeded store catalogue carries a single current rate,
-- which is wrong for any date before the most recent change.
with source as (

    select * from {{ source('jaffle_sim', 'raw_tax_rate_changes') }}

),

renamed as (

    select
        store_id,
        cast(effective_date as date) as effective_date,
        cast(tax_rate as double)     as tax_rate

    from source

)

select
    *,
    lead(effective_date) over (
        partition by store_id order by effective_date
    ) as next_effective_date

from renamed
