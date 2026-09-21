-- Reference catalogue. opened_at is deliberately NOT taken from here -- see
-- stg_jaffle_shop__store_openings and the note in _sources.yml.
with source as (

    select * from {{ ref('raw_stores') }}

),

renamed as (

    select
        id                          as store_id,
        name                        as store_name,
        cast(tax_rate as double)    as tax_rate

    from source

)

select * from renamed
