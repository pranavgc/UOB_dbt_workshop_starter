with source as (

    select * from {{ source('jaffle_sim', 'raw_price_snapshots') }}

),

renamed as (

    select
        store_id,
        sku,
        cast(price_date as date)        as price_date,
        {{ cents_to_dollars('price') }} as list_price,
        price                           as list_price_cents

    from source

)

select * from renamed
