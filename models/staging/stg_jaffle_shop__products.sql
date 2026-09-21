-- Reference catalogue, loaded as a dbt seed rather than generated.
with source as (

    select * from {{ ref('raw_products') }}

),

renamed as (

    select
        sku,
        name                                as product_name,
        type                                as product_type,
        {{ cents_to_dollars('price') }}     as product_price,
        price                               as product_price_cents,
        description                         as product_description

    from source

)

select * from renamed
