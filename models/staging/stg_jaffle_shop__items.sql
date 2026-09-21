with source as (

    select * from {{ source('jaffle_sim', 'raw_items') }}

),

renamed as (

    select
        id          as order_item_id,
        order_id,
        sku

    from source

)

select * from renamed
