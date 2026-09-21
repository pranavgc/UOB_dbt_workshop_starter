with source as (

    select * from {{ ref('raw_supplies') }}

),

renamed as (

    select
        id                              as supply_id,
        name                            as supply_name,
        {{ cents_to_dollars('cost') }}  as supply_cost,
        cast(perishable as boolean)     as is_perishable,
        sku

    from source

)

select * from renamed
