with source as (

    select * from {{ source('jaffle_sim', 'raw_featured_placements') }}

),

renamed as (

    select
        store_id,
        sku,
        cast(placement_date as date)    as placement_date,
        campaign_id

    from source

)

select * from renamed
