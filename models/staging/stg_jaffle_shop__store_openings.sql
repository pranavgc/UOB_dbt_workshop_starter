with source as (

    select * from {{ source('jaffle_sim', 'raw_store_openings') }}

),

renamed as (

    select
        store_id,
        cast(opened_at as date) as opened_at

    from source

)

select * from renamed
