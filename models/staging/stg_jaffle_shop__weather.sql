with source as (

    select * from {{ source('jaffle_sim', 'raw_weather') }}

),

renamed as (

    select
        store_id,
        cast(weather_date as date)      as weather_date,
        cast(temperature_c as double)   as temperature_c

    from source

)

select * from renamed
