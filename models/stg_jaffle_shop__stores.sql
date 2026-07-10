
with source as (

    select * from {{ source('jaffle_shop', 'raw_stores') }}

),

renamed as (

    select
    id as store_id,
    name as store_name,
    timestamp(opened_at) as opened_at,
    cast(tax_rate as numeric) as tax_rate
from source

)

select * from renamed

