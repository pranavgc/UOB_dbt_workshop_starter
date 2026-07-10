
with source as (

    select * from {{ source('jaffle_shop', 'raw_supplies') }}

),

renamed as (

    select
    id as supply_id,
    name as supply_name,
    {{ cents_to_dollars('cost') }} as supply_cost,
    cast(perishable as bool) as is_perishable,
    sku
from source
)

select * from renamed

