with stores as (

    select * from {{ ref('stg_jaffle_shop__stores') }}

),

openings as (

    select * from {{ ref('stg_jaffle_shop__store_openings') }}

)

select
    stores.store_id,
    stores.store_name,
    stores.tax_rate,
    openings.opened_at

from stores
inner join openings
    on openings.store_id = stores.store_id
