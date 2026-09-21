-- Every ROWS-framed window downstream assumes one row per store x sku per
-- calendar day with no gaps. This asserts it, because a silent gap would change
-- what "trailing 7 days" means without changing any code.

with dated as (

    select
        store_id,
        sku,
        day_index,
        lag(day_index) over (partition by store_id, sku order by day_index) as prev_day_index
    from {{ ref('int_store_day_sku_demand') }}

)

select *
from dated
where prev_day_index is not null
  and day_index - prev_day_index <> 1
