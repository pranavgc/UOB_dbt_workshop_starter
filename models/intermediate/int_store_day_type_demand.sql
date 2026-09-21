{{ config(materialized='table') }}

-- Grain: store x product_type x day.
--
-- This exists for one reason: featured placements are scheduled off trailing
-- demand at the store x product-type level, so this is the level at which the
-- confounder has to be measured. Without it, the featuring coefficient cannot
-- be de-confounded downstream.

select
    store_id,
    product_type,
    order_date,
    day_index,
    sum(units_sold)                             as type_units_sold,
    max(case when is_featured then 1 else 0 end) as any_featured

from {{ ref('int_store_day_sku_demand') }}
group by 1, 2, 3, 4
