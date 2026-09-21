{{ config(materialized='view') }}

-- Bill of materials, rolled up to a unit cost per sku.
--
-- ASSUMPTION, stated because it is not self-evident: every row in the supplies
-- table is consumed once per unit sold, so the unit cost of a sku is the sum of
-- its supply costs. The source has no quantity-per-unit column, so this is the
-- only consistent reading of it. The original workshop mart made the same
-- assumption silently; it is written down here instead.

with supplies as (

    select * from {{ ref('stg_jaffle_shop__supplies') }}

)

select
    sku,
    sum(supply_cost)                                            as unit_cost,
    count(*)                                                    as n_supply_lines,
    sum(case when is_perishable then supply_cost else 0 end)    as perishable_cost,
    max(case when is_perishable then 1 else 0 end) = 1          as has_perishable_inputs

from supplies
group by 1
