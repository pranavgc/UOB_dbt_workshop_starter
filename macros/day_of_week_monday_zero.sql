{# Portable weekday index: 0 = Monday ... 6 = Sunday.

   Warehouses disagree here. DuckDB's EXTRACT(dayofweek) returns 0 for Sunday,
   BigQuery's returns 1 for Sunday. Dispatching keeps every downstream model and
   the Python feature code on one convention. #}

{% macro day_of_week_monday_zero(column_name) -%}
    {{ return(adapter.dispatch('day_of_week_monday_zero')(column_name)) }}
{%- endmacro %}

{% macro default__day_of_week_monday_zero(column_name) -%}
    ((cast(extract(dayofweek from {{ column_name }}) as integer) + 6) % 7)
{%- endmacro %}

{% macro bigquery__day_of_week_monday_zero(column_name) -%}
    mod(extract(dayofweek from {{ column_name }}) + 5, 7)
{%- endmacro %}
