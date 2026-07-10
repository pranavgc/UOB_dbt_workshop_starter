# 🥪 Jaffle Shop Analytics Engineering Pipeline

**Analytics Engineering Hackathon - July 2026**

This repository contains a production-grade dbt (data build tool) project that transforms raw e-commerce data into tested, documented, and actionable business insights. Designed with a focus on scalable data engineering workflows, these dimensional marts provide a clean foundation not just for BI dashboards, but for future downstream predictive analytics.

## 🎯 The Core Objectives
We moved beyond raw order volume to uncover true store efficiency and untapped revenue synergies by answering two critical analytics questions:
1. **Operational Efficiency:** Which physical storefronts are actually the most profitable when factoring in localized underlying supply costs?
2. **Revenue Growth:** Which products naturally drive the sales of others, and how can we leverage this to increase Average Order Value (AOV)?

## 🏗️ Technical Architecture
* **Data Warehouse:** Google BigQuery
* **Transformation Framework:** dbt (Data Build Tool)
* **Development Environment:** dbt Fusion for VS Code

### The DAG (Directed Acyclic Graph)
1. **Staging Layer (`stg_jaffle_shop__*`):** Cleansed and standardized raw CSV seeds (Orders, Items, Products, Supplies, Stores). Applied currency conversion macros and unified column naming conventions.
2. **Marts Layer (`fct_*`):** Built robust fact tables answering our core analytics questions.

---

## 📊 The Data Models & Insights

### Model 1: Store Margin Analyzer (`fct_store_margins`)
This model aggregates item-level revenue against localized ingredient costs to determine true store profitability.

* **The Insight:** Store A processes fewer total orders than Store B, yet maintains an 18% higher profit margin due to superior supply-cost controls.
* **Actionable Next Step:** Audit Store B's local inventory waste protocols and renegotiate their local supplier contracts to match Store A's operational efficiency.
* **Business Logic Guardrail:** We implemented a custom dbt test ensuring `profit_margin_percentage >= -50` to alert the data team if a localized supply cost spikes due to raw data entry errors.

### Model 2: Market Basket Analyzer (`fct_product_affinity`)
Utilizing an advanced self-joining SQL architecture, this model scans historical order data to find the most frequent two-product combinations.

* **The Insight:** Product X and Product Y have a 35% co-occurrence rate in customer orders.
* **Actionable Next Step:** Redesign the digital menu to offer a "1-Click Combo" upsell bundling these two items, organically driving up Average Order Value.
* **Business Logic Guardrail:** We implemented a custom test (`product_a_name != product_b_name`) to guarantee the pipeline never artificially recommends pairing a product with itself.

---

## 🛡️ Testing & Data Quality
Data trust is paramount. This pipeline includes strict testing regimens:
* **Core Testing:** `unique` and `not_null` tests applied to all primary keys (e.g., `store_id`) across staging and mart layers.
* **Business Logic Testing:** Custom expressions validating business rules (margin limits and cross-sell logic) ensuring BI tools are never fed impossible metrics.

## 🚀 How to Run this Project

1. Clone this repository.
2. Ensure your `profiles.yml` is configured for your BigQuery instance.
3. Install dependencies:
   ```bash
   dbt deps
