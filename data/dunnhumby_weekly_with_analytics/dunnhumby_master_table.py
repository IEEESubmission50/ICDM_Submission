from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import text
import os
import sys
from typing import List, Dict, Tuple
try:
    current_file_dir = Path(__file__).resolve().parent
    project_root = current_file_dir.parent.parent
    config_path = project_root / 'configs'
    project_dir_guess = Path(__file__).resolve().parent.parent.parent
    src_dir_for_config = project_dir_guess / 'src'
    if not src_dir_for_config.exists():
        script_dir = Path(__file__).resolve().parent
        configs_dir_alt = script_dir.parent / 'configs'
        if not configs_dir_alt.exists():
            configs_dir_alt = script_dir.parent.parent / 'configs'
        if configs_dir_alt.exists():
            sys.path.insert(0, str(configs_dir_alt))
            print(f'Attempting to import DB_CONFIG from: {configs_dir_alt}')
        else:
            sys.path.insert(0, str(src_dir_for_config))
            print(
                f'Attempting to import DB_CONFIG from: {src_dir_for_config} (fallback)'
                )
    else:
        sys.path.insert(0, str(src_dir_for_config))
        print(f'Attempting to import DB_CONFIG from: {src_dir_for_config}')
    from db_access_config import DB_CONFIG
    print(f'Successfully imported DB_CONFIG.')
except ImportError as e:
    print(
        f'Error: Could not import DB_CONFIG after trying multiple common paths: {e}'
        )
    print(
        'Please ensure db_access_config.py is in your PYTHONPATH, or adjust script paths.'
        )
    print(
        'Using placeholder DB_CONFIG. Update with your actual configuration if proceeding.'
        )
    DB_CONFIG = {'user': 'your_user', 'password': 'your_password', 'host':
        'localhost', 'port': 3306, 'database': 'your_database'}
except Exception as e_gen:
    print(f'An unexpected error occurred during DB_CONFIG import: {e_gen}')
    sys.exit(1)
MASTER_TABLE_NAME = 'dunnhumby_master_data'
REMOTE_END_WEEK = 52
HOST_START_WEEK = 53
HOST_RAW_DATA_END_WEEK = 66
LARGE_BASKET_QTY_THRESHOLD = 15


def get_engine():
    cfg = DB_CONFIG
    uri = (
        f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        )
    try:
        engine = sa.create_engine(uri, pool_recycle=3600, connect_args={
            'connect_timeout': 300})
        with engine.connect() as connection:
            print(f"Successfully connected to database: {cfg['database']}")
        return engine
    except Exception as e:
        print(f'Error creating database engine: {e}')
        sys.exit(1)


def execute_sql(engine, sql_statement: str, step_name: str):
    print(f'\n--- Executing SQL for: {step_name} ---')
    try:
        with engine.begin() as connection:
            connection.execute(text(sql_statement))
        print(f"Step '{step_name}' executed successfully.")
    except Exception as e:
        print(f"Error executing SQL for step '{step_name}': {e}")
        print(
            f'Problematic SQL (or part of it for brevity):\n{sql_statement[:2000]}...'
            )
        raise


def get_column_definitions_for_alter(columns_to_add: List[str],
    default_type='FLOAT NULL') ->List[str]:
    tpl = 'ADD COLUMN `{}` ' + default_type
    return [tpl.format(col) for col in columns_to_add]


def build_remote_features_ctes_and_select(remote_end_week: int,
    large_basket_qty_threshold: int) ->Tuple[str, str]:
    remote_ctes_list = [
        f"""households_in_scope AS (
            SELECT DISTINCT CAST(household_key AS CHAR(255)) as household_key 
            FROM transaction_data 
            WHERE CAST(WEEK_NO AS UNSIGNED) <= {remote_end_week} AND household_key IS NOT NULL
        )"""
        ,
        f"""demographics AS (
            SELECT 
                CAST(hs.household_key AS CHAR(255)) as household_key, 
                h.classification_1, h.classification_2, h.classification_3, h.HOMEOWNER_DESC,
                h.classification_5, h.classification_4, h.KID_CATEGORY_DESC
            FROM hh_demographic h 
            JOIN households_in_scope hs ON CAST(h.household_key AS CHAR(255)) = hs.household_key
        )"""
        ,
        f"""baskets_remote AS (
            SELECT CAST(household_key AS CHAR(255)) as household_key, BASKET_ID, STORE_ID, CAST(WEEK_NO AS UNSIGNED) AS WEEK_NO, CAST(DAY AS UNSIGNED) AS DAY,
                   SUM(CAST(QUANTITY AS DECIMAL(10,2))) AS basket_qty, SUM(CAST(SALES_VALUE AS DECIMAL(10,2))) AS basket_spend
            FROM transaction_data WHERE CAST(WEEK_NO AS UNSIGNED) <= {remote_end_week}
            GROUP BY household_key, BASKET_ID, STORE_ID, WEEK_NO, DAY
        )"""
        ,
        f"""trip_stats_remote AS (
            SELECT household_key, COUNT(DISTINCT WEEK_NO) AS remote_total_weeks_shopped, COUNT(*) AS remote_total_trips,
                   SUM(basket_spend) AS remote_total_spend, SUM(basket_qty) AS remote_total_units,
                   SUM(basket_spend) / NULLIF(COUNT(*), 0) AS remote_avg_spend_per_trip,
                   SUM(basket_qty) / NULLIF(COUNT(*), 0) AS remote_avg_units_per_trip,
                   SUM(basket_spend) / NULLIF(SUM(basket_qty), 0) AS remote_avg_price_per_unit,
                   SUM(CASE WHEN basket_qty >= {large_basket_qty_threshold} THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS remote_pct_large_basket_trips,
                   MIN(DAY) as remote_first_purchase_day, MAX(DAY) as remote_last_purchase_day,
                   MAX(DAY) - MIN(DAY) AS remote_shopping_span_days,
                   ({remote_end_week} * 7) - MAX(DAY) AS remote_days_since_last_purchase_at_wk{remote_end_week},
                   AVG(next_purchase_day - DAY) AS remote_avg_inter_purchase_days
            FROM (SELECT *, LEAD(DAY, 1) OVER (PARTITION BY household_key ORDER BY DAY) as next_purchase_day FROM baskets_remote) br
            GROUP BY household_key
        )"""
        ,
        f"""weekly_agg_remote AS (
            SELECT household_key, WEEK_NO, SUM(basket_spend) AS weekly_spend, COUNT(DISTINCT BASKET_ID) AS weekly_trips, SUM(basket_qty) AS weekly_units
            FROM baskets_remote GROUP BY household_key, WEEK_NO
        )"""
        ,
        f"""overall_weekly_stats_remote AS (
            SELECT household_key,
                   AVG(weekly_spend) AS remote_avg_weekly_spend, MAX(weekly_spend) AS remote_max_weekly_spend, MIN(weekly_spend) AS remote_min_weekly_spend, STDDEV_SAMP(weekly_spend) AS remote_stddev_weekly_spend,
                   AVG(weekly_trips) AS remote_avg_weekly_trips, MAX(weekly_trips) AS remote_max_weekly_trips, MIN(weekly_trips) AS remote_min_weekly_trips, STDDEV_SAMP(weekly_trips) AS remote_stddev_weekly_trips,
                   AVG(weekly_units) AS remote_avg_weekly_units, MAX(weekly_units) AS remote_max_weekly_units, MIN(weekly_units) AS remote_min_weekly_units, STDDEV_SAMP(weekly_units) AS remote_stddev_weekly_units,
                   COUNT(DISTINCT CASE WHEN weekly_trips > 0 THEN WEEK_NO ELSE NULL END) AS remote_num_active_weeks,
                   COUNT(DISTINCT CASE WHEN weekly_trips > 0 THEN WEEK_NO ELSE NULL END) / {remote_end_week}.0 AS remote_pct_active_weeks
            FROM weekly_agg_remote GROUP BY household_key
        )"""
        ,
        f"""temporal_halves_remote AS (
            SELECT household_key,
                   AVG(CASE WHEN WEEK_NO <= ({remote_end_week}/2) THEN weekly_spend ELSE NULL END) AS remote_avg_spend_first_half,
                   AVG(CASE WHEN WEEK_NO > ({remote_end_week}/2) AND WEEK_NO <= {remote_end_week} THEN weekly_spend ELSE NULL END) AS remote_avg_spend_second_half,
                   AVG(CASE WHEN WEEK_NO <= ({remote_end_week}/2) THEN weekly_trips ELSE NULL END) AS remote_avg_trips_first_half,
                   AVG(CASE WHEN WEEK_NO > ({remote_end_week}/2) AND WEEK_NO <= {remote_end_week} THEN weekly_trips ELSE NULL END) AS remote_avg_trips_second_half
            FROM weekly_agg_remote GROUP BY household_key
        )"""
        ,
        f"""promo_brand_stats_remote AS (
            SELECT t.household_key,
                   SUM(ABS(CAST(t.RETAIL_DISC AS DECIMAL(10,2))) + ABS(CAST(t.COUPON_DISC AS DECIMAL(10,2))) + ABS(CAST(t.COUPON_MATCH_DISC AS DECIMAL(10,2)))) /
                       NULLIF(SUM(ABS(CAST(t.SALES_VALUE AS DECIMAL(10,2)) - CAST(t.RETAIL_DISC AS DECIMAL(10,2)) - CAST(t.COUPON_MATCH_DISC AS DECIMAL(10,2)))), 0) AS remote_promo_sensitivity,
                   SUM(CASE WHEN p.BRAND = 'National' THEN (CAST(t.SALES_VALUE AS DECIMAL(10,2)) - CAST(t.RETAIL_DISC AS DECIMAL(10,2)) - CAST(t.COUPON_MATCH_DISC AS DECIMAL(10,2))) ELSE 0 END) /
                       NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2)) - CAST(t.RETAIL_DISC AS DECIMAL(10,2)) - CAST(t.COUPON_MATCH_DISC AS DECIMAL(10,2))), 0) AS remote_pct_national_brand_spend,
                   SUM(CASE WHEN p.BRAND = 'Private' THEN (CAST(t.SALES_VALUE AS DECIMAL(10,2)) - CAST(t.RETAIL_DISC AS DECIMAL(10,2)) - CAST(t.COUPON_MATCH_DISC AS DECIMAL(10,2))) ELSE 0 END) /
                       NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2)) - CAST(t.RETAIL_DISC AS DECIMAL(10,2)) - CAST(t.COUPON_MATCH_DISC AS DECIMAL(10,2))), 0) AS remote_pct_private_brand_spend,
                   SUM(CASE WHEN p.BRAND = 'National' THEN CAST(t.QUANTITY AS DECIMAL(10,2)) ELSE 0 END) / NULLIF(SUM(CAST(t.QUANTITY AS DECIMAL(10,2))),0) AS remote_pct_national_brand_units,
                   SUM(CASE WHEN p.BRAND = 'Private' THEN CAST(t.QUANTITY AS DECIMAL(10,2)) ELSE 0 END) / NULLIF(SUM(CAST(t.QUANTITY AS DECIMAL(10,2))),0) AS remote_pct_private_brand_units
            FROM transaction_data t JOIN product p ON t.PRODUCT_ID = CAST(p.PRODUCT_ID AS CHAR(50))
            WHERE CAST(t.WEEK_NO AS UNSIGNED) <= {remote_end_week} GROUP BY t.household_key
        )"""
        ,
        f"""product_prefs_remote AS (
            SELECT t.household_key, COUNT(DISTINCT t.PRODUCT_ID) AS remote_distinct_product_ids, COUNT(DISTINCT p.DEPARTMENT) AS remote_distinct_departments, COUNT(DISTINCT p.COMMODITY_DESC) AS remote_distinct_commodities,
                   SUM(CASE WHEN p.DEPARTMENT = 'GROCERY' THEN CAST(t.SALES_VALUE AS DECIMAL(10,2)) ELSE 0 END) / NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2))),0) AS remote_spend_share_grocery,
                   SUM(CASE WHEN p.DEPARTMENT = 'PRODUCE' THEN CAST(t.SALES_VALUE AS DECIMAL(10,2)) ELSE 0 END) / NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2))),0) AS remote_spend_share_produce,
                   SUM(CASE WHEN p.DEPARTMENT = 'MEAT-PCKGD' THEN CAST(t.SALES_VALUE AS DECIMAL(10,2)) ELSE 0 END) / NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2))),0) AS remote_spend_share_meat,
                   SUM(CASE WHEN p.DEPARTMENT = 'DRUG GM' THEN CAST(t.SALES_VALUE AS DECIMAL(10,2)) ELSE 0 END) / NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2))),0) AS remote_spend_share_drugstore
            FROM transaction_data t JOIN product p ON t.PRODUCT_ID = CAST(p.PRODUCT_ID AS CHAR(50))
            WHERE CAST(t.WEEK_NO AS UNSIGNED) <= {remote_end_week} GROUP BY t.household_key
        )"""
        ,
        f"""monthly_quarterly_agg_remote AS (
             SELECT household_key, (WEEK_NO - 1) DIV 4 + 1 AS month_num, (WEEK_NO - 1) DIV 13 + 1 AS quarter_num, weekly_spend, weekly_trips
             FROM weekly_agg_remote
        )"""
        ,
        f"""pivoted_fourier_remote AS (
            SELECT household_key,
                   AVG(CASE WHEN month_num = 1 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m1, AVG(CASE WHEN month_num = 2 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m2,
                   AVG(CASE WHEN month_num = 3 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m3, AVG(CASE WHEN month_num = 4 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m4,
                   AVG(CASE WHEN month_num = 5 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m5, AVG(CASE WHEN month_num = 6 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m6,
                   AVG(CASE WHEN month_num = 7 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m7, AVG(CASE WHEN month_num = 8 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m8,
                   AVG(CASE WHEN month_num = 9 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m9, AVG(CASE WHEN month_num = 10 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m10,
                   AVG(CASE WHEN month_num = 11 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m11, AVG(CASE WHEN month_num = 12 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m12,
                   AVG(CASE WHEN month_num = 13 THEN weekly_spend ELSE NULL END) as remote_avg_spend_m13,
                   AVG(CASE WHEN quarter_num = 1 THEN weekly_spend ELSE NULL END) as remote_avg_spend_q1, AVG(CASE WHEN quarter_num = 2 THEN weekly_spend ELSE NULL END) as remote_avg_spend_q2,
                   AVG(CASE WHEN quarter_num = 3 THEN weekly_spend ELSE NULL END) as remote_avg_spend_q3, AVG(CASE WHEN quarter_num = 4 THEN weekly_spend ELSE NULL END) as remote_avg_spend_q4
            FROM monthly_quarterly_agg_remote GROUP BY household_key
        )"""
        ,
        f"""causal_influence_remote AS (
            SELECT t.household_key,
                   SUM(CASE WHEN (ca.display IS NOT NULL AND ca.display <> '0' AND ca.display <> '') THEN CAST(t.SALES_VALUE AS DECIMAL(10,2)) ELSE 0 END) /
                       NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2))),0) AS remote_pct_spend_displayed,
                   SUM(CASE WHEN (ca.mailer IS NOT NULL AND ca.mailer <> '0' AND ca.mailer <> '') THEN CAST(t.SALES_VALUE AS DECIMAL(10,2)) ELSE 0 END) /
                       NULLIF(SUM(CAST(t.SALES_VALUE AS DECIMAL(10,2))),0) AS remote_pct_spend_mailer
            FROM transaction_data t 
            LEFT JOIN causal_data ca ON t.PRODUCT_ID = ca.PRODUCT_ID AND t.STORE_ID = ca.STORE_ID AND CAST(t.WEEK_NO AS UNSIGNED) = CAST(ca.WEEK_NO AS UNSIGNED)
            WHERE CAST(t.WEEK_NO AS UNSIGNED) <= {remote_end_week} GROUP BY t.household_key
        )"""
        ]
    remote_ctes_sql = ',\n'.join(remote_ctes_list)
    select_clause_sql = f"""
        h_scope.household_key,
        d.classification_1, d.classification_2, d.classification_3, d.HOMEOWNER_DESC, d.classification_5, d.classification_4, d.KID_CATEGORY_DESC,
        tsr.remote_total_weeks_shopped, tsr.remote_total_trips, tsr.remote_total_spend,
        tsr.remote_total_units, tsr.remote_avg_spend_per_trip, tsr.remote_avg_units_per_trip,
        tsr.remote_avg_price_per_unit, tsr.remote_pct_large_basket_trips,
        tsr.remote_first_purchase_day, tsr.remote_last_purchase_day, 
        tsr.remote_shopping_span_days, tsr.remote_days_since_last_purchase_at_wk{remote_end_week},
        tsr.remote_avg_inter_purchase_days,
        owsr.remote_avg_weekly_spend, owsr.remote_max_weekly_spend, owsr.remote_min_weekly_spend, owsr.remote_stddev_weekly_spend,
        owsr.remote_avg_weekly_trips, owsr.remote_max_weekly_trips, owsr.remote_min_weekly_trips, owsr.remote_stddev_weekly_trips,
        owsr.remote_avg_weekly_units, owsr.remote_max_weekly_units, owsr.remote_min_weekly_units, owsr.remote_stddev_weekly_units,
        owsr.remote_num_active_weeks, owsr.remote_pct_active_weeks,
        thr.remote_avg_spend_first_half, thr.remote_avg_spend_second_half, thr.remote_avg_trips_first_half, thr.remote_avg_trips_second_half,
        (thr.remote_avg_spend_second_half / NULLIF(thr.remote_avg_spend_first_half,0)) - 1 AS remote_spend_growth_rate_halves,
        (thr.remote_avg_trips_second_half / NULLIF(thr.remote_avg_trips_first_half,0)) - 1 AS remote_trips_growth_rate_halves,
        pbsr.remote_promo_sensitivity, pbsr.remote_pct_national_brand_spend, pbsr.remote_pct_private_brand_spend, pbsr.remote_pct_national_brand_units, pbsr.remote_pct_private_brand_units,
        ppr.remote_distinct_product_ids, ppr.remote_distinct_departments, ppr.remote_distinct_commodities,
        ppr.remote_spend_share_grocery, ppr.remote_spend_share_produce, ppr.remote_spend_share_meat, ppr.remote_spend_share_drugstore,
        pfr.remote_avg_spend_m1, pfr.remote_avg_spend_m2, pfr.remote_avg_spend_m3, pfr.remote_avg_spend_m4, pfr.remote_avg_spend_m5, pfr.remote_avg_spend_m6,
        pfr.remote_avg_spend_m7, pfr.remote_avg_spend_m8, pfr.remote_avg_spend_m9, pfr.remote_avg_spend_m10, pfr.remote_avg_spend_m11, pfr.remote_avg_spend_m12, pfr.remote_avg_spend_m13,
        pfr.remote_avg_spend_q1, pfr.remote_avg_spend_q2, pfr.remote_avg_spend_q3, pfr.remote_avg_spend_q4,
        cir.remote_pct_spend_displayed, cir.remote_pct_spend_mailer
    FROM households_in_scope h_scope
    LEFT JOIN demographics d ON h_scope.household_key = d.household_key 
    LEFT JOIN trip_stats_remote tsr ON h_scope.household_key = tsr.household_key
    LEFT JOIN overall_weekly_stats_remote owsr ON h_scope.household_key = owsr.household_key
    LEFT JOIN temporal_halves_remote thr ON h_scope.household_key = thr.household_key
    LEFT JOIN promo_brand_stats_remote pbsr ON h_scope.household_key = pbsr.household_key
    LEFT JOIN product_prefs_remote ppr ON h_scope.household_key = ppr.household_key
    LEFT JOIN pivoted_fourier_remote pfr ON h_scope.household_key = pfr.household_key
    LEFT JOIN causal_influence_remote cir ON h_scope.household_key = cir.household_key
    WHERE tsr.household_key IS NOT NULL
    """
    return remote_ctes_sql, select_clause_sql


def create_initial_table_with_remote_features(engine):
    step_name = 'Create Initial Table with Remote Features'
    print(f'--- Starting Step: {step_name} ---')
    execute_sql(engine, f'DROP TABLE IF EXISTS {MASTER_TABLE_NAME};',
        'Drop existing master table')
    ctes_sql, select_sql = build_remote_features_ctes_and_select(
        REMOTE_END_WEEK, LARGE_BASKET_QTY_THRESHOLD)
    create_sql = f"""
    CREATE TABLE {MASTER_TABLE_NAME} AS
    WITH
    {ctes_sql}
    SELECT
    {select_sql};
    """
    execute_sql(engine, create_sql, 'Create table with remote features')
    execute_sql(engine,
        f'ALTER TABLE {MASTER_TABLE_NAME} MODIFY household_key VARCHAR(255) NOT NULL, ADD PRIMARY KEY (household_key);'
        , 'Add PK to master table')
    print(f'--- Finished Step: {step_name} ---')


def get_host_pivoted_column_names(host_start_week: int,
    host_raw_data_end_week: int) ->List[str]:
    cols = []
    metrics = ['trips', 'units', 'spend', 'private_brand_units',
        'private_brand_spend', 'national_brand_units',
        'national_brand_spend', 'spend_on_displayed_items',
        'spend_on_mailer_items', 'num_distinct_products']
    for week_num in range(host_start_week, host_raw_data_end_week + 1):
        for metric_name in metrics:
            cols.append(f'host_week_{week_num}_{metric_name}')
    return cols


def add_and_populate_host_features(engine):
    step_name = 'Add and Populate Host Pivoted Features'
    print(f'--- Starting Step: {step_name} ---')
    host_col_names = get_host_pivoted_column_names(HOST_START_WEEK,
        HOST_RAW_DATA_END_WEEK)
    add_column_clauses = get_column_definitions_for_alter(host_col_names,
        default_type='DECIMAL(18,2) NULL')
    if add_column_clauses:
        alter_sql = (
            f"ALTER TABLE {MASTER_TABLE_NAME} {', '.join(add_column_clauses)};"
            )
        execute_sql(engine, alter_sql,
            'Add host feature columns to master table')
    host_pivot_metrics_defs = [('trips', 'COUNT(DISTINCT BASKET_ID)'), (
        'units', 'SUM(CAST(QUANTITY AS DECIMAL(18,2)))'), ('spend',
        'SUM(CAST(SALES_VALUE AS DECIMAL(18,2)))'), ('private_brand_units',
        "SUM(CASE WHEN p.BRAND = 'Private' THEN CAST(t.QUANTITY AS DECIMAL(18,2)) ELSE 0 END)"
        ), ('private_brand_spend',
        "SUM(CASE WHEN p.BRAND = 'Private' THEN CAST(t.SALES_VALUE AS DECIMAL(18,2)) ELSE 0 END)"
        ), ('national_brand_units',
        "SUM(CASE WHEN p.BRAND = 'National' THEN CAST(t.QUANTITY AS DECIMAL(18,2)) ELSE 0 END)"
        ), ('national_brand_spend',
        "SUM(CASE WHEN p.BRAND = 'National' THEN CAST(t.SALES_VALUE AS DECIMAL(18,2)) ELSE 0 END)"
        ), ('spend_on_displayed_items',
        "SUM(CASE WHEN (cd.display IS NOT NULL AND cd.display <> '0' AND cd.display <> '') THEN CAST(t.SALES_VALUE AS DECIMAL(18,2)) ELSE 0 END)"
        ), ('spend_on_mailer_items',
        "SUM(CASE WHEN (cd.mailer IS NOT NULL AND cd.mailer <> '0' AND cd.mailer <> '') THEN CAST(t.SALES_VALUE AS DECIMAL(18,2)) ELSE 0 END)"
        ), ('num_distinct_products', 'COUNT(DISTINCT t.PRODUCT_ID)')]
    host_pivot_selects = []
    for week_num in range(HOST_START_WEEK, HOST_RAW_DATA_END_WEEK + 1):
        for metric_name, _ in host_pivot_metrics_defs:
            host_pivot_selects.append(
                f'COALESCE(MAX(CASE WHEN th.WEEK_NO = {week_num} THEN th.{metric_name}_wk ELSE NULL END), 0) AS host_week_{week_num}_{metric_name}'
                )
    update_cte_sql = f"""
    WITH weekly_transactions_host_raw_intermediate AS (
        SELECT CAST(t.household_key AS CHAR(255)) as household_key, CAST(t.WEEK_NO AS UNSIGNED) AS WEEK_NO,
            {', '.join([f'{expr} AS {name}_wk' for name, expr in host_pivot_metrics_defs])}
        FROM transaction_data t
        LEFT JOIN product p ON t.PRODUCT_ID = CAST(p.PRODUCT_ID AS CHAR(50)) 
        LEFT JOIN causal_data cd ON t.PRODUCT_ID = cd.PRODUCT_ID AND t.STORE_ID = cd.STORE_ID AND CAST(t.WEEK_NO AS UNSIGNED) = CAST(cd.WEEK_NO AS UNSIGNED)
        WHERE CAST(t.WEEK_NO AS UNSIGNED) BETWEEN {HOST_START_WEEK} AND {HOST_RAW_DATA_END_WEEK}
        GROUP BY t.household_key, CAST(t.WEEK_NO AS UNSIGNED)
    ),
    pivoted_host_data_for_update AS (
        SELECT th.household_key, {', '.join(host_pivot_selects)}
        FROM weekly_transactions_host_raw_intermediate th
        GROUP BY th.household_key
    )
    UPDATE {MASTER_TABLE_NAME} m
    JOIN pivoted_host_data_for_update phu ON m.household_key = phu.household_key
    SET {', '.join([f'm.`{col}` = phu.`{col}`' for col in host_col_names])};
    """
    execute_sql(engine, update_cte_sql, 'Populate host feature columns')
    print(f'--- Finished Step: {step_name} ---')


def add_qid_columns_to_master_table(engine):
    step_name = 'Add QID Columns to Master Table'
    print(f'--- Starting Step: {step_name} ---')
    alter_sql = f"""
    ALTER TABLE {MASTER_TABLE_NAME} 
    ADD COLUMN qi_household_budget_qid INT DEFAULT NULL,
    ADD COLUMN qi_family_type_qid INT DEFAULT NULL;
    """
    execute_sql(engine, alter_sql, 'Add QID columns to master table')
    update_sql = f"""
    UPDATE {MASTER_TABLE_NAME}
    SET 
        qi_household_budget_qid = 
            /* Household-budget QID = income_band*10 + owner_code */
            (10 * CASE
                WHEN classification_3 IN ('Level1','Level2') THEN 1
                WHEN classification_3 IN ('Level3','Level4') THEN 2
                WHEN classification_3 IN ('Level5','Level6') THEN 3
                WHEN classification_3 IN ('Level7','Level8','Level9') THEN 4
                WHEN classification_3 IN ('Level10','Level11','Level12') THEN 5
                ELSE 9
            END) /* 9 => default band */
            + CASE
                WHEN HOMEOWNER_DESC IN ('Homeowner','Probable Owner') THEN 1
                WHEN HOMEOWNER_DESC IN ('Renter','Probable Renter') THEN 2
                ELSE 3
            END,

        qi_family_type_qid = 
            /* Family-type QID = marital_code*10 + kid_code */
            (10 * CASE
                WHEN classification_2 = 'X' THEN 1
                WHEN classification_2 = 'Y' THEN 2
                ELSE 3
            END) /* 3 => other/unknown */
            + CASE
                WHEN KID_CATEGORY_DESC IN ('None/Unknown','0') THEN 0
                WHEN KID_CATEGORY_DESC IN ('1','1 Child') THEN 1
                WHEN KID_CATEGORY_DESC IN ('2','2 Children') THEN 2
                WHEN KID_CATEGORY_DESC IN ('3+','3') THEN 3
                ELSE 4
            END;
    """
    execute_sql(engine, update_sql, 'Populate QID columns with derived values')
    print(f'--- Finished Step: {step_name} ---')


def main():
    print(
        'Dunnhumby Master Table Generation Script - Multi-Step (Schema Corrected & Week 66)'
        )
    print(
        '---------------------------------------------------------------------------------'
        )
    engine = get_engine()
    if not engine:
        sys.exit(1)
    create_initial_table_with_remote_features(engine)
    add_and_populate_host_features(engine)
    add_qid_columns_to_master_table(engine)
    print(
        '\n---------------------------------------------------------------------------------'
        )
    print('Script finished.')
    print(
        f"The '{MASTER_TABLE_NAME}' table should now be available and populated in steps."
        )
    print('Please verify the table structure and data.')


if __name__ == '__main__':
    main()
