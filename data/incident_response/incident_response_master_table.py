import sys
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import text, exc as sa_exc, inspect
try:
    here = Path(__file__).resolve().parent
    for probe in (here.parent / 'configs', here.parent.parent / 'src', here
        .parent.parent / 'configs', here):
        if probe.exists():
            sys.path.insert(0, str(probe))
    from db_access_config import DB_CONFIG
except Exception as exc:
    print('FATAL: cannot import DB_CONFIG –', exc)
    sys.exit(1)
RAW_TABLE = 'incident_event_log'
MASTER_TABLE = 'incident_master_features'


def engine():
    cfg = DB_CONFIG
    uri = (
        f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        )
    eng = sa.create_engine(uri, pool_recycle=3600, connect_args={
        'connect_timeout': 300})
    with eng.connect() as con:
        con.execute(text('SELECT 1'))
    return eng


def exec_sql(eng: sa.Engine, sql: str):
    with eng.begin() as con:
        con.execute(text(sql))


def create_master(eng: sa.Engine):
    exec_sql(eng, f'DROP TABLE IF EXISTS {MASTER_TABLE};')
    ddl = f"""
    CREATE TABLE {MASTER_TABLE} (
        number VARCHAR(255) PRIMARY KEY,
        opened_at_raw TEXT,
        opened_at_clean DATETIME NULL,
        resolved_at_raw TEXT,
        resolved_at_clean DATETIME NULL,
        closed_at_raw TEXT,
        closed_at_clean DATETIME NULL,

        contact_type_raw TEXT,
        contact_type_clean VARCHAR(50),
        category_raw TEXT,
        category_clean VARCHAR(100),
        subcategory_raw TEXT,
        subcategory_clean VARCHAR(100),
        location_raw TEXT,
        location_clean VARCHAR(100),
        urgency_raw TEXT,
        urgency_clean TINYINT,
        priority_raw TEXT,
        priority_clean TINYINT,
        impact_raw TEXT,
        impact_clean TINYINT,
        assignment_group_raw TEXT,
        assignment_group_clean VARCHAR(100),
        assigned_to_raw TEXT,
        assigned_to_clean VARCHAR(100),
        reassignment_count INT,
        reopen_count INT,
        sys_mod_count INT,
        made_sla_raw TEXT,
        made_sla_clean TINYINT,
        knowledge_raw TEXT,
        knowledge_clean TINYINT,
        u_priority_confirmation_raw TEXT,
        u_priority_confirmation_clean TINYINT,
        problem_id_raw TEXT,
        problem_id_clean VARCHAR(100),
        rfc_raw TEXT,
        rfc_clean VARCHAR(100),
        vendor_raw TEXT,
        vendor_clean VARCHAR(100),
        caused_by_raw TEXT,
        caused_by_clean VARCHAR(100),
        closed_code_raw TEXT,
        closed_code_clean VARCHAR(50),
        active_raw TEXT,
        active_clean TINYINT,
        target_y_fast_6h TINYINT,
        target_y_slow_96h TINYINT,
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    exec_sql(eng, ddl)


def populate_master(eng: sa.Engine):
    sql = f"""
    INSERT INTO {MASTER_TABLE} (
        number, opened_at_raw, opened_at_clean,
        resolved_at_raw, resolved_at_clean,
        closed_at_raw, closed_at_clean,
        contact_type_raw, contact_type_clean,
        category_raw, category_clean,
        subcategory_raw, subcategory_clean,
        location_raw, location_clean,
        urgency_raw, urgency_clean,
        priority_raw, priority_clean,
        impact_raw, impact_clean,
        assignment_group_raw, assignment_group_clean,
        assigned_to_raw, assigned_to_clean,
        reassignment_count, reopen_count, sys_mod_count,
        made_sla_raw, made_sla_clean,
        knowledge_raw, knowledge_clean,
        u_priority_confirmation_raw, u_priority_confirmation_clean,
        problem_id_raw, problem_id_clean,
        rfc_raw, rfc_clean,
        vendor_raw, vendor_clean,
        caused_by_raw, caused_by_clean,
        closed_code_raw, closed_code_clean,
        active_raw, active_clean,
        target_y_fast_6h, target_y_slow_96h
    )
    SELECT
        f.number,
        f.opened_at_raw,
        f.opened_at_clean,
        f.resolved_at_raw,
        f.resolved_at_clean,
        f.closed_at_raw,
        f.closed_at_clean,

        l.contact_type,
        NULLIF(TRIM(l.contact_type), '?'),
        l.category,
        NULLIF(TRIM(l.category), '?'),
        l.subcategory,
        NULLIF(TRIM(l.subcategory), '?'),
        l.location,
        NULLIF(TRIM(l.location), '?'),
        l.urgency,
        CAST(NULLIF(SUBSTRING_INDEX(l.urgency,' ',1),'?') AS SIGNED),
        l.priority,
        CAST(NULLIF(SUBSTRING_INDEX(l.priority,' ',1),'?') AS SIGNED),
        l.impact,
        CAST(NULLIF(SUBSTRING_INDEX(l.impact,' ',1),'?')  AS SIGNED),
        l.assignment_group,
        NULLIF(TRIM(l.assignment_group), '?'),
        l.assigned_to,
        NULLIF(TRIM(l.assigned_to), '?'),
        l.reassignment_count,
        l.reopen_count,
        l.sys_mod_count,
        l.made_sla,
        CASE WHEN LOWER(TRIM(l.made_sla))='true'  THEN 1
             WHEN LOWER(TRIM(l.made_sla))='false' THEN 0 END,
        l.knowledge,
        CASE WHEN LOWER(TRIM(l.knowledge))='true'  THEN 1
             WHEN LOWER(TRIM(l.knowledge))='false' THEN 0 END,
        l.u_priority_confirmation,
        CASE WHEN LOWER(TRIM(l.u_priority_confirmation))='true'  THEN 1
             WHEN LOWER(TRIM(l.u_priority_confirmation))='false' THEN 0 END,
        l.problem_id,
        NULLIF(TRIM(l.problem_id), '?'),
        l.rfc,
        NULLIF(TRIM(l.rfc), '?'),
        l.vendor,
        NULLIF(TRIM(l.vendor), '?'),
        l.caused_by,
        NULLIF(TRIM(l.caused_by), '?'),
        l.closed_code,
        NULLIF(TRIM(l.closed_code), '?'),
        l.active,
        CASE WHEN LOWER(TRIM(l.active))='true'  THEN 1
             WHEN LOWER(TRIM(l.active))='false' THEN 0 END,

      
        /* targets */
        CASE WHEN f.resolved_at_clean IS NOT NULL
                  AND TIMESTAMPDIFF(HOUR,f.opened_at_clean,f.resolved_at_clean) <= 6
             THEN 1 ELSE 0 END,
        CASE WHEN f.resolved_at_clean IS NOT NULL
                  AND TIMESTAMPDIFF(HOUR,f.opened_at_clean,f.resolved_at_clean) >  96
             THEN 1 ELSE 0 END
        
    FROM (
        /* earliest timestamps per ticket */
        SELECT  number,
                MIN(opened_at)   AS opened_at_raw,
                MIN(resolved_at) AS resolved_at_raw,
                MIN(closed_at)   AS closed_at_raw,
                MIN(STR_TO_DATE(NULLIF(opened_at,   '?'), '%d/%m/%Y %H:%i')) AS opened_at_clean,
                MIN(STR_TO_DATE(NULLIF(resolved_at, '?'), '%d/%m/%Y %H:%i')) AS resolved_at_clean,
                MIN(STR_TO_DATE(NULLIF(closed_at,   '?'), '%d/%m/%Y %H:%i')) AS closed_at_clean
        FROM    {RAW_TABLE}
        GROUP BY number
    ) f
    JOIN (
        /* deterministic “latest” row per ticket */
        SELECT e1.*
        FROM   {RAW_TABLE} e1
        JOIN (
            SELECT number, MAX(sys_mod_count) AS max_mod
            FROM   {RAW_TABLE}
            GROUP BY number
        ) m ON m.number = e1.number AND m.max_mod = e1.sys_mod_count
        LEFT JOIN {RAW_TABLE} e2
               ON  e2.number = e1.number
               AND e2.sys_mod_count = e1.sys_mod_count
               AND STR_TO_DATE(NULLIF(e2.sys_updated_at,'?'), '%d/%m/%Y %H:%i')
                   > STR_TO_DATE(NULLIF(e1.sys_updated_at,'?'), '%d/%m/%Y %H:%i')
        WHERE  e2.number IS NULL      /* keep the row with latest sys_updated_at */
    ) l ON l.number = f.number;
    """
    exec_sql(eng, sql)


LABEL_DEFS = {'target_y_fast_6h': 'TINYINT', 'target_y_slow_96h': 'TINYINT'}


def populate_labels_for_host(engine, use_pandas=True):
    with engine.begin() as conn:
        cols = {c['name'] for c in inspect(engine).get_columns(MASTER_TABLE)}
        for col in LABEL_DEFS:
            if col in cols:
                conn.execute(text(
                    f'ALTER TABLE {MASTER_TABLE} DROP COLUMN {col}'))
        for col, typ in LABEL_DEFS.items():
            conn.execute(text(
                f'ALTER TABLE {MASTER_TABLE} ADD COLUMN {col} {typ} NULL'))
    if not use_pandas:
        raise NotImplementedError('use_pandas=True is the fast path')
    _bulk_update_labels(engine, _compute_labels_df(engine))


def _ensure_label_cols(conn, existing_cols):
    for col, typ in LABEL_DEFS.items():
        if col not in existing_cols:
            conn.execute(text(
                f'ALTER TABLE {MASTER_TABLE} ADD COLUMN {col} {typ} NULL'))


def _compute_labels_df(engine):
    import pandas as pd
    q = f"""
        SELECT number, opened_at_clean, resolved_at_clean,
               reassignment_count
        FROM {MASTER_TABLE}
    """
    df = pd.read_sql(q, engine, parse_dates=['opened_at_clean',
        'resolved_at_clean'])
    diff_hrs = (df.resolved_at_clean - df.opened_at_clean).dt.total_seconds(
        ).div(3600)
    dow = df.opened_at_clean.dt.dayofweek
    df['target_y_fast_6h'] = (df.resolved_at_clean.notna() & (diff_hrs <= 6)
        ).astype('int8')
    df['target_y_slow_96h'] = (df.resolved_at_clean.notna() & (diff_hrs > 96)
        ).astype('int8')
    return df[['number'] + list(LABEL_DEFS)]


def _bulk_update_labels(engine, df, chunk=50000):
    tpl = (
        'UPDATE {tbl} SET target_y_fast_6h=:f, target_y_slow_96h=:sWHERE number=:n'
        .format(tbl=MASTER_TABLE))
    with engine.begin() as conn:
        for start in range(0, len(df), chunk):
            batch = df.iloc[start:start + chunk]
            conn.execute(text(tpl), [{'n': r.number, 'f': int(r.
                target_y_fast_6h), 's': int(r.target_y_slow_96h)} for r in
                batch.itertuples()])


def _prepare_index_columns(conn):
    for col in ('category', 'location'):
        dtype = conn.execute(text(
            """SELECT DATA_TYPE
                    FROM information_schema.columns
                    WHERE table_schema = DATABASE()
                      AND table_name = 'incident_event_log'
                      AND column_name = :col"""
            ), {'col': col}).scalar()
        if dtype and dtype.lower() == 'text':
            conn.execute(text(
                f"""ALTER TABLE incident_event_log
                    MODIFY {col} VARCHAR(191)
                    CHARACTER SET utf8mb4
                    COLLATE utf8mb4_0900_ai_ci"""
                ))


def _ensure_indexes(conn):
    stmts = [
        'CREATE INDEX IF NOT EXISTS idx_event_cat ON incident_event_log(category)'
        ,
        'CREATE INDEX IF NOT EXISTS idx_event_loc ON incident_event_log(location)'
        ]
    for ddl in stmts:
        conn.execute(text(ddl))


def main():
    eng = engine()
    populate_labels_for_host(eng)


if __name__ == '__main__':
    main()
