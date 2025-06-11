import json
import logging
from pathlib import Path
import sys
from pathlib import Path
import sys
try:
    project_src = current_file = Path(__file__).resolve().parent.parent.parent
    src_path = project_src / 'src'
    sys.path.insert(0, str(src_path))
    from db_access_config import DB_CONFIG
    from sqlalchemy import create_engine, text, exc as sqlalchemy_exc, inspect as sqlalchemy_inspect
except ImportError:
    print(
        'Error: Could not import DB_CONFIG or sqlalchemy. Ensure db_access_config.py is accessible and sqlalchemy is installed.'
        )
    DB_CONFIG = None
    sys.exit(1)
except Exception as e:
    print(f'An unexpected error occurred during imports: {e}')
    DB_CONFIG = None
    sys.exit(1)
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
EVALUATION_SPEC_PATH = Path('evaluation_spec.json')


def get_db_engine(db_cfg):
    if not db_cfg:
        logger.error(
            'Database configuration not available. Cannot create engine.')
        return None
    try:
        uri = (
            f"mysql+mysqlconnector://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
            )
        engine = create_engine(uri, pool_recycle=3600, connect_args={
            'connect_timeout': 10})
        with engine.connect() as connection:
            logger.info(
                f"Successfully created and tested database engine for {db_cfg['database']}."
                )
        return engine
    except sqlalchemy_exc.OperationalError as e:
        logger.error(
            f"Database connection error for {db_cfg.get('database', 'N/A')} at {db_cfg.get('host', 'N/A')}:{db_cfg.get('port', 'N/A')}: {e}"
            )
        if 'Access denied' in str(e):
            logger.error('Please check database username and password.')
        elif 'Unknown database' in str(e):
            logger.error(
                f"Database '{db_cfg.get('database')}' does not exist or is not accessible."
                )
        return None
    except Exception as e:
        logger.error(f'Error creating database engine: {e}')
        return None


def validate_columns_exist(engine, master_table_name: str,
    view_column_source_list: list, all_master_cols_in_spec: list,
    view_name_for_log: str) ->bool:
    if not view_column_source_list:
        logger.error(
            f"Error: The column source list for view '{view_name_for_log}' (e.g., HOST_DATA_ALL_COLS or REMOTE_ALL_COLS) is empty in the spec."
            )
        logger.error(
            'A view must select at least one column from the master table. Please define columns for this view in the spec.'
            )
        return False
    missing_in_spec_master_list = [col for col in view_column_source_list if
        col not in all_master_cols_in_spec]
    if missing_in_spec_master_list:
        logger.error(
            f"Error: For view '{view_name_for_log}', the following columns specified in its source list (e.g., HOST_DATA_ALL_COLS) are NOT listed in 'ALL_MASTER_TABLE_COLS' in the spec: {missing_in_spec_master_list}"
            )
        logger.error(
            "Please ensure 'ALL_MASTER_TABLE_COLS' accurately reflects all columns in the master table and that view column lists only pick from these."
            )
        return False
    try:
        inspector = sqlalchemy_inspect(engine)
        db_master_columns = [col_info['name'] for col_info in inspector.
            get_columns(master_table_name)]
        missing_in_db = [col for col in view_column_source_list if col not in
            db_master_columns]
        if missing_in_db:
            logger.error(
                f"Error: For view '{view_name_for_log}', the following columns specified in its source list do not exist in the master table '{master_table_name}' in the database: {missing_in_db}"
                )
            logger.error(
                f"Columns found in database table '{master_table_name}': {db_master_columns}"
                )
            return False
    except Exception as e:
        logger.error(
            f"Error inspecting columns for master table '{master_table_name}': {e}"
            )
        return False
    return True


def create_view(engine, view_name: str, master_table_name: str,
    view_column_source_list: list, all_master_cols_in_spec: list):
    if not validate_columns_exist(engine, master_table_name,
        view_column_source_list, all_master_cols_in_spec, view_name):
        logger.error(
            f"Column validation failed for view '{view_name}'. Skipping view creation."
            )
        return False
    cols_str = ', '.join([f'`{col}`' for col in view_column_source_list])
    sql_view_statement = (
        f'CREATE OR REPLACE VIEW `{view_name}` AS SELECT {cols_str} FROM `{master_table_name}`'
        )
    try:
        with engine.connect() as connection:
            connection.execute(text(sql_view_statement))
            connection.commit()
        logger.info(
            f"Successfully created/replaced view '{view_name}' with {len(view_column_source_list)} columns from '{master_table_name}'."
            )
        return True
    except Exception as e:
        logger.error(f"Error creating view '{view_name}': {e}")
        logger.error(f'Failed SQL: {sql_view_statement}')
        return False


def main():
    if not EVALUATION_SPEC_PATH.exists():
        logger.error(f'Evaluation spec file not found: {EVALUATION_SPEC_PATH}')
        sys.exit(1)
    try:
        with open(EVALUATION_SPEC_PATH, 'r', encoding='utf-8') as f:
            spec = json.load(f)
    except Exception as e:
        logger.error(f'Error loading or parsing {EVALUATION_SPEC_PATH}: {e}')
        sys.exit(1)
    db_engine = get_db_engine(DB_CONFIG)
    if not db_engine:
        sys.exit(1)
    try:
        master_table = spec['database']['MASTER_TABLE_NAME']
        host_view_name = spec['database']['TABLE_HOST']
        remote_view_name = spec['database']['TABLE_REMOTE']
        surrogate_cols = spec['columns'].get('FOR_SURROGATE_HOST_DATA_ALL_COLS'
            , [])
        host_view_cols_source_list = list(dict.fromkeys(spec['columns'][
            'HOST_DATA_ALL_COLS'] + surrogate_cols))
        remote_view_cols_source_list = spec['columns']['REMOTE_ALL_COLS']
        all_master_cols_in_spec = spec['columns']['ALL_MASTER_TABLE_COLS']
    except KeyError as e:
        logger.error(f'Missing critical key in {EVALUATION_SPEC_PATH}: {e}')
        sys.exit(1)
    logger.info(f"Master table: '{master_table}'")
    logger.info(
        f"Host view to create: '{host_view_name}' using columns defined in 'HOST_DATA_ALL_COLS'"
        )
    logger.info(
        f"Remote view to create: '{remote_view_name}' using columns defined in 'REMOTE_ALL_COLS'"
        )
    success_host = create_view(db_engine, host_view_name, master_table,
        host_view_cols_source_list, all_master_cols_in_spec)
    success_remote = create_view(db_engine, remote_view_name, master_table,
        remote_view_cols_source_list, all_master_cols_in_spec)
    if success_host and success_remote:
        logger.info('View creation process completed successfully.')
    else:
        logger.error('View creation process encountered errors.')
    if db_engine:
        db_engine.dispose()


if __name__ == '__main__':
    main()
