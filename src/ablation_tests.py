import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss, average_precision_score
try:
    from l2 import _evaluate_and_save_l2_metrics
    from utils import preprocess_features_for_lgbm, train_lgbm_model, sanitize_feature_names, get_predictions, evaluate_predictions, standardise_key, load_json_config, NpEncoder, ensure_dir_exists, save_artifact, load_artifact
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        'Could not import all shared utilities. Some functions might be redefined locally or unavailable.'
        )
if 'logger' not in globals():
    logging.basicConfig(level=logging.INFO, format=
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger = logging.getLogger(__name__)
ABLATION_OUTPUT_SUBDIR = '5_ablation_outputs'
GLOBAL_SEED = 42


def _prepare_ablation_dataframe(df_full_features: pd.DataFrame,
    df_target_full: pd.DataFrame, l1_splits: Dict, vfl_key: str, target_col:
    str, task_name: str) ->Tuple[Optional[pd.DataFrame], Optional[pd.Series
    ], Optional[pd.DataFrame], Optional[pd.Series]]:
    logger.info(f'[{task_name}] Preparing data for ablation...')
    df_full_features = df_full_features.copy()
    df_target_full = df_target_full.copy()
    for _df in (df_full_features, df_target_full):
        if vfl_key not in _df.columns:
            logger.error(
                f"[{task_name}] VFL_KEY '{vfl_key}' missing. Aborting data prep."
                )
            return None, None, None, None
        _df[vfl_key] = _df[vfl_key].astype(str)
    df_full_features = standardise_key(df_full_features, vfl_key).reset_index(
        drop=True)
    df_target_full = standardise_key(df_target_full, vfl_key).reset_index(drop
        =True)
    df_merged = pd.merge(df_full_features, df_target_full[[vfl_key,
        target_col]], on=vfl_key, how='inner')
    if df_merged.empty:
        logger.error(
            f'[{task_name}] Data is empty after merging features and target. Aborting.'
            )
        return None, None, None, None
    df_merged = standardise_key(df_merged, vfl_key)
    train_idx_keys = pd.Index(list(map(str, l1_splits.get('train_idx', []))))
    test_idx_keys = pd.Index(list(map(str, l1_splits.get('test_idx', []))))
    train_idx_present = train_idx_keys.intersection(df_merged.index)
    test_idx_present = test_idx_keys.intersection(df_merged.index)
    if train_idx_present.empty:
        logger.error(
            f'[{task_name}] Training split is empty after aligning L1 indices. Aborting.'
            )
        return None, None, None, None
    df_train = df_merged.loc[train_idx_present].copy().reset_index(drop=True)
    y_train = df_train.pop(target_col)
    df_test = pd.DataFrame()
    y_test = pd.Series(dtype=y_train.dtype)
    if not test_idx_present.empty:
        df_test = df_merged.loc[test_idx_present].copy().reset_index(drop=True)
        y_test = df_test.pop(target_col)
    else:
        logger.warning(f'[{task_name}] Test split is empty for ablation.')
    return df_train, y_train, df_test, y_test


def run_ablation1_weighted_host0(task_name: str, task_def: Dict, configs:
    Dict, eval_spec: Dict, paths: Dict):
    logger.info(f'[{task_name}] === Starting Ablation 1: Weighted Host0 ===')
    l1_cfg = configs['l1']
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    target_col = eval_spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    ablation_base_dir = paths['s1_in'].parent / ABLATION_OUTPUT_SUBDIR
    ablation_out_dir = ablation_base_dir / 'ablation1_weighted_host0'
    ensure_dir_exists(ablation_out_dir)
    df_host_with_all_surrogates = load_artifact(paths['s1_in'] / l1_cfg[
        'artifact_names']['host_data_with_local_surrogates_csv'])
    host_native_num_cols = eval_spec['columns'][
        'HOST_NUMERIC_COLS_FOR_MODELING']
    host_native_cat_cols = eval_spec['columns'][
        'HOST_CATEGORICAL_COLS_FOR_MODELING']
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    native_cols = [vfl_key] + host_native_num_cols + host_native_cat_cols
    df_host_native_full = df_host_with_all_surrogates[[col for col in
        native_cols if col in df_host_with_all_surrogates.columns]]
    df_target_full = load_artifact(paths['s1_in'] / l1_cfg['artifact_names'
        ]['host_target_cols_csv'])
    l1_splits = load_artifact(paths['s1_in'] /
        'host_train_val_test_indices.json')
    host0_bat = load_artifact(paths['s1_in'] / l1_cfg['artifact_names'][
        'host0_bat_model_pkl'])
    error_leaf_ids_list = load_artifact(paths['s1_in'] / l1_cfg[
        'artifact_names']['host_error_leaf_ids_json'])
    error_leaf_ids = set(map(int, error_leaf_ids_list)
        ) if error_leaf_ids_list else set()
    if (df_host_native_full is None or df_target_full is None or l1_splits is
        None or host0_bat is None):
        logger.error(
            f'[{task_name}] Missing critical artifacts for Ablation 1. Aborting.'
            )
        return
    df_train_native, y_train, df_test_native, y_test = (
        _prepare_ablation_dataframe(df_host_native_full, df_target_full,
        l1_splits, vfl_key, target_col, task_name + '_Ablation1'))
    if df_train_native is None:
        return
    host_num_cols = [c for c in eval_spec['columns'][
        'HOST_NUMERIC_COLS_FOR_MODELING'] if c in df_train_native.columns and
        c != vfl_key]
    host_cat_cols = [c for c in eval_spec['columns'][
        'HOST_CATEGORICAL_COLS_FOR_MODELING'] if c in df_train_native.
        columns and c != vfl_key]
    sample_weights = pd.Series(1.0, index=df_train_native.index)
    if error_leaf_ids and not df_train_native.empty:
        train_features_for_bat = df_train_native[host_num_cols + host_cat_cols]
        if hasattr(host0_bat, 'bat_preprocessor_'
            ) and train_features_for_bat.shape[1] > 0:
            try:
                X_bat_transformed = host0_bat.bat_preprocessor_.transform(
                    train_features_for_bat)
                leaf_id_preds_train = host0_bat.apply(X_bat_transformed).ravel(
                    )
                error_mask_train = np.isin(leaf_id_preds_train, list(
                    error_leaf_ids))
                sample_weights[error_mask_train] = 2.0
                logger.info(
                    f'[{task_name}] Applied sample weights. {error_mask_train.sum()} instances upweighted.'
                    )
            except Exception as e:
                logger.error(
                    f'[{task_name}] Failed to apply Host0 BAT for sample weighting: {e}. Using uniform weights.'
                    )
        else:
            logger.warning(
                f'[{task_name}] Host0 BAT preprocessor missing or no features for BAT. Using uniform weights.'
                )
    X_train_proc, preprocessor = preprocess_features_for_lgbm(df_train_native
        [host_num_cols + host_cat_cols], host_num_cols, host_cat_cols,
        f'{task_name}_Ablation1_Host0Weighted', fit_mode=True)
    if X_train_proc is None or X_train_proc.empty:
        logger.error(
            f'[{task_name}] Preprocessing failed for Ablation 1. Aborting.')
        return
    host0_lgbm_params = l1_cfg['host_model_lgbm_params'].copy()
    weighted_host0_model = train_lgbm_model(X_train_proc, y_train,
        host0_lgbm_params, task_def['type'], sample_weight=sample_weights)
    save_artifact(weighted_host0_model, ablation_out_dir /
        'ablation1_weighted_host0_model.pkl', 'pkl')
    if preprocessor:
        save_artifact(preprocessor, ablation_out_dir /
            'ablation1_weighted_host0_preproc.pkl', 'pkl')
    if not df_test_native.empty:
        X_test_proc, _ = preprocess_features_for_lgbm(df_test_native[
            host_num_cols + host_cat_cols], host_num_cols, host_cat_cols,
            f'{task_name}_Ablation1_Host0Weighted_Test', fit_mode=False,
            existing_num_preprocessor=preprocessor)
        if X_test_proc is not None and not X_test_proc.empty:
            preds_test = get_predictions(weighted_host0_model, X_test_proc,
                task_def['type'])
            metrics_cfg_list = configs['l2'].get('evaluation', {}).get(
                'metrics', ['roc_auc_score'])
            _evaluate_and_save_l2_metrics(y_test, pd.Series(preds_test,
                index=y_test.index), task_def['type'], metrics_cfg_list, 
                ablation_out_dir / 'ablation1_metrics.json', task_name +
                '_Ablation1')
    logger.info(
        f'[{task_name}] Finished Ablation 1. Results in {ablation_out_dir}')


def run_ablation2_rf_with_surrogates(task_name: str, task_def: Dict,
    configs: Dict, eval_spec: Dict, paths: Dict):
    logger.info(
        f'[{task_name}] === Starting Ablation 2: Random Forest with Surrogates ==='
        )
    l1_cfg = configs['l1']
    l2_cfg = configs['l2']
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    target_col = eval_spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    ablation_out_dir = paths['s1_in'
        ].parent / ABLATION_OUTPUT_SUBDIR / 'ablation2_rf_with_surrogates'
    ensure_dir_exists(ablation_out_dir)
    df_host_with_surrogates = load_artifact(paths['s1_in'] / l1_cfg[
        'artifact_names']['host_data_with_local_surrogates_csv'])
    df_target_full = load_artifact(paths['s1_in'] / l1_cfg['artifact_names'
        ]['host_target_cols_csv'])
    l1_splits = load_artifact(paths['s1_in'] / l1_cfg['artifact_names'][
        'split_indices_json'])
    if (df_host_with_surrogates is None or df_target_full is None or 
        l1_splits is None):
        logger.error(
            f'[{task_name}] Missing critical artifacts for Ablation 2. Aborting.'
            )
        return
    df_train_aug, y_train, df_test_aug, y_test = _prepare_ablation_dataframe(
        df_host_with_surrogates, df_target_full, l1_splits, vfl_key,
        target_col, task_name + '_Ablation2')
    if df_train_aug is None:
        return
    all_features = [col for col in df_train_aug.columns if col != vfl_key]
    numeric_features_rf = df_train_aug[all_features].select_dtypes(include=
        np.number).columns.tolist()
    categorical_features_rf = df_train_aug[all_features].select_dtypes(exclude
        =np.number).columns.tolist()
    df_train_rf_proc = df_train_aug.copy()
    if numeric_features_rf:
        num_imputer = SimpleImputer(strategy='median')
        df_train_rf_proc[numeric_features_rf] = num_imputer.fit_transform(
            df_train_rf_proc[numeric_features_rf])
    if categorical_features_rf:
        for col in categorical_features_rf:
            df_train_rf_proc[col] = df_train_rf_proc[col].astype(str).fillna(
                '!MISSING!')
        df_train_rf_proc = pd.get_dummies(df_train_rf_proc, columns=
            categorical_features_rf, dummy_na=False)
    cols_for_rf_train = [c for c in df_train_rf_proc.columns if c !=
        vfl_key and c != target_col]
    df_train_rf_proc = df_train_rf_proc[cols_for_rf_train]
    df_train_rf_proc = df_train_rf_proc.loc[:, ~df_train_rf_proc.columns.
        duplicated()]
    df_train_rf_proc.columns = sanitize_feature_names(df_train_rf_proc.columns)
    train_cols = df_train_rf_proc.columns.tolist()
    rf_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 
        5, 'random_state': GLOBAL_SEED, 'n_jobs': -1, 'class_weight':
        'balanced_subsample'}
    rf_model = RandomForestClassifier(**rf_params)
    logger.info(
        f'[{task_name}] Training RandomForest for Ablation 2 with {df_train_rf_proc.shape[1]} features.'
        )
    rf_model.fit(df_train_rf_proc, y_train)
    save_artifact(rf_model, ablation_out_dir / 'ablation2_rf_model.pkl', 'pkl')
    if not df_test_aug.empty:
        df_test_rf_proc = df_test_aug.copy()
        if numeric_features_rf:
            df_test_rf_proc[numeric_features_rf] = num_imputer.transform(
                df_test_rf_proc[numeric_features_rf])
        if categorical_features_rf:
            for col in categorical_features_rf:
                df_test_rf_proc[col] = df_test_rf_proc[col].astype(str).fillna(
                    '!MISSING!')
            df_test_rf_proc = pd.get_dummies(df_test_rf_proc, columns=
                categorical_features_rf, dummy_na=False)
        df_test_rf_proc.columns = sanitize_feature_names(df_test_rf_proc.
            columns)
        df_test_rf_proc = df_test_rf_proc.reindex(columns=df_train_rf_proc.
            columns, fill_value=0)
        cols_for_rf_test = [c for c in df_test_rf_proc.columns if c !=
            vfl_key and c != target_col]
        df_test_rf_proc = df_test_rf_proc[cols_for_rf_test]
        preds_test_rf = get_predictions(rf_model, df_test_rf_proc, task_def
            ['type'])
        metrics_cfg_list = configs['l2'].get('evaluation', {}).get('metrics',
            ['roc_auc_score'])
        _evaluate_and_save_l2_metrics(y_test, pd.Series(preds_test_rf,
            index=y_test.index), task_def['type'], metrics_cfg_list, 
            ablation_out_dir / 'ablation2_metrics.json', task_name +
            '_Ablation2')
    logger.info(
        f'[{task_name}] Finished Ablation 2. Results in {ablation_out_dir}')


def main():
    script_dir = Path(__file__).resolve().parent
    l1_cfg = load_json_config(script_dir / 'l1_config.json')
    l2_cfg = load_json_config(script_dir / 'l2_config.json')
    eval_spec = load_json_config(script_dir / 'evaluation_spec.json')
    if not all([l1_cfg, l2_cfg, eval_spec]):
        logger.error('Failed to load configuration files')
        sys.exit(1)
    global GLOBAL_SEED
    GLOBAL_SEED = eval_spec.get('seed', l1_cfg.get('seed', l2_cfg.get(
        'seed', 42)))
    np.random.seed(GLOBAL_SEED)
    logger.info(f'Global seed for ablations: {GLOBAL_SEED}')
    configs = {'l1': l1_cfg, 'l2': l2_cfg}
    for task_name, task_def in eval_spec.get('tasks', {}).items():
        logger.info(f'===== Running Ablations for Task: {task_name} =====')
        task_def['type'] = task_def.get('type', 'binary')
        dataset_output_root = Path(eval_spec['dataset_output_dir_name'])
        task_root = dataset_output_root / task_name
        paths = {'s1_in': task_root / l1_cfg[
            'script1_output_subdir_template'].rstrip('/')}
        ensure_dir_exists(paths['s1_in'].parent / ABLATION_OUTPUT_SUBDIR)
        try:
            logger.info(f'[{task_name}] Running Ablation 1...')
            run_ablation1_weighted_host0(task_name, task_def, configs,
                eval_spec, paths)
            logger.info(f'[{task_name}] Running Ablation 2...')
            run_ablation2_rf_with_surrogates(task_name, task_def, configs,
                eval_spec, paths)
        except Exception as e:
            logger.error(f"Ablation error for task '{task_name}': {e}",
                exc_info=True)
    logger.info('All ablation tests completed.')


if __name__ == '__main__':
    main()
