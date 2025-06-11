import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
from collections import defaultdict
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy.stats import t
from sqlalchemy import create_engine, text
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss, average_precision_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from utils import standardise_key, sanitize_feature_names, load_artifact, sanitize_feature_name, load_json_config, ensure_dir_exists, NpEncoder
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _load_l1_defined_indices(task_name: str, spec: Dict, base_cfg: Dict,
    task_output_root_dir: Path) ->Tuple[Optional[pd.Index], Optional[pd.
    Index], Optional[pd.Index]]:
    l1_subdir_template = base_cfg.get('l1_script_output_subdir_template',
        '1_teacher_discovery_vfl/')
    split_file_name = base_cfg.get('l1_split_indices_artifact_name',
        'host_train_val_test_indices.json')
    l1_splits_path = (task_output_root_dir / task_name / l1_subdir_template /
        split_file_name)
    l1_splits_data = load_artifact(l1_splits_path)
    if not l1_splits_data:
        logger.error(
            f'[{task_name}] Failed to load L1 splits from: {l1_splits_path}')
        return None, None, None
    train_keys = pd.Index(map(str, l1_splits_data.get('train_idx', [])))
    val_keys = pd.Index(map(str, l1_splits_data.get('val_idx', [])))
    test_keys = pd.Index(map(str, l1_splits_data.get('test_idx', [])))
    if train_keys.empty or test_keys.empty:
        logger.error(
            f'[{task_name}] L1 train or test splits are empty. Cannot proceed.'
            )
        return None, None, None
    logger.info(
        f'[{task_name}] Loaded L1 splits: Train({len(train_keys)}), Val({len(val_keys)}), Test({len(test_keys)})'
        )
    return train_keys, val_keys, test_keys


def _fit_and_evaluate_model_on_split(model_factory: Callable, X_train: pd.
    DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    is_clf: bool, cfg: Dict, task: str, mode: str) ->Tuple[Dict[str, Any],
    Optional[List[Tuple[str, float]]]]:
    if X_train.empty or y_train.empty:
        logger.warning(
            f'[{task}-{mode}] Training data empty. Skipping model fit/eval.')
        return {m: np.nan for m in cfg['evaluation_metrics'] if not m.
            endswith(('_std', '_ci_low', '_ci_up'))}, None
    model = model_factory()
    is_lgbm = isinstance(model, (LGBMClassifier, LGBMRegressor))
    X_train_proc, X_test_proc = X_train.copy(), X_test.copy()
    if is_lgbm:
        for col in X_train_proc.select_dtypes(include=['object']).columns:
            X_train_proc[col] = X_train_proc[col].astype('category')
        for col in X_test_proc.select_dtypes(include=['object']).columns:
            X_test_proc[col] = X_test_proc[col].astype('category')
        X_train_proc.columns = sanitize_feature_names(X_train_proc.columns)
        X_test_proc.columns = sanitize_feature_names(X_test_proc.columns)
    elif isinstance(model, MLPRegressor):
        prep = _create_sklearn_preprocessor(X_train_proc, task, mode)
        try:
            X_train_proc = pd.DataFrame(prep.fit_transform(X_train_proc),
                columns=sanitize_feature_names(prep.get_feature_names_out()
                ), index=X_train_proc.index)
            X_test_proc = pd.DataFrame(prep.transform(X_test_proc), columns
                =sanitize_feature_names(prep.get_feature_names_out()),
                index=X_test_proc.index
                ) if not X_test_proc.empty else pd.DataFrame(columns=
                X_train_proc.columns)
        except Exception as e_prep:
            logger.error(
                f'[{task}-{mode}] Preprocessing error for single split: {e_prep}. Returning NaNs.'
                )
            return {m: np.nan for m in cfg['evaluation_metrics'] if not m.
                endswith(('_std', '_ci_low', '_ci_up'))}, None
    fit_success, top_features = _fit_model(model, X_train_proc, y_train,
        task, mode, fold=0)
    if not fit_success:
        return {m: np.nan for m in cfg['evaluation_metrics'] if not m.
            endswith(('_std', '_ci_low', '_ci_up'))}, None
    if X_test_proc.empty or y_test.empty:
        logger.warning(
            f'[{task}-{mode}] Test data empty for evaluation. Returning NaNs.')
        return {m: np.nan for m in cfg['evaluation_metrics'] if not m.
            endswith(('_std', '_ci_low', '_ci_up'))}, top_features
    labels, probas = _get_predictions(model, X_test_proc, is_clf, task,
        mode, fold=0)
    test_metrics = _calculate_fold_metrics(y_test, labels, probas, is_clf,
        cfg['evaluation_metrics'], task, mode, fold=0)
    final_metrics = {k: v for k, v in test_metrics.items() if not k.
        endswith(('_std', '_ci_low', '_ci_up'))}
    return final_metrics, top_features


def _finalize_metrics_for_single_eval(test_set_metrics: Dict[str, float],
    feature_importances: Optional[List[Tuple[str, float]]],
    metrics_cfg_list: List[str]) ->Dict[str, Any]:
    final_metrics = {}
    for metric_name in metrics_cfg_list:
        if metric_name.endswith(('_std', '_ci_low', '_ci_up')):
            final_metrics[metric_name] = np.nan
        else:
            final_metrics[metric_name] = test_set_metrics.get(metric_name,
                np.nan)
    if feature_importances:
        if feature_importances and isinstance(feature_importances[0], tuple):
            final_metrics['top_5_features_by_importance'] = [{'feature':
                sanitize_feature_name(f), 'importance': float(v)} for f, v in
                feature_importances[:5]]
        else:
            final_metrics['top_5_features_by_importance'
                ] = feature_importances[:5]
    else:
        final_metrics['top_5_features_by_importance'] = []
    return final_metrics


def _process_simple_baseline_on_split(df_train_val: pd.DataFrame,
    y_train_val: pd.Series, df_test: pd.DataFrame, y_test: pd.Series, sets:
    Dict[str, List[str]], cfg: Dict, spec: Dict, task_name: str, is_clf:
    bool, mode_name: str) ->Dict[str, Any]:
    logger.info(
        f'[{task_name}] Processing Simple Baseline (on split): {mode_name}')
    X_train_val = _prepare_feature_df_for_mode(df_train_val, mode_name,
        sets, task_name)
    X_test = _prepare_feature_df_for_mode(df_test, mode_name, sets, task_name)
    if X_train_val.empty:
        logger.warning(
            f'[{task_name}-{mode_name}] X_train_val empty. Returning NaN metrics.'
            )
        nan_metrics = {m: np.nan for m in cfg['evaluation_metrics'] if not
            m.endswith(('_std', '_ci_low', '_ci_up'))}
        return _finalize_metrics_for_single_eval(nan_metrics, None, cfg[
            'evaluation_metrics'])
    lgbm_params = _get_lgbm_params(cfg)
    model_factory = lambda : LGBMClassifier(**lgbm_params
        ) if is_clf else LGBMRegressor(**lgbm_params)
    test_metrics, top_features = _fit_and_evaluate_model_on_split(model_factory
        , X_train_val, y_train_val, X_test, y_test, is_clf, cfg, task_name,
        mode_name)
    return _finalize_metrics_for_single_eval(test_metrics, top_features,
        cfg['evaluation_metrics'])


def _save_metrics_to_json(metrics: Dict, path: Path, task: str, mode: str):
    ensure_dir_exists(path.parent)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, cls=NpEncoder)
        logger.info(f'[{task}-{mode}] Metrics saved to {path}')
    except Exception as e:
        logger.error(f'[{task}-{mode}] Error saving metrics: {e}', exc_info
            =True)


def _process_transfer_learning_baseline_on_split(df_train_val_full: pd.
    DataFrame, y_train_val_full: pd.Series, df_test_full: pd.DataFrame,
    y_test_full: pd.Series, sets: Dict[str, List[str]], cfg: Dict, spec:
    Dict, task_name: str, is_clf: bool, mode_name: str, transfer_type: str
    ) ->Dict[str, Any]:
    logger.info(
        f'[{task_name}] Processing Transfer Baseline (on split): {mode_name} ({transfer_type})'
        )
    vfl_key = spec['column_identifiers']['VFL_KEY']
    kb_feat_cols = [c for c in sets['all_kb'] if c in df_train_val_full.columns
        ]
    X_kb_train_val = df_train_val_full[kb_feat_cols].copy()
    host_feat_cols = [c for c in sets['all_host'] if c in df_train_val_full
        .columns]
    X_host_train_val = df_train_val_full[host_feat_cols].copy()
    oof_kb_derived_for_train_val = pd.DataFrame(index=X_kb_train_val.index)
    cv_splitter_internal = _get_cv_splitter(cfg['n_splits_cv'], cfg['seed'],
        is_clf, y_train_val_full)
    first_fold_processed = False
    for fold_idx, (tr_fold_indices, val_fold_indices) in enumerate(
        cv_splitter_internal.split(X_kb_train_val, y_train_val_full if 
        is_clf and y_train_val_full.nunique() > 1 else None)):
        X_kb_tr_fold = X_kb_train_val.iloc[tr_fold_indices]
        y_tr_fold = y_train_val_full.iloc[tr_fold_indices]
        X_kb_val_fold = X_kb_train_val.iloc[val_fold_indices]
        _, kb_derived_val_fold_outputs = _train_kb_encoder_or_teacher(
            X_kb_tr_fold, y_tr_fold, X_kb_val_fold, is_clf, cfg, task_name,
            f'{mode_name}_oof_gen', transfer_type)
        if (kb_derived_val_fold_outputs is None or
            kb_derived_val_fold_outputs.empty):
            logger.warning(
                f'[{task_name}-{mode_name}] OOF KB outputs failed for fold {fold_idx}. Skipping OOF for these instances.'
                )
            continue
        if not first_fold_processed:
            for col_name in kb_derived_val_fold_outputs.columns:
                oof_kb_derived_for_train_val[col_name] = np.nan
            first_fold_processed = True
        for col_name in kb_derived_val_fold_outputs.columns:
            if col_name in oof_kb_derived_for_train_val.columns:
                oof_kb_derived_for_train_val.loc[X_kb_val_fold.index, col_name
                    ] = kb_derived_val_fold_outputs[col_name].values
            else:
                logger.warning(
                    f'[{task_name}-{mode_name}] Column mismatch {col_name} in OOF generation.'
                    )
    for col in oof_kb_derived_for_train_val.columns:
        if oof_kb_derived_for_train_val[col].isnull().any():
            oof_kb_derived_for_train_val[col] = oof_kb_derived_for_train_val[
                col].fillna(oof_kb_derived_for_train_val[col].median())
            logger.warning(
                f'[{task_name}-{mode_name}] NaNs found in OOF KB derived column {col}, filled with median.'
                )
    if not first_fold_processed or oof_kb_derived_for_train_val.empty:
        logger.error(
            f'[{task_name}-{mode_name}] No OOF KB-derived features were generated for train+val. Cannot proceed.'
            )
        nan_metrics = {m: np.nan for m in cfg['evaluation_metrics'] if not
            m.endswith(('_std', '_ci_low', '_ci_up'))}
        return _finalize_metrics_for_single_eval(nan_metrics, None, cfg[
            'evaluation_metrics'])
    X_kb_test = df_test_full[[c for c in sets['all_kb'] if c in
        df_test_full.columns]].copy()
    _, kb_derived_for_test = _train_kb_encoder_or_teacher(X_kb_train_val,
        y_train_val_full, X_kb_test, is_clf, cfg, task_name,
        f'{mode_name}_test_gen', transfer_type)
    if kb_derived_for_test is None:
        logger.warning(
            f'[{task_name}-{mode_name}] KB derived features for test set failed. Using zeros.'
            )
        kb_derived_for_test = pd.DataFrame(0, index=X_kb_test.index,
            columns=oof_kb_derived_for_train_val.columns)
    mimic_model_tuple = _train_host_mimic_mlp(X_host_train_val,
        oof_kb_derived_for_train_val, cfg, task_name, mode_name)
    if mimic_model_tuple is None:
        logger.error(
            f'[{task_name}-{mode_name}] Mimic training failed. Cannot proceed.'
            )
        nan_metrics = {m: np.nan for m in cfg['evaluation_metrics'] if not
            m.endswith(('_std', '_ci_low', '_ci_up'))}
        return _finalize_metrics_for_single_eval(nan_metrics, None, cfg[
            'evaluation_metrics'])
    host_mimic_model, host_mimic_preprocessor = mimic_model_tuple
    mimic_preds_for_train_val = _get_host_mimic_predictions(host_mimic_model,
        host_mimic_preprocessor, X_host_train_val, list(
        oof_kb_derived_for_train_val.columns), task_name, mode_name)
    X_host_test = df_test_full[[c for c in sets['all_host'] if c in
        df_test_full.columns]].copy()
    mimic_preds_for_test = _get_host_mimic_predictions(host_mimic_model,
        host_mimic_preprocessor, X_host_test, list(
        oof_kb_derived_for_train_val.columns), task_name, mode_name)
    X_student_train = _prepare_feature_df_for_mode(X_host_train_val,
        mode_name, sets, task_name, pred_logits=mimic_preds_for_train_val.
        iloc[:, 0] if transfer_type == 'logits' and not
        mimic_preds_for_train_val.empty else None, pred_embeds=
        mimic_preds_for_train_val if transfer_type != 'logits' and not
        mimic_preds_for_train_val.empty else None)
    X_student_test = _prepare_feature_df_for_mode(X_host_test, mode_name,
        sets, task_name, pred_logits=mimic_preds_for_test.iloc[:, 0] if 
        transfer_type == 'logits' and not mimic_preds_for_test.empty else
        None, pred_embeds=mimic_preds_for_test if transfer_type != 'logits' and
        not mimic_preds_for_test.empty else None)
    if X_student_train.empty:
        logger.error(
            f'[{task_name}-{mode_name}] X_student_train empty after feature prep. Cannot proceed.'
            )
        nan_metrics = {m: np.nan for m in cfg['evaluation_metrics'] if not
            m.endswith(('_std', '_ci_low', '_ci_up'))}
        return _finalize_metrics_for_single_eval(nan_metrics, None, cfg[
            'evaluation_metrics'])
    lgbm_student_params = _get_lgbm_params(cfg)
    student_model_factory = lambda : LGBMClassifier(**lgbm_student_params
        ) if is_clf else LGBMRegressor(**lgbm_student_params)
    test_metrics, top_features = _fit_and_evaluate_model_on_split(
        student_model_factory, X_student_train, y_train_val_full,
        X_student_test, y_test_full, is_clf, cfg, task_name,
        f'{mode_name}_student_final')
    return _finalize_metrics_for_single_eval(test_metrics, top_features,
        cfg['evaluation_metrics'])


DB_ENGINE_CACHE: Dict[str, Any] = {}


def _get_db_engine(module_name: str, path_str: Optional[str]) ->Optional[Any]:
    key = f'{module_name}_{path_str}'
    if key in DB_ENGINE_CACHE:
        return DB_ENGINE_CACHE[key]
    try:
        if path_str and path_str not in sys.path:
            sys.path.insert(0, path_str)
        module = __import__(module_name)
        cfg = getattr(module, 'DB_CONFIG', None)
        if not cfg:
            logger.error(f"DB_CONFIG not in '{module_name}.py'")
            return None
        uri = (
            f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
            )
        engine = create_engine(uri, pool_recycle=3600, connect_args={
            'connect_timeout': 10})
        DB_ENGINE_CACHE[key] = engine
        logger.info(f"DB engine created for {cfg['database']}")
        return engine
    except Exception as e:
        logger.error(f'Error creating DB engine for {module_name}: {e}',
            exc_info=True)
        return None


def _read_sql_to_df(query: str, engine: Any, task: str) ->pd.DataFrame:
    if not engine:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(text(query), engine)
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].tolist()
            logger.warning(
                f'[{task}] Duplicate SQL cols: {dupes}. Keeping first.')
            df = df.loc[:, ~df.columns.duplicated()]
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].apply(lambda x: isinstance(x, Decimal)).any():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.info(f'[{task}] Loaded SQL data, shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'[{task}] Error loading SQL data: {e}', exc_info=True)
        return pd.DataFrame()


def _parse_target_def(def_str: str, task: str) ->Callable:
    try:
        return eval(def_str)
    except Exception as e:
        logger.error(f"[{task}] Error parsing target def '{def_str}': {e}")
        raise


def _apply_target_def(df: pd.DataFrame, func: Callable, task: str, name: str
    ) ->pd.Series:
    try:
        return pd.Series(np.asarray(func(df.copy())), index=df.index, name=name
            )
    except Exception as e:
        logger.error(f'[{task}] Error applying target def: {e}', exc_info=True)
        raise


def _filter_and_clean_data(df: pd.DataFrame, y: pd.Series, spec: Dict, task:
    str) ->Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.
    Index]]:
    nan_mask = y.isna()
    if nan_mask.all():
        logger.warning(f'[{task}] All target values NaN. Skipping.')
        return None, None, None
    df_c = df.loc[~nan_mask].copy()
    y_c = y.loc[~nan_mask].copy()
    orig_idx = df_c.index.copy()
    leaks = set(spec['tasks'][task].get('leak_columns', []))
    leaks.update(spec['global_settings'].get('GLOBAL_LEAK_COLS', []))
    df_c.drop(columns=[c for c in leaks if c in df_c.columns], inplace=True,
        errors='ignore')
    target_name = spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    if target_name in df_c.columns:
        df_c.drop(columns=[target_name], inplace=True, errors='ignore')
    vfl_key = spec['column_identifiers']['VFL_KEY']
    if vfl_key in df_c.columns and df_c.index.name == vfl_key:
        df_c.drop(columns=[vfl_key], inplace=True, errors='ignore')
    if df_c.empty:
        logger.warning(f'[{task}] DataFrame empty after NaN/leak removal.')
        return None, None, None
    return df_c, y_c, orig_idx


def _process_initial_data_for_task(task_name: str, task_def: Dict, spec:
    Dict, engine: Any) ->Tuple[Optional[pd.DataFrame], Optional[pd.Series],
    Optional[pd.Index]]:
    query = task_def['sql_query'].format(TABLE_HOST=spec['database'][
        'TABLE_HOST'], TABLE_REMOTE=spec['database']['TABLE_REMOTE'])
    target_func = _parse_target_def(task_def['target_definition'], task_name)
    df_orig = _read_sql_to_df(query, engine, task_name)
    if df_orig.empty:
        return None, None, None
    vfl_key = spec['column_identifiers']['VFL_KEY']
    if vfl_key not in df_orig.columns:
        logger.error(
            f"[{task_name}] VFL_KEY '{vfl_key}' not in data. Cols: {df_orig.columns.tolist()}"
            )
        return None, None, None
    if df_orig[vfl_key].duplicated().any():
        logger.warning(
            f"[{task_name}] Duplicate VFL_KEY '{vfl_key}'. Keeping first.")
        df_orig = df_orig.drop_duplicates(subset=[vfl_key], keep='first')
    df_orig = standardise_key(df_orig, vfl_key)
    target_col_name = spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    y_series = _apply_target_def(df_orig, target_func, task_name,
        target_col_name)
    return _filter_and_clean_data(df_orig, y_series, spec, task_name)


def _get_cols_from_spec(key: str, spec_cols: Dict, available_cols: List[str
    ], task: str) ->List[str]:
    cols = [c for c in spec_cols.get(key, []) if c in available_cols]
    if not cols and spec_cols.get(key):
        logger.debug(
            f"[{task}] Spec key '{key}': none of {spec_cols.get(key)} in available data."
            )
    return cols


def _identify_feature_sets(df_cols: pd.Index, spec: Dict, task: str) ->Dict[
    str, List[str]]:
    spec_cols = spec['columns']
    non_feature_cols = {spec['column_identifiers']['VFL_KEY'], spec[
        'global_settings']['TARGET_COLUMN_NAME_PROCESSED']}
    non_feature_cols.add(spec['tasks'][task].get('target_raw_column'))
    potential = [c for c in df_cols if c not in non_feature_cols and c is not
        None]
    sets = {'host_num': _get_cols_from_spec(
        'HOST_NUMERIC_COLS_FOR_MODELING', spec_cols, potential, task),
        'host_cat': _get_cols_from_spec(
        'HOST_CATEGORICAL_COLS_FOR_MODELING', spec_cols, potential, task),
        'host_derived_num': _get_cols_from_spec(
        'DERIVED_HOST_FEATURES_FOR_MODELING', spec_cols, potential, task),
        'kb_num': _get_cols_from_spec('REMOTE_NUMERIC_COLS_FOR_MODELING',
        spec_cols, potential, task), 'kb_cat': _get_cols_from_spec(
        'REMOTE_CATEGORICAL_COLS_FOR_MODELING', spec_cols, potential, task)}
    sets['all_host'] = list(set(sets['host_num'] + sets['host_cat'] + sets[
        'host_derived_num']))
    sets['all_kb'] = list(set(sets['kb_num'] + sets['kb_cat']))
    logger.info(
        f"[{task}] Features: Host={len(sets['all_host'])}, KB={len(sets['all_kb'])}"
        )
    return sets


def _impute_and_cast_features(X: pd.DataFrame, task: str, mode: str
    ) ->pd.DataFrame:
    Xc = X.copy()
    for col in Xc.columns:
        if Xc[col].isnull().any():
            if pd.api.types.is_numeric_dtype(Xc[col]):
                fill_val = Xc[col].median()
            else:
                mv = Xc[col].mode()
                fill_val = mv.iloc[0] if not mv.empty and pd.notna(mv.iloc[0]
                    ) else 'Missing'
            Xc[col] = Xc[col].fillna(fill_val)
        if not pd.api.types.is_numeric_dtype(Xc[col]) and Xc[col
            ].dtype.name != 'category':
            Xc[col] = Xc[col].astype('category')
    return Xc


def _prepare_feature_df_for_mode(df_full: pd.DataFrame, mode: str, sets:
    Dict[str, List[str]], task: str, pred_logits: Optional[pd.Series]=None,
    pred_embeds: Optional[pd.DataFrame]=None) ->pd.DataFrame:
    cols_map = {'host_only_all_features': sets['all_host'],
        'oracle_all_data': list(set(sets['all_host'] + sets['all_kb'])),
        'kb_pca_embeddings_fedonce_style': sets['all_host'],
        'kb_nat_embeddings': sets['all_host'],
        'student_with_distilled_logits': sets['all_host']}
    cols = cols_map.get(mode, [])
    final_cols = [c for c in cols if c in df_full.columns]
    if not final_cols and cols:
        logger.warning(
            f'[{task}-{mode}] No specified cols {cols} in DataFrame.')
    X = df_full[final_cols].copy() if final_cols else pd.DataFrame(index=
        df_full.index)
    if mode in ['kb_pca_embeddings_fedonce_style', 'kb_nat_embeddings'
        ] and pred_embeds is not None:
        X = pd.concat([X, pred_embeds.reindex(X.index)], axis=1)
    elif mode == 'student_with_distilled_logits' and pred_logits is not None:
        X['predicted_logit_feature'] = pred_logits.reindex(X.index)
    return _impute_and_cast_features(X, task, mode)


def _get_lgbm_params(cfg: Dict) ->Dict:
    return cfg.get('lgbm_params', {})


def _get_mlp_params(cfg: Dict) ->Dict:
    return cfg.get('mlp_regressor_params', {})


def _get_cv_splitter(n_splits: int, seed: int, is_clf: bool, y: Optional[pd
    .Series]=None) ->Any:
    if is_clf and y is not None and y.nunique() > 1:
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    else:
        return KFold(n_splits, shuffle=True, random_state=seed)


def _calculate_metric_stats(scores: List[float], n_splits: int, conf: float
    =0.95) ->Tuple[float, float, float, float]:
    valid_scores = [s for s in scores if pd.notna(s)]
    if not valid_scores:
        return np.nan, np.nan, np.nan, np.nan
    mean, std = float(np.mean(valid_scores)), float(np.std(valid_scores,
        ddof=1) if len(valid_scores) > 1 else 0.0)
    ci_low, ci_up = mean, mean
    if n_splits > 1 and std > 1e-09:
        std_err = std / np.sqrt(n_splits)
        dof = n_splits - 1
        if dof > 0:
            try:
                t_crit = t.ppf((1 + conf) / 2, dof)
                ci_low, ci_up = float(mean - t_crit * std_err), float(mean +
                    t_crit * std_err)
            except Exception:
                pass
    return mean, std, ci_low, ci_up


def _create_sklearn_preprocessor(X_fold: pd.DataFrame, task: str, mode: str
    ) ->ColumnTransformer:
    num_feats = X_fold.select_dtypes(include=np.number).columns.tolist()
    cat_feats = X_fold.select_dtypes(include=['object', 'category']
        ).columns.tolist()
    for col in cat_feats:
        if X_fold[col].dtype == 'bool':
            logger.debug(
                f"[{task}-{mode}] Converting boolean column '{col}' to string for OHE."
                )
            X_fold[col] = X_fold[col].astype(str)
    transformers = []
    if num_feats:
        transformers.append(('num', Pipeline([('imp', SimpleImputer(
            strategy='median')), ('std', StandardScaler())]), num_feats))
    if cat_feats:
        transformers.append(('cat', Pipeline([('imp', SimpleImputer(
            strategy='most_frequent')), ('ohe', OneHotEncoder(
            handle_unknown='ignore', sparse_output=False))]), cat_feats))
    if not transformers:
        logger.warning(
            f'[{task}-{mode}] No numeric or categorical features for sklearn preprocessor.'
            )
        return ColumnTransformer([('passthrough', 'passthrough', list(
            X_fold.columns))], sparse_threshold=0)
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0
        )


def _fit_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, task:
    str, mode: str, fold: int) ->Tuple[bool, Optional[List[Tuple[str, float]]]
    ]:
    try:
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        model.fit(X_train, y_train)
        top_features = None
        if hasattr(model, 'feature_importances_') and hasattr(model,
            'feature_name_'):
            sorted_fi = sorted(zip(model.feature_name_, model.
                feature_importances_), key=lambda t: -t[1])[:5]
            top_features = [(name, float(imp)) for name, imp in sorted_fi]
            logger.info(f'[{task}-{mode}-F{fold}] Top-5 FI: {sorted_fi}')
        return True, top_features
    except Exception as e:
        logger.error(f'[{task}-{mode}-F{fold}] Model fit error: {e}',
            exc_info=True)
        return False, None


def _get_predictions(model: Any, X_test: pd.DataFrame, is_clf: bool, task:
    str, mode: str, fold: int) ->Tuple[Optional[np.ndarray], Optional[np.
    ndarray]]:
    if X_test.empty:
        return None, None
    try:
        if isinstance(model, (LGBMClassifier, LGBMRegressor)
            ) and not isinstance(X_test, np.ndarray):
            X_test = X_test.copy()
            for col in X_test.select_dtypes(include=['object']).columns:
                X_test[col] = X_test[col].astype('category')
            X_test.columns = sanitize_feature_names(X_test.columns)
        pred_label = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1] if is_clf and hasattr(
            model, 'predict_proba') else None
        return pred_label, pred_proba
    except Exception as e:
        logger.error(f'[{task}-{mode}-F{fold}] Predict error: {e}',
            exc_info=True)
        return None, None


def _calculate_fold_metrics(y_true: pd.Series, y_pred_label: Optional[np.
    ndarray], y_pred_proba: Optional[np.ndarray], is_clf: bool,
    metric_names: List[str], task: str, mode: str, fold: int) ->Dict[str, float
    ]:
    fold_scores: Dict[str, float] = {name: np.nan for name in metric_names if
        not name.endswith(('_std', '_ci_low', '_ci_up'))}
    if y_pred_label is None and y_pred_proba is None:
        return fold_scores
    metric_fns: Dict[str, Callable] = {'roc_auc_score': roc_auc_score,
        'auprc_score': average_precision_score, 'brier_score_loss':
        brier_score_loss, 'accuracy_score': accuracy_score, 'f1_score':
        f1_score, 'precision_score': precision_score, 'recall_score':
        recall_score, 'mean_absolute_error': mean_absolute_error}
    for name in fold_scores.keys():
        if name not in metric_fns:
            continue
        try:
            if is_clf:
                if name in ['roc_auc_score', 'auprc_score']:
                    if y_pred_proba is not None and y_true.nunique() > 1:
                        fold_scores[name] = metric_fns[name](y_true,
                            y_pred_proba)
                elif name == 'brier_score_loss':
                    if y_pred_proba is not None:
                        fold_scores[name] = metric_fns[name](y_true,
                            y_pred_proba)
                elif y_pred_label is not None:
                    kwargs = {'zero_division': 0} if name in ['f1_score',
                        'precision_score', 'recall_score'] else {}
                    fold_scores[name] = metric_fns[name](y_true,
                        y_pred_label, **kwargs)
            elif not is_clf and y_pred_label is not None and name == 'mean_absolute_error':
                fold_scores[name] = metric_fns[name](y_true, y_pred_label)
        except Exception as e:
            logger.warning(f'[{task}-{mode}-F{fold}] Metric {name} error: {e}')
    return fold_scores


def _finalize_metrics(all_fold_scores: Dict[str, List[float]], metrics_cfg:
    List[str], n_splits: int) ->Dict[str, Any]:
    final_metrics: Dict[str, Any] = {}
    for name_base in metrics_cfg:
        if name_base.endswith(('_std', '_ci_low', '_ci_up')):
            continue
        scores = all_fold_scores.get(name_base, [])
        mean, std, ci_l, ci_u = _calculate_metric_stats(scores, n_splits)
        final_metrics[name_base] = mean
        if f'{name_base}_std' in metrics_cfg:
            final_metrics[f'{name_base}_std'] = std
        if f'{name_base}_ci_low' in metrics_cfg:
            final_metrics[f'{name_base}_ci_low'] = ci_l
        if f'{name_base}_ci_up' in metrics_cfg:
            final_metrics[f'{name_base}_ci_up'] = ci_u
    return final_metrics


def _run_model_cv_and_evaluate(model_factory: Callable, X: pd.DataFrame, y:
    pd.Series, is_clf: bool, cfg: Dict, task: str, mode: str) ->Dict[str, Any]:
    cv = _get_cv_splitter(cfg['n_splits_cv'], cfg['seed'], is_clf, y)
    fold_scores_acc = defaultdict(list)
    fold_feature_importances = []
    y_s = y.reset_index(drop=True)
    X_df = X.reset_index(drop=True)
    if X_df.empty:
        logger.warning(
            f'[{task}-{mode}] X_df empty for CV. Returning NaN metrics.')
        return _finalize_metrics({}, cfg['evaluation_metrics'], cfg[
            'n_splits_cv'])
    for i, (train_idx, test_idx) in enumerate(cv.split(X_df, y_s if is_clf and
        y_s.nunique() > 1 else None)):
        X_tr, X_te = X_df.iloc[train_idx].copy(), X_df.iloc[test_idx].copy()
        y_tr, y_te = y_s.iloc[train_idx], y_s.iloc[test_idx]
        model = model_factory()
        is_lgbm = isinstance(model, (LGBMClassifier, LGBMRegressor))
        if is_lgbm:
            X_tr_proc, X_te_proc = X_tr.copy(), X_te.copy()
            for col in X_tr.select_dtypes(include=['object']).columns:
                X_tr_proc[col] = X_tr_proc[col].astype('category')
            for col in X_te.select_dtypes(include=['object']).columns:
                X_te_proc[col] = X_te_proc[col].astype('category')
            X_tr_proc.columns = sanitize_feature_names(X_tr_proc.columns)
            X_te_proc.columns = sanitize_feature_names(X_te_proc.columns)
        elif isinstance(model, MLPRegressor):
            if X_tr.empty:
                logger.warning(
                    f'[{task}-{mode}-F{i + 1}] X_tr empty for MLP. Skipping fold.'
                    )
                continue
            prep = _create_sklearn_preprocessor(X_tr, task, mode)
            try:
                X_tr_proc = pd.DataFrame(prep.fit_transform(X_tr), columns=
                    sanitize_feature_names(prep.get_feature_names_out()),
                    index=X_tr.index)
                X_te_proc = pd.DataFrame(prep.transform(X_te), columns=
                    sanitize_feature_names(prep.get_feature_names_out()),
                    index=X_te.index) if not X_te.empty else pd.DataFrame(
                    columns=X_tr_proc.columns)
            except ValueError as e_prep:
                logger.error(
                    f'[{task}-{mode}-F{i + 1}] Preprocessing error: {e_prep}. Skipping.'
                    )
                continue
        else:
            X_tr_proc, X_te_proc = X_tr, X_te
        fit_success, top_features = _fit_model(model, X_tr_proc, y_tr, task,
            mode, i + 1)
        if not fit_success:
            continue
        if top_features:
            fold_feature_importances.append(top_features)
        labels, probas = _get_predictions(model, X_te_proc, is_clf, task,
            mode, i + 1)
        fold_m = _calculate_fold_metrics(y_te, labels, probas, is_clf, cfg[
            'evaluation_metrics'], task, mode, i + 1)
        for m_name, m_val in fold_m.items():
            fold_scores_acc[m_name].append(m_val)
    final_metrics = _finalize_metrics(fold_scores_acc, cfg[
        'evaluation_metrics'], cfg['n_splits_cv'])
    if fold_feature_importances:
        fi_dict = defaultdict(list)
        for fold_fi in fold_feature_importances:
            for feat_name, imp in fold_fi:
                fi_dict[feat_name].append(imp)
        avg_fi = [(feat, np.mean(imps)) for feat, imps in fi_dict.items()]
        avg_fi.sort(key=lambda x: -x[1])
        final_metrics['top_5_features_by_importance'] = [{'feature': feat,
            'importance': float(imp)} for feat, imp in avg_fi[:5]]
    return final_metrics


def _train_kb_encoder_or_teacher(X_kb_tr: pd.DataFrame, y_tr: pd.Series,
    X_kb_te: pd.DataFrame, is_clf: bool, cfg: Dict, task: str, mode: str,
    type: str) ->Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if X_kb_tr.empty:
        logger.warning(f'[{task}-{mode}] X_kb_train empty for {type}.')
        return None, None
    kb_prep = _create_sklearn_preprocessor(X_kb_tr, task, f'{mode}_kb_prep')
    X_kb_tr_proc = pd.DataFrame(kb_prep.fit_transform(X_kb_tr), columns=
        sanitize_feature_names(kb_prep.get_feature_names_out()), index=
        X_kb_tr.index)
    X_kb_te_proc = pd.DataFrame(kb_prep.transform(X_kb_te), columns=
        sanitize_feature_names(kb_prep.get_feature_names_out()), index=
        X_kb_te.index) if not X_kb_te.empty else pd.DataFrame(columns=
        X_kb_tr_proc.columns)
    if X_kb_tr_proc.shape[1] == 0:
        logger.warning(
            f'[{task}-{mode}] KB features 0 cols post-prep for {type}.')
        return None, None
    if type == 'logits':
        lgbm_p = _get_lgbm_params(cfg)
        model = LGBMClassifier(**lgbm_p) if is_clf else LGBMRegressor(**lgbm_p)
        fit_success, _ = _fit_model(model, X_kb_tr_proc, y_tr, task,
            f'{mode}_kb_teacher', 0)
        if not fit_success:
            return None, None
        tr_out = model.predict_proba(X_kb_tr_proc)[:, 1
            ] if is_clf else model.predict(X_kb_tr_proc)
        te_out = (model.predict_proba(X_kb_te_proc)[:, 1] if is_clf else
            model.predict(X_kb_te_proc)
            ) if not X_kb_te_proc.empty else np.array([])
        return pd.DataFrame(tr_out, index=X_kb_tr.index, columns=['kb_logit']
            ), pd.DataFrame(te_out, index=X_kb_te.index, columns=['kb_logit'])
    dim_cfg = cfg.get('pca_params' if type == 'pca' else
        'nat_embedding_params', {})
    enc_dim = min(dim_cfg.get('encoder_dim', 16), X_kb_tr_proc.shape[1], 
        X_kb_tr_proc.shape[0] - 1 if X_kb_tr_proc.shape[0] > 1 else 1)
    if enc_dim < 1:
        enc_dim = 1
    if X_kb_tr_proc.shape[0] <= enc_dim:
        logger.warning(
            f'[{task}-{mode}] Not enough samples for {type} dim {enc_dim}.')
        return None, None
    encoder: Any = None
    if type == 'pca':
        encoder = PCA(n_components=enc_dim, random_state=cfg['seed'])
    elif type == 'nat':
        rng = np.random.RandomState(cfg['seed'])
        targets = rng.rand(X_kb_tr_proc.shape[0], enc_dim)
        mlp_p = _get_mlp_params(cfg)
        mlp_p['max_iter'] = dim_cfg.get('mlp_max_iter_for_nat_encoder',
            mlp_p.get('max_iter', 300))
        encoder = MLPRegressor(**mlp_p)
        encoder.fit(X_kb_tr_proc, targets)
        tr_emb = encoder.predict(X_kb_tr_proc)
        te_emb = encoder.predict(X_kb_te_proc
            ) if not X_kb_te_proc.empty else np.array([])
        cols = [f'nat_emb_{i}' for i in range(enc_dim)]
        return pd.DataFrame(tr_emb, index=X_kb_tr.index, columns=cols
            ), pd.DataFrame(te_emb, index=X_kb_te.index, columns=cols)
    tr_emb = encoder.fit_transform(X_kb_tr_proc)
    te_emb = encoder.transform(X_kb_te_proc
        ) if not X_kb_te_proc.empty else np.array([])
    cols = [f'pca_emb_{i}' for i in range(enc_dim)]
    return pd.DataFrame(tr_emb, index=X_kb_tr.index, columns=cols
        ), pd.DataFrame(te_emb, index=X_kb_te.index, columns=cols)


def _train_host_mimic_mlp(X_host_tr: pd.DataFrame, kb_derived_tr_outputs:
    pd.DataFrame, cfg: Dict, task: str, mode: str) ->Optional[Tuple[
    Pipeline, ColumnTransformer]]:
    if X_host_tr.empty or kb_derived_tr_outputs.empty:
        logger.warning(
            f'[{task}-{mode}] Host or KB-derived training data empty for mimic.'
            )
        return None
    host_mimic_prep = _create_sklearn_preprocessor(X_host_tr, task,
        f'{mode}_mimic_prep')
    X_host_tr_proc = pd.DataFrame(host_mimic_prep.fit_transform(X_host_tr),
        columns=sanitize_feature_names(host_mimic_prep.
        get_feature_names_out()), index=X_host_tr.index)
    mlp_p = _get_mlp_params(cfg)
    host_mimic_model = MLPRegressor(**mlp_p)
    fit_success, _ = _fit_model(host_mimic_model, X_host_tr_proc,
        kb_derived_tr_outputs, task, f'{mode}_mimic', 0)
    if not fit_success:
        return None
    return host_mimic_model, host_mimic_prep


def _get_host_mimic_predictions(host_mimic_model: MLPRegressor,
    host_mimic_preprocessor: ColumnTransformer, X_host_data: pd.DataFrame,
    original_kb_output_cols: List[str], task: str, mode: str) ->pd.DataFrame:
    if X_host_data.empty:
        return pd.DataFrame(columns=original_kb_output_cols)
    X_host_proc = pd.DataFrame(host_mimic_preprocessor.transform(
        X_host_data), columns=sanitize_feature_names(
        host_mimic_preprocessor.get_feature_names_out()), index=X_host_data
        .index)
    preds = host_mimic_model.predict(X_host_proc)
    return pd.DataFrame(preds, index=X_host_data.index, columns=
        original_kb_output_cols)


def _run_task_baselines(task_name: str, task_def: Dict, spec: Dict,
    base_cfg: Dict, engine: Any, dataset_output_dir: Path):
    is_clf = task_def['type'] == 'binary'
    df_full_orig, y_full_orig, _ = _process_initial_data_for_task(task_name,
        task_def, spec, engine)
    if df_full_orig is None or y_full_orig is None:
        logger.error(f'[{task_name}] Data loading failed. Skipping baselines.')
        return
    y_full_orig.index = y_full_orig.index.astype(str)
    vfl_key = spec['column_identifiers']['VFL_KEY']
    df_full_orig = standardise_key(df_full_orig, vfl_key)
    train_keys, val_keys, test_keys = _load_l1_defined_indices(task_name,
        spec, base_cfg, dataset_output_dir)
    if not train_keys.size or not test_keys.size:
        logger.error(
            f'[{task_name}] Could not load or use L1 defined splits. Skipping baselines.'
            )
        return
    train_val_keys = train_keys
    all_l1_keys = train_val_keys.union(test_keys)
    df_full_aligned = df_full_orig[df_full_orig.index.isin(all_l1_keys)].copy()
    y_full_aligned = y_full_orig[y_full_orig.index.isin(all_l1_keys)].copy()
    if df_full_aligned.empty:
        logger.error(
            f'[{task_name}] Data empty after aligning with L1 split keys. Skipping baselines.'
            )
        return
    df_train_val = df_full_aligned.loc[df_full_aligned.index.isin(
        train_val_keys)].copy()
    y_train_val = y_full_aligned.loc[y_full_aligned.index.isin(train_val_keys)
        ].copy()
    df_test = df_full_aligned.loc[df_full_aligned.index.isin(test_keys)].copy()
    y_test = y_full_aligned.loc[y_full_aligned.index.isin(test_keys)].copy()
    if (df_train_val.empty or y_train_val.empty or df_test.empty or y_test.
        empty):
        logger.error(
            f'[{task_name}] One or more data partitions (train_val, test) are empty after L1 split. Skipping.'
            )
        return
    sets = _identify_feature_sets(df_full_orig.columns, spec, task_name)
    task_out_dir = dataset_output_dir / task_name / base_cfg[
        'baseline_output_subdir']
    ensure_dir_exists(task_out_dir)
    for mode_name in base_cfg.get('modes_to_run', []):
        logger.info(
            f'[{task_name}] ----- Running Baseline (on L1 Split): {mode_name} -----'
            )
        metrics: Dict[str, Any] = {'error':
            f'Mode {mode_name} not implemented or failed.'}
        try:
            if mode_name in ['oracle_all_data', 'host_only_all_features']:
                metrics = _process_simple_baseline_on_split(df_train_val,
                    y_train_val, df_test, y_test, sets, base_cfg, spec,
                    task_name, is_clf, mode_name)
            elif mode_name in ['kb_pca_embeddings_fedonce_style',
                'kb_nat_embeddings', 'student_with_distilled_logits']:
                transfer_type = ('pca' if 'pca' in mode_name else 'nat' if 
                    'nat' in mode_name else 'logits')
                metrics = _process_transfer_learning_baseline_on_split(
                    df_train_val, y_train_val, df_test, y_test, sets,
                    base_cfg, spec, task_name, is_clf, mode_name, transfer_type
                    )
            else:
                logger.warning(
                    f"[{task_name}] Mode '{mode_name}' not explicitly handled for L1 split evaluation."
                    )
        except Exception as e:
            logger.error(f'[{task_name}-{mode_name}] Unhandled exception: {e}',
                exc_info=True)
            metrics = {'error': f'Unhandled exception: {str(e)}'}
        finally:
            metrics_file_name = base_cfg['artifact_names']['metrics_template'
                ].format(mode_name=mode_name)
            _save_metrics_to_json(metrics, task_out_dir / metrics_file_name,
                task_name, mode_name)


def main():
    parser = argparse.ArgumentParser(description='Run baseline models.')
    parser.add_argument('--config', type=str, default=
        'baselines_config.json', help='Baselines config JSON.')
    parser.add_argument('--task', type=str, help='Specific task name to run.')
    args = parser.parse_args()
    base_cfg = load_json_config(Path(args.config))
    if not base_cfg:
        sys.exit('Failed to load baselines_config.json')
    spec_path_str = base_cfg.get('evaluation_spec_file')
    if not spec_path_str:
        logger.error("'evaluation_spec_file' missing in baselines_config.")
        sys.exit(1)
    spec_path = Path(args.config).parent / spec_path_str
    if not spec_path.is_file():
        spec_path = Path(spec_path_str)
    eval_spec = load_json_config(spec_path)
    if not eval_spec:
        sys.exit(f'Failed to load evaluation_spec from: {spec_path}')
    np.random.seed(base_cfg.get('seed', 42))
    logger.info(f"Global seed: {base_cfg.get('seed', 42)}")
    db_module = base_cfg.get('db_config_module', eval_spec.get(
        'db_config_module'))
    if not db_module:
        logger.error('db_config_module not specified.')
        sys.exit(1)
    db_path_cand = next((str(p.parent) for p in [spec_path, Path(args.
        config), Path(__file__)] if (p.parent / f'{db_module}.py').exists()
        ), None)
    db_engine = _get_db_engine(db_module, db_path_cand)
    if not db_engine:
        sys.exit('Failed to create DB engine.')
    out_root_str = eval_spec.get('dataset_output_dir_name')
    if not out_root_str:
        logger.error("'dataset_output_dir_name' missing in eval_spec.")
        sys.exit(1)
    out_parent_dir = Path(out_root_str)
    ensure_dir_exists(out_parent_dir)
    tasks = eval_spec.get('tasks', {})
    if args.task:
        tasks = {args.task: tasks[args.task]} if args.task in tasks else {}
    if not tasks:
        logger.info('No tasks to process.')
        sys.exit(0)
    for name, definition in tasks.items():
        logger.info(f'\n========== PROCESSING TASK: {name} ==========')
        definition['type'] = definition.get('type', 'binary')
        _run_task_baselines(name, definition, eval_spec, base_cfg,
            db_engine, out_parent_dir)
    if db_engine:
        db_engine.dispose()
    logger.info('Baseline group script finished.')


if __name__ == '__main__':
    main()
