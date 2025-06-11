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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss, average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import sanitize_feature_names, get_predictions, evaluate_predictions, standardise_key, load_json_config, NpEncoder, ensure_dir_exists, save_artifact, load_artifact, preprocess_features_for_lgbm
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
L2_CONFIG_FILE = Path('l2_config.json')
EVAL_SPEC_FILE_KEY_IN_L2_CONFIG = 'evaluation_spec_file'
GLOBAL_CONFIG_SEED = 42


def _get_spec_feature_lists(spec: Dict, host_feat_cfg_keys: Dict) ->Tuple[
    List[str], List[str]]:
    num_key = host_feat_cfg_keys.get('native_numeric_key',
        'HOST_NUMERIC_COLS_FOR_MODELING')
    cat_key = host_feat_cfg_keys.get('native_categorical_key',
        'HOST_CATEGORICAL_COLS_FOR_MODELING')
    cols_spec = spec.get('columns', {})
    return cols_spec.get(num_key, []), cols_spec.get(cat_key, [])


def _create_numeric_preprocessing_pipeline() ->Pipeline:
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('scl',
        StandardScaler())])


def _impute_and_cast_categoricals(df: pd.DataFrame, active_cat_cols: List[
    str], task_name: str) ->pd.DataFrame:
    df_proc = df.copy()
    if not active_cat_cols:
        return df_proc
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_proc[active_cat_cols] = cat_imputer.fit_transform(df_proc[
        active_cat_cols].astype(str))
    for col in active_cat_cols:
        df_proc[col] = df_proc[col].astype('category')
    logger.debug(
        f'[{task_name}] Imputed and cast {len(active_cat_cols)} categorical features.'
        )
    return df_proc


def train_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, params:
    Dict, task_type: str, X_val: Optional[pd.DataFrame]=None, y_val:
    Optional[pd.Series]=None, sample_weight: Optional[pd.Series]=None) ->Any:
    fit_params, model_params = {'sample_weight': sample_weight
        } if sample_weight is not None else {}, params.copy()
    X_train_sanitized = X_train.copy()
    if 'ErrorBoost' in params.get('task_name_for_debug', ''):
        logger.info(
            f'DEBUG: ErrorBoost training features (X_train columns before sanitization): {X_train.columns.tolist()}'
            )
    X_train_sanitized.columns = sanitize_feature_names(X_train.columns.tolist()
        )
    if 'ErrorBoost' in params.get('task_name_for_debug', ''):
        logger.info(
            f'DEBUG: ErrorBoost training features (X_train_sanitized columns): {X_train_sanitized.columns.tolist()}'
            )
    original_categorical_feature_names = [col for col in X_train.columns if
        isinstance(X_train[col].dtype, pd.CategoricalDtype)]
    sanitized_categorical_for_lgbm = sanitize_feature_names(
        original_categorical_feature_names)
    fit_params['categorical_feature'] = sanitized_categorical_for_lgbm
    if (X_val is not None and y_val is not None and not X_val.empty and
        model_params.get('early_stopping_rounds')):
        X_val_sanitized = X_val.copy()
        X_val_sanitized.columns = sanitize_feature_names(X_val.columns.tolist()
            )
        fit_params['eval_set'] = [(X_val_sanitized, y_val)]
        fit_params['eval_metric'] = 'auc' if task_type == 'binary' else 'mae'
        fit_params['callbacks'] = [lgb.early_stopping(model_params.pop(
            'early_stopping_rounds'), verbose=-1)]
    model_class = (lgb.LGBMClassifier if task_type == 'binary' else lgb.
        LGBMRegressor)
    model = model_class(**model_params)
    model.fit(X_train_sanitized, y_train, **fit_params)
    return model


def _extract_lgbm_top_features(model: Any, top_n: int=5) ->List[Dict]:
    if hasattr(model, 'feature_importances_') and hasattr(model,
        'feature_name_'):
        pairs = sorted(zip(model.feature_name_, model.feature_importances_),
            key=lambda x: x[1], reverse=True)
        return [{'feature': name, 'importance': float(imp)} for name, imp in
            pairs[:top_n]]
    return []


def evaluate_model_l2(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
    metrics_cfg: List[str], task_type: str, task_name: str) ->Dict[str, Any]:
    results: Dict[str, Any] = {}
    X_test_sanitized = X_test.copy()
    X_test_sanitized.columns = sanitize_feature_names(X_test.columns.tolist())
    y_pred_proba = get_predictions(model, X_test_sanitized, task_type)
    y_pred_labels = (y_pred_proba > 0.5).astype(int
        ) if task_type == 'binary' and y_pred_proba is not None else None
    metric_map = {'roc_auc_score': lambda : roc_auc_score(y_test,
        y_pred_proba) if y_test.nunique() > 1 else 0.5, 'auprc_score': lambda :
        average_precision_score(y_test, y_pred_proba) if y_test.nunique() >
        1 else y_test.mean(), 'accuracy_score': lambda : accuracy_score(
        y_test, y_pred_labels), 'f1_score': lambda : f1_score(y_test,
        y_pred_labels, zero_division=0), 'precision_score': lambda :
        precision_score(y_test, y_pred_labels, zero_division=0),
        'recall_score': lambda : recall_score(y_test, y_pred_labels,
        zero_division=0), 'brier_score_loss': lambda : brier_score_loss(
        y_test, y_pred_proba)}
    for name in metrics_cfg:
        if name in metric_map and (task_type == 'binary' or name not in [
            'roc_auc_score', 'auprc_score', 'brier_score_loss']):
            try:
                results[name] = metric_map[name]()
            except Exception as e:
                logger.warning(f'[{task_name}] Metric {name} error: {e}')
                results[name] = np.nan
        elif name.endswith(('_ci_low', '_ci_up', '_std')):
            results[name] = np.nan
    results['top_5_features_by_importance'] = _extract_lgbm_top_features(model)
    logger.info(
        f'[{task_name}] Metrics: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items() if pd.notna(v)})}'
        )
    return results


def _harmonise_categorical_dtypes(df: pd.DataFrame) ->pd.DataFrame:
    for col in df.select_dtypes(include='category').columns:
        current_categories = df[col].cat.categories.astype(str).tolist()
        df[col] = df[col].astype(str).astype(pd.CategoricalDtype(categories
            =current_categories))
    return df


def _get_all_relevant_features(df_augmented_host: pd.DataFrame, eval_spec:
    Dict, l2_cfg: Dict, task_root: Path, task_name: str) ->Tuple[List[str],
    List[str], List[str], List[str]]:
    host_feat_cfg_keys_for_native = eval_spec.get('columns', {})
    num_key = 'HOST_NUMERIC_COLS_FOR_MODELING'
    cat_key = 'HOST_CATEGORICAL_COLS_FOR_MODELING'
    host_native_num_spec = host_feat_cfg_keys_for_native.get(num_key, [])
    host_native_cat_spec = host_feat_cfg_keys_for_native.get(cat_key, [])
    host_native_num = [c for c in host_native_num_spec if c in
        df_augmented_host.columns]
    host_native_cat = [c for c in host_native_cat_spec if c in
        df_augmented_host.columns]
    configured_surr_prefixes = l2_cfg.get('surr_feature_prefixes', (
        'p_hat_rule_', 'surr_', 'surrx_', 'pre_surrx_', 'pre_remote_surrx_',
        'llm_'))
    surr_prefixes_tuple = tuple(configured_surr_prefixes) if isinstance(
        configured_surr_prefixes, list) else configured_surr_prefixes
    surr_cols = [c for c in df_augmented_host.columns if any(c.startswith(
        prefix) for prefix in surr_prefixes_tuple)]
    try:
        sr_file_key = 'l1_selected_remote_rules_filename'
        sr_filename = l2_cfg['artifact_names'].get(sr_file_key,
            'selected_remote_rules_for_S_hat_training.json')
        l1_output_subdir = l2_cfg['script1_input_subdir_template']
        sr_file = task_root / l1_output_subdir / sr_filename
        if sr_file.exists():
            with open(sr_file, 'r', encoding='utf-8') as fh:
                rules_data = json.load(fh)
            expected_rule_ids = set(rules_data.keys())
            missing_phat_rules = {f'p_hat_rule_{rid}' for rid in
                expected_rule_ids} - set(surr_cols)
            if missing_phat_rules:
                logger.warning(
                    f"[{task_name}] Missing expected p_hat_rule_ features for rules: {', '.join(sorted(missing_phat_rules))}"
                    )
        else:
            logger.warning(
                f'[{task_name}] Selected remote rules file not found for surrogate check: {sr_file}'
                )
    except Exception as e:
        logger.debug(
            f'[{task_name}] Surrogate presence check skipped or failed: {e}')

    def is_categorical_or_string(series):
        return isinstance(series.dtype, pd.CategoricalDtype
            ) or pd.api.types.is_string_dtype(series
            ) or pd.api.types.is_object_dtype(series)
    all_surr_num = [c for c in surr_cols if c in df_augmented_host.columns and
        pd.api.types.is_numeric_dtype(df_augmented_host[c])]
    all_surr_cat = [c for c in surr_cols if c in df_augmented_host.columns and
        is_categorical_or_string(df_augmented_host[c])]
    logger.info(
        f'[{task_name}] Identified native features: {len(host_native_num)} numeric, {len(host_native_cat)} categorical.'
        )
    logger.info(
        f'[{task_name}] Identified surrogate features (prefixes {surr_prefixes_tuple}): {len(all_surr_num)} numeric, {len(all_surr_cat)} categorical.'
        )
    return host_native_num, host_native_cat, all_surr_num, all_surr_cat


def _determine_error_leaf_coverage(df: pd.DataFrame, host0_bat_model: Any,
    error_leaf_ids: set[int], num_cols: list[str], cat_cols: list[str],
    return_leaf_ids: bool=False) ->np.ndarray:
    if df.empty:
        return np.array([], dtype=int) if return_leaf_ids else np.zeros(len
            (df), dtype=bool)
    feat_cols = [c for c in num_cols + cat_cols if c in df.columns]
    if not feat_cols or not hasattr(host0_bat_model, 'bat_preprocessor_'):
        return np.array([], dtype=int) if return_leaf_ids else np.zeros(len
            (df), dtype=bool)
    X_trans = host0_bat_model.bat_preprocessor_.transform(df[feat_cols])
    leaf_mat = host0_bat_model.apply(X_trans)
    if leaf_mat.ndim == 2:
        leaf_ids = leaf_mat[:, 0]
    else:
        leaf_ids = leaf_mat.ravel()
    if return_leaf_ids:
        return leaf_ids.astype(int)
    if not error_leaf_ids:
        return np.zeros(len(df), dtype=bool)
    return np.isin(leaf_ids, list(error_leaf_ids))


def _define_l2_training_partitions(df_host_train_with_surrogates: pd.
    DataFrame, y_train_full: pd.Series, host0_bat_model: Any,
    error_leaf_ids: set[int], host_native_num: List[str], host_native_cat:
    List[str], task_name: str) ->Tuple[pd.DataFrame, pd.Series]:
    logger.info(
        f'[{task_name}] Defining L2 training partitions based on pre-computed error leaf coverage...'
        )
    error_mask_train = _determine_error_leaf_coverage(
        df_host_train_with_surrogates, host0_bat_model, error_leaf_ids,
        host_native_num, host_native_cat)
    if isinstance(error_mask_train, np.ndarray):
        error_mask_train = pd.Series(error_mask_train, index=
            df_host_train_with_surrogates.index)
    df_eb_train_raw = df_host_train_with_surrogates.copy()
    y_eb_train = y_train_full.copy()
    logger.info(
        '[%s] Error-Boost train rows: %d (Host0 rows (not in error boost for training): %d)'
        , task_name, error_mask_train.sum(), (~error_mask_train).sum())
    return df_eb_train_raw, y_eb_train


def _make_ensemble_predictions(df_test: pd.DataFrame, errorboost_model:
    Optional[Any], errorboost_preproc: Optional[Any], errorboost_num: List[
    str], errorboost_cat: List[str], host0_model: Any, host0_preproc: Any,
    host0_num: List[str], host0_cat: List[str], host0_bat_model: Any,
    all_error_leaf_ids: Set[int], segment_model_decision_map: Dict[int, str
    ], task_type: str, task_name: str, vfl_key_col: str) ->Tuple[pd.Series,
    np.ndarray, pd.Series]:
    logger.info(
        f'[{task_name}] Making ensemble predictions with segment-specific model routing...'
        )
    df_test_for_bat = df_test.copy()
    if vfl_key_col not in df_test_for_bat.columns:
        df_test_for_bat[vfl_key_col] = df_test_for_bat.index
    leaf_id_predictions_test = pd.Series(_determine_error_leaf_coverage(
        df_test_for_bat, host0_bat_model, all_error_leaf_ids, host0_num,
        host0_cat, return_leaf_ids=True), index=df_test.index)
    preds = pd.Series(np.nan, index=df_test.index)
    potential_eb_path_mask = leaf_id_predictions_test.isin(all_error_leaf_ids)
    actual_model_routed_to = pd.Series('host0', index=df_test.index)
    host0_direct_mask = ~potential_eb_path_mask
    if host0_direct_mask.any():
        df_h0_direct_test = df_test.loc[host0_direct_mask]
        h0_native_feats_test = [f for f in host0_num + host0_cat if f in
            df_h0_direct_test.columns]
        X_h0_direct, _ = preprocess_features_for_lgbm(df_h0_direct_test[
            h0_native_feats_test], host0_num, host0_cat,
            f'{task_name}_Host0_Direct_Inference', fit_mode=False,
            existing_num_preprocessor=host0_preproc)
        if X_h0_direct is not None and not X_h0_direct.empty:
            preds.loc[host0_direct_mask] = get_predictions(host0_model,
                X_h0_direct, task_type)
            actual_model_routed_to.loc[host0_direct_mask] = 'host0_direct'
        else:
            logger.warning(
                f'[{task_name}] Host0_Direct path preprocessing empty. Defaulting predictions.'
                )
            preds.loc[host0_direct_mask] = 0.0
    final_eb_path_mask = pd.Series(False, index=df_test.index)
    for leaf_id in all_error_leaf_ids:
        current_leaf_mask_test = leaf_id_predictions_test == leaf_id
        if not current_leaf_mask_test.any():
            continue
        chosen_model_for_segment = segment_model_decision_map.get(leaf_id,
            'host0')
        df_segment_test = df_test.loc[current_leaf_mask_test]
        if (chosen_model_for_segment == 'errorboost' and errorboost_model
             is not None):
            eb_feats_test = [f for f in errorboost_num + errorboost_cat if 
                f in df_segment_test.columns]
            X_eb_segment, _ = preprocess_features_for_lgbm(df_segment_test[
                eb_feats_test], errorboost_num, errorboost_cat,
                f'{task_name}_ErrorBoost_Seg_{leaf_id}_Inference', fit_mode
                =False, existing_num_preprocessor=errorboost_preproc)
            if X_eb_segment is not None and not X_eb_segment.empty:
                preds.loc[current_leaf_mask_test] = get_predictions(
                    errorboost_model, X_eb_segment, task_type)
                actual_model_routed_to.loc[current_leaf_mask_test
                    ] = f'errorboost_leaf_{leaf_id}'
                final_eb_path_mask.loc[current_leaf_mask_test] = True
            else:
                logger.warning(
                    f'[{task_name}] EB preprocessing for leaf {leaf_id} empty. Defaulting to Host0 for this segment.'
                    )
                chosen_model_for_segment = 'host0'
        if chosen_model_for_segment == 'host0':
            h0_native_feats_test_seg = [f for f in host0_num + host0_cat if
                f in df_segment_test.columns]
            X_h0_segment, _ = preprocess_features_for_lgbm(df_segment_test[
                h0_native_feats_test_seg], host0_num, host0_cat,
                f'{task_name}_Host0_Seg_{leaf_id}_Inference', fit_mode=
                False, existing_num_preprocessor=host0_preproc)
            if X_h0_segment is not None and not X_h0_segment.empty:
                preds.loc[current_leaf_mask_test] = get_predictions(host0_model
                    , X_h0_segment, task_type)
                actual_model_routed_to.loc[current_leaf_mask_test
                    ] = f'host0_switched_leaf_{leaf_id}'
            else:
                logger.warning(
                    f'[{task_name}] Host0_Switched path preprocessing for leaf {leaf_id} empty. Defaulting predictions.'
                    )
                preds.loc[current_leaf_mask_test] = 0.0
    preds.fillna(0.0, inplace=True)
    num_eb_final_path = final_eb_path_mask.sum()
    num_h0_final_path = (~final_eb_path_mask).sum()
    total_test = len(df_test)
    logger.info(
        f'[{task_name}] Test Set Final Routing: {num_eb_final_path}/{total_test} to ErrorBoost, {num_h0_final_path}/{total_test} to Host0 (direct or switched).'
        )
    return preds, final_eb_path_mask, actual_model_routed_to


def _load_l2_main_input_artifacts(paths: Dict[str, Path], l2_cfg: Dict,
    task_name: str) ->Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame],
    Optional[Dict]]:
    logger.info(f'[{task_name}] Loading main L2 input artifacts...')
    aug_data_fn = l2_cfg['artifact_names'].get(
        's1_augmented_host_data_filename')
    df_aug_host_full = load_artifact(paths['s1_in'] / aug_data_fn,
        'Augmented Host Data')
    target_fn = l2_cfg['artifact_names']['l1_host_target_cols_filename']
    df_target_full = load_artifact(paths['s1_in'] / target_fn,
        'Host Target Cols')
    splits_fn = l2_cfg['artifact_names']['l1_split_indices_filename']
    l1_splits = load_artifact(paths['s1_in'] / splits_fn, 'L1 Split Indices')
    if any(x is None for x in (df_aug_host_full, df_target_full, l1_splits)):
        logger.error(
            f'[{task_name}] One or more critical L2 input artifacts missing. Aborting.'
            )
        return None, None, None
    return df_aug_host_full, df_target_full, l1_splits


def _load_l2_model_artifacts(paths: Dict[str, Path], l2_cfg: Dict,
    task_name: str) ->Tuple[Optional[Any], Optional[Any], Optional[Any],
    Optional[Set[int]]]:
    logger.info(
        f'[{task_name}] Loading L2 model artifacts (Host0 models, error leaves)...'
        )
    h0_model_fn = l2_cfg['artifact_names'].get('l1_host0_model_filename')
    host0_model = load_artifact(paths['s1_in'] / h0_model_fn,
        'Host0 LGBM Model')
    h0_preproc_fn = l2_cfg['artifact_names'].get(
        'l1_host0_preprocessor_filename')
    host0_preproc = load_artifact(paths['s1_in'] / h0_preproc_fn,
        'Host0 LGBM Preprocessor')
    h0_bat_fn = l2_cfg['artifact_names'].get('l1_host0_bat_model_filename')
    host0_bat = load_artifact(paths['s1_in'] / h0_bat_fn, 'Host0 BAT Model')
    error_leaf_ids: Set[int] = set()
    err_leaf_fn = l2_cfg['artifact_names'].get(
        'l1_host_error_leaf_ids_filename')
    error_leaf_path = paths['s1_in'] / err_leaf_fn
    if error_leaf_path.exists():
        loaded_ids = load_artifact(error_leaf_path, 'L1 Error-leaf list')
        if isinstance(loaded_ids, list) and loaded_ids:
            error_leaf_ids = set(map(int, loaded_ids))
            logger.info(
                f'[{task_name}] Loaded {len(error_leaf_ids)} error-leaf IDs from L1.'
                )
    if not error_leaf_ids:
        logger.warning(
            f'[{task_name}] No valid error_leaf_ids loaded from L1 ({error_leaf_path}). ErrorBoost may not be effective.'
            )
    if any(x is None for x in (host0_model, host0_preproc, host0_bat)):
        logger.error(
            f'[{task_name}] One or more Host0 model artifacts missing. Aborting.'
            )
        return None, None, None, set()
    return host0_model, host0_preproc, host0_bat, error_leaf_ids


def _prepare_l2_input_dataframe(df_aug_host: pd.DataFrame, df_target: pd.
    DataFrame, vfl_key: str, target_col: str, task_name: str) ->Optional[pd
    .DataFrame]:
    logger.debug(f'[{task_name}] Preparing L2 input DataFrame...')
    for df in [df_aug_host, df_target]:
        if vfl_key not in df.columns:
            if df.index.name == vfl_key:
                df.reset_index(inplace=True)
            else:
                logger.error(
                    f"[{task_name}] VFL_KEY '{vfl_key}' missing from a DataFrame."
                    )
                return None
        df[vfl_key] = df[vfl_key].astype(str)
    df_aug_host = standardise_key(df_aug_host, vfl_key).reset_index(drop=True)
    df_target = standardise_key(df_target, vfl_key).reset_index(drop=True)
    df_aug_host = df_aug_host.drop_duplicates(subset=[vfl_key], keep='first')
    df_target = df_target.drop_duplicates(subset=[vfl_key], keep='first')
    df_full = pd.merge(df_aug_host, df_target[[vfl_key, target_col]], on=
        vfl_key, how='inner')
    if df_full.empty:
        logger.error(
            f'[{task_name}] DataFrame empty after merging aug-host and target.'
            )
        return None
    df_full = standardise_key(df_full, vfl_key)
    logger.info(
        f'[{task_name}] Prepared L2 input DataFrame. Shape: {df_full.shape}')
    return df_full


def _validate_and_get_l2_split_indices(l1_splits: Dict, df_full_input: pd.
    DataFrame, vfl_key: str, task_name: str) ->Tuple[Optional[pd.Index],
    Optional[pd.Index]]:
    logger.debug(f'[{task_name}] Validating L2 split indices...')
    df_key_set = set(df_full_input[vfl_key].astype(str))
    validated_indices = {}
    for split_name in ['train', 'test']:
        keys_raw_str = list(map(str, l1_splits.get(f'{split_name}_idx', [])))
        if not keys_raw_str:
            logger.warning(
                f"[{task_name}] L1 supplied empty '{split_name}_idx'.")
            validated_indices[split_name] = pd.Index([])
            continue
        matched_keys = [k for k in keys_raw_str if k in df_key_set]
        if not matched_keys and keys_raw_str:
            logger.error(
                f"[{task_name}] L1 '{split_name}_idx' has ZERO OVERLAP with L2 data on VFL_KEY. Aborting task."
                )
            return None, None
        if len(matched_keys) < 0.95 * len(keys_raw_str) and keys_raw_str:
            logger.warning(
                f'[{task_name}] Only {len(matched_keys)} / {len(keys_raw_str)} L1 {split_name}_idx keys matched L2 data.'
                )
        validated_indices[split_name] = pd.Index(matched_keys)
    if validated_indices['train'].empty:
        logger.error(
            f'[{task_name}] Training data partition is empty after index validation. Aborting task.'
            )
        return None, None
    return validated_indices['train'], validated_indices['test']


def _prepare_l2_feature_lists(df_full_input: pd.DataFrame, eval_spec: Dict,
    l2_cfg: Dict, paths: Dict, task_name: str, no_interactions_flag: bool,
    vfl_key: str, target_col: str) ->Tuple[List[str], List[str], List[str],
    List[str]]:
    logger.debug(f'[{task_name}] Preparing L2 feature lists...')
    dataset_output_root = Path(eval_spec['dataset_output_dir_name'])
    task_root_for_features = dataset_output_root / task_name
    h_num_all, h_cat_all, s_num_all, s_cat_all = _get_all_relevant_features(
        df_full_input, eval_spec, l2_cfg, task_root_for_features, task_name)
    leak_cols = {vfl_key, target_col}
    h_num = [c for c in h_num_all if c not in leak_cols]
    h_cat = [c for c in h_cat_all if c not in leak_cols]
    all_surr_unfiltered = [c for c in s_num_all + s_cat_all if c not in
        leak_cols]
    surr_filtered = [f for f in all_surr_unfiltered if not (
        no_interactions_flag and f.startswith('surrx_'))]
    s_num = [c for c in surr_filtered if c in df_full_input.columns and pd.
        api.types.is_numeric_dtype(df_full_input[c])]
    s_cat = [c for c in surr_filtered if c in df_full_input.columns and not
        pd.api.types.is_numeric_dtype(df_full_input[c])]
    logger.info(
        f'[{task_name}] Interaction flag (no_interactions_flag={no_interactions_flag!r}): {len(s_num)} num-surr, {len(s_cat)} cat-surr.'
        )
    return h_num, h_cat, s_num, s_cat


def _train_errorboost_model_l2(df_eb_train_raw: pd.DataFrame, y_eb_train:
    pd.Series, eb_num_feats: List[str], eb_cat_feats: List[str], task_def:
    Dict, l2_cfg: Dict, paths: Dict, task_name: str, sample_weight:
    Optional[pd.Series]=None) ->Tuple[Optional[Any], Optional[Any]]:
    logger.info(f'[{task_name}] Training ErrorBoost LGBM model...')
    if df_eb_train_raw.empty:
        logger.warning(
            f'[{task_name}] ErrorBoost training data is empty. Model will not be trained.'
            )
        return None, None
    current_eb_num = [f for f in eb_num_feats if f in df_eb_train_raw.columns]
    current_eb_cat = [f for f in eb_cat_feats if f in df_eb_train_raw.columns]
    if not current_eb_num and not current_eb_cat:
        logger.error(
            f'[{task_name}] No features available in ErrorBoost training data for modeling.'
            )
        return None, None
    X_eb_train_proc, eb_preproc = preprocess_features_for_lgbm(df_eb_train_raw
        [current_eb_num + current_eb_cat], current_eb_num, current_eb_cat,
        f'{task_name}_ErrorBoost', fit_mode=True)
    if X_eb_train_proc is None or X_eb_train_proc.empty:
        logger.warning(
            f'[{task_name}] Preprocessing ErrorBoost data resulted in empty set. Model not trained.'
            )
        return None, None
    eb_lgbm_params = l2_cfg.get('errorboost_lgbm_params', {}).copy()
    eb_lgbm_params['task_name_for_debug'] = f'{task_name}_ErrorBoost'
    aligned_y_eb_train_for_model = y_eb_train.reindex(X_eb_train_proc.index)
    if aligned_y_eb_train_for_model.isnull().any():
        num_missing = aligned_y_eb_train_for_model.isnull().sum()
        logger.warning(
            f'[{task_name}] {num_missing} target values became NaN after re-indexing y_eb_train to X_eb_train_proc. Filling with mode.'
            )
        if not aligned_y_eb_train_for_model.dropna().empty:
            mode_val = aligned_y_eb_train_for_model.mode()[0]
            aligned_y_eb_train_for_model = aligned_y_eb_train_for_model.fillna(
                mode_val)
        else:
            logger.error(
                f'[{task_name}] All target values for ErrorBoost training are NaN after re-indexing. Cannot train model.'
                )
            return None, None
    aligned_sample_weight = None
    if sample_weight is not None:
        if not X_eb_train_proc.empty:
            aligned_sample_weight = sample_weight.reindex(X_eb_train_proc.index
                )
            if aligned_sample_weight.isnull().any():
                logger.warning(
                    f'[{task_name}] Nulls found in aligned sample weights for ErrorBoost. Filling with 1.0.'
                    )
                aligned_sample_weight = aligned_sample_weight.fillna(1.0)
        else:
            logger.warning(
                f'[{task_name}] X_eb_train_proc is empty, cannot align sample_weight for ErrorBoost.'
                )
    errorboost_model = train_lgbm_model(X_eb_train_proc,
        aligned_y_eb_train_for_model, eb_lgbm_params, task_def['type'],
        sample_weight=aligned_sample_weight)
    if errorboost_model and not X_eb_train_proc.empty:
        logger.info(
            f'[{task_name}] Evaluating ErrorBoost model on its training partition (error segments)...'
            )
        eb_train_preds_proba = get_predictions(errorboost_model,
            X_eb_train_proc, task_def['type'])
        eb_metrics = {}
        if task_def['type'] == 'binary':
            if aligned_y_eb_train_for_model.nunique() > 1:
                eb_metrics['roc_auc_on_eb_train'] = roc_auc_score(
                    aligned_y_eb_train_for_model, eb_train_preds_proba)
                eb_metrics['auprc_on_eb_train'] = average_precision_score(
                    aligned_y_eb_train_for_model, eb_train_preds_proba)
            else:
                eb_metrics['roc_auc_on_eb_train'] = 0.5
                eb_metrics['auprc_on_eb_train'
                    ] = aligned_y_eb_train_for_model.mean(
                    ) if not aligned_y_eb_train_for_model.empty else 0.0
            eb_metrics['brier_on_eb_train'] = brier_score_loss(
                aligned_y_eb_train_for_model, eb_train_preds_proba)
        else:
            eb_metrics['mae_on_eb_train'] = mean_absolute_error(
                aligned_y_eb_train_for_model, eb_train_preds_proba)
        log_eb_metrics = {k: (round(v, 4) if isinstance(v, (float, np.
            floating)) else v) for k, v in eb_metrics.items()}
        logger.info(
            f'[{task_name}] ErrorBoost performance on its training data: {json.dumps(log_eb_metrics)}'
            )
        eb_metrics_path = paths['s3_out'] / l2_cfg['artifact_names'][
            'errorboost_train_metrics_filename']
        save_artifact(eb_metrics, eb_metrics_path, artifact_type='json')
    out_paths_s3 = paths['s3_out']
    save_artifact(errorboost_model, out_paths_s3 / l2_cfg['artifact_names']
        ['errorboost_lgbm_model_filename'])
    if eb_preproc:
        save_artifact(eb_preproc, out_paths_s3 / l2_cfg['artifact_names'][
            'errorboost_lgbm_preprocessor_filename'])
    logger.info(f'[{task_name}] ErrorBoost model and preprocessor saved.')
    return errorboost_model, eb_preproc


def _load_l2_validation_data_for_switchback(paths: Dict[str, Path], l2_cfg:
    Dict, eval_spec: Dict, df_full_input_indexed_by_key: pd.DataFrame,
    task_name: str) ->Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    logger.info(
        f'[{task_name}] Loading and preparing validation data for switch-back decisions...'
        )
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    target_col = eval_spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    l1_val_feat_fn = l2_cfg['artifact_names'][
        'l1_host_val_for_l2_switchback_features_filename']
    df_l1_val_feats_ids = load_artifact(paths['s1_in'] / l1_val_feat_fn,
        'L1 Validation Features for Switchback')
    l1_val_target_fn = l2_cfg['artifact_names'][
        'l1_host_val_for_l2_switchback_target_filename']
    df_l1_val_target = load_artifact(paths['s1_in'] / l1_val_target_fn,
        'L1 Validation Target for Switchback')
    if df_l1_val_feats_ids is None or df_l1_val_target is None:
        logger.error(
            f'[{task_name}] Missing L1 validation data for switch-back. Cannot proceed.'
            )
        return None, None
    df_l1_val_feats_ids = standardise_key(df_l1_val_feats_ids, vfl_key)
    df_l1_val_target = standardise_key(df_l1_val_target, vfl_key)
    y_val_switchback = df_l1_val_target.set_index(vfl_key)[target_col]
    val_vfl_keys = df_l1_val_feats_ids.index
    df_l2_val_all_feats = df_full_input_indexed_by_key.loc[
        df_full_input_indexed_by_key.index.intersection(val_vfl_keys)].copy()
    if 'host0_leaf_id' in df_l1_val_feats_ids.columns:
        df_l2_val_all_feats['host0_leaf_id'] = df_l1_val_feats_ids[
            'host0_leaf_id'].reindex(df_l2_val_all_feats.index)
    else:
        logger.error(
            f"[{task_name}] 'host0_leaf_id' missing in L1 validation features. Switch-back may be impaired."
            )
        return None, None
    if df_l2_val_all_feats.empty or y_val_switchback.empty:
        logger.warning(
            f'[{task_name}] Validation data for switch-back is empty after processing.'
            )
        return None, None
    y_val_switchback = y_val_switchback.reindex(df_l2_val_all_feats.index
        ).dropna()
    df_l2_val_all_feats = df_l2_val_all_feats.loc[y_val_switchback.index]
    logger.info(
        f'[{task_name}] Prepared L2 validation data for switch-back. Shape: {df_l2_val_all_feats.shape}'
        )
    return df_l2_val_all_feats, y_val_switchback


def _evaluate_segment_performance(df_segment_val: pd.DataFrame,
    y_segment_val: pd.Series, model: Any, preprocessor: Any, num_cols: List
    [str], cat_cols: List[str], task_def_type: str, metric_to_optimize: str,
    model_name_log_prefix: str, task_name: str) ->float:
    if df_segment_val.empty or y_segment_val.empty:
        logger.warning(
            f'[{task_name}] {model_name_log_prefix}: Segment validation data empty.'
            )
        return 0.0 if metric_to_optimize == 'roc_auc_score' else float('inf')
    model_features = [col for col in num_cols + cat_cols if col in
        df_segment_val.columns]
    if not model_features:
        logger.warning(
            f'[{task_name}] {model_name_log_prefix}: No features for model in segment data.'
            )
        return 0.0 if metric_to_optimize == 'roc_auc_score' else float('inf')
    X_segment_val_raw = df_segment_val[model_features]
    X_segment_val_proc, _ = preprocess_features_for_lgbm(X_segment_val_raw,
        num_cols, cat_cols, f'{task_name}_{model_name_log_prefix}_SegVal',
        fit_mode=False, existing_num_preprocessor=preprocessor)
    if X_segment_val_proc is None or X_segment_val_proc.empty:
        logger.warning(
            f'[{task_name}] {model_name_log_prefix}: Preprocessing segment validation data resulted in empty set.'
            )
        return 0.0 if metric_to_optimize == 'roc_auc_score' else float('inf')
    y_segment_val_aligned = y_segment_val.reindex(X_segment_val_proc.index)
    if y_segment_val_aligned.isnull().any():
        y_segment_val_aligned = y_segment_val_aligned.fillna(
            y_segment_val_aligned.mode()[0])
    preds = get_predictions(model, X_segment_val_proc, task_def_type)
    metric_alias = {'roc_auc_score': 'auc', 'auprc_score': 'auprc',
        'brier_score_loss': 'brier', 'f1_score': 'f1', 'precision_score':
        'precision', 'recall_score': 'recall', 'accuracy_score': 'accuracy'}
    eval_metric_name = metric_alias.get(metric_to_optimize,
        metric_to_optimize.lower().replace('_score', ''))
    score = evaluate_predictions(y_segment_val_aligned, preds,
        task_def_type, metric=eval_metric_name)
    if pd.isna(score):
        logger.warning(
            f'[{task_name}] {model_name_log_prefix}: Metric {metric_to_optimize} is NaN for segment. Using default bad score.'
            )
        score = (0.0 if task_def_type == 'binary' and eval_metric_name ==
            'auc' else float('inf'))
    return score


def _determine_optimal_model_for_segments(df_l2_val_all_feats: pd.DataFrame,
    y_val_switchback: pd.Series, host0_model: Any, host0_preproc: Any,
    host_native_num: List[str], host_native_cat: List[str],
    errorboost_model: Any, errorboost_preproc: Any, eb_num_feats: List[str],
    eb_cat_feats: List[str], all_error_leaf_ids: Set[int], task_def: Dict,
    l2_cfg: Dict, paths: Dict, task_name: str) ->Tuple[Dict[int, str], Dict
    [int, Dict[str, float]]]:
    logger.info(
        f'[{task_name}] Determining optimal model for error segments...')
    segment_model_decision_map: Dict[int, str] = {}
    segment_performance_details: Dict[int, Dict[str, float]] = {}
    metric_to_optimize = l2_cfg['evaluation']['metrics'][0] if l2_cfg[
        'evaluation']['metrics'] else 'roc_auc_score'
    for leaf_id in all_error_leaf_ids:
        segment_mask = df_l2_val_all_feats['host0_leaf_id'] == leaf_id
        df_segment_val = df_l2_val_all_feats[segment_mask]
        y_segment_val = y_val_switchback[segment_mask]
        if df_segment_val.empty or y_segment_val.empty or len(y_segment_val
            .unique()) < 2:
            logger.warning(
                f'[{task_name}] Segment for leaf_id {leaf_id} too small or uninformative in validation for comparison. Defaulting to Host0.'
                )
            segment_model_decision_map[leaf_id] = 'host0'
            segment_performance_details[leaf_id] = {'host0_perf': np.nan,
                'errorboost_perf': np.nan, 'chosen': 'host0', 'reason':
                'small_or_uninformative_val_segment'}
            continue
        host0_perf = _evaluate_segment_performance(df_segment_val,
            y_segment_val, host0_model, host0_preproc, host_native_num,
            host_native_cat, task_def['type'], metric_to_optimize, 'Host0',
            task_name)
        errorboost_perf = (0.0 if metric_to_optimize == 'roc_auc_score' else
            float('inf'))
        if errorboost_model:
            errorboost_perf = _evaluate_segment_performance(df_segment_val,
                y_segment_val, errorboost_model, errorboost_preproc,
                eb_num_feats, eb_cat_feats, task_def['type'],
                metric_to_optimize, 'ErrorBoost', task_name)
        else:
            logger.warning(
                f'[{task_name}] ErrorBoost model is None. Leaf {leaf_id} will use Host0.'
                )
        chosen_model = 'host0'
        reason = ''
        if errorboost_model:
            if metric_to_optimize in ['roc_auc_score', 'auprc_score',
                'f1_score', 'recall_score', 'precision_score', 'accuracy_score'
                ]:
                if errorboost_perf > host0_perf:
                    chosen_model = 'errorboost'
                    reason = (
                        f'ErrorBoost better ({errorboost_perf:.4f} > {host0_perf:.4f})'
                        )
                else:
                    reason = (
                        f'Host0 better or equal ({host0_perf:.4f} >= {errorboost_perf:.4f})'
                        )
            elif errorboost_perf < host0_perf:
                chosen_model = 'errorboost'
                reason = (
                    f'ErrorBoost better ({errorboost_perf:.4f} < {host0_perf:.4f})'
                    )
            else:
                reason = (
                    f'Host0 better or equal ({host0_perf:.4f} <= {errorboost_perf:.4f})'
                    )
        else:
            reason = 'ErrorBoost model None, defaulting to Host0'
        segment_model_decision_map[leaf_id] = chosen_model
        segment_performance_details[leaf_id] = {'host0_perf': host0_perf,
            'errorboost_perf': errorboost_perf if errorboost_model else np.
            nan, 'chosen_model': chosen_model, 'metric_optimized':
            metric_to_optimize, 'reason_for_choice': reason,
            'validation_segment_size': len(df_segment_val)}
        logger.info(
            f"[{task_name}] Leaf {leaf_id}: Host0 perf={host0_perf:.4f}, ErrorBoost perf={errorboost_perf if errorboost_model else 'N/A':.4f}. Chosen: {chosen_model}. Reason: {reason}"
            )
    map_artifact_name = l2_cfg['artifact_names'][
        'segment_model_decision_map_filename']
    save_artifact(segment_model_decision_map, paths['s3_out'] /
        map_artifact_name, artifact_type='json')
    perf_details_name = l2_cfg['artifact_names'][
        'segment_performance_comparison_filename']
    save_artifact(segment_performance_details, paths['s3_out'] /
        perf_details_name, artifact_type='json')
    return segment_model_decision_map, segment_performance_details


def _evaluate_and_save_l2_metrics(y_true: pd.Series, y_pred_proba: pd.
    Series, task_def_type: str, metrics_config_list: List[str], output_path:
    Path, task_name: str):
    logger.info(f'[{task_name}] Evaluating final L2 ensemble predictions...')
    if y_true.empty or y_pred_proba.empty or y_true.shape[0
        ] != y_pred_proba.shape[0]:
        logger.warning(
            f'[{task_name}] Cannot evaluate L2 metrics: y_true or y_pred_proba is empty or shapes mismatch.'
            )
        metrics_to_save = {metric: np.nan for metric in metrics_config_list}
    else:
        y_pred_labels = (y_pred_proba > 0.5).astype(int
            ) if task_def_type == 'binary' else None
        metrics_to_save = {}
        metric_map = {'roc_auc_score': lambda : roc_auc_score(y_true,
            y_pred_proba) if y_true.nunique() > 1 else 0.5, 'auprc_score': 
            lambda : average_precision_score(y_true, y_pred_proba) if 
            y_true.nunique() > 1 else y_true.mean(), 'accuracy_score': lambda :
            accuracy_score(y_true, y_pred_labels) if y_pred_labels is not
            None else np.nan, 'f1_score': lambda : f1_score(y_true,
            y_pred_labels, zero_division=0) if y_pred_labels is not None else
            np.nan, 'precision_score': lambda : precision_score(y_true,
            y_pred_labels, zero_division=0) if y_pred_labels is not None else
            np.nan, 'recall_score': lambda : recall_score(y_true,
            y_pred_labels, zero_division=0) if y_pred_labels is not None else
            np.nan, 'brier_score_loss': lambda : brier_score_loss(y_true,
            y_pred_proba)}
        for name in metrics_config_list:
            if name in metric_map and (task_def_type == 'binary' or name not in
                ['roc_auc_score', 'auprc_score', 'brier_score_loss']):
                try:
                    metrics_to_save[name] = metric_map[name]()
                except Exception as e:
                    logger.warning(f'[{task_name}] Metric {name} error: {e}')
                    metrics_to_save[name] = np.nan
            elif name.endswith(('_ci_low', '_ci_up', '_std')):
                metrics_to_save[name] = np.nan
    log_metrics = {k: (round(v, 4) if isinstance(v, (float, np.floating)) else
        v) for k, v in metrics_to_save.items() if pd.notna(v)}
    logger.info(
        f'[{task_name}] Final L2 Ensemble Metrics: {json.dumps(log_metrics)}')
    save_artifact(metrics_to_save, output_path, artifact_type='json')


def _analyze_routing_performance(y_true: pd.Series, y_pred: pd.Series, mask:
    np.ndarray, path_name: str, task_type: str) ->Dict[str, Any]:
    y_path = y_true[mask]
    pred_path = y_pred[mask]
    if y_path.empty or pred_path.empty:
        return {}
    metrics = {f'{path_name}_count': len(y_path), f'{path_name}_mean_pred':
        float(pred_path.mean()), f'{path_name}_actual_prevalence': float(
        y_path.mean())}
    if task_type == 'binary' and y_path.nunique() > 1:
        metrics[f'{path_name}_auc'] = roc_auc_score(y_path, pred_path)
    elif task_type == 'regression':
        metrics[f'{path_name}_mae'] = mean_absolute_error(y_path, pred_path)
    return metrics


def _compute_test_routing_summary(y_test: pd.Series, final_preds: pd.Series,
    actual_model_routed_series: pd.Series, task_type: str, task_name: str
    ) ->Dict[str, Any]:
    summary = {}
    for model_path_name in actual_model_routed_series.unique():
        mask = actual_model_routed_series == model_path_name
        path_metrics = _analyze_routing_performance(y_test, final_preds,
            mask, model_path_name, task_type)
        summary.update(path_metrics)


def _calculate_errorboost_sample_weights(df_eb_train_indexed: pd.DataFrame,
    selected_rules_info: Dict[str, Dict], weighting_config: Dict, task_name:
    str) ->pd.Series:
    logger.info(
        f"[{task_name}] Calculating ErrorBoost sample weights (mode: {weighting_config.get('pi_mode')})..."
        )
    w_default = weighting_config.get('w_tilde_default', 1.0)
    if not weighting_config.get('enabled', False):
        logger.info(
            f'[{task_name}] Sample weighting is disabled. Using uniform default weights ({w_default}).'
            )
        return pd.Series(w_default, index=df_eb_train_indexed.index)
    if not selected_rules_info:
        logger.warning(
            f"[{task_name}] No selected_rules_info provided for 'rule_gain_proportional' weighting. Using default weights."
            )
        return pd.Series(w_default, index=df_eb_train_indexed.index)
    instance_propensity = pd.Series(0.0, index=df_eb_train_indexed.index)
    all_gains = [details.get('f1_k', 0.0) for details in
        selected_rules_info.values() if isinstance(details, dict)]
    max_overall_gain = max(all_gains) if all_gains else 0.0
    if max_overall_gain <= 0:
        logger.warning(
            f'[{task_name}] Max overall rule gain is <= 0 ({max_overall_gain:.4f}). Cannot use gain for proportional weighting. Using default weights.'
            )
        return pd.Series(w_default, index=df_eb_train_indexed.index)
    for rule_id, rule_details in selected_rules_info.items():
        if not isinstance(rule_details, dict):
            continue
        rule_gain = rule_details.get('f1_k', 0.0)
        covered_keys_for_rule = map(str, rule_details.get(
            'covered_instance_ids_by_remote', []))
        for vfl_key in covered_keys_for_rule:
            if vfl_key in instance_propensity.index:
                instance_propensity.loc[vfl_key] = max(instance_propensity.
                    loc[vfl_key], rule_gain)
    scaled_propensity = instance_propensity / max_overall_gain
    clip_max_factor = weighting_config.get('clip_max_factor', 5.0)
    additional_weight_range = w_default * (clip_max_factor - 1.0)
    final_weights = w_default + scaled_propensity * additional_weight_range
    min_final_weight = weighting_config.get('min_weight_after_clip', 0.1)
    max_abs_weight = w_default * clip_max_factor
    final_weights = final_weights.clip(lower=min_final_weight, upper=
        max_abs_weight)
    logger.info(
        f'[{task_name}] ErrorBoost sample weights calculated. Min: {final_weights.min():.2f}, Max: {final_weights.max():.2f}, Mean: {final_weights.mean():.2f}'
        )
    return final_weights


def generate_prediction_audit_output(df: pd.DataFrame, y_true: pd.Series,
    y_pred_prob: pd.Series, routed_by: pd.Series, vfl_key: str, output_dir:
    Path, task_name: str, threshold: float=0.5) ->None:
    df = df.copy()
    df['predicted_proba'] = y_pred_prob
    df['predicted_label'] = (y_pred_prob >= threshold).astype(int)
    df['true_label'] = y_true
    df['routed_by'] = routed_by
    df['correct'] = (df['predicted_label'] == df['true_label']).astype(int)
    false_negatives = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)
        ]
    false_positives = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)
        ]
    key_cols = [vfl_key, 'predicted_proba', 'predicted_label', 'true_label',
        'routed_by', 'correct']
    audit_cols = key_cols + [c for c in df.columns if c.startswith((
        'llm_surr_', 'p_hat_rule_'))]
    fn_path = output_dir / f'{task_name}_false_negatives.csv'
    fp_path = output_dir / f'{task_name}_false_positives.csv'
    false_negatives[audit_cols].to_csv(fn_path, index=False)
    false_positives[audit_cols].to_csv(fp_path, index=False)
    return {'false_negatives_saved': str(fn_path), 'false_positives_saved':
        str(fp_path), 'false_negatives_count': len(false_negatives),
        'false_positives_count': len(false_positives)}


def process_task_l2_ensemble(task_name: str, task_def: Dict, l2_cfg: Dict,
    eval_spec: Dict, paths: Dict, no_interactions_flag: bool=False):
    logger.info(
        f'[{task_name}] L2 Ensemble (Local Surrogates). Interactions excluded: {no_interactions_flag}.'
        )
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    target_col = eval_spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    df_aug_host_full, df_target_full, l1_splits = (
        _load_l2_main_input_artifacts(paths, l2_cfg, task_name))
    if df_aug_host_full is None:
        return
    host0_model, host0_preproc, host0_bat, error_leaf_ids = (
        _load_l2_model_artifacts(paths, l2_cfg, task_name))
    if host0_model is None:
        return
    df_full_input = _prepare_l2_input_dataframe(df_aug_host_full,
        df_target_full, vfl_key, target_col, task_name)
    if df_full_input is None:
        return
    df_full_input = df_full_input.set_index(vfl_key, drop=False)
    y_full = df_full_input[target_col].copy()
    y_full = df_full_input[target_col]
    df_full_input = _harmonise_categorical_dtypes(df_full_input.drop(
        columns=[target_col]))
    df_full_input[target_col] = y_full
    train_vfl_keys, test_vfl_keys = _validate_and_get_l2_split_indices(
        l1_splits, df_full_input, vfl_key, task_name)
    if train_vfl_keys is None:
        return
    df_full_input_indexed = df_full_input.set_index(vfl_key, drop=False)
    df_l2_val_all_feats, y_val_switchback = None, None
    val_vfl_keys_from_l1 = l1_splits.get('val_idx', [])
    if val_vfl_keys_from_l1:
        val_vfl_keys_from_l1_str = [str(k) for k in val_vfl_keys_from_l1]
        intersecting_val_keys = df_full_input_indexed.index.intersection(pd
            .Index(val_vfl_keys_from_l1_str))
        if not intersecting_val_keys.empty:
            df_l2_val_all_feats_temp = df_full_input_indexed.loc[
                intersecting_val_keys].copy()
            y_val_switchback_temp = y_full.loc[intersecting_val_keys].copy()
            l1_val_feat_fn = l2_cfg['artifact_names'][
                'l1_host_val_for_l2_switchback_features_filename']
            df_l1_val_feat_info = load_artifact(paths['s1_in'] /
                l1_val_feat_fn, 'L1 Val Feats for Leaf IDs')
            if df_l1_val_feat_info is not None:
                df_l1_val_feat_info = standardise_key(df_l1_val_feat_info,
                    vfl_key)
                if 'host0_leaf_id' in df_l1_val_feat_info.columns:
                    df_l2_val_all_feats_temp['host0_leaf_id'
                        ] = df_l1_val_feat_info['host0_leaf_id'].reindex(
                        df_l2_val_all_feats_temp.index)
                    df_l2_val_all_feats = df_l2_val_all_feats_temp.dropna(
                        subset=['host0_leaf_id'])
                    y_val_switchback = y_val_switchback_temp.loc[
                        df_l2_val_all_feats.index]
                else:
                    logger.warning(
                        f"[{task_name}] 'host0_leaf_id' not in L1 validation feature artifact. Switch-back impaired."
                        )
            else:
                logger.warning(
                    f'[{task_name}] Failed to load L1 validation features artifact. Switch-back impaired.'
                    )
        else:
            logger.warning(
                f'[{task_name}] No overlap between L1 validation keys and L2 full data. Switch-back impaired.'
                )
    else:
        logger.warning(
            f'[{task_name}] L1 validation indices not found. Switch-back impaired.'
            )
    df_train_with_surr = df_full_input_indexed.loc[train_vfl_keys].copy()
    y_train = y_full.loc[train_vfl_keys].copy()
    df_test_with_surr = pd.DataFrame()
    y_test = pd.Series(dtype=y_full.dtype)
    if test_vfl_keys is not None and not test_vfl_keys.empty:
        df_test_with_surr = df_full_input_indexed.loc[test_vfl_keys].copy()
        y_test = y_full.loc[test_vfl_keys].copy()
    else:
        logger.warning(
            f'[{task_name}] Test set is empty based on L1 indices or validation.'
            )
    df_train_with_surr.reset_index(drop=True, inplace=True)
    if not df_test_with_surr.empty:
        df_test_with_surr.reset_index(drop=True, inplace=True)
    host_native_num, host_native_cat, all_surr_num, all_surr_cat = (
        _prepare_l2_feature_lists(df_full_input, eval_spec, l2_cfg, paths,
        task_name, no_interactions_flag, vfl_key, target_col))
    feature_stats = {'native_numeric_count': len(host_native_num),
        'native_categorical_count': len(host_native_cat),
        'surrogate_numeric_count': len(all_surr_num),
        'surrogate_categorical_count': len(all_surr_cat),
        'surrogate_prefixes': list(l2_cfg.get('surr_feature_prefixes', (
        'p_hat_rule_', 'surr_', 'surrx_')))}
    stats_path = paths['s3_out'] / l2_cfg['artifact_names'][
        'feature_discovery_stats_filename']
    save_artifact(feature_stats, stats_path, artifact_type='json')
    df_train_with_surr_indexed = df_train_with_surr.set_index(vfl_key, drop
        =False)
    df_eb_train_raw, y_eb_train = _define_l2_training_partitions(
        df_train_with_surr_indexed, y_train, host0_bat, error_leaf_ids,
        host_native_num, host_native_cat, task_name)
    sample_weights_eb = None
    if l2_cfg.get('sample_weighting', {}).get('enabled', False):
        l1_rules_filename = l2_cfg['artifact_names'][
            'l1_selected_remote_rules_filename']
        l1_rules_path = paths['s1_in'] / l1_rules_filename
        selected_rules_info = load_artifact(l1_rules_path,
            'L1 Selected Remote Rules for S-hat')
        if selected_rules_info:
            current_eb_train_index_name = df_eb_train_raw.index.name
            if current_eb_train_index_name != vfl_key:
                if vfl_key not in df_eb_train_raw.columns:
                    logger.error(
                        f"[{task_name}] VFL_KEY '{vfl_key}' not in df_eb_train_raw columns for re-indexing. Skipping sample weighting."
                        )
                else:
                    df_eb_train_raw_indexed_for_weights = (df_eb_train_raw.
                        set_index(vfl_key, drop=False))
                    sample_weights_eb = _calculate_errorboost_sample_weights(
                        df_eb_train_raw_indexed_for_weights,
                        selected_rules_info, l2_cfg['sample_weighting'],
                        task_name)
                    sample_weights_eb = sample_weights_eb.reindex(
                        df_eb_train_raw.index)
            else:
                sample_weights_eb = _calculate_errorboost_sample_weights(
                    df_eb_train_raw, selected_rules_info, l2_cfg[
                    'sample_weighting'], task_name)
        else:
            logger.warning(
                f'[{task_name}] Could not load L1 selected remote rules. Sample weighting skipped.'
                )
        if sample_weights_eb is not None and l2_cfg.get('sample_weighting', {}
            ).get('enabled', False):
            weights_artifact_key = 'errorboost_sample_weights'
            weights_artifact_name = l2_cfg['artifact_names'].get(
                weights_artifact_key)
            if weights_artifact_name:
                weights_to_save_dict = {str(idx): weight for idx, weight in
                    sample_weights_eb.items()}
                weights_output_path = paths['s3_out'] / weights_artifact_name
                save_artifact(weights_to_save_dict, weights_output_path,
                    artifact_type='json')
                logger.info(
                    f'[{task_name}] Saved ErrorBoost sample weights to {weights_output_path}'
                    )
            else:
                logger.warning(
                    f"[{task_name}] Artifact name for '{weights_artifact_key}' not defined in l2_config.json. Skipping save."
                    )
    eb_num_feats = host_native_num + all_surr_num
    eb_cat_feats = host_native_cat + all_surr_cat
    errorboost_model, errorboost_preprocessor = _train_errorboost_model_l2(
        df_eb_train_raw, y_eb_train, eb_num_feats, eb_cat_feats, task_def,
        l2_cfg, paths, task_name, sample_weight=sample_weights_eb)
    if df_test_with_surr.empty:
        logger.warning(
            f'[{task_name}] Test data is empty. Skipping ensemble inference and evaluation.'
            )
        metrics_path = paths['s3_out'] / l2_cfg['artifact_names'][
            'final_ensemble_metrics_filename']
        _evaluate_and_save_l2_metrics(pd.Series(dtype='float64'), pd.Series
            (dtype='float64'), task_def['type'], l2_cfg['evaluation'][
            'metrics'], metrics_path, task_name)
        logger.info(
            f'[{task_name}] L2 Ensemble processing finished due to empty test set.'
            )
        return
    if errorboost_model is None:
        logger.warning(
            f'[{task_name}] ErrorBoost model not available. Ensemble will rely solely on Host0.'
            )
    segment_model_decision_map = {}
    segment_performance_details = {}
    if (df_l2_val_all_feats is not None and y_val_switchback is not None and
        len(error_leaf_ids) > 0):
        segment_model_decision_map, segment_performance_details = (
            _determine_optimal_model_for_segments(df_l2_val_all_feats,
            y_val_switchback, host0_model, host0_preproc, host_native_num,
            host_native_cat, errorboost_model, errorboost_preprocessor,
            eb_num_feats, eb_cat_feats, error_leaf_ids, task_def, l2_cfg,
            paths, task_name))
    else:
        logger.warning(
            f'[{task_name}] Skipping segment model optimization due to missing validation data or no error leaves.'
            )
        if errorboost_model:
            for leaf_id_val in error_leaf_ids:
                segment_model_decision_map[int(leaf_id_val)] = 'errorboost'
        else:
            for leaf_id_val in error_leaf_ids:
                segment_model_decision_map[int(leaf_id_val)] = 'host0'
    map_artifact_name = l2_cfg['artifact_names'][
        'segment_model_decision_map_filename']
    save_artifact(segment_model_decision_map, paths['s3_out'] /
        map_artifact_name, artifact_type='json', desc=
        'Segment Model Decision Map')
    if segment_performance_details:
        perf_details_name = l2_cfg['artifact_names'][
            'segment_performance_comparison_filename']
        save_artifact(segment_performance_details, paths['s3_out'] /
            perf_details_name, artifact_type='json', desc=
            'Segment Performance Comparison')
    df_test_with_surr_indexed = df_test_with_surr.set_index(vfl_key, drop=False
        )
    final_preds_test, final_eb_path_mask, actual_model_routed_series = (
        _make_ensemble_predictions(df_test_with_surr_indexed,
        errorboost_model, errorboost_preprocessor, eb_num_feats,
        eb_cat_feats, host0_model, host0_preproc, host_native_num,
        host_native_cat, host0_bat, error_leaf_ids,
        segment_model_decision_map, task_def['type'], task_name, vfl_key))
    if final_preds_test is None or final_preds_test.empty:
        logger.error(
            f'[{task_name}] Failed to make ensemble predictions or predictions are empty. Aborting.'
            )
        return
    test_routing_summary = _compute_test_routing_summary(y_test,
        final_preds_test, actual_model_routed_series, task_def['type'],
        task_name)
    if test_routing_summary:
        logger.info(
            f'[{task_name}] Test Set Performance Breakdown: {json.dumps(test_routing_summary, cls=NpEncoder)}'
            )
        routing_summary_path = paths['s3_out'] / l2_cfg['artifact_names'][
            'test_routing_breakdown_filename']
        save_artifact(test_routing_summary, routing_summary_path,
            artifact_type='json')
    metrics_output_path = paths['s3_out'] / l2_cfg['artifact_names'][
        'final_ensemble_metrics_filename']
    _evaluate_and_save_l2_metrics(y_test, final_preds_test, task_def['type'
        ], l2_cfg['evaluation']['metrics'], metrics_output_path, task_name)
    generate_prediction_audit_output(df=df_test_with_surr_indexed, y_true=
        y_test, y_pred_prob=final_preds_test, routed_by=
        actual_model_routed_series, vfl_key=vfl_key, output_dir=paths[
        's3_out'], task_name=task_name)
    logger.info(
        f"[{task_name}] L2 Ensemble processing finished. Artifacts in {paths['s3_out']}"
        )


def main():
    script_dir = Path(__file__).resolve().parent
    l2_cfg = load_json_config(script_dir / 'l2_config.json')
    eval_spec = load_json_config(script_dir / 'evaluation_spec.json')
    if not l2_cfg or not eval_spec:
        logger.error('Failed to load configuration files')
        sys.exit(1)
    global GLOBAL_CONFIG_SEED
    GLOBAL_CONFIG_SEED = l2_cfg.get('seed', 42)
    np.random.seed(GLOBAL_CONFIG_SEED)
    logger.info(f'Global seed set to: {GLOBAL_CONFIG_SEED}')
    dataset_output_root = Path(eval_spec['dataset_output_dir_name'])
    stage_templates = {'s1_in': l2_cfg['script1_input_subdir_template'].
        rstrip('/'), 's3_out': l2_cfg['script3_output_subdir_template'].
        rstrip('/')}
    for task_name, task_def in eval_spec.get('tasks', {}).items():
        logger.info(f'===== L2 Processing Task: {task_name} =====')
        task_def['type'] = task_def.get('type', 'binary')
        task_root = dataset_output_root / task_name
        task_paths = {key: (task_root / Path(subdir)) for key, subdir in
            stage_templates.items()}
        ensure_dir_exists(task_paths['s3_out'])
        try:
            process_task_l2_ensemble(task_name, task_def, l2_cfg, eval_spec,
                task_paths, no_interactions_flag=False)
        except Exception as e:
            logger.error(f"L2 exception for task '{task_name}': {e}",
                exc_info=True)
    logger.info('L2 script finished all tasks.')


if __name__ == '__main__':
    main()
