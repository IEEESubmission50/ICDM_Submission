from string import Template
from sklearn.tree import export_text
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import shap
from sqlalchemy import create_engine, text
from utils import tag_feature_origin, get_base_feature_name_for_penalty, _create_numeric_preprocessing_pipeline, _impute_and_cast_categoricals, preprocess_features_for_lgbm, _create_bat_preprocessor, train_bat_model, _initialize_lgbm_training_config, _prepare_lgbm_training_data_and_cats, _apply_lgbm_dynamic_hyperparams_for_penalty, _construct_lgbm_feature_penalties, _finalize_lgbm_model_params_for_constructor, _setup_lgbm_early_stopping_and_eval, _instantiate_and_fit_lgbm_model, train_lgbm_model, _robust_zscore, analyze_shap_by_origin, calculate_binary_classification_metrics, get_s_hat_performance_stats, load_json_config, ensure_dir_exists, save_artifact, load_artifact, standardise_key, ensure_sorted_unique_key, _get_lgbm_cat_feature_names, canonicalize_feature_name, canonicalize_feature_dict, _fmt, format_schema_for_llm_prompt, _safe_to_numeric, get_predictions, evaluate_predictions, split_data_and_save_indices, sanitize_feature_names, sanitize_feature_name, pretty_print_shap
import warnings
import re
import numpy as np
import pandas as pd
import shap
import json
from string import Template
import json, textwrap
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, f1_score, precision_score, recall_score
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', message='LightGBM binary classifier*',
    module='shap')
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
GLOBAL_CONFIG_SEED = 42
L1_CONFIG_FILE = Path('l1_config.json')
EVAL_SPEC_FILE = Path('evaluation_spec.json')
eval_spec: Dict[str, Any]
l1_cfg: Dict[str, Any]


def _generate_pre_remote_host_surrogates(df_input: pd.DataFrame,
    host_native_num_cols: List[str], host_native_cat_cols: List[str],
    host0_lgbm_model: Any, host0_lgbm_preprocessor: Any, l1_cfg: Dict,
    task_name: str, vfl_key: str, fixed_source_cols: Optional[List[str]]=
    None, out_dir: Optional[Path]=None) ->Tuple[pd.DataFrame, List[str],
    List[str]]:
    cfg = l1_cfg.get('pre_remote_surrogate_generation', {})
    if not cfg.get('enabled', False):
        return df_input.copy(), [], []
    logger.info(
        f'[{task_name}] Generating pre-remote host surrogate features...')
    df_augmented = df_input.copy()
    new_surrogate_names = []
    method = cfg.get('method', 'robustify_all_numeric_host_features')
    prefix = cfg.get('new_surrogate_prefix', 'pre_surrx_')
    if fixed_source_cols:
        features_to_robustify = [c for c in fixed_source_cols if c in
            df_augmented.columns]
        method = 'fixed_list_replay'
    else:
        features_to_robustify = []
    if method == 'shap_driven_robustify_on_error_set':
        top_n_shap = cfg.get('top_n_shap_for_host_robustification', 3)
        if df_input.empty:
            logger.warning(
                f'[{task_name}] Input for SHAP-driven pre-remote surrogates is empty. Skipping.'
                )
            return df_augmented, new_surrogate_names
        X_error_set_raw = df_input[host_native_num_cols + host_native_cat_cols]
        X_error_set_proc, _ = preprocess_features_for_lgbm(X_error_set_raw,
            host_native_num_cols, host_native_cat_cols,
            f'{task_name}_Host0ErrorSetForSHAP', fit_mode=False,
            existing_num_preprocessor=host0_lgbm_preprocessor)
        if X_error_set_proc is None or X_error_set_proc.empty:
            logger.warning(
                f'[{task_name}] Preprocessing for SHAP-driven pre-remote surrogates failed. Robustifying all numeric instead.'
                )
            method = 'robustify_all_numeric_host_features'
        else:
            try:
                explainer = shap.TreeExplainer(host0_lgbm_model)
                shap_values_eb = explainer.shap_values(X_error_set_proc)
                shap_matrix = shap_values_eb[1] if isinstance(shap_values_eb,
                    list) and len(shap_values_eb) == 2 else shap_values_eb
                mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
                feature_shap_pairs = sorted(zip(X_error_set_proc.columns,
                    mean_abs_shap), key=lambda x: x[1], reverse=True)
                for proc_feat_name, _ in feature_shap_pairs[:top_n_shap]:
                    original_name = canonicalize_feature_name(proc_feat_name)
                    if original_name in host_native_num_cols:
                        features_to_robustify.append(original_name)
                logger.info(
                    f'[{task_name}] Identified top {top_n_shap} host features for pre-remote robustification via SHAP: {features_to_robustify}'
                    )
            except Exception as e:
                logger.warning(
                    f'[{task_name}] SHAP calculation for pre-remote surrogates failed: {e}. Robustifying all numeric instead.'
                    )
                method = 'robustify_all_numeric_host_features'
    if method == 'robustify_all_numeric_host_features':
        features_to_robustify = [col for col in host_native_num_cols if col in
            df_augmented.columns]
        logger.info(
            f'[{task_name}] Robustifying all {len(features_to_robustify)} available host numeric features for pre-remote surrogates.'
            )
    if not features_to_robustify:
        logger.warning(
            f'[{task_name}] No host numeric features identified to robustify for pre-remote surrogates.'
            )
    for feat_name in features_to_robustify:
        if feat_name not in df_augmented.columns:
            logger.warning(
                f"[{task_name}] Feature '{feat_name}' for pre-remote robustification not found in DataFrame. Skipping."
                )
            continue
        feat_series_raw = df_augmented[feat_name]
        feat_series_num = _safe_to_numeric(feat_series_raw)
        if feat_series_num is None:
            logger.debug(
                f"[{task_name}] Skipping pre-remote robustification for non-numeric feature '{feat_name}'."
                )
            continue
        z_scores = _robust_zscore(feat_series_num)
        surr_col_name = f'{prefix}anti_{sanitize_feature_name(feat_name)}'
        df_augmented[surr_col_name] = 1.0 / (1.0 + np.exp(z_scores.clip(-10,
            10)))
        new_surrogate_names.append(surr_col_name)
    logger.info(
        f'[{task_name}] Generated {len(new_surrogate_names)} pre-remote host surrogate features.'
        )
    if not fixed_source_cols and out_dir is not None:
        art_key = 'pre_remote_surrogate_source_cols_json'
        fn = out_dir / l1_cfg['artifact_names'].get(art_key,
            'pre_remote_surrogate_source_cols.json')
        save_artifact(features_to_robustify, fn, art_key, 'json')
    return df_augmented, new_surrogate_names, features_to_robustify


def _load_prompt_template(template_filename: str, task_name: str) ->Optional[
    Template]:
    script_dir = Path(__file__).resolve().parent
    template_path = script_dir / template_filename
    if not template_path.exists():
        template_path = script_dir / 'prompts' / template_filename
        if not template_path.exists():
            logger.error(
                f'[{task_name}] Prompt template file not found: {template_filename}'
                )
            return None
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return Template(f.read())
    except Exception as e:
        logger.error(
            f'[{task_name}] Error loading prompt template {template_path}: {e}'
            )
        return None


def _read_artifact_txt(path: Path, task: str) ->str:
    try:
        txt = path.read_text()[:20000]
        return txt or 'N/A'
    except Exception as e:
        logger.warning('[%s] Could not read %s: %s', task, path.name, e)
        return 'N/A'


def _summarise_error_segments(host_err_ids: List[int], y_err: pd.Series,
    task: str) ->str:
    pos_rate = y_err.mean() if not y_err.empty else 0.0
    return (
        f'Error-set rows: {len(y_err)} | positive-rate: {pos_rate:.3f} | Host-0 BAT leaf ids flagged as error: {sorted(host_err_ids)[:10]}...'
        )


def _get_llm_context_task_info(eval_spec: dict, task_name: str) ->str:
    return eval_spec.get('tasks', {}).get(task_name, {}).get('description',
        'N/A')


def _get_llm_context_target_rule_info(selected_rules: dict) ->tuple[str, str]:
    target_id = next(iter(selected_rules), 'N/A')
    rule_info = selected_rules.get(target_id, {})
    description = rule_info.get('human_rule_text',
        'No rule description available.')
    return target_id, description


def _format_rules_for_llm(selected_rules: dict) ->str:
    if not selected_rules:
        return 'None selected – remote model added no rules.'
    lines = []
    for rid, meta in selected_rules.items():
        support = meta.get('support')
        lift = meta.get('lift')
        support_str = f'{support:.3f}' if isinstance(support, (int, float)
            ) else '?'
        lift_str = f'{lift:.2f}' if isinstance(lift, (int, float)) else '?'
        lines.append(
            f"{rid} | {support_str} | {lift_str} | {meta.get('human_rule_text', '')[:60]}"
            )
    return '\n'.join(lines)


def _format_stats_for_llm(df: pd.DataFrame, y_proxy: pd.Series, num_cols:
    list[str], cat_cols: list[str]) ->str:
    mi_scores = {}
    for col in (num_cols + cat_cols):
        if col not in df.columns:
            continue
        col_series = df[col]
        valid_mask = col_series.notna() & y_proxy.notna()
        if valid_mask.sum() == 0:
            continue
        if col_series.dtype == 'O' or isinstance(col_series.dtype, pd.
            CategoricalDtype):
            mi_scores[col] = mutual_info_score(col_series[valid_mask],
                y_proxy[valid_mask])
        else:
            binned = pd.qcut(col_series[valid_mask], 10, duplicates='drop')
            mi_scores[col] = mutual_info_score(binned, y_proxy[valid_mask])
    top_mi = sorted(mi_scores.items(), key=lambda item: item[1], reverse=True)[
        :8]
    return '\n'.join([f'- {col:<30} {score:0.4f}' for col, score in top_mi])


def _format_examples_for_llm(df: pd.DataFrame, cols: list[str],
    random_state: int) ->str:
    return df[cols].sample(5, random_state=random_state).to_markdown(index=
        False)


def _parse_bat_rules(bat_model: Any) ->Dict[int, str]:
    tree = bat_model.tree_
    feature_names = bat_model.bat_feature_names_
    node_paths = {}
    leaf_rules = {}

    def recurse(node_id, path_conditions):
        if tree.feature[node_id] == -2:
            leaf_rules[node_id] = ' and '.join(path_conditions)
            return
        name = feature_names[tree.feature[node_id]]
        threshold = f'{tree.threshold[node_id]:.4f}'
        left_path = path_conditions + [f'{name} <= {threshold}']
        recurse(tree.children_left[node_id], left_path)
        right_path = path_conditions + [f'{name} > {threshold}']
        recurse(tree.children_right[node_id], right_path)
    recurse(0, [])
    return leaf_rules


def _format_schema_for_llm(df: pd.DataFrame, cols: list[str],
    column_descriptions: dict, max_examples: int=3) ->str:
    lines = []
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        dtype_str = str(series.dtype)
        unique_vals = series.unique()
        if pd.api.types.is_numeric_dtype(series.dtype) and len(unique_vals
            ) > max_examples * 2:
            sample_vals = (
                f'Min:{series.min():.2f}, Median:{series.median():.2f}, Max:{series.max():.2f}'
                )
        else:
            sample_vals = ', '.join(map(str, unique_vals[:max_examples]))
            if len(unique_vals) > max_examples:
                sample_vals += ', ...'
        desc = column_descriptions.get(col, 'No description provided.')
        lines.append(
            f'- `{col}` (type: {dtype_str}, e.g., [{sample_vals}]). Desc: {desc}'
            )
    return '\n'.join(lines) if lines else '(No features to describe)'


def _prepare_llm_prompt_and_data_for_s_hat_surrogates(selected_rules: dict,
    df_host_train: pd.DataFrame, host_native_num_cols: list[str],
    host_native_cat_cols: list[str], pre_remote_surrogate_names: list[str],
    eval_spec: dict, l1_cfg: dict, task_name: str, out_dir: Path, vfl_key:
    str, remote_decision_tree_text: str=''):
    tmpl_path = Path(l1_cfg['llm_s_hat_prompt_generation'][
        'prompt_template_file'])
    template = _load_prompt_template(str(tmpl_path), task_name)
    if not template:
        return
    target_id = next(iter(selected_rules), None)
    target_rule = selected_rules.get(target_id, {})
    supp_rules_block = []
    for rid, meta in selected_rules.items():
        if rid == target_id:
            continue
        rule_info = f"""  - Rule ID: {rid}
    Logic: {meta.get('human_rule_text')}
    Lift: {meta.get('lift', 0):.2f}, Support: {meta.get('remote_rule_support', 0)}"""
        supp_rules_block.append(rule_info)
    all_host_cols = (host_native_num_cols + host_native_cat_cols +
        pre_remote_surrogate_names)
    substitutions = {'task_description': eval_spec['tasks'][task_name][
        'description'], 'target_remote_rule_id': target_id or 'N/A',
        'target_remote_rule_logic': target_rule.get('human_rule_text',
        'N/A'), 'supplemental_rules_block': '\n'.join(supp_rules_block) if
        supp_rules_block else 'No supplemental rules.',
        'remote_decision_tree_text': textwrap.indent(
        remote_decision_tree_text.strip(), '# '), 'host_data_schema':
        _format_schema_for_llm(df_host_train, all_host_cols, eval_spec.get(
        'column_descriptions', {})), 'num_features_to_generate': l1_cfg.get
        ('llm_s_hat_prompt_generation', {}).get('num_features_to_generate', 8)}
    prompt_txt = template.safe_substitute(substitutions)
    prompt_path = out_dir / l1_cfg['artifact_names']['llm_s_hat_prompt_txt']
    prompt_path.write_text(prompt_txt, encoding='utf-8')
    cols_for_llm = [vfl_key] + list(dict.fromkeys(all_host_cols))
    csv_path = out_dir / l1_cfg['artifact_names'][
        'host_data_for_llm_s_hat_input_csv']
    df_host_train[cols_for_llm].to_csv(csv_path, index=False)
    logger.info(
        f'[{task_name}] Saved rich LLM prompt and data for surrogate generation.'
        )
    return prompt_path


def get_db_engine(db_config_module_name: str):
    try:
        db_cfg = getattr(__import__(db_config_module_name), 'DB_CONFIG')
        uri = (
            f"mysql+mysqlconnector://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
            )
        engine = create_engine(uri, pool_recycle=3600, connect_args={
            'connect_timeout': 10})
        logger.info(f"DB engine created for {db_cfg['database']}")
        return engine
    except Exception as e:
        logger.error(f'Error creating DB engine: {e}', exc_info=True)
        return None


def load_data_from_db(sql_query: str, db_engine: Any, task_name: str
    ) ->pd.DataFrame:
    if not db_engine:
        return pd.DataFrame()
    logger.info(f'[{task_name}] Executing SQL: {sql_query[:200]}...')
    try:
        with db_engine.connect() as conn:
            df = pd.read_sql_query(text(sql_query), conn)
        df = df.loc[:, ~df.columns.duplicated()]
        logger.info(f'[{task_name}] Loaded SQL data, shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'[{task_name}] Error loading SQL data: {e}')
        return pd.DataFrame()


def parse_target_definition(target_def_str: str, task_name: str):
    try:
        return eval(target_def_str)
    except Exception as e:
        logger.error(
            f"[{task_name}] Error parsing target def '{target_def_str}': {e}")
        raise


def apply_target_definition(df: pd.DataFrame, target_func, task_name: str,
    target_col_name: str) ->pd.Series:
    try:
        return pd.Series(np.asarray(target_func(df.copy())), index=df.index,
            name=target_col_name)
    except Exception as e:
        logger.error(f'[{task_name}] Error applying target def: {e}')
        raise


def _load_and_index_raw_data(sql_query: str, db_engine: Any, task_name: str,
    vfl_key: str) ->Optional[pd.DataFrame]:
    df_raw = load_data_from_db(sql_query, db_engine, task_name)
    if df_raw.empty:
        return None
    df_raw = standardise_key(df_raw, vfl_key)
    return df_raw


def _apply_target_and_filter_nans(df_indexed: pd.DataFrame, target_func,
    task_name: str, target_col: str, vfl_key: str) ->Tuple[Optional[pd.
    DataFrame], Optional[pd.Series]]:
    y_full = apply_target_definition(df_indexed.copy(), target_func,
        task_name, target_col)
    valid_keys_series = y_full.dropna()
    if valid_keys_series.empty:
        logger.warning(f'[{task_name}] Target empty after dropna.')
        return None, None
    valid_keys = valid_keys_series.index
    df_features = df_indexed.loc[valid_keys].copy()
    df_features = standardise_key(df_features, vfl_key)
    return df_features, y_full.loc[valid_keys]


def load_and_prepare_initial_data_for_task(task_name: str, task_def: Dict,
    spec: Dict, engine: Any) ->Optional[Tuple[pd.DataFrame, pd.Series]]:
    vfl_key = spec['column_identifiers']['VFL_KEY']
    sql = task_def['sql_query'].format(TABLE_HOST=spec['database'][
        'TABLE_HOST'], TABLE_REMOTE=spec['database']['TABLE_REMOTE'])
    df_indexed = _load_and_index_raw_data(sql, engine, task_name, vfl_key)
    if df_indexed is None:
        return None
    target_func = parse_target_definition(task_def['target_definition'],
        task_name)
    target_col = spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    return _apply_target_and_filter_nans(df_indexed, target_func, task_name,
        target_col, vfl_key)


def _select_host_native_features(df_task_data: pd.DataFrame, eval_spec:
    Dict, task_def: Dict) ->pd.DataFrame:
    host_num = eval_spec['columns']['HOST_NUMERIC_COLS_FOR_MODELING']
    host_cat = eval_spec['columns']['HOST_CATEGORICAL_COLS_FOR_MODELING']
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    cols_to_keep = list(set(host_num + host_cat + [vfl_key]))
    task_leaks = set(task_def.get('leak_columns', []))
    global_leaks = set(eval_spec['global_settings'].get('GLOBAL_LEAK_COLS', [])
        )
    all_leaks = task_leaks.union(global_leaks)
    final_cols = [col for col in cols_to_keep if col in df_task_data.
        columns and col not in all_leaks]
    return df_task_data[final_cols].copy()


def _extract_lgbm_top_features(model: Any, top_n: int=5) ->List[Dict]:
    if hasattr(model, 'feature_importances_') and hasattr(model,
        'feature_name_'):
        pairs = sorted(zip(model.feature_name_, model.feature_importances_),
            key=lambda x: x[1], reverse=True)
        return [{'feature': name, 'importance': float(imp)} for name, imp in
            pairs[:top_n]]
    return []


def train_and_save_host0_lgbm_and_preprocessor(X_train_raw: pd.DataFrame,
    y_train: pd.Series, X_val_raw: pd.DataFrame, y_val: pd.Series,
    host_num_cols: List[str], host_cat_cols: List[str], l1_cfg: Dict,
    task_def: Dict, out_dir: Path, task_name: str) ->Tuple[Optional[Any],
    Optional[Any]]:
    X_train_proc, preproc = preprocess_features_for_lgbm(X_train_raw,
        host_num_cols, host_cat_cols, f'{task_name}_Host0', fit_mode=True)
    if X_train_proc is None or not validate_preprocessed_features(X_train_proc,
        X_train_raw.shape, f'{task_name}_Host0'):
        logger.error(f'[{task_name}] Host0 preprocessing failed')
        return None, None
    X_val_proc, _ = preprocess_features_for_lgbm(X_val_raw, host_num_cols,
        host_cat_cols, f'{task_name}_Host0Val', fit_mode=False,
        existing_num_preprocessor=preproc)
    if X_val_proc is None:
        X_val_proc = pd.DataFrame()
    model = train_lgbm_model(X_train_proc, y_train, l1_cfg[
        'host_model_lgbm_params'], task_def['type'], X_val_proc, y_val)
    save_artifact(model, out_dir / l1_cfg['artifact_names'][
        'host0_model_pkl'], 'pkl')
    if preproc is not None:
        save_artifact(preproc, out_dir / l1_cfg['artifact_names'][
            'host0_preprocessor_pkl'], 'pkl')
    y_hat = get_predictions(model, X_val_proc, task_def['type'])
    metrics = _compute_binary_or_reg_metrics(y_val, y_hat, task_def['type'])
    metrics['top_features'] = _extract_lgbm_top_features(model)
    save_artifact(metrics, out_dir / l1_cfg['artifact_names'][
        'host0_metrics_json'], 'json')
    logger.info(
        f"[{task_name}] Host0 LGBM Val {metrics['validation_metric']:.4f}")
    return model, preproc


def _compute_binary_or_reg_metrics(y_true, y_pred, task_type):
    if task_type == 'binary':
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss
        y_bin = (y_pred > 0.5).astype(int)
        return {'roc_auc_score': float(roc_auc_score(y_true, y_pred)),
            'auprc_score': float(average_precision_score(y_true, y_pred)),
            'accuracy_score': float(accuracy_score(y_true, y_bin)),
            'f1_score': float(f1_score(y_true, y_bin)), 'precision_score':
            float(precision_score(y_true, y_bin)), 'recall_score': float(
            recall_score(y_true, y_bin)), 'brier_score_loss': float(
            brier_score_loss(y_true, y_pred)), 'validation_metric': float(
            roc_auc_score(y_true, y_pred))}
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mse = mean_squared_error(y_true, y_pred)
        return {'mae_score': float(mean_absolute_error(y_true, y_pred)),
            'mse_score': float(mse), 'rmse_score': float(np.sqrt(mse)),
            'r2_score': float(r2_score(y_true, y_pred)),
            'validation_metric': float(mean_absolute_error(y_true, y_pred))}


def train_and_save_host0_bat_model(X_train_raw: pd.DataFrame, y_train: pd.
    Series, host_num_cols: List[str], host_cat_cols: List[str], l1_cfg:
    Dict, task_def: Dict, out_dir: Path, task_name: str) ->Optional[Any]:
    model = train_bat_model(X_train_raw, y_train, l1_cfg[
        'host_model_bat_params'], task_def['type'], l1_cfg['seed'],
        host_num_cols, host_cat_cols)
    if not model:
        return None
    save_artifact(model, out_dir / l1_cfg['artifact_names'][
        'host0_bat_model_pkl'], artifact_type='pkl')
    if hasattr(model, 'tree_') and model.bat_feature_names_:
        rules = export_text(model, feature_names=model.bat_feature_names_)
        save_artifact(rules, out_dir / l1_cfg['artifact_names'][
            'host0_bat_rules_txt'], artifact_type='text')
    return model


def _get_host0_leaf_val_performance(df_host_val_with_leaves: pd.DataFrame,
    y_val: pd.Series, host0_lgbm_model: Any, host0_lgbm_preprocessor: Any,
    host_num_cols: List[str], host_cat_cols: List[str], task_type: str
    ) ->pd.DataFrame:
    X_val_raw = df_host_val_with_leaves[host_num_cols + host_cat_cols]
    X_val_proc, _ = preprocess_features_for_lgbm(X_val_raw, host_num_cols,
        host_cat_cols, 'Host0_Val_Perf', fit_mode=False,
        existing_num_preprocessor=host0_lgbm_preprocessor)
    df_perf = df_host_val_with_leaves[['host0_leaf_id']].copy()
    df_perf['y_true'] = y_val
    df_perf['y_pred_proba'] = get_predictions(host0_lgbm_model, X_val_proc,
        task_type)
    metric = 'auc' if task_type == 'binary' else 'mae'
    leaf_metrics = df_perf.groupby('host0_leaf_id').apply(lambda g:
        evaluate_predictions(g['y_true'], g['y_pred_proba'], task_type,
        metric), include_groups=False).rename(f'host0_val_{metric}'
        ).reset_index()
    leaf_counts = df_perf['host0_leaf_id'].value_counts().rename('val_samples'
        ).rename_axis('host0_leaf_id').reset_index()
    return leaf_metrics.merge(leaf_counts, on='host0_leaf_id')


def _get_instances_for_host_error_set(df_host_train_with_leaves: pd.
    DataFrame, y_train: pd.Series, error_leaf_ids: List[int]) ->Tuple[pd.
    DataFrame, pd.Series]:
    error_set_mask = df_host_train_with_leaves['host0_leaf_id'].isin(
        error_leaf_ids)
    if not error_set_mask.any():
        logger.warning('No instances found in error leaves')
        return pd.DataFrame(), pd.Series()
    df_error_set = df_host_train_with_leaves.loc[error_set_mask].copy()
    y_error_set = y_train.loc[df_error_set.index].copy()
    df_error_set = standardise_key(df_error_set, eval_spec[
        'column_identifiers']['VFL_KEY'])
    return df_error_set, y_error_set


def apply_k_anonymization(df_to_anonymize: pd.DataFrame, qi_cols: List[str],
    k_value: int, enabled: bool=True) ->pd.DataFrame:
    if not enabled or k_value <= 1 or not qi_cols:
        return df_to_anonymize
    df = df_to_anonymize.copy()
    numeric_qis = [c for c in qi_cols if pd.api.types.is_numeric_dtype(df[c])]
    key_view = df[qi_cols].astype(str)
    grp_sizes = key_view.groupby(list(key_view.columns))[key_view.columns[0]
        ].transform('size')
    mask_small = grp_sizes < k_value
    for col in qi_cols:
        if col in numeric_qis:
            df.loc[mask_small, col] = np.nan
        else:
            df[col] = df[col].astype(str)
            df.loc[mask_small, col] = f'Other_{col}'
    for col in numeric_qis:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in qi_cols:
        if col not in numeric_qis:
            df[col] = df[col].astype(str)
    return df


def _prepare_data_for_remote_model(df_host_augmented_kAnon: pd.DataFrame,
    y_target_set: pd.Series, df_remote_global: pd.DataFrame, eval_spec:
    Dict, task_def: Dict, task_name: str, original_qi_cols: List[str],
    pre_remote_surrogate_names: List[str]) ->Optional[Tuple[pd.DataFrame,
    pd.DataFrame, pd.Series, List[str], List[str], List[str], Dict[str, str
    ], Any]]:
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    df_host_augmented_kAnon = standardise_key(df_host_augmented_kAnon, vfl_key
        ).reset_index(drop=True)
    df_remote_global = standardise_key(df_remote_global, vfl_key).reset_index(
        drop=True)
    cols_from_host_for_merge = [vfl_key
        ] + original_qi_cols + pre_remote_surrogate_names
    cols_from_host_for_merge = list(dict.fromkeys(cols_from_host_for_merge))
    cols_from_host_for_merge = [col for col in cols_from_host_for_merge if 
        col in df_host_augmented_kAnon.columns]
    df_merged = pd.merge(df_host_augmented_kAnon[cols_from_host_for_merge],
        df_remote_global.drop_duplicates(subset=[vfl_key]), on=vfl_key, how
        ='inner')
    if df_merged.empty:
        logger.warning(
            f'[{task_name}] Merged error set with remote data is empty.')
        return None
    df_merged = standardise_key(df_merged, vfl_key)
    common_idx = df_merged.index.intersection(y_target_set.index)
    if common_idx.empty:
        logger.warning(
            f'[{task_name}] No common index for remote data and target.')
        return None
    df_raw_merged_aligned = df_merged.loc[common_idx].copy()
    y_aligned = y_target_set.loc[common_idx].copy()
    remote_num_spec = eval_spec['columns']['REMOTE_NUMERIC_COLS_FOR_MODELING']
    remote_cat_spec = eval_spec['columns'][
        'REMOTE_CATEGORICAL_COLS_FOR_MODELING']
    leak_cols = set(task_def.get('leak_columns', [])).union(set(eval_spec[
        'global_settings'].get('GLOBAL_LEAK_COLS', [])))
    active_true_remote_num = [c for c in remote_num_spec if c in
        df_raw_merged_aligned.columns and c not in leak_cols]
    active_true_remote_cat = [c for c in remote_cat_spec if c in
        df_raw_merged_aligned.columns and c not in leak_cols]
    true_remote_feature_names = active_true_remote_num + active_true_remote_cat
    active_host_qis = [c for c in original_qi_cols if c in
        df_raw_merged_aligned.columns and c not in leak_cols]
    active_pre_remote_surrogates = [c for c in pre_remote_surrogate_names if
        c in df_raw_merged_aligned.columns and c not in leak_cols]
    all_raw_features_for_model = list(dict.fromkeys(active_host_qis +
        active_pre_remote_surrogates + true_remote_feature_names))
    feat_origin_map = {c: tag_feature_origin(c, set(active_host_qis), set(
        active_pre_remote_surrogates)) for c in all_raw_features_for_model}
    model_num_cols = [c for c in all_raw_features_for_model if pd.api.types
        .is_numeric_dtype(df_raw_merged_aligned[c])]
    model_cat_cols = [c for c in all_raw_features_for_model if not pd.api.
        types.is_numeric_dtype(df_raw_merged_aligned[c])]
    if not model_num_cols and not model_cat_cols:
        logger.warning(f'[{task_name}] No features for remote model.')
        return None
    X_input_for_lgbm = df_raw_merged_aligned[model_num_cols + model_cat_cols]
    X_processed_lgbm, preprocessor = preprocess_features_for_lgbm(
        X_input_for_lgbm, model_num_cols, model_cat_cols,
        f'{task_name}_RemoteErrorSetLGBM', fit_mode=True)
    if X_processed_lgbm is None or not validate_preprocessed_features(
        X_processed_lgbm, X_input_for_lgbm.shape,
        f'{task_name}_RemoteErrorSetLGBM'):
        logger.error(
            f'[{task_name}] Remote error set preprocessing failed or returned empty'
            )
        return None
    return (df_raw_merged_aligned, X_processed_lgbm, y_aligned,
        model_num_cols, model_cat_cols, true_remote_feature_names,
        feat_origin_map, preprocessor)


def _train_remote_teacher_on_error_set(X_proc_err_set_lgbm: pd.DataFrame,
    y_err_set_aligned: pd.Series, task_type: str, l1_cfg: Dict,
    host0_metric_on_err_set: float, task_name: str, feat_origin_map: Dict[
    str, str], raw_model_num_cols: List[str], raw_model_cat_cols: List[str]
    ) ->Optional[Tuple[Any, float]]:
    remote_lgbm_config_params = l1_cfg['remote_party_analysis_params'][
        'remote_teacher_lgbm_params'].copy()
    remote_lgbm_config_params['task_name_for_debug'
        ] = f'{task_name}_RemoteTeacherLGBM'
    remote_model = train_lgbm_model(X_proc_err_set_lgbm, y_err_set_aligned,
        params=remote_lgbm_config_params, task_type=task_type,
        feature_origin_map=feat_origin_map, original_numeric_cols=
        raw_model_num_cols, original_categorical_cols=raw_model_cat_cols)
    metric_type = 'auc' if task_type == 'binary' else 'mae'
    remote_metric = evaluate_predictions(y_err_set_aligned, get_predictions
        (remote_model, X_proc_err_set_lgbm, task_type), task_type, metric_type)
    min_lift = l1_cfg['remote_party_analysis_params'][
        'min_lift_over_host0_on_error_set']
    improved = (task_type == 'binary' and remote_metric > 
        host0_metric_on_err_set + min_lift or task_type == 'regression' and
        remote_metric < host0_metric_on_err_set - min_lift)
    logger.info(
        f'[{task_name}] Remote Teacher on Error Set: Metric={remote_metric:.4f}, Host0_on_ErrorSet_Metric={host0_metric_on_err_set:.4f}. Improved: {improved}'
        )
    if remote_lgbm_config_params.get('feature_penalty_mode', 'off') != 'off':
        try:
            explainer = shap.TreeExplainer(remote_model)
            shap_values_matrix = explainer.shap_values(X_proc_err_set_lgbm,
                check_additivity=False)
            if isinstance(shap_values_matrix, list) and len(shap_values_matrix
                ) == 2:
                shap_values_for_analysis = shap_values_matrix[1]
            else:
                shap_values_for_analysis = shap_values_matrix
            shap_origin_fractions = analyze_shap_by_origin(
                shap_values_for_analysis, X_proc_err_set_lgbm.columns.
                tolist(), feat_origin_map, task_name)
            logger.info(
                f'[{task_name}] Remote LGBM SHAP mass distribution by origin: {shap_origin_fractions}'
                )
            remote_true_shap_frac = shap_origin_fractions.get('remote_true',
                0.0)
            if remote_true_shap_frac > 0.65:
                logger.warning(
                    f"[{task_name}] >65% ({remote_true_shap_frac:.2%}) SHAP mass still from 'remote_true' features despite penalty mode '{remote_lgbm_config_params.get('feature_penalty_mode')}'. Consider ↑ penalty or ↓ feature_fraction in remote_teacher_lgbm_params."
                    )
        except Exception as e:
            logger.error(
                f'[{task_name}] SHAP analysis for remote teacher failed: {e}')
    if not improved:
        logger.info(
            f'[{task_name}] Remote teacher model did not meet minimum lift criteria. Not using for rule generation.'
            )
        pass
    return remote_model, remote_metric


def _train_and_save_remote_bat_on_error_set(X_raw_for_bat: pd.DataFrame,
    y_err_set_aligned: pd.Series, task_type: str, l1_cfg: Dict, out_dir:
    Path, task_name: str, model_num_cols: List[str], model_cat_cols: List[str]
    ) ->Optional[Any]:
    bat_config_params = l1_cfg['remote_party_analysis_params'][
        'remote_teacher_bat_params'].copy()
    lgbm_penalty_mode = l1_cfg['remote_party_analysis_params'][
        'remote_teacher_lgbm_params'].get('feature_penalty_mode', 'off')
    dynamic_bat_params_key = 'dynamic_hyperparams_on_penalty'
    if (lgbm_penalty_mode != 'off' and dynamic_bat_params_key in
        bat_config_params):
        dynamic_params = bat_config_params.pop(dynamic_bat_params_key)
        logger.info(
            f'[{task_name}] Applying dynamic BAT hyperparameters due to LGBM penalty mode: {dynamic_params}'
            )
        if 'max_depth' in dynamic_params:
            bat_config_params['max_depth'] = min(dynamic_params['max_depth'
                ], bat_config_params['max_depth'])
        if 'min_samples_leaf' in dynamic_params:
            bat_config_params['min_samples_leaf'] = max(dynamic_params[
                'min_samples_leaf'], bat_config_params['min_samples_leaf'])
    features_for_bat_training = model_num_cols + model_cat_cols
    features_present_in_df = [f for f in features_for_bat_training if f in
        X_raw_for_bat.columns]
    final_model_num_cols = [f for f in model_num_cols if f in
        features_present_in_df]
    final_model_cat_cols = [f for f in model_cat_cols if f in
        features_present_in_df]
    model = train_bat_model(X_raw_for_bat[final_model_num_cols +
        final_model_cat_cols], y_err_set_aligned, bat_config_params,
        task_type, l1_cfg['seed'], final_model_num_cols, final_model_cat_cols)
    if not model:
        return None
    save_artifact(model, out_dir / l1_cfg['artifact_names'][
        'remote_bat_error_set_model_pkl'], artifact_type='pkl')
    if hasattr(model, 'tree_') and hasattr(model, 'bat_feature_names_'
        ) and model.bat_feature_names_:
        rules_text = export_text(model, feature_names=model.bat_feature_names_)
        save_artifact(rules_text, out_dir / l1_cfg['artifact_names'][
            'remote_bat_error_set_rules_txt'], artifact_type='text')
        logger.info(
            f'[{task_name}] Saved Remote BAT rules text using sanitized feature names.'
            )
    else:
        logger.warning(
            f'[{task_name}] Could not save Remote BAT rules text (model invalid or no feature names).'
            )
    return model


def _extract_remote_rules_from_error_set_bat(remote_bat_model: Any,
    X_raw_for_bat: pd.DataFrame, y_target_error_set: pd.Series, l1_cfg:
    Dict, task_name: str, vfl_key: str) ->List[Dict]:
    if not hasattr(remote_bat_model, 'tree_'):
        return []
    parsed_rules = _parse_bat_rules(remote_bat_model)
    X_proc_for_bat = remote_bat_model.bat_preprocessor_.transform(X_raw_for_bat
        )
    leaf_ids = remote_bat_model.apply(X_proc_for_bat)
    all_insights = []
    min_support = l1_cfg['remote_party_analysis_params'][
        'min_remote_rule_support']
    baseline_perf = y_target_error_set.mean()
    for leaf_id in np.unique(leaf_ids):
        rule_mask = leaf_ids == leaf_id
        if rule_mask.sum() < min_support:
            continue
        y_covered = y_target_error_set[rule_mask]
        p_y_rule = y_covered.mean() if not y_covered.empty else 0.0
        all_insights.append({'rule_id': f'R{leaf_id}', 'human_rule_text':
            parsed_rules.get(leaf_id, 'Rule logic not parsed.'),
            'remote_rule_support': int(rule_mask.sum()),
            'p_y_given_remote_rule': p_y_rule, 'lift': p_y_rule / (
            baseline_perf + 1e-06), 'covered_instance_ids_by_remote':
            X_raw_for_bat.index[rule_mask].tolist()})
    logger.info(
        f'[{task_name}] Extracted {len(all_insights)} remote rules meeting support criteria.'
        )
    return all_insights


def _calculate_pareto_scores_for_remote_rule(rule_insight: Dict, task_type:
    str, pareto_params: Dict, error_set_baseline_y_mean: float) ->Tuple[
    float, float]:
    p_y_rule = rule_insight['p_y_given_remote_rule']
    support = rule_insight['remote_rule_support']
    f1_k = abs(p_y_rule - error_set_baseline_y_mean) * support
    rule_text_len = len(rule_insight.get('remote_rule_text',
        'RemoteBAT_Leaf_X'))
    f2_k = 1.0 / (1.0 + rule_text_len / 50.0)
    return f1_k, f2_k


def perform_pareto_optimization_for_remote_rules(all_remote_rule_insights:
    List[Dict], l1_cfg: Dict, task_def: Dict, task_name: str,
    error_set_baseline_y_mean: float, out_dir: Path) ->Dict[str, Dict]:
    if not all_remote_rule_insights:
        return {}
    pareto_params = l1_cfg['pareto_optimization_params']
    if not pareto_params.get('enabled', False):
        sorted_rules = sorted(all_remote_rule_insights, key=lambda r: r.get
            ('remote_rule_support', 0) * abs(r.get('p_y_given_remote_rule',
            0) - error_set_baseline_y_mean), reverse=True)
        top_n = pareto_params.get('keep_top_n_remote_rules', 5)
        return {r['rule_id']: r for r in sorted_rules[:top_n]}
    pareto_data = []
    for insight in all_remote_rule_insights:
        f1, f2 = _calculate_pareto_scores_for_remote_rule(insight, task_def
            ['type'], pareto_params, error_set_baseline_y_mean)
        pareto_data.append({'rule_id': insight['rule_id'], 'f1_k': f1,
            'f2_k': f2, **insight})
    if not pareto_data:
        return {}
    df_pareto = pd.DataFrame(pareto_data).set_index('rule_id')
    df_pareto['f1_norm'] = _linearly_scale_series(df_pareto['f1_k'])
    df_pareto['f2_norm'] = _linearly_scale_series(df_pareto['f2_k'])
    df_pareto['score'] = pareto_params.get('w_gain', 0.6) * df_pareto['f1_norm'
        ] + pareto_params.get('w_quality', 0.4) * df_pareto['f2_norm']
    selected_rules_df = df_pareto.nlargest(pareto_params.get(
        'keep_top_n_remote_rules', 10), 'score')
    selected_details = {idx: row.to_dict() for idx, row in
        selected_rules_df.iterrows()}
    save_artifact(selected_rules_df.reset_index().to_dict(orient='records'),
        out_dir / l1_cfg['artifact_names']['pareto_analysis_details_json'],
        artifact_type='json')
    logger.info(
        f'[{task_name}] Pareto selected {len(selected_details)} remote rules.')
    return selected_details


def _extract_shap_from_lgbm_model(model: Any, X_processed_lgbm: pd.
    DataFrame, top_n: int) ->Optional[Dict[str, float]]:
    if not hasattr(model, 'predict'):
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_matrix = explainer.shap_values(X_processed_lgbm)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[1] if len(shap_matrix
                ) > 1 else shap_matrix[0]
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        raw_map = dict(zip(X_processed_lgbm.columns, mean_abs))
        canonical = canonicalize_feature_dict(raw_map)
        return dict(sorted(canonical.items(), key=lambda kv: kv[1], reverse
            =True)[:top_n])
    except Exception as exc:
        logger.warning('SHAP extraction failed: %s', exc)
        return None


def _get_host0_perf_on_rule_coverage(host0_model: Any, host0_preprocessor:
    Any, df_host_data_for_coverage: pd.DataFrame, y_data_for_coverage: pd.
    Series, covered_vfl_keys: List[str], host_num_cols: List[str],
    host_cat_cols: List[str], task_type: str) ->Dict:
    if not covered_vfl_keys:
        return {'error': 'No covered instances'}
    df_host_data_for_coverage = standardise_key(df_host_data_for_coverage,
        eval_spec['column_identifiers']['VFL_KEY'])
    df_subset_raw = df_host_data_for_coverage.loc[df_host_data_for_coverage
        .index.isin(covered_vfl_keys)][host_num_cols + host_cat_cols]
    y_subset = y_data_for_coverage.loc[y_data_for_coverage.index.isin(
        covered_vfl_keys)]
    if df_subset_raw.empty or y_subset.empty:
        return {'error': 'Empty subset after filtering'}
    X_subset_proc, _ = preprocess_features_for_lgbm(df_subset_raw,
        host_num_cols, host_cat_cols, 'Host0_RuleCoverage', fit_mode=False,
        existing_num_preprocessor=host0_preprocessor)
    if X_subset_proc is None:
        return {'error': 'Preprocessing failed for subset'}
    y_pred_proba_subset = get_predictions(host0_model, X_subset_proc, task_type
        )
    if y_pred_proba_subset.size == 0:
        return {'error': 'No predictions for subset'}
    metrics = {'size': len(y_subset), 'target_prevalence': float(y_subset.
        mean()) if not y_subset.empty else 0.0}
    if task_type == 'binary':
        if y_subset.nunique() > 1:
            metrics['auc'] = float(roc_auc_score(y_subset, y_pred_proba_subset)
                )
            y_pred_labels = (y_pred_proba_subset > 0.5).astype(int)
            metrics['precision'] = float(precision_score(y_subset,
                y_pred_labels, zero_division=0))
            metrics['recall'] = float(recall_score(y_subset, y_pred_labels,
                zero_division=0))
        else:
            metrics['auc'] = 0.5
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
    else:
        metrics['mae'] = float(mean_absolute_error(y_subset,
            y_pred_proba_subset))
    return metrics


def validate_preprocessed_features(X_proc: pd.DataFrame, original_shape:
    tuple, task_name: str):
    if X_proc.shape[0] != original_shape[0]:
        logger.error(
            f'[{task_name}] Row mismatch after preprocessing: {X_proc.shape[0]} vs {original_shape[0]}'
            )
        return False
    if X_proc.isna().any().any():
        logger.warning(f'[{task_name}] NaNs in preprocessed features')
    return True


def _persist_l2_switchback_validation(df_val_with_leaves: pd.DataFrame,
    y_val: pd.Series, host_num_cols: list[str], host_cat_cols: list[str],
    vfl_key: str, out_dir: Path, l1_cfg: dict, eval_spec: dict, task_name: str
    ) ->None:
    if df_val_with_leaves.empty or y_val.empty:
        logger.warning(
            '[%s] Validation data for L2 switch-back is empty. Skipping save.',
            task_name)
        return
    df_val = standardise_key(df_val_with_leaves.copy(), vfl_key)
    if 'host0_leaf_id' not in df_val.columns:
        raise KeyError(
            "'host0_leaf_id' missing in validation frame – cannot persist switch-back features"
            )
    cols: list[str] = [vfl_key]
    cols += [c for c in host_num_cols if c in df_val.columns]
    cols += [c for c in host_cat_cols if c in df_val.columns]
    cols += ['host0_leaf_id']
    cols = list(dict.fromkeys(cols))
    feat_path = out_dir / l1_cfg['artifact_names'][
        'host_val_for_l2_switchback_features_csv']
    save_artifact(df_val[cols].reset_index(drop=True), feat_path,
        artifact_type='csv', desc='Host validation features for L2 switch-back'
        )
    y_aligned = y_val.reindex(df_val.index)
    tgt_path = out_dir / l1_cfg['artifact_names'][
        'host_val_for_l2_switchback_target_csv']
    save_artifact(pd.DataFrame({vfl_key: y_aligned.index.astype(str),
        eval_spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']:
        y_aligned.values}), tgt_path, artifact_type='csv', desc=
        'Host validation target for L2 switch-back')
    logger.info('[%s] L2 switch-back artifacts saved → %s | %s', task_name,
        feat_path.name, tgt_path.name)


def _init_task_processing(eval_spec: Dict, l1_cfg: Dict, task_name: str
    ) ->Tuple[Path, str]:
    out_dir = Path(eval_spec['dataset_output_dir_name']) / task_name / l1_cfg[
        'script1_output_subdir_template']
    ensure_dir_exists(out_dir)
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    return out_dir, vfl_key


def _load_and_split_host_data(task_name: str, task_def: Dict, eval_spec:
    Dict, l1_cfg: Dict, db_engine: Any, vfl_key: str) ->Optional[Tuple[pd.
    DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.
    Series, pd.Series, pd.Series, List[str], List[str], List[str]]]:
    data_load_result = load_and_prepare_initial_data_for_task(task_name,
        task_def, eval_spec, db_engine)
    if not data_load_result:
        return None
    df_task_data_full, y_full_task = data_load_result
    df_host_native_full = _select_host_native_features(df_task_data_full,
        eval_spec, task_def)
    df_host_native_full = standardise_key(df_host_native_full, vfl_key)
    out_dir_for_split = Path(eval_spec['dataset_output_dir_name']
        ) / task_name / l1_cfg['script1_output_subdir_template']
    df_host_train, df_host_val, df_host_test, y_train, y_val, y_test = (
        split_data_and_save_indices(df_host_native_full, y_full_task,
        task_def['type'], l1_cfg, out_dir_for_split, task_name))
    if df_host_train.empty:
        logger.error(f'[{task_name}] Host training data empty after split.')
        return None
    host_num_cols = [c for c in eval_spec['columns'][
        'HOST_NUMERIC_COLS_FOR_MODELING'] if c in df_host_train.columns]
    host_cat_cols = [c for c in eval_spec['columns'][
        'HOST_CATEGORICAL_COLS_FOR_MODELING'] if c in df_host_train.columns]
    train_vfl_keys = df_host_train.index.astype(str).tolist()
    val_vfl_keys = df_host_val.index.astype(str).tolist()
    test_vfl_keys = df_host_test.index.astype(str).tolist()
    return (df_host_native_full, y_full_task, df_host_train, df_host_val,
        y_train, y_val, host_num_cols, host_cat_cols, train_vfl_keys,
        val_vfl_keys, test_vfl_keys)


def _prepare_l2_target_and_llm_input(df_host_native_full: pd.DataFrame,
    y_full_task: pd.Series, eval_spec: Dict, l1_cfg: Dict, out_dir: Path,
    vfl_key: str):
    df_surrogate_input = standardise_key(df_host_native_full.copy(), vfl_key)
    target_col = eval_spec['global_settings']['TARGET_COLUMN_NAME_PROCESSED']
    y_full_task.index = y_full_task.index.astype(str)
    target_df = pd.DataFrame({vfl_key: df_surrogate_input.index, target_col:
        y_full_task.reindex(df_surrogate_input.index)})
    target_df = standardise_key(target_df, vfl_key)
    save_artifact(target_df, out_dir / l1_cfg['artifact_names'][
        'host_target_cols_csv'], artifact_type='csv')


def _train_initial_host_models(df_host_train: pd.DataFrame, y_train: pd.
    Series, df_host_val: pd.DataFrame, y_val: pd.Series, host_num_cols:
    List[str], host_cat_cols: List[str], l1_cfg: Dict, task_def: Dict,
    out_dir: Path, task_name: str) ->Tuple[Optional[Any], Optional[Any],
    Optional[Any]]:
    host0_lgbm, host0_lgbm_preproc = (
        train_and_save_host0_lgbm_and_preprocessor(df_host_train[
        host_num_cols + host_cat_cols], y_train, df_host_val[host_num_cols +
        host_cat_cols], y_val, host_num_cols, host_cat_cols, l1_cfg,
        task_def, out_dir, task_name))
    if not host0_lgbm or not host0_lgbm_preproc:
        return None, None, None
    host0_bat = train_and_save_host0_bat_model(df_host_train[host_num_cols +
        host_cat_cols], y_train, host_num_cols, host_cat_cols, l1_cfg,
        task_def, out_dir, task_name)
    if not host0_bat:
        return host0_lgbm, host0_lgbm_preproc, None
    return host0_lgbm, host0_lgbm_preproc, host0_bat


def _annotate_data_with_host0_leaves(host0_bat: Any, df_host_train: pd.
    DataFrame, df_host_val: pd.DataFrame, host_num_cols: List[str],
    host_cat_cols: List[str]) ->Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_bat_proc = host0_bat.bat_preprocessor_.transform(df_host_train[
        host_num_cols + host_cat_cols])
    df_host_train_with_leaves = df_host_train.copy()
    df_host_train_with_leaves['host0_leaf_id'] = host0_bat.apply(
        X_train_bat_proc)
    X_val_bat_proc = host0_bat.bat_preprocessor_.transform(df_host_val[
        host_num_cols + host_cat_cols])
    df_host_val_with_leaves = df_host_val.copy()
    df_host_val_with_leaves['host0_leaf_id'] = host0_bat.apply(X_val_bat_proc)
    return df_host_train_with_leaves, df_host_val_with_leaves


def _identify_error_segment(df_host_train_with_leaves: pd.DataFrame,
    df_host_val_with_leaves: pd.DataFrame, y_val: pd.Series, host0_lgbm:
    Any, host0_lgbm_preproc: Any, host_num_cols: List[str], host_cat_cols:
    List[str], task_def: Dict, l1_cfg: Dict) ->List[int]:
    host0_leaf_val_metrics = _get_host0_leaf_val_performance(
        df_host_val_with_leaves, y_val, host0_lgbm, host0_lgbm_preproc,
        host_num_cols, host_cat_cols, task_def['type'])
    err_thresh_key = 'host0_leaf_auc_error_threshold' if task_def['type'
        ] == 'binary' else 'host0_leaf_mae_error_threshold_factor'
    err_thresh = l1_cfg['segment_identification_params'].get(err_thresh_key,
        0.65 if task_def['type'] == 'binary' else 1.2)
    min_samples = l1_cfg['segment_identification_params'].get(
        'min_leaf_samples_for_error_segment', 40)
    leaf_sizes = df_host_train_with_leaves['host0_leaf_id'].value_counts(
        ).rename_axis('host0_leaf_id').reset_index(name='train_samples')
    eligible = host0_leaf_val_metrics.merge(leaf_sizes, on='host0_leaf_id'
        ).query('train_samples >= @min_samples')
    if task_def['type'] == 'binary':
        error_leaf_ids = eligible.loc[eligible['host0_val_auc'] <
            err_thresh, 'host0_leaf_id'].tolist()
    else:
        error_leaf_ids = eligible.loc[eligible['host0_val_mae'] >
            err_thresh, 'host0_leaf_id'].tolist()
    if not error_leaf_ids and not eligible.empty:
        sort_col, asc = ('host0_val_auc', True) if task_def['type'
            ] == 'binary' else ('host0_val_mae', False)
        error_leaf_ids = eligible.sort_values(sort_col, ascending=asc).head(1)[
            'host0_leaf_id'].tolist()
    return error_leaf_ids


def _extract_and_augment_host_error_set(df_host_train_with_leaves: pd.
    DataFrame, y_train: pd.Series, error_leaf_ids: List[int], host_num_cols:
    List[str], host_cat_cols: List[str], host0_lgbm: Any,
    host0_lgbm_preproc: Any, l1_cfg: Dict, task_name: str, vfl_key: str,
    df_host_native_full: pd.DataFrame, out_dir: Path) ->Tuple[Optional[pd.
    DataFrame], Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
    df_host_error_set, y_host_error_set = _get_instances_for_host_error_set(
        df_host_train_with_leaves, y_train, error_leaf_ids)
    if df_host_error_set.empty:
        logger.info(
            f'[{task_name}] Host error set empty. Skipping pre-remote surrogate generation for remote and remote steps.'
            )
        save_artifact(standardise_key(df_host_native_full.copy(), vfl_key),
            out_dir / l1_cfg['artifact_names'][
            'host_data_with_local_surrogates_csv'], artifact_type='csv',
            desc='Host data (no surrogates as error set was empty)')
        return None, None, None, []
    df_host_error_set_augmented_for_remote = df_host_error_set.copy()
    pre_remote_surr_names_for_remote = []
    pre_remote_surr_cfg = l1_cfg.get('pre_remote_surrogate_generation', {})
    if pre_remote_surr_cfg.get('enabled', False):
        (df_host_error_set_augmented_for_remote,
            pre_remote_surr_names_for_remote, _) = (
            _generate_pre_remote_host_surrogates(df_host_error_set,
            host_num_cols, host_cat_cols, host0_lgbm, host0_lgbm_preproc,
            l1_cfg, task_name, vfl_key, fixed_source_cols=None, out_dir=
            out_dir))
    else:
        logger.info(
            f'[{task_name}] Pre-remote surrogate generation (for remote interaction) skipped.'
            )
    return (df_host_error_set, df_host_error_set_augmented_for_remote,
        y_host_error_set, pre_remote_surr_names_for_remote)


def _persist_error_ids_and_switchback_data(error_leaf_ids: List[int],
    df_host_val_with_leaves: pd.DataFrame, y_val: pd.Series, host_num_cols:
    List[str], host_cat_cols: List[str], vfl_key: str, out_dir: Path,
    l1_cfg: Dict, eval_spec: Dict, task_name: str):
    _persist_l2_switchback_validation(df_host_val_with_leaves, y_val,
        host_num_cols, host_cat_cols, vfl_key, out_dir, l1_cfg, eval_spec,
        task_name)
    try:
        save_artifact(error_leaf_ids, out_dir / l1_cfg['artifact_names'][
            'host_error_leaf_ids_json'], artifact_type='json')
        logger.info(
            f'[{task_name}] Saved {len(error_leaf_ids)} error leaf IDs.')
    except Exception as exc:
        logger.error(f'[{task_name}] Could not save error_leaf_ids – {exc}')


def _prepare_data_for_remote_processing(df_host_error_set_augmented_pre_remote:
    pd.DataFrame, y_host_error_set: pd.Series, eval_spec: Dict, l1_cfg:
    Dict, task_def: Dict, task_name: str, vfl_key: str, db_engine: Any,
    pre_remote_surrogate_names: List[str]) ->Optional[Tuple[pd.DataFrame,
    pd.DataFrame, pd.Series, List[str], List[str], List[str], Dict[str, str
    ], Any]]:
    qi_cols = eval_spec['column_identifiers']['QI_COLS']
    active_qi_cols = [c for c in qi_cols if c in
        df_host_error_set_augmented_pre_remote.columns]
    df_kAnon = apply_k_anonymization(df_host_error_set_augmented_pre_remote,
        active_qi_cols, l1_cfg['k_anonymization_params']['k_value'],
        enabled=l1_cfg['k_anonymization_params']['enabled'])
    df_kAnon = standardise_key(df_kAnon, vfl_key)
    df_remote_global = load_data_from_db(
        f"SELECT * FROM {eval_spec['database']['TABLE_REMOTE']}", db_engine,
        task_name)
    if df_remote_global.empty:
        logger.error(f'[{task_name}] Global remote data empty.')
        return None
    df_remote_global = standardise_key(df_remote_global, vfl_key)
    return _prepare_data_for_remote_model(df_kAnon, y_host_error_set,
        df_remote_global, eval_spec, task_def, task_name, active_qi_cols,
        pre_remote_surrogate_names)


def _train_and_evaluate_remote_teacher_models(prep_result_remote_tuple:
    Tuple, host0_lgbm: Any, host0_lgbm_preproc: Any,
    df_host_error_set_original: pd.DataFrame, host_num_cols: List[str],
    host_cat_cols: List[str], task_def: Dict, l1_cfg: Dict, task_name: str,
    out_dir: Path) ->Tuple[Optional[Any], Optional[Any]]:
    (df_raw_merged, X_proc_lgbm, y_aligned, remote_model_num_raw,
        remote_model_cat_raw, _, feat_origin_map, _) = prep_result_remote_tuple
    X_h0_err_raw = df_host_error_set_original.loc[y_aligned.index, 
        host_num_cols + host_cat_cols]
    X_h0_err_proc, _ = preprocess_features_for_lgbm(X_h0_err_raw,
        host_num_cols, host_cat_cols, f'{task_name}_Host0ErrSetRemote',
        fit_mode=False, existing_num_preprocessor=host0_lgbm_preproc)
    metric = 'auc' if task_def['type'] == 'binary' else 'mae'
    h0_metric = evaluate_predictions(y_aligned, get_predictions(host0_lgbm,
        X_h0_err_proc, task_def['type']), task_def['type'], metric)
    rem_lgbm_res = _train_remote_teacher_on_error_set(X_proc_lgbm,
        y_aligned, task_def['type'], l1_cfg, h0_metric, task_name,
        feat_origin_map, remote_model_num_raw, remote_model_cat_raw)
    rem_lgbm_model = rem_lgbm_res[0] if rem_lgbm_res else None
    X_raw_for_bat = df_raw_merged[remote_model_num_raw + remote_model_cat_raw]
    rem_bat_model = _train_and_save_remote_bat_on_error_set(X_raw_for_bat,
        y_aligned, task_def['type'], l1_cfg, out_dir, task_name,
        remote_model_num_raw, remote_model_cat_raw)
    return rem_lgbm_model, rem_bat_model


def _extract_rules_and_perform_pareto(remote_bat_model: Any,
    df_raw_merged_err_set: pd.DataFrame, y_err_set_aligned: pd.Series,
    remote_lgbm_model: Optional[Any], X_proc_err_set_lgbm: pd.DataFrame,
    l1_cfg: Dict, task_name: str, vfl_key: str, task_def: Dict, out_dir: Path
    ) ->Optional[Dict[str, Dict]]:
    if not remote_bat_model:
        return None
    all_insights = _extract_remote_rules_from_error_set_bat(remote_bat_model,
        df_raw_merged_err_set, y_err_set_aligned, l1_cfg, task_name, vfl_key)
    if not all_insights:
        logger.info(f'[{task_name}] No remote rules extracted.')
        return None
    save_artifact(all_insights, out_dir / l1_cfg['artifact_names'][
        'all_remote_rule_insights_json'], artifact_type='json')
    baseline_y = y_err_set_aligned.mean(
        ) if not y_err_set_aligned.empty else 0.5
    selected_rules = perform_pareto_optimization_for_remote_rules(all_insights,
        l1_cfg, task_def, task_name, baseline_y, out_dir)
    if not selected_rules:
        logger.info(f'[{task_name}] No remote rules by Pareto.')
        return None
    save_artifact(selected_rules, out_dir / l1_cfg['artifact_names'][
        'selected_remote_rules_for_S_hat_training_json'], artifact_type='json')
    return selected_rules


def _generate_and_save_final_local_surrogates(df_host_native_full: pd.
    DataFrame, final_rules_for_llm: Optional[Dict[str, Dict]],
    selected_rule_ids: Optional[List[str]], host_native_num_cols: List[str],
    host_native_cat_cols: List[str], host0_lgbm_model: Optional[Any],
    host0_lgbm_preproc: Optional[Any], vfl_key: str, task_name: str,
    out_dir: Path, l1_cfg: Dict):
    logger.info(
        f'[{task_name}] Starting final local surrogate feature generation (p-hats, robustified, interactions, and global pre-remote)...'
        )
    df_augmented_locally = standardise_key(df_host_native_full.copy(), vfl_key)
    all_generated_surrogate_names = []
    pre_remote_surr_cfg_final = l1_cfg.get('pre_remote_surrogate_generation',
        {})
    if pre_remote_surr_cfg_final.get('enabled', False
        ) and host0_lgbm_model and host0_lgbm_preproc:
        art_key = 'pre_remote_surrogate_source_cols_json'
        src_list_file = out_dir / l1_cfg['artifact_names'].get(art_key,
            'pre_remote_surrogate_source_cols.json')
        fixed_cols = []
        if src_list_file.exists():
            fixed_cols = json.load(open(src_list_file, 'r'))
            logger.info(
                f'[{task_name}] Replaying {len(fixed_cols)} pre-remote surrogate source cols saved from error-set.'
                )
        (df_augmented_locally, new_pre_remote,
            source_feats_robustified_for_pre_remote, _) = (
            _generate_pre_remote_host_surrogates(df_augmented_locally,
            host_native_num_cols, host_native_cat_cols, host0_lgbm_model,
            host0_lgbm_preproc, l1_cfg, task_name, vfl_key,
            fixed_source_cols=fixed_cols, out_dir=out_dir))
        all_generated_surrogate_names.extend(new_pre_remote)
        logger.info(
            f"[{task_name}] Added {len(new_pre_remote)} 'pre_remote_surrx_' features to final L1 output."
            )
    elif pre_remote_surr_cfg_final.get('enabled', False):
        logger.warning(
            f'[{task_name}] Pre-remote surrogate generation for final L1 output enabled, but Host0 model/preprocessor missing. Skipping.'
            )
    df_augmented_locally = df_augmented_locally.loc[:, ~
        df_augmented_locally.columns.duplicated(keep='first')]
    logger.info(
        f'[{task_name}] Generated/updated a total of {len(all_generated_surrogate_names)} local surrogates. Final L1 output shape: {df_augmented_locally.shape}'
        )
    save_artifact(df_augmented_locally, out_dir / l1_cfg['artifact_names'][
        'host_data_with_local_surrogates_csv'], artifact_type='csv')


def process_task_vfl_t3_final(task_name: str, task_def: Dict, eval_spec:
    Dict, l1_cfg: Dict, db_engine: Any):
    logger.info(
        f'[{task_name}] Starting L1 processing (generates inputs for LLM S-hat surrogates)...'
        )
    out_dir, vfl_key = _init_task_processing(eval_spec, l1_cfg, task_name)
    load_split_res = _load_and_split_host_data(task_name, task_def,
        eval_spec, l1_cfg, db_engine, vfl_key)
    if not load_split_res:
        return
    (df_host_native_full, y_full_task, df_host_train, df_host_val, y_train,
        y_val, host_native_num_cols, host_native_cat_cols, train_vfl_keys,
        val_vfl_keys, test_vfl_keys) = load_split_res
    _prepare_l2_target_and_llm_input(df_host_native_full, y_full_task,
        eval_spec, l1_cfg, out_dir, vfl_key)
    h0_models_tuple = _train_initial_host_models(df_host_train, y_train,
        df_host_val, y_val, host_native_num_cols, host_native_cat_cols,
        l1_cfg, task_def, out_dir, task_name)
    if not h0_models_tuple or not all(h0_models_tuple):
        return
    host0_lgbm, host0_lgbm_preproc, host0_bat = h0_models_tuple
    df_h_train_lf, df_h_val_lf = _annotate_data_with_host0_leaves(host0_bat,
        df_host_train, df_host_val, host_native_num_cols, host_native_cat_cols)
    err_leaf_ids = _identify_error_segment(df_h_train_lf, df_h_val_lf,
        y_val, host0_lgbm, host0_lgbm_preproc, host_native_num_cols,
        host_native_cat_cols, task_def, l1_cfg)
    _persist_error_ids_and_switchback_data(err_leaf_ids, df_h_val_lf, y_val,
        host_native_num_cols, host_native_cat_cols, vfl_key, out_dir,
        l1_cfg, eval_spec, task_name)
    df_host_native_full_with_pre_surr = df_host_native_full.copy()
    pre_remote_surrogate_names = []
    source_feats_robustified_for_pre_remote = []
    pre_remote_surr_cfg = l1_cfg.get('pre_remote_surrogate_generation', {})
    if pre_remote_surr_cfg.get('enabled', False):
        (df_host_native_full_with_pre_surr, pre_remote_surrogate_names,
            source_feats_robustified_for_pre_remote) = (
            _generate_pre_remote_host_surrogates(df_host_native_full,
            host_native_num_cols, host_native_cat_cols, host0_lgbm,
            host0_lgbm_preproc, l1_cfg, task_name, vfl_key, out_dir=out_dir))
    save_artifact(standardise_key(df_host_native_full_with_pre_surr.copy(),
        vfl_key), out_dir / l1_cfg['artifact_names'][
        'host_native_full_with_pre_surr_csv'], artifact_type='csv', desc=
        'Full host data with native and global pre-remote surrogates')
    df_host_train_with_pre_surr = df_host_native_full_with_pre_surr.loc[
        train_vfl_keys].copy()
    error_set_train_vfl_keys = df_h_train_lf[df_h_train_lf['host0_leaf_id']
        .isin(err_leaf_ids)].index
    df_host_error_set_for_remote = pd.DataFrame()
    if not error_set_train_vfl_keys.empty:
        df_host_error_set_for_remote = df_host_train_with_pre_surr.loc[
            df_host_train_with_pre_surr.index.intersection(
            error_set_train_vfl_keys)].copy()
    y_host_error_set_for_remote = y_train.loc[y_train.index.intersection(
        df_host_error_set_for_remote.index)].copy()
    if df_host_error_set_for_remote.empty:
        logger.info(
            f'[{task_name}] Host error set (from augmented train data) is empty. Skipping remote interaction.'
            )
        save_artifact({}, out_dir / l1_cfg['artifact_names'][
            'selected_remote_rules_for_S_hat_training_json'], artifact_type
            ='json')
        _prepare_llm_prompt_and_data_for_s_hat_surrogates({},
            df_host_train_with_pre_surr, host_native_num_cols,
            host_native_cat_cols, pre_remote_surrogate_names, eval_spec,
            l1_cfg, task_name, out_dir, vfl_key)
        return
    active_qi_cols = [c for c in eval_spec['column_identifiers']['QI_COLS'] if
        c in df_host_error_set_for_remote.columns]
    prep_rem_res_tuple = _prepare_data_for_remote_processing(
        df_host_error_set_for_remote, y_host_error_set_for_remote,
        eval_spec, l1_cfg, task_def, task_name, vfl_key, db_engine,
        pre_remote_surrogate_names)
    selected_rules = None
    if not prep_rem_res_tuple:
        logger.warning(
            f'[{task_name}] Remote data preparation failed. No remote rules will be generated.'
            )
    else:
        (df_raw_merged, X_proc_lgbm_rem, y_aligned_rem,
            remote_model_num_cols_raw, remote_model_cat_cols_raw, _,
            feat_origin_map, remote_lgbm_preprocessor) = prep_rem_res_tuple
        df_host_error_set_native_for_h0_eval = df_host_native_full.loc[
            df_host_native_full.index.intersection(y_aligned_rem.index)].copy()
        rem_lgbm, rem_bat = _train_and_evaluate_remote_teacher_models(
            prep_rem_res_tuple, host0_lgbm, host0_lgbm_preproc,
            df_host_error_set_native_for_h0_eval, host_native_num_cols,
            host_native_cat_cols, task_def, l1_cfg, task_name, out_dir)
        selected_rules = _extract_rules_and_perform_pareto(rem_bat,
            df_raw_merged, y_aligned_rem, rem_lgbm, X_proc_lgbm_rem, l1_cfg,
            task_name, vfl_key, task_def, out_dir)
    if not selected_rules:
        logger.info(
            f'[{task_name}] No remote rules selected. Preparing LLM prompt with empty rule set.'
            )
        selected_rules = {}
    _prepare_llm_prompt_and_data_for_s_hat_surrogates(selected_rules,
        df_host_train_with_pre_surr, host_native_num_cols,
        host_native_cat_cols, pre_remote_surrogate_names, eval_spec, l1_cfg,
        task_name, out_dir, vfl_key)
    logger.info(
        f'[{task_name}] L1 processing (up to LLM prompt generation) finished.')


def _linearly_scale_series(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-09)


def main():
    global l1_cfg, eval_spec
    logger.info('Starting L1 script (T3-Final Flow) execution...')
    script_dir = Path(__file__).resolve().parent
    l1_cfg = load_json_config(script_dir / L1_CONFIG_FILE)
    eval_spec = load_json_config(script_dir / EVAL_SPEC_FILE)
    global GLOBAL_CONFIG_SEED
    GLOBAL_CONFIG_SEED = l1_cfg['seed']
    np.random.seed(GLOBAL_CONFIG_SEED)
    logger.info(f'Global seed set to: {GLOBAL_CONFIG_SEED}')
    ensure_dir_exists(Path(eval_spec['dataset_output_dir_name']))
    db_engine = get_db_engine(eval_spec['db_config_module'])
    if not db_engine:
        sys.exit('DB engine failed.')
    for task_name, task_def in eval_spec['tasks'].items():
        logger.info(f'===== Processing task: {task_name} =====')
        try:
            process_task_vfl_t3_final(task_name, task_def, eval_spec,
                l1_cfg, db_engine)
        except Exception as e:
            logger.error(f"Unhandled exception for task '{task_name}': {e}",
                exc_info=True)
    if db_engine:
        db_engine.dispose()
    logger.info('L1 script (T3-Final) finished all tasks.')


if __name__ == '__main__':
    main()
