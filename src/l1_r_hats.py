import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import itertools
import joblib
import numpy as np
import pandas as pd
import shap
from utils import tag_feature_origin, get_base_feature_name_for_penalty, _create_numeric_preprocessing_pipeline, _impute_and_cast_categoricals, preprocess_features_for_lgbm, _create_bat_preprocessor, train_bat_model, _initialize_lgbm_training_config, _prepare_lgbm_training_data_and_cats, _apply_lgbm_dynamic_hyperparams_for_penalty, _construct_lgbm_feature_penalties, _finalize_lgbm_model_params_for_constructor, _setup_lgbm_early_stopping_and_eval, _instantiate_and_fit_lgbm_model, train_lgbm_model, _robust_zscore, analyze_shap_by_origin, calculate_binary_classification_metrics, get_s_hat_performance_stats, load_json_config, ensure_dir_exists, save_artifact, load_artifact, standardise_key, ensure_sorted_unique_key, _get_lgbm_cat_feature_names, canonicalize_feature_name, canonicalize_feature_dict, _fmt, format_schema_for_llm_prompt, _safe_to_numeric, get_predictions, evaluate_predictions, split_data_and_save_indices, sanitize_feature_names, sanitize_feature_name, pretty_print_shap
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
EVAL_SPEC_FILE = Path('evaluation_spec.json')
L1_CONFIG_FILE = Path('l1_config.json')


def _create_single_phat_interaction(df: pd.DataFrame, col1_name: str,
    col2_name: str, interaction_type: str, prefix: str, epsilon: float=1e-06
    ) ->Tuple[pd.Series, str]:
    p1_id = col1_name.split('p_hat_rule_')[-1]
    p2_id = col2_name.split('p_hat_rule_')[-1]
    new_col_name = f'{prefix}phat_{interaction_type}_{p1_id}_vs_{p2_id}'
    if interaction_type == 'ratio':
        interaction_series = df[col1_name] / (df[col2_name] + epsilon)
    elif interaction_type == 'sum':
        interaction_series = df[col1_name] + df[col2_name]
    else:
        raise ValueError(
            f'Unsupported p-hat interaction type: {interaction_type}')
    return interaction_series, new_col_name


def _create_host_phat_interaction(df: pd.DataFrame, host_col_name: str,
    phat_col_name: str, prefix: str) ->Tuple[pd.Series, str]:
    host_feat_id = sanitize_feature_name(host_col_name)
    p_hat_id = phat_col_name.split('p_hat_rule_')[-1]
    new_col_name = f'{prefix}hostinter_{host_feat_id}_x_{p_hat_id}'
    interaction_series = df[host_col_name].fillna(0) * df[phat_col_name]
    return interaction_series, new_col_name


def _generate_interaction_features(df_base: pd.DataFrame, p_hat_col_names:
    list[str], host_numeric_cols: list[str], task_name: str, cfg_type_c: Dict
    ) ->tuple[pd.DataFrame, list[str]]:
    logger.info(
        f'[{task_name}] Generating Type C: Interaction/Ratio Features...')
    df_with_interactions = df_base.copy()
    interaction_col_names = []
    epsilon = 1e-06
    interaction_prefix = cfg_type_c.get('prefix', 'surrx_cross_')
    max_degree = cfg_type_c.get('max_interaction_degree', 2)
    top_n_phats = cfg_type_c.get('top_n_phat_for_interactions', len(
        p_hat_col_names))
    phat_interaction_types = cfg_type_c.get('phat_interaction_types', [
        'ratio', 'sum'])
    host_interaction_cfg = cfg_type_c.get('host_feature_interactions', {})
    logger.info(
        f'[{task_name}] Using interaction prefix: {interaction_prefix}, max_degree: {max_degree}, top_n_phats: {top_n_phats}'
        )
    if not p_hat_col_names:
        logger.info(
            f'[{task_name}] No p_hat_rule_ columns available – skipping Type C features.'
            )
        return df_with_interactions, interaction_col_names
    phats_to_interact = p_hat_col_names[:min(top_n_phats, len(p_hat_col_names))
        ]
    if max_degree >= 2 and len(phats_to_interact) >= 2:
        logger.info(
            f'[{task_name}] Generating pairwise interactions for top {len(phats_to_interact)} p-hat features: {phats_to_interact}'
            )
        for col1, col2 in itertools.combinations(phats_to_interact, 2):
            if 'ratio' in phat_interaction_types:
                series, name = _create_single_phat_interaction(
                    df_with_interactions, col1, col2, 'ratio',
                    interaction_prefix, epsilon)
                df_with_interactions[name] = series
                interaction_col_names.append(name)
            if 'sum' in phat_interaction_types:
                series, name = _create_single_phat_interaction(
                    df_with_interactions, col1, col2, 'sum', interaction_prefix
                    )
                df_with_interactions[name] = series
                interaction_col_names.append(name)
    if host_interaction_cfg.get('enabled', False
        ) and host_numeric_cols and phats_to_interact:
        num_host_to_use = host_interaction_cfg.get('num_host_features_to_use',
            1)
        num_phats_for_host_interact = host_interaction_cfg.get(
            'num_phats_to_use_with_host', 1)
        selected_host_cols = host_numeric_cols[:min(num_host_to_use, len(
            host_numeric_cols))]
        selected_phats_for_host = phats_to_interact[:min(
            num_phats_for_host_interact, len(phats_to_interact))]
        logger.info(
            f'[{task_name}] Generating host-p_hat interactions using {selected_host_cols} and {selected_phats_for_host}'
            )
        for host_col in selected_host_cols:
            for phat_col in selected_phats_for_host:
                series, name = _create_host_phat_interaction(
                    df_with_interactions, host_col, phat_col,
                    interaction_prefix)
                df_with_interactions[name] = series
                interaction_col_names.append(name)
    for col in interaction_col_names:
        if col in df_with_interactions.columns and df_with_interactions[col
            ].isnull().any():
            df_with_interactions[col] = df_with_interactions[col].replace([
                np.inf, -np.inf], np.nan)
            df_with_interactions[col] = df_with_interactions[col].fillna(0.0)
    logger.info(
        f'[{task_name}] Added {len(interaction_col_names)} Type C interaction features.'
        )
    return df_with_interactions, interaction_col_names


def _generate_type_A_host0_shap_surrogates(df_base: pd.DataFrame,
    host0_lgbm_model: Any, host0_lgbm_preprocessor: Any,
    host_native_num_cols: List[str], host_native_cat_cols: List[str],
    l1_r_hats_cfg: Dict, task_name: str) ->Tuple[pd.DataFrame, List[str]]:
    df_augmented = df_base.copy()
    new_type_a_surrogates = []
    cfg = l1_r_hats_cfg.get('type_A_host0_shap_surrogates', {})
    if not cfg.get('enabled', False):
        logger.info(
            f'[{task_name}] Type A (Host0 SHAP) surrogate generation disabled.'
            )
        return df_augmented, new_type_a_surrogates
    top_n = cfg.get('top_n_host0_shap_features', 3)
    method = cfg.get('method', 'robustify')
    prefix = cfg.get('prefix', 'h0_shap_surrx_')
    X_native_raw = df_base[host_native_num_cols + host_native_cat_cols]
    if X_native_raw.empty:
        logger.warning(
            f'[{task_name}] Native feature set for Host0 SHAP is empty. Skipping Type A.'
            )
        return df_augmented, new_type_a_surrogates
    X_native_proc, _ = preprocess_features_for_lgbm(X_native_raw,
        host_native_num_cols, host_native_cat_cols,
        f'{task_name}_Host0_ForTypeAShAP', fit_mode=False,
        existing_num_preprocessor=host0_lgbm_preprocessor)
    if X_native_proc is None or X_native_proc.empty:
        logger.warning(
            f'[{task_name}] Preprocessing for Host0 SHAP failed. Skipping Type A.'
            )
        return df_augmented, new_type_a_surrogates
    try:
        explainer = shap.TreeExplainer(host0_lgbm_model)
        shap_values_host0 = explainer.shap_values(X_native_proc)
        shap_matrix = shap_values_host0[1] if isinstance(shap_values_host0,
            list) and len(shap_values_host0) == 2 else shap_values_host0
        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        processed_feature_names = X_native_proc.columns.tolist()
        feature_shap_pairs = sorted(zip(processed_feature_names,
            mean_abs_shap), key=lambda x: x[1], reverse=True)
        features_to_process_raw_names = []
        for proc_feat_name, _ in feature_shap_pairs[:top_n]:
            original_name = get_base_feature_name_for_penalty(str(
                proc_feat_name))
            if original_name in host_native_num_cols:
                features_to_process_raw_names.append(original_name)
        logger.info(
            f'[{task_name}] Top Host0 SHAP numeric features for Type A: {features_to_process_raw_names}'
            )
        for feat_name in features_to_process_raw_names:
            if feat_name not in df_augmented.columns:
                continue
            safe_feat_name = sanitize_feature_name(feat_name)
            if method == 'robustify':
                feat_series_num = _safe_to_numeric(df_augmented[feat_name])
                if feat_series_num is None:
                    continue
                z_scores = _robust_zscore(feat_series_num)
                surr_col_name = f'{prefix}anti_{safe_feat_name}'
                df_augmented[surr_col_name] = 1.0 / (1.0 + np.exp(z_scores.
                    clip(-10, 10)))
                new_type_a_surrogates.append(surr_col_name)
            elif method == 'mask':
                surr_col_name = f'{prefix}masked_{safe_feat_name}'
                median_val = df_augmented[feat_name].median()
                df_augmented[surr_col_name] = median_val
                logger.info(
                    f"[{task_name}] Masking feature '{feat_name}' (column name: {surr_col_name}) - effectively removing its variance by setting to median for downstream models that use this new name."
                    )
                new_type_a_surrogates.append(surr_col_name)
    except Exception as e:
        logger.error(
            f'[{task_name}] Error generating Type A (Host0 SHAP) surrogates: {e}'
            , exc_info=True)
    logger.info(
        f'[{task_name}] Generated {len(new_type_a_surrogates)} Type A (Host0 SHAP-based) features.'
        )
    return df_augmented, new_type_a_surrogates


def _extract_s_hat_shap_summary(model: Any, X_processed: pd.DataFrame,
    task_name: str, rule_id: str) ->Optional[Dict]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_processed)
        shap_matrix = shap_vals[1] if isinstance(shap_vals, list) and len(
            shap_vals) == 2 else shap_vals
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        return canonicalize_feature_dict(dict(zip(X_processed.columns,
            mean_abs)))
    except Exception as exc:
        logger.warning(
            f'[{task_name}] Rule {rule_id}: SHAP for S-hat failed – {exc}')
        return None


def _make_binary_target(df: pd.DataFrame, covered_keys: List[str], rule_id: str
    ) ->pd.Series:
    y = pd.Series(0, index=df.index, name=f'target_{rule_id}')
    y.loc[df.index.intersection(covered_keys)] = 1
    return y


def _split_for_s_hat(X: pd.DataFrame, y: pd.Series, seed: int) ->Tuple[pd.
    DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split
    strat = y if y.nunique() > 1 and y.value_counts(normalize=True).min(
        ) > 0.1 else None
    return train_test_split(X, y, test_size=0.2, random_state=seed,
        stratify=strat)


def _harmonise_categorical_dtypes(df_train: pd.DataFrame, df_val: pd.
    DataFrame, cat_cols: List[str], task_tag: str) ->Tuple[pd.DataFrame, pd
    .DataFrame]:
    for col in cat_cols:
        if col not in df_train.columns or col not in df_val.columns:
            continue
        cats = sorted(list(set(df_train[col].astype('category').cat.
            categories) | set(df_val[col].astype('category').cat.categories)))
        df_train[col] = df_train[col].astype(pd.CategoricalDtype(categories
            =cats))
        df_val[col] = df_val[col].astype(pd.CategoricalDtype(categories=cats))
        logger.debug("[%s] Column '%s' → categories: %s", task_tag, col,
            cats[:10])
    return df_train, df_val


def _save_s_hat_artifacts(model: Any, preproc: Any, stats: Dict, out_dir:
    Path, cfg: Dict, rule_id: str):
    save_artifact(model, out_dir / cfg['mini_lgbm_S_hat_model_template_pkl'
        ].format(rule_id=rule_id), 'pkl')
    if preproc:
        save_artifact(preproc, out_dir / cfg[
            'mini_lgbm_S_hat_preprocessor_template_pkl'].format(rule_id=
            rule_id), 'pkl')
    save_artifact(stats, out_dir / cfg[
        'S_hat_distribution_stats_template_json'].format(rule_id=rule_id),
        'json')


def _train_host_calibration_model_S_hat_j(df_host_train: pd.DataFrame,
    y_train_main: pd.Series, rule_covered_vfl_keys: List[str],
    host_num_cols: List[str], host_cat_cols: List[str], mini_lgbm_params:
    Dict, task_name: str, rule_id: str, seed: int, out_dir: Path,
    artifacts_cfg: Dict) ->Optional[Tuple[Any, Any, Dict, Optional[Dict]]]:
    y_rule = _make_binary_target(df_host_train, rule_covered_vfl_keys, rule_id)
    if y_rule.nunique() < 2 or y_rule.sum() < 5:
        logger.warning(
            f'[{task_name}] Rule {rule_id}: S-hat target has <2 classes or <5 positives. Skipping.'
            )
        return None
    X_tr_raw, X_v_raw, y_tr_rule, y_v_rule = _split_for_s_hat(df_host_train
        [host_num_cols + host_cat_cols], y_rule, seed)
    X_train_proc, preproc = preprocess_features_for_lgbm(X_tr_raw,
        host_num_cols, host_cat_cols, f'{task_name}_SHat_{rule_id}Train',
        fit_mode=True)
    if X_train_proc is None:
        return None
    X_val_proc, _ = preprocess_features_for_lgbm(X_v_raw, host_num_cols,
        host_cat_cols, f'{task_name}_SHat_{rule_id}Val', fit_mode=False,
        existing_num_preprocessor=preproc)
    if X_val_proc is None:
        X_val_proc = pd.DataFrame()
    if not X_train_proc.empty and not X_val_proc.empty:
        X_train_proc, X_val_proc = _harmonise_categorical_dtypes(X_train_proc,
            X_val_proc, host_cat_cols, f'{task_name}_SHat_{rule_id}')
    s_hat_model = train_lgbm_model(X_train_proc, y_tr_rule,
        mini_lgbm_params, 'binary', X_val_proc, y_v_rule)
    X_full_proc = pd.concat([X_train_proc, X_val_proc]
        ) if not X_val_proc.empty else X_train_proc
    X_full_proc.columns = sanitize_feature_names(X_full_proc.columns)
    y_full_rule = pd.concat([y_tr_rule, y_v_rule]
        ) if not X_val_proc.empty else y_tr_rule
    preds_full = get_predictions(s_hat_model, X_full_proc, 'binary')
    stats = get_s_hat_performance_stats(preds_full, y_full_rule.loc[
        X_full_proc.index], rule_id)
    shap_summary = _extract_s_hat_shap_summary(s_hat_model, X_full_proc,
        task_name, rule_id)
    _save_s_hat_artifacts(s_hat_model, preproc, stats, out_dir,
        artifacts_cfg, rule_id)
    logger.info(
        f"[{task_name}] Rule {rule_id}: S-hat training done. AUC: {stats.get('s_hat_roc_auc', 'N/A'):.4f}, AUPRC: {stats.get('s_hat_auprc', 'N/A'):.4f}"
        )
    return s_hat_model, preproc, stats, shap_summary


def _train_r_hat_models_and_add_phats(df_full: pd.DataFrame, selected_rules:
    dict, host_num_cols: list[str], host_cat_cols: list[str], vfl_key: str,
    l1_cfg: dict, task_name: str, out_dir: Path) ->tuple[pd.DataFrame, list
    [str]]:
    phat_cols: list[str] = []
    if not selected_rules:
        logger.warning(
            f'[{task_name}] No selected rules found. Skipping R-hat model training.'
            )
        return df_full, phat_cols
    mini_params = l1_cfg['host_calibration_model_params'][
        'mini_vfl_lgbm_params']
    art = l1_cfg['artifact_names']
    for rule_id, rule_meta in selected_rules.items():
        covered_keys = rule_meta['covered_instance_ids_by_remote']
        result = _train_host_calibration_model_S_hat_j(df_full, pd.Series(
            dtype=float), covered_keys, host_num_cols, host_cat_cols,
            mini_params, task_name, rule_id, l1_cfg['seed'], out_dir, art)
        auc = result[2].get('s_hat_roc_auc', 0)
        if auc < 0.6:
            logger.info('skipped because of the low AUC: %s', auc)
            continue
        m_path = out_dir / art['mini_lgbm_S_hat_model_template_pkl'].format(
            rule_id=rule_id)
        p_path = out_dir / art['mini_lgbm_S_hat_preprocessor_template_pkl'
            ].format(rule_id=rule_id)
        if not (m_path.exists() and p_path.exists()):
            continue
        model = joblib.load(m_path)
        pre = joblib.load(p_path)
        feats = [c for c in host_num_cols + host_cat_cols if c in df_full.
            columns]
        X_raw = df_full[feats].copy()
        X_proc, _ = preprocess_features_for_lgbm(X_raw, host_num_cols,
            host_cat_cols, f'{task_name}_RHatPred_{rule_id}', fit_mode=
            False, existing_num_preprocessor=pre)
        if X_proc is None or X_proc.empty:
            continue
        preds = get_predictions(model, X_proc, 'binary')
        col_name = f'p_hat_rule_{rule_id}'
        df_full[col_name] = pd.Series(preds, index=X_proc.index)
        phat_cols.append(col_name)
    return df_full, phat_cols


def process_task_l1_r_hats(task_name: str, task_def: Dict, eval_spec: Dict,
    l1_cfg: Dict):
    logger.info(
        f'[{task_name}] Starting L1 R-hats and Final Surrogate Generation...')
    l1_output_dir_template = l1_cfg['script1_output_subdir_template']
    l1_out_dir = Path(eval_spec['dataset_output_dir_name']
        ) / task_name / l1_output_dir_template
    vfl_key = eval_spec['column_identifiers']['VFL_KEY']
    base_data_for_final_surrs_path = l1_out_dir / l1_cfg['artifact_names'][
        'host_native_full_with_pre_surr_csv']
    df_host_base_with_pre_surr = load_artifact(base_data_for_final_surrs_path)
    if df_host_base_with_pre_surr is None:
        logger.error(
            f'[{task_name}] Base host data with pre-remote surrogates not found at {base_data_for_final_surrs_path}. Aborting l1_r_hats.'
            )
        return
    df_host_base_with_pre_surr = standardise_key(df_host_base_with_pre_surr,
        vfl_key)
    llm_s_hat_surrs_path = l1_out_dir / l1_cfg['artifact_names'][
        'llm_s_hat_surrogates_csv']
    df_augmented_locally = df_host_base_with_pre_surr.copy()
    llm_phat_cols = []
    if llm_s_hat_surrs_path.exists():
        df_llm_s_hats = load_artifact(llm_s_hat_surrs_path)
        if df_llm_s_hats is not None and not df_llm_s_hats.empty:
            df_llm_s_hats = standardise_key(df_llm_s_hats, vfl_key)
            llm_phat_cols = [c for c in df_llm_s_hats.columns if c != vfl_key]
            df_augmented_locally = df_augmented_locally.join(df_llm_s_hats.
                set_index(vfl_key)[llm_phat_cols], how='left')
            dup_cols = [col for col in df_augmented_locally.columns if col.
                endswith('_dup')]
            if dup_cols:
                df_augmented_locally.drop(columns=dup_cols, inplace=True)
            df_augmented_locally = standardise_key(df_augmented_locally,
                vfl_key)
            logger.info(
                f'[{task_name}] Merged {len(llm_phat_cols)} LLM-generated S-hat surrogates.'
                )
        else:
            logger.warning(
                f'[{task_name}] LLM S-hat surrogates file empty at {llm_s_hat_surrs_path}. Proceeding without them.'
                )
    else:
        logger.warning(
            f'[{task_name}] LLM S-hat surrogates file not found at {llm_s_hat_surrs_path}. Proceeding without them.'
            )
    selected_rules = load_artifact(l1_out_dir / l1_cfg['artifact_names'][
        'selected_remote_rules_for_S_hat_training_json'])
    if selected_rules is None:
        logger.warning(
            f'[{task_name}] Could not load selected remote rules. Will proceed without R-hat models.'
            )
    host_native_num_cols = [c for c in eval_spec['columns'][
        'HOST_NUMERIC_COLS_FOR_MODELING'] if c in df_augmented_locally.columns]
    host_native_cat_cols = [c for c in eval_spec['columns'][
        'HOST_CATEGORICAL_COLS_FOR_MODELING'] if c in df_augmented_locally.
        columns]
    current_all_cols = df_augmented_locally.columns.tolist()
    s_hat_input_num_cols = []
    s_hat_input_cat_cols = []
    s_hat_input_num_cols.extend(host_native_num_cols)
    s_hat_input_cat_cols.extend(host_native_cat_cols)
    pre_surrx_cols = [col for col in current_all_cols if col.startswith(
        'pre_surrx_')]
    for col in pre_surrx_cols:
        if pd.api.types.is_numeric_dtype(df_augmented_locally[col]):
            if col not in s_hat_input_num_cols:
                s_hat_input_num_cols.append(col)
        elif col not in s_hat_input_cat_cols:
            s_hat_input_cat_cols.append(col)
    for col in llm_phat_cols:
        if pd.api.types.is_numeric_dtype(df_augmented_locally[col]):
            if col not in s_hat_input_num_cols:
                s_hat_input_num_cols.append(col)
        else:
            if col not in s_hat_input_cat_cols:
                s_hat_input_cat_cols.append(col)
            logger.warning(
                f'[{task_name}] LLM p_hat column {col} is not numeric, treating as categorical for S-hat.'
                )
    s_hat_input_num_cols = list(dict.fromkeys(s_hat_input_num_cols))
    s_hat_input_cat_cols = list(dict.fromkeys(s_hat_input_cat_cols))
    logger.info(
        f'[{task_name}] S-hat training will use {len(s_hat_input_num_cols)} numeric and {len(s_hat_input_cat_cols)} categorical features.'
        )
    df_augmented_locally, new_phat_cols = _train_r_hat_models_and_add_phats(
        df_augmented_locally, selected_rules, s_hat_input_num_cols,
        s_hat_input_cat_cols, vfl_key, l1_cfg, task_name, l1_out_dir)
    logger.info('[%s] Added %d new p_hat_rule_* features.', task_name, len(
        new_phat_cols))
    l1_r_hats_surr_cfg = l1_cfg.get('final_surrogate_generation_in_l1_r_hats',
        {})
    host0_lgbm = load_artifact(l1_out_dir / l1_cfg['artifact_names'][
        'host0_model_pkl'])
    host0_lgbm_preproc = load_artifact(l1_out_dir / l1_cfg['artifact_names'
        ]['host0_preprocessor_pkl'])
    if host0_lgbm and host0_lgbm_preproc:
        df_augmented_locally, type_a_cols = (
            _generate_type_A_host0_shap_surrogates(df_augmented_locally,
            host0_lgbm, host0_lgbm_preproc, host_native_num_cols,
            host_native_cat_cols, l1_r_hats_surr_cfg, task_name))
    else:
        logger.warning(
            f'[{task_name}] Host0 model/preprocessor not loaded. Skipping Type A Host0 SHAP surrogates.'
            )
    type_c_config = l1_r_hats_surr_cfg.get('type_C_interaction_surrogates', {})
    phats_for_interaction = new_phat_cols if new_phat_cols else llm_phat_cols
    if phats_for_interaction and type_c_config.get('enabled', False):
        logger.info(
            f"[{task_name}] Using p-hats for Type C interactions: {'S-hat model outputs (new_phat_cols)' if new_phat_cols else 'LLM outputs (llm_phat_cols)'}"
            )
        df_augmented_locally, type_c_cols = _generate_interaction_features(
            df_augmented_locally, phats_for_interaction,
            host_native_num_cols, task_name, type_c_config)
    else:
        logger.info(
            f'[{task_name}] Skipping Type C interaction surrogate generation.')
    df_augmented_locally = df_augmented_locally.loc[:, ~
        df_augmented_locally.columns.duplicated(keep='first')]
    logger.info(
        f'[{task_name}] L1 R-hats: Final augmented shape for L2: {df_augmented_locally.shape}'
        )
    df_to_save = df_augmented_locally.reset_index(drop=True)
    save_artifact(df_to_save, l1_out_dir / l1_cfg['artifact_names'][
        'host_data_with_local_surrogates_csv'], artifact_type='csv', desc=
        'Final host data with all local (native, pre-remote, LLM S-hat, Type A/C) surrogates'
        )
    logger.info(f'[{task_name}] L1 R-hats processing complete.')


def main_l1_r_hats():
    logger.info('Starting L1 R-Hats script execution...')
    script_dir = Path(__file__).resolve().parent
    l1_cfg_path = script_dir / L1_CONFIG_FILE
    eval_spec_path = script_dir / EVAL_SPEC_FILE
    l1_cfg_loaded = load_json_config(l1_cfg_path)
    eval_spec_loaded = load_json_config(eval_spec_path)
    if not l1_cfg_loaded or not eval_spec_loaded:
        logger.error(
            'Failed to load L1 config or evaluation spec for l1_r_hats.py')
        sys.exit(1)
    global GLOBAL_CONFIG_SEED
    GLOBAL_CONFIG_SEED = l1_cfg_loaded.get('seed', 42)
    np.random.seed(GLOBAL_CONFIG_SEED)
    logger.info(f'Global seed set to: {GLOBAL_CONFIG_SEED}')
    for task_name, task_def in eval_spec_loaded.get('tasks', {}).items():
        logger.info(f'===== L1 R-Hats: Processing task: {task_name} =====')
        try:
            process_task_l1_r_hats(task_name, task_def, eval_spec_loaded,
                l1_cfg_loaded)
        except Exception as e:
            logger.error(
                f"Unhandled exception in L1 R-Hats for task '{task_name}': {e}"
                , exc_info=True)
    logger.info('L1 R-Hats script finished all tasks.')


if __name__ == '__main__':
    main_l1_r_hats()
