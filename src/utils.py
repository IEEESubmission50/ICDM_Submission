import re
import sys
import joblib
import json, logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Set
import pandas.api.types as pdt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, f1_score, precision_score, recall_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
logger = logging.getLogger(__name__)
import warnings, shap
warnings.filterwarnings('ignore', message=
    'LightGBM binary classifier with TreeExplainer shap values output has changed.*'
    , category=UserWarning, module='shap')


def tag_feature_origin(col: str, qi_cols: Set[str], pre_remote_surr_cols:
    Set[str]) ->str:
    if col in qi_cols:
        return 'host_qi'
    if col in pre_remote_surr_cols:
        return 'host_surr'
    return 'remote_true'


def get_base_feature_name_for_penalty(transformed_name: str) ->str:
    name = str(transformed_name)
    if '__' in name:
        parts = name.split('__', 1)
        transformer_prefix = parts[0]
        processed_name_part = parts[1]
        base_name = processed_name_part
        if transformer_prefix.startswith('cat'
            ) or transformer_prefix.startswith('onehot'):
            name_parts_for_ohe_strip = processed_name_part.rsplit('_', 1)
            if len(name_parts_for_ohe_strip) > 1:
                base_name = name_parts_for_ohe_strip[0]
        return canonicalize_feature_name(base_name)
    return canonicalize_feature_name(name)


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


def preprocess_features_for_lgbm(df: pd.DataFrame, numeric_cols: List[str],
    categorical_cols: List[str], task_name: str, fit_mode: bool=True,
    existing_num_preprocessor: Optional[ColumnTransformer]=None,
    exclude_cols: Optional[List[str]]=None) ->Tuple[Optional[pd.DataFrame],
    Optional[ColumnTransformer]]:
    exclude_cols = exclude_cols or []
    if df.empty:
        logger.warning(
            f'[{task_name}] Input DataFrame to preprocess_features_for_lgbm is empty.'
            )
        return pd.DataFrame(), existing_num_preprocessor
    df_proc = df.drop(columns=[c for c in exclude_cols if c in df.columns and
        c != df.index.name], errors='ignore').copy()
    coerced_num_cols = []
    for col in numeric_cols:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
    active_num = [c for c in numeric_cols if c in df_proc.columns and pd.
        api.types.is_numeric_dtype(df_proc[c])]
    active_cat = [c for c in categorical_cols if c in df_proc.columns]
    df_proc = _impute_and_cast_categoricals(df_proc, active_cat, task_name)
    processed_num_df = pd.DataFrame(index=df_proc.index)
    num_preprocessor = existing_num_preprocessor
    if active_num:
        num_pipeline = _create_numeric_preprocessing_pipeline()
        if fit_mode or not num_preprocessor:
            num_preprocessor = ColumnTransformer([('num', num_pipeline,
                active_num)], remainder='drop')
            processed_array = num_preprocessor.fit_transform(df_proc[
                active_num])
        else:
            try:
                processed_array = num_preprocessor.transform(df_proc[
                    active_num])
            except ValueError as ve:
                logger.error(
                    f'[{task_name}] Error transforming numeric features: {ve}. Check feature consistency.'
                    )
                return None, num_preprocessor
        num_feature_names_out = sanitize_feature_names(num_preprocessor.
            get_feature_names_out())
        processed_num_df = pd.DataFrame(processed_array, columns=
            num_feature_names_out, index=df_proc.index)
        for col in processed_num_df.columns:
            processed_num_df[col] = pd.to_numeric(processed_num_df[col],
                errors='coerce')
    df_cat_part = df_proc[active_cat].copy()
    if not processed_num_df.empty and not df_cat_part.empty:
        final_df = pd.concat([processed_num_df, df_cat_part], axis=1)
    elif not processed_num_df.empty:
        final_df = processed_num_df
    elif not df_cat_part.empty:
        final_df = df_cat_part
    else:
        final_df = pd.DataFrame(index=df_proc.index)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    logger.info(
        f'[{task_name}] Preprocessed for LGBM. Shape: {final_df.shape}. Columns: {final_df.columns.tolist()[:10]}...'
        )
    return final_df, num_preprocessor


def _create_bat_preprocessor(num_cols: List[str], cat_cols: List[str]
    ) ->ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(
            strategy='median')), ('scaler', StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(
            strategy='most_frequent')), ('onehot', OneHotEncoder(
            handle_unknown='ignore', sparse_output=False))]), cat_cols))
    return ColumnTransformer(transformers, remainder='drop')


def train_bat_model(X_train_raw: pd.DataFrame, y_train: pd.Series,
    bat_params: Dict, task_type: str, seed: int, num_cols: List[str],
    cat_cols: List[str]) ->Optional[Any]:
    if X_train_raw.empty:
        logger.warning('Empty X_train_raw for BAT.')
        return None
    X_train_bat = X_train_raw.copy()
    for col in cat_cols:
        X_train_bat[col] = X_train_bat[col].astype(str).fillna('!MISSING!')
    bat_preprocessor = _create_bat_preprocessor(num_cols, cat_cols)
    try:
        X_train_bat_proc = bat_preprocessor.fit_transform(X_train_bat)
        bat_feature_names = sanitize_feature_names(bat_preprocessor.
            get_feature_names_out())
    except Exception as e:
        logger.error(f'BAT preprocessing error: {e}')
        return None
    params = {**bat_params, 'random_state': seed}
    model = DecisionTreeClassifier(**params
        ) if task_type == 'binary' else DecisionTreeRegressor(**params)
    model.fit(X_train_bat_proc, y_train)
    model.bat_preprocessor_ = bat_preprocessor
    model.bat_feature_names_ = bat_feature_names
    return model


def _initialize_lgbm_training_config(params: Dict, sample_weight: Optional[
    pd.Series]) ->Tuple[Dict, Dict]:
    model_params = params.copy()
    fit_params = {}
    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight
    return model_params, fit_params


def _prepare_lgbm_training_data_and_cats(X_train: pd.DataFrame) ->Tuple[pd.
    DataFrame, List[str]]:
    X_train_model_input = X_train.copy()
    lgbm_cat_features = [col for col in X_train_model_input.columns if
        isinstance(X_train_model_input[col].dtype, pd.CategoricalDtype)]
    return X_train_model_input, lgbm_cat_features


def _apply_lgbm_dynamic_hyperparams_for_penalty(model_params: Dict,
    task_name_for_debug: str) ->Dict:
    penalty_config_key = 'dynamic_hyperparams_on_penalty'
    penalty_mode = model_params.get('feature_penalty_mode', 'off')
    if penalty_mode != 'off' and penalty_config_key in model_params:
        dynamic_params = model_params.pop(penalty_config_key)
        logger.info(
            f'[{task_name_for_debug}] Applying dynamic hyperparameters for penalty mode: {dynamic_params}'
            )
        if 'feature_fraction' in dynamic_params:
            model_params['colsample_bytree'] = dynamic_params[
                'feature_fraction']
        if 'bagging_fraction' in dynamic_params:
            model_params['subsample'] = dynamic_params['bagging_fraction']
        if 'n_estimators' in dynamic_params:
            model_params['n_estimators'] = dynamic_params['n_estimators']
        if 'min_child_samples' in dynamic_params:
            model_params['min_child_samples'] = dynamic_params[
                'min_child_samples']
    return model_params


def _construct_lgbm_feature_penalties(model_params: Dict,
    X_train_model_input: pd.DataFrame, feature_origin_map: Optional[Dict[
    str, str]], task_name_for_debug: str) ->Dict:
    if model_params.get('feature_penalty_mode', 'off'
        ) != 'off' and feature_origin_map:
        penalties = []
        remote_mult = model_params.get('remote_feature_penalty_multiplier', 1.0
            )
        surr_mult = model_params.get('surrogate_feature_penalty_multiplier',
            1.0)
        qi_mult = model_params.get('host_qi_penalty_multiplier', 1.0)
        for processed_col_name in X_train_model_input.columns:
            base_raw_name = get_base_feature_name_for_penalty(
                processed_col_name)
            origin = feature_origin_map.get(base_raw_name, 'remote_true')
            if origin == 'remote_true':
                penalties.append(remote_mult)
            elif origin == 'host_surr':
                penalties.append(surr_mult)
            elif origin == 'host_qi':
                penalties.append(qi_mult)
            else:
                penalties.append(1.0)
        if len(penalties) == X_train_model_input.shape[1]:
            model_params['feature_penalty'] = penalties
            logger.info(
                f'[{task_name_for_debug}] Applied feature penalties. Example: {penalties[:10]}'
                )
        else:
            logger.error(
                f'[{task_name_for_debug}] Penalty vector length mismatch. Penalties not applied.'
                )
    return model_params


def _finalize_lgbm_model_params_for_constructor(model_params: Dict) ->Dict:
    model_params.pop('feature_penalty_mode', None)
    model_params.pop('remote_feature_penalty_multiplier', None)
    model_params.pop('surrogate_feature_penalty_multiplier', None)
    model_params.pop('host_qi_penalty_multiplier', None)
    model_params.pop('dynamic_hyperparams_on_penalty', None)
    return model_params


def _setup_lgbm_early_stopping_and_eval(model_params: Dict, fit_params:
    Dict, task_type: str, X_val: Optional[pd.DataFrame]=None, y_val:
    Optional[pd.Series]=None) ->Tuple[Dict, Dict]:
    if (X_val is not None and y_val is not None and not X_val.empty and
        model_params.get('early_stopping_rounds')):
        X_val_model_input = X_val.copy()
        X_val_model_input.columns = sanitize_feature_names(X_val.columns.
            tolist())
        fit_params['eval_set'] = [(X_val_model_input, y_val)]
        fit_params['eval_metric'] = 'auc' if task_type == 'binary' else 'mae'
        fit_params['callbacks'] = [lgb.early_stopping(model_params.pop(
            'early_stopping_rounds'), verbose=-1)]
    return model_params, fit_params


def _instantiate_and_fit_lgbm_model(X_train_model_input: pd.DataFrame,
    y_train: pd.Series, model_params: Dict, fit_params: Dict, task_type:
    str, task_name_for_debug: str) ->Any:
    model_class = (lgb.LGBMClassifier if task_type == 'binary' else lgb.
        LGBMRegressor)
    model = model_class(**model_params)
    try:
        model.fit(X_train_model_input, y_train, **fit_params)
    except Exception as e:
        logger.error(
            f'[{task_name_for_debug}] Model training failed: {e}. Params: {model_params}, Fit_Params_Keys: {fit_params.keys()}'
            )
        if 'feature_penalty' in model_params:
            logger.error(
                f"Feature penalty (sample): {model_params['feature_penalty'][:20]}"
                )
        if 'categorical_feature' in fit_params:
            logger.error(
                f"Categorical features: {fit_params['categorical_feature']}")
        raise e
    return model


def train_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, params:
    Dict, task_type: str, X_val: Optional[pd.DataFrame]=None, y_val:
    Optional[pd.Series]=None, sample_weight: Optional[pd.Series]=None,
    feature_origin_map: Optional[Dict[str, str]]=None,
    original_numeric_cols: Optional[List[str]]=None,
    original_categorical_cols: Optional[List[str]]=None) ->Any:
    model_params, fit_params = _initialize_lgbm_training_config(params,
        sample_weight)
    X_train_model_input, lgbm_cat_features = (
        _prepare_lgbm_training_data_and_cats(X_train))
    if lgbm_cat_features:
        fit_params['categorical_feature'] = lgbm_cat_features
    task_name_for_debug = model_params.get('task_name_for_debug', 'LGBM')
    model_params = _apply_lgbm_dynamic_hyperparams_for_penalty(model_params,
        task_name_for_debug)
    model_params = _construct_lgbm_feature_penalties(model_params,
        X_train_model_input, feature_origin_map, task_name_for_debug)
    model_params = _finalize_lgbm_model_params_for_constructor(model_params)
    model_params, fit_params = _setup_lgbm_early_stopping_and_eval(model_params
        , fit_params, task_type, X_val, y_val)
    model = _instantiate_and_fit_lgbm_model(X_train_model_input, y_train,
        model_params, fit_params, task_type, task_name_for_debug)
    return model


def _robust_zscore(series: pd.Series) ->pd.Series:
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std < 1e-06:
        return pd.Series(0.0, index=series.index).fillna(0.0)
    z = (series - mean) / std
    return z.fillna(0.0)


def analyze_shap_by_origin(shap_values_matrix: np.ndarray,
    processed_feature_names: List[str], feature_origin_map: Dict[str, str],
    task_name: str) ->Dict[str, float]:
    if shap_values_matrix.ndim != 2 or len(processed_feature_names
        ) != shap_values_matrix.shape[1]:
        logger.error(
            f'[{task_name}] SHAP matrix and feature names mismatch for origin analysis.'
            )
        return {'error_shap_analysis': 1.0}
    abs_shap_mean_per_feature = np.abs(shap_values_matrix).mean(axis=0)
    origin_totals: Dict[str, float] = {'host_qi': 0.0, 'host_surr': 0.0,
        'remote_true': 0.0}
    unmapped_shap_total = 0.0
    for i, processed_feat_name in enumerate(processed_feature_names):
        base_name = get_base_feature_name_for_penalty(processed_feat_name)
        origin = feature_origin_map.get(base_name, 'remote_true')
        if base_name not in feature_origin_map and '_' in base_name:
            potential_original = base_name.rsplit('_', 1)[0]
            origin = feature_origin_map.get(potential_original, 'remote_true')
        if origin in origin_totals:
            origin_totals[origin] += abs_shap_mean_per_feature[i]
        else:
            logger.warning(
                f"[{task_name}] Unmapped feature origin for '{processed_feat_name}' (base: '{base_name}'). Attributing its SHAP to 'remote_true'."
                )
            origin_totals['remote_true'] += abs_shap_mean_per_feature[i]
    total_shap_mass = sum(origin_totals.values())
    if total_shap_mass == 0:
        logger.warning(
            f'[{task_name}] Total SHAP mass is zero for origin analysis.')
        return {k: (0.0) for k in origin_totals}
    return {k: (v / total_shap_mass) for k, v in origin_totals.items()}


def calculate_binary_classification_metrics(y_true: pd.Series, y_pred_proba:
    np.ndarray, prefix: str='') ->Dict[str, float]:
    metrics = {}
    if y_true.nunique() > 1:
        metrics[f'{prefix}roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics[f'{prefix}auprc'] = average_precision_score(y_true,
            y_pred_proba)
    else:
        metrics[f'{prefix}roc_auc'] = 0.5
        metrics[f'{prefix}auprc'] = y_true.mean()
    metrics[f'{prefix}brier_score'] = brier_score_loss(y_true, y_pred_proba)
    return metrics


def get_s_hat_performance_stats(preds: np.ndarray, y_true: pd.Series,
    rule_id: str) ->Dict[str, Any]:
    stats = {'s_hat_rule_id': rule_id, 's_hat_target_prevalence': y_true.mean()
        }
    if preds.size > 0:
        stats.update({'s_hat_pred_mean': float(preds.mean()),
            's_hat_pred_std': float(preds.std()), 's_hat_pred_min': float(
            preds.min()), 's_hat_pred_max': float(preds.max()),
            's_hat_pred_q50': float(np.percentile(preds, 50))})
        perf_metrics = calculate_binary_classification_metrics(y_true,
            preds, 's_hat_')
        stats.update(perf_metrics)
    else:
        stats.update({'s_hat_pred_mean': np.nan, 's_hat_pred_std': np.nan,
            's_hat_roc_auc': 0.5, 's_hat_auprc': y_true.mean() if not
            y_true.empty else np.nan, 's_hat_brier_score': np.nan})
    return stats


def load_json_config(config_path: Path) ->Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f'Error loading config {config_path}: {e}')
        sys.exit(1)


def ensure_dir_exists(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class NpEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o) if pd.notna(o) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if pd.isna(o):
            return None
        return super().default(o)


def save_artifact(data: Any, path: Path, desc: str='artifact',
    artifact_type: Optional[str]=None):
    ensure_dir_exists(path.parent)
    effective_type = artifact_type if artifact_type else path.suffix.lower(
        ).replace('.', '')
    try:
        if effective_type == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, cls=NpEncoder)
        elif effective_type == 'text':
            with open(path, 'w', encoding='utf-8') as f:
                f.write(data)
        elif effective_type == 'pkl':
            joblib.dump(data, path)
        elif effective_type == 'csv' and isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            logger.error(f"Unsupported save type '{effective_type}' for {path}"
                )
            return
        logger.info(f'Saved {desc} ({effective_type}) to {path}')
    except Exception as e:
        logger.error(f'Error saving {desc} to {path}: {e}')


def load_artifact(path: Path, desc: str='artifact') ->Optional[Any]:
    if not path.exists():
        logger.error(f'{desc.capitalize()} file not found: {path}')
        return None
    try:
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif path.suffix == '.pkl':
            return joblib.load(path)
        elif path.suffix == '.csv':
            return pd.read_csv(path)
        else:
            logger.error(f'Unsupported load type for {path}')
            return None
    except Exception as e:
        logger.error(f'Error loading {desc} from {path}: {e}')
        return None


def standardise_key(df: pd.DataFrame, key: str) ->pd.DataFrame:
    df = df.copy()
    if (key in df.columns and df.index.name == key and df[key].dtype ==
        'object' and df.index.dtype == 'object' and not df.index.duplicated
        ().any()):
        return df
    if key not in df.columns:
        if key in df.index.names:
            df[key] = df.index
        else:
            raise KeyError(f'{key} not found as column or index')
    df[key] = df[key].astype(str)
    if key not in df.index.names or df.index.name != key:
        df.set_index(key, inplace=True, drop=False)
    df.index = df.index.astype(str)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
    return df


def ensure_sorted_unique_key(df: pd.DataFrame, key: str) ->pd.DataFrame:
    return standardise_key(df, key)


def _get_lgbm_cat_feature_names(model, fallback_df=None):
    if hasattr(model, 'categorical_feature_name_'
        ) and model.categorical_feature_name_:
        return model.categorical_feature_name_
    if hasattr(model, 'categorical_feature_') and model.categorical_feature_:
        return model.categorical_feature_
    if fallback_df is not None:
        return [c for c in fallback_df.columns if isinstance(fallback_df[c]
            .dtype, pd.CategoricalDtype)]
    return []


NUM_PREFIX = 'num_'


def canonicalize_feature_name(name: str) ->str:
    return name[len(NUM_PREFIX):] if name.startswith(NUM_PREFIX) else name


def canonicalize_feature_dict(raw_dict: Dict[str, float]) ->Dict[str, float]:
    canon: Dict[str, float] = {}
    for k, v in raw_dict.items():
        ck = canonicalize_feature_name(k)
        canon[ck] = canon.get(ck, 0.0) + float(v)
    return canon


def _fmt(val: Any, digits: int=4) ->str:
    if val is None:
        return 'N/A'
    try:
        if isinstance(val, (float, int, np.floating, np.integer)):
            if np.isnan(val):
                return 'N/A'
            return ('{0:.' + str(digits) + 'f}').format(val)
    except Exception:
        pass
    return str(val)


def format_schema_for_llm_prompt(df: pd.DataFrame, feature_list: List[str],
    column_descriptions: Optional[Dict[str, str]]=None, max_examples: int=3
    ) ->str:
    column_descriptions = column_descriptions or {}
    lines: List[str] = []
    for col in feature_list:
        df_col_name = col
        if col not in df.columns and 'num_' + col in df.columns:
            df_col_name = 'num_' + col
        if df_col_name not in df.columns:
            continue
        series = df[df_col_name].dropna()
        dtype_str = str(series.dtype)
        sample_vals = ', '.join(map(str, series.unique()[:max_examples])
            ) or 'N/A'
        desc = column_descriptions.get(col, column_descriptions.get(
            df_col_name, 'N/A'))
        lines.append('  - `{}` (dtype: {}, examples: [{}]). Description: {}'
            .format(col, dtype_str, sample_vals, desc))
    return '\n'.join(lines) if lines else '  (No matching features)'


def _safe_to_numeric(series: pd.Series) ->(pd.Series | None):
    if pdt.is_numeric_dtype(series):
        return series
    coerced = pd.to_numeric(series, errors='coerce')
    if coerced.notna().any():
        return coerced
    midpoints = series.str.extract('(\\d+)[^\\d]+(\\d+)').dropna()
    if not midpoints.empty:
        midpoint = midpoints.astype(float).mean(axis=1)
        coerced = pd.Series(midpoint, index=series.index)
        return coerced
    return None


def get_predictions(model: Any, X: Union[pd.DataFrame, np.ndarray],
    task_type: str) ->np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X.columns = sanitize_feature_names(X.columns)
    if task_type == 'binary':
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)
    else:
        return model.predict(X)


def evaluate_predictions(y_true: Union[pd.Series, np.ndarray], y_pred: np.
    ndarray, task_type: str, metric: str='auc') ->float:
    metric = metric.lower()
    try:
        if task_type == 'binary':
            if metric == 'auc':
                return float(roc_auc_score(y_true, y_pred)) if len(np.
                    unique(y_true)) > 1 else 0.5
            elif metric == 'auprc':
                return float(average_precision_score(y_true, y_pred)) if len(np
                    .unique(y_true)) > 1 else 0.0
            elif metric == 'brier':
                return float(brier_score_loss(y_true, y_pred))
            elif metric == 'f1':
                return float(f1_score(y_true, (y_pred > 0.5).astype(int),
                    zero_division=0))
            elif metric == 'precision':
                return float(precision_score(y_true, (y_pred > 0.5).astype(
                    int), zero_division=0))
            elif metric == 'recall':
                return float(recall_score(y_true, (y_pred > 0.5).astype(int
                    ), zero_division=0))
            else:
                logger.warning("Unknown metric '%s' for binary.", metric)
                return 0.0
        if metric in ('mae', 'l1'):
            return float(mean_absolute_error(y_true, y_pred))
        logger.warning("Unknown metric '%s' for regression; using MAE.", metric
            )
        return float(mean_absolute_error(y_true, y_pred))
    except Exception as exc:
        logger.error('Metric %s failed: %s', metric, exc)
        return np.nan


def split_data_and_save_indices(df_features: pd.DataFrame, y: pd.Series,
    task_type: str, l1_cfg: Dict[str, Any], out_dir: Path, task_name: str
    ) ->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.
    Series, pd.Series]:
    split_cfg = l1_cfg['data_split_params']
    rng = l1_cfg.get('seed', 2024)
    test_size = split_cfg.get('test_size', 0.1)
    val_size_total = split_cfg.get('val_size', 0.1)
    stratify_all = y if task_type == 'binary' and y.nunique() > 1 else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_features,
        y, test_size=test_size, random_state=rng, stratify=stratify_all)
    remaining_frac = 1.0 - test_size
    val_frac = val_size_total / remaining_frac if remaining_frac > 0 else 0.1
    val_frac = max(min(val_frac, 0.5), 0.05)
    stratify_tv = y_train_val if task_type == 'binary' and y_train_val.nunique(
        ) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(X_train_val,
        y_train_val, test_size=val_frac, random_state=rng, stratify=stratify_tv
        )
    y_train.index = X_train.index
    y_val.index = X_val.index
    y_test.index = X_test.index
    idx_json = {'train_idx': [str(i) for i in X_train.index], 'val_idx': [
        str(i) for i in X_val.index], 'test_idx': [str(i) for i in X_test.
        index]}
    out_path = out_dir / l1_cfg['artifact_names']['split_indices_json']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as fp:
        json.dump(idx_json, fp, indent=2)
    logger.info('[%s] Split sizes – train=%d | val=%d | test=%d', task_name,
        len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


ILLEGAL_CHARS_REGEX = re.compile('[^a-zA-Z0-9_]')
REPEATED_UNDERSCORE_REGEX = re.compile('__+')


def sanitize_feature_names(names):
    return [sanitize_feature_name(name) for name in names]


def sanitize_feature_name(name):
    name = str(name)
    name = name.replace('=', '_eq_')
    name = name.replace('<', '_lt_')
    name = name.replace('>', '_gt_')
    name = name.replace(' ', '_')
    name = name.replace('/', '_div_')
    name = name.replace('+', '_plus_')
    name = name.replace('-', '_minus_')
    name = name.replace('.', '_dot_')
    name = name.replace('(', '_')
    name = name.replace(')', '_')
    name = name.replace('[', '_')
    name = name.replace(']', '_')
    name = name.replace('{', '_')
    name = name.replace('}', '_')
    name = name.replace('"', '_')
    name = name.replace('\\', '_')
    name = name.replace(':', '_')
    name = name.replace(',', '_')
    name = name.replace('!', '_excl_')
    name = name.replace('?', '_q_')
    name = name.replace('&', '_and_')
    name = name.replace('%', '_pct_')
    name = name.replace('*', '_mult_')
    name = name.replace('#', '_num_')
    name = name.replace('@', '_at_')
    sanitized = ILLEGAL_CHARS_REGEX.sub('_', name)
    sanitized = REPEATED_UNDERSCORE_REGEX.sub('_', sanitized)
    sanitized = sanitized.strip('_')
    if not sanitized:
        sanitized = 'unnamed_feature'
    return sanitized


def pretty_print_shap(shap_dict: dict, top_n: int=10, logger_obj=None):
    if not shap_dict:
        (logger_obj or print)('No SHAP values to display.')
        return
    head = f"{'Feature':<40} | {'|SHAP|':>10}"
    bar = '-' * len(head)
    rows = [head, bar]
    for k, v in sorted(shap_dict.items(), key=lambda kv: kv[1], reverse=True)[:
        top_n]:
        rows.append(f'{k:<40} | {v:10.4f}')
    msg = '\n'.join(rows)
    if logger_obj:
        logger_obj.info('\n' + msg)
    else:
        print(msg)
