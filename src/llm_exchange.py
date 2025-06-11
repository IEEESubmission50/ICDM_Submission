import argparse
import json
import logging
import shutil
import sys
import textwrap
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, Optional
import requests
import pandas as pd
import importlib.util
import traceback
from string import Template
import textwrap
import logging
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
import pandas as pd
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


def load_json_config(config_path: str) ->Dict:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f'Configuration file not found: {config_path}')
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(
            f'Error decoding JSON from configuration file: {config_path}')
        sys.exit(1)


def ensure_dir_exists(path: Path) ->None:
    path.mkdir(parents=True, exist_ok=True)


def call_llm_api(prompt: str, llm_config: Dict) ->str:
    TOGETHER_API_KEY = llm_config.get('api_key')
    MODEL_NAME = llm_config.get('model_name')
    TOGETHER_API_URL = llm_config.get('api_url')
    LLM_TIMEOUT = llm_config.get('timeout_seconds', 120)
    headers = {'Authorization': f'Bearer {TOGETHER_API_KEY}',
        'Content-Type': 'application/json'}
    payload = {'model': MODEL_NAME, 'messages': [{'role': 'user', 'content':
        prompt}], 'temperature': llm_config.get('temperature', 0),
        'max_tokens': llm_config.get('max_tokens', 8192)}
    logger.info(
        f'Calling LLM API for provider: Together AI, model: {MODEL_NAME}')
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=
            payload, timeout=LLM_TIMEOUT)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            logger.error(
                f'API call failed with status code {response.status_code}: {response.text}'
                )
            return (
                f'Error: API call failed with status {response.status_code}. Please check logs.'
                )
    except requests.exceptions.Timeout:
        logger.error(f'API call timed out after {LLM_TIMEOUT} seconds')
        return (
            'Error: API call timed out. Please try again or increase timeout value.'
            )
    except Exception as e:
        logger.error(f'Unexpected error during API call: {str(e)}')
        return f'Error: {str(e)}'


def extract_python_code(llm_response: str) ->Optional[str]:
    if '```python' in llm_response:
        start = llm_response.find('```python') + len('```python\n')
        end = llm_response.rfind('```')
        if start < end:
            return llm_response[start:end].strip()
    logger.warning(
        'Could not find ```python ... ``` block, attempting to use the whole response as code. This might be unreliable.'
        )
    return llm_response.strip()


def get_code_for_task(task_name: str, config: Dict, eval_spec: Dict,
    force_regenerate: bool=False):
    logger.info(f'[{task_name}] Mode: get_code')
    base_output_dir_for_dataset = Path(eval_spec['dataset_output_dir_name'])
    script1_output_dir = base_output_dir_for_dataset / task_name / config[
        'script1_input_subdir_template']
    script2_output_dir = base_output_dir_for_dataset / task_name / config[
        'script2_output_subdir_template']
    ensure_dir_exists(script2_output_dir)
    prompt_file_path = script1_output_dir / config[
        'prompt_filename_from_script1']
    generated_code_path = script2_output_dir / config['generated_code_filename'
        ]
    if generated_code_path.exists() and not force_regenerate:
        logger.info(
            f'[{task_name}] Code file {generated_code_path} already exists. Skipping generation. Use --force to regenerate.'
            )
        return
    if not prompt_file_path.exists():
        logger.error(f'[{task_name}] Prompt file not found: {prompt_file_path}'
            )
        return
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
    logger.info(f'[{task_name}] Sending prompt to LLM...')
    llm_response = call_llm_api(prompt_content, config['llm_api_config'])
    python_code = extract_python_code(llm_response)
    if python_code:
        with open(generated_code_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
        logger.info(
            f'[{task_name}] Generated code saved to {generated_code_path}')
    else:
        logger.error(
            f'[{task_name}] Could not extract Python code from LLM response.')
        raw_response_path = (script2_output_dir /
            'llm_raw_response_get_code.txt')
        with open(raw_response_path, 'w', encoding='utf-8') as f:
            f.write(llm_response)
        logger.info(
            f'[{task_name}] Raw LLM response saved to {raw_response_path}')


def execute_generated_code_for_task(task_name: str, cfg: Dict, eval_spec: Dict
    ) ->bool:
    logger.info(f'[{task_name}] Mode: execute_code')
    base_output_dir_for_dataset = Path(eval_spec['dataset_output_dir_name'])

    def _task_subdir_path(subdir_template_key: str) ->Path:
        return base_output_dir_for_dataset / task_name / cfg[
            subdir_template_key]

    def _write_wrapper(tmp: Path, data: Path, code: Path, out_: Path,
        vfl_key: str) ->Path:
        tpl = Template(textwrap.dedent(
            """
            import pandas as pd
            import sys, importlib.util, logging, traceback

            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - SUB - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
            logger = logging.getLogger(__name__)

            VFL_KEY     = "${vfl_key}"
            DATA_PATH   = r"${data}"
            CODE_PATH   = r"${code}"
            OUTPUT_PATH = r"${out_}"

            def main():
                try:
                    # ------------------------------------------------------------------
                    df = pd.read_csv(DATA_PATH)
                    df[VFL_KEY] = df[VFL_KEY].astype(str)
                    logger.info(f"Loaded data: shape={df.shape}")

                    spec = importlib.util.spec_from_file_location("surrogate_mod", CODE_PATH)
                    mod  = importlib.util.module_from_spec(spec)
                    sys.modules["surrogate_mod"] = mod
                    spec.loader.exec_module(mod)

                    if not hasattr(mod, "apply_surrogates"):
                        logger.error("apply_surrogates() missing in %s", CODE_PATH)
                        sys.exit(1)

                    df2 = mod.apply_surrogates(df.copy())
                    logger.info(f"Augmented data: shape={df2.shape}")

                    # ------------------------------------------------------------------
                    if VFL_KEY not in df2.columns:
                        df2[VFL_KEY] = df[VFL_KEY]

                    # cast & sanitise column names
                    df2.columns = (
                        df2.columns.astype(str)
                                   .str.strip()
                                   .str.replace(r"[^\\w]", "_", regex=True)
                                   .str.replace(r"__+", "_", regex=True)
                    )
                    logger.info(f"Sanitised column names: {list(df2.columns)}")

                    dup_cols = df2.columns[df2.columns.duplicated()].unique().tolist()
                    if dup_cols:
                        raise ValueError(f"Duplicate column names after sanitisation: {dup_cols}")

                    if df2[VFL_KEY].duplicated().any():
                        raise ValueError(f"Duplicate {VFL_KEY}s after surrogate gen")

                    df2[VFL_KEY] = df2[VFL_KEY].astype(str)
                    df2.to_csv(OUTPUT_PATH, index=False)
                    logger.info(f"Successfully wrote augmented data to {OUTPUT_PATH}")
                    sys.exit(0)

                except Exception as err:
                    logger.error("Wrapper execution failed: %s", err, exc_info=True)
                    print("TRACEBACK_START\\n" + traceback.format_exc() + "\\nTRACEBACK_END", file=sys.stderr)
                    sys.exit(1)

            if __name__ == "__main__":
                main()
            """
            ))
        content = tpl.substitute(data=str(data), code=str(code), out_=str(
            out_), vfl_key=vfl_key)
        wrapper_script_path = tmp / 'wrapper.py'
        wrapper_script_path.write_text(content, encoding='utf-8')
        logger.info('Generated wrapper script at: %s', wrapper_script_path)
        return wrapper_script_path

    def _run_subproc(wrapper: Path) ->subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(wrapper)],
            capture_output=True, text=True, check=False, timeout=cfg.get(
            'subprocess_timeout_seconds', 600), cwd=wrapper.parent,
            encoding='utf-8')

    def _log_failure(err_path: Path, proc: subprocess.CompletedProcess) ->None:
        err_path.write_text(
            f"""Return code: {proc.returncode}

STDOUT:
{proc.stdout}

STDERR:
{proc.stderr}"""
            , encoding='utf-8')
        logger.error('[%s] Execution failed; details in %s', task_name,
            err_path)
    script1_dir = _task_subdir_path('script1_input_subdir_template')
    script2_dir = _task_subdir_path('script2_output_subdir_template')
    ensure_dir_exists(script2_dir)
    in_csv = script1_dir / cfg['host_data_input_csv_from_script1']
    gen_code = script2_dir / cfg['generated_code_filename']
    out_csv = script2_dir / cfg['augmented_host_data_output_csv_name']
    err_log = script2_dir / cfg['error_log_filename']
    if err_log.exists():
        err_log.unlink()
    for pth, label in [(in_csv, 'Input CSV'), (gen_code, 'Generated code')]:
        if not pth.exists():
            logger.error('[%s] %s not found: %s', task_name, label, pth)
            return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_in = shutil.copy(in_csv, tmp_path / in_csv.name)
        tmp_code = shutil.copy(gen_code, tmp_path / gen_code.name)
        tmp_out = tmp_path / out_csv.name
        vfl_key = eval_spec['column_identifiers']['VFL_KEY']
        wrapper = _write_wrapper(tmp_path, Path(tmp_in), Path(tmp_code),
            tmp_out, vfl_key)
        proc = _run_subproc(wrapper)
        if proc.returncode == 0 and tmp_out.exists():
            shutil.move(str(tmp_out), out_csv)
            logger.info('[%s] Success; augmented data at %s', task_name,
                out_csv)
            return True
        _log_failure(err_log, proc)
        return False


def fix_generated_code_for_task(task_name: str, config: Dict, eval_spec: Dict):
    logger.info(f'[{task_name}] Mode: fix_code')
    base_output_dir_for_dataset = Path(eval_spec['dataset_output_dir_name'])
    script1_output_dir = base_output_dir_for_dataset / task_name / config[
        'script1_input_subdir_template']
    script2_output_dir = base_output_dir_for_dataset / task_name / config[
        'script2_output_subdir_template']
    generated_code_path = script2_output_dir / config['generated_code_filename'
        ]
    error_log_path = script2_output_dir / config['error_log_filename']
    prompt_template_path = Path(config['fix_code_prompt_template_file'])
    if not prompt_template_path.is_file():
        prompt_template_path = Path(__file__).resolve().parent / config[
            'fix_code_prompt_template_file']
    original_prompt_path = script1_output_dir / config[
        'prompt_filename_from_script1']
    if not generated_code_path.exists():
        logger.error(
            f'[{task_name}] No generated code file found at {generated_code_path} to fix.'
            )
        return False
    if not error_log_path.exists():
        logger.warning(
            f'[{task_name}] No error log file found at {error_log_path}. Assuming code is correct or not yet executed.'
            )
        return True
    if not prompt_template_path.exists():
        logger.error(
            f'[{task_name}] Fix code prompt template not found: {prompt_template_path}'
            )
        return False
    original_llm_prompt_content = 'Original prompt context not available.'
    if original_prompt_path.exists():
        with open(original_prompt_path, 'r', encoding='utf-8') as f:
            original_llm_prompt_content = f.read()
    else:
        logger.error(
            f'[{task_name}] Original LLM prompt not found: {original_prompt_path}. Cannot provide full context for fixing.'
            )
    with open(generated_code_path, 'r', encoding='utf-8') as f:
        faulty_code = f.read()
    with open(error_log_path, 'r', encoding='utf-8') as f:
        error_message = f.read()
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        fix_prompt_template = f.read()
    task_description = 'N/A'
    host_native_numeric_features_schema = 'N/A'
    host_native_categorical_features_schema = 'N/A'
    kb_insights_summary = (
        'Details about KB features were provided in the original prompt.')
    try:
        task_desc_marker = 'improve a predictive model for the task: '
        if task_desc_marker in original_llm_prompt_content:
            task_description = original_llm_prompt_content.split(
                task_desc_marker)[1].split('\n')[0]
        host_native_num_marker = (
            'The host data has the following native numeric features:\n')
        if host_native_num_marker in original_llm_prompt_content:
            host_native_numeric_features_schema = (original_llm_prompt_content
                .split(host_native_num_marker)[1].split('\n\n')[0].strip())
        host_native_cat_marker = (
            'And the following native categorical features:\n')
        if host_native_cat_marker in original_llm_prompt_content:
            host_native_categorical_features_schema = (
                original_llm_prompt_content.split(host_native_cat_marker)[1
                ].split('Analysis of difficult-to-predict')[0].strip())
    except Exception as e:
        logger.warning(
            f'Could not parse all details from original prompt for fix_code: {e}'
            )
    fix_prompt = fix_prompt_template.format(task_description=
        task_description, host_native_numeric_features_schema=
        host_native_numeric_features_schema,
        host_native_categorical_features_schema=
        host_native_categorical_features_schema, kb_insights_summary=
        kb_insights_summary, faulty_code=faulty_code, error_message=
        error_message)
    logger.info(f'[{task_name}] Sending code and error to LLM for fixing...')
    llm_response = call_llm_api(fix_prompt, config['llm_api_config'])
    fixed_code = extract_python_code(llm_response)
    if fixed_code:
        with open(generated_code_path, 'w', encoding='utf-8') as f:
            f.write(fixed_code)
        logger.info(
            f"[{task_name}] LLM provided fixed code. Saved to {generated_code_path}. Please re-run 'execute_code' mode."
            )
        return True
    else:
        logger.error(
            f'[{task_name}] Could not extract fixed Python code from LLM response.'
            )
        raw_response_path = (script2_output_dir /
            'llm_raw_response_fix_code.txt')
        with open(raw_response_path, 'w', encoding='utf-8') as f:
            f.write(llm_response)
        logger.info(
            f'[{task_name}] Raw LLM fix response saved to {raw_response_path}')
        return False


def main():
    mode, task_arg, force_regen = 'all', None, False
    force_regen = True
    cfg = load_json_config('llm_exchange_config.json')
    spec_file_path_str = cfg['evaluation_spec_file']
    spec_file_path = Path(spec_file_path_str)
    if not spec_file_path.is_file():
        spec_file_path = Path(__file__).resolve().parent / spec_file_path.name
    if not spec_file_path.exists():
        logger.error(
            f'Evaluation spec file not found at {spec_file_path_str} or {spec_file_path}'
            )
        sys.exit(1)
    with open(spec_file_path, 'r', encoding='utf-8') as f:
        eval_spec = json.load(f)
    tasks_to_process = {}
    if task_arg and task_arg in eval_spec['tasks']:
        tasks_to_process = {task_arg: eval_spec['tasks'][task_arg]}
    elif not task_arg:
        tasks_to_process = eval_spec['tasks']
    else:
        logger.error(
            f"Specified task '{task_arg}' not found in {spec_file_path}")
        sys.exit(1)
    for task_name_loop in tasks_to_process:
        if mode in ('get_code', 'all'):
            get_code_for_task(task_name_loop, cfg, eval_spec, force_regen)
        if mode in ('execute_code', 'all'):
            execute_generated_code_for_task(task_name_loop, cfg, eval_spec)
        if mode == 'fix_code':
            for attempt in range(cfg.get('max_fix_attempts', 3)):
                logger.info(f'[{task_name_loop}] Fix attempt {attempt + 1}')
                fix_attempted = fix_generated_code_for_task(task_name_loop,
                    cfg, eval_spec)
                if not fix_attempted:
                    logger.error(
                        f'[{task_name_loop}] LLM could not provide a fix. Stopping fix attempts.'
                        )
                    break
                execution_successful = execute_generated_code_for_task(
                    task_name_loop, cfg, eval_spec)
                if execution_successful:
                    logger.info(
                        f'[{task_name_loop}] Code execution successful after fix attempt {attempt + 1}.'
                        )
                    error_log_path = Path(eval_spec['dataset_output_dir_name']
                        ) / task_name_loop / cfg[
                        'script2_output_subdir_template'] / cfg[
                        'error_log_filename']
                    if error_log_path.exists():
                        error_log_path.unlink()
                        logger.info(
                            f'[{task_name_loop}] Cleared error log: {error_log_path}'
                            )
                    break
                else:
                    logger.warning(
                        f'[{task_name_loop}] Code execution failed after fix attempt {attempt + 1}.'
                        )
                    if attempt == cfg.get('max_fix_attempts', 3) - 1:
                        logger.error(
                            f'[{task_name_loop}] Max fix attempts reached. Code remains broken.'
                            )
    logger.info('Script 2 (LLM Exchange) processing finished.')


if __name__ == '__main__':
    main()
