import streamlit as st
import pandas as pd
import json
import os
import random
import copy # Для глубокого копирования словарей/списков
from math import floor, ceil, isnan
from typing import Dict, Any, Optional, List, Literal, Tuple, Set

# =============================================
# --- Конфигурация и Константы ---
# =============================================

# --- Файлы и Данные ---
JSONL_FILE_PATH = 'vkusvill_data_enriched_v4.jsonl' # <<< Используем V4 файл
ANALYSIS_FIELD_V3 = 'product_analysis_v3' # Ключ для данных V3
ENRICHMENT_V4_KEY = 'enrichment_v4' # Ключ для данных V4
ANALYSIS_PREFIX = "analysis_" # Префикс для V3 полей
V4_PREFIX = 'v4_'             # Префикс для V4 полей
NUTRITION_PREFIX = "nutrition_"
MULTI_PORTION_SPLIT_BONUS = 0.75

# Базовые колонки
BREADCRUMBS_COL = 'breadcrumbs'
URL_COL = 'url'
NAME_COL = 'name'
IMAGE_COL = 'image_url'
WEIGHT_VALUE_COL = 'weight_value'
WEIGHT_UNIT_COL = 'weight_unit'

# Колонки КБЖУ
NUTRITION_BASE_COL = f'{NUTRITION_PREFIX}basis'
NUTRITION_COLS_BASE = {'kcal': f'{NUTRITION_PREFIX}kcal', 'protein': f'{NUTRITION_PREFIX}protein', 'fat': f'{NUTRITION_PREFIX}fat', 'carbs': f'{NUTRITION_PREFIX}carbs'}
TOTAL_NUTRITION_COLUMNS = {'kcal': 'total_kcal', 'protein': 'total_protein', 'fat': 'total_fat', 'carbs': 'total_carbs'}
CALCULABLE_COLUMN = 'calculable'
NUTRIENT_KEYS: List[str] = ['kcal', 'protein', 'fat', 'carbs'] # Для итерации

# Категории и Роли
READY_MEAL_CATEGORIES: Set[str] = {"Готовая еда", "Готовая ета"}
UNLIKELY_BREAKFAST_ROLES: Set[str] = {'Primary Protein Source', 'Complete Meal'}
SNACK_ROLES: Set[str] = {'Snack', 'Fruit/Dessert'}

# Колонки V4
V4_PORTIONS_COL = f'{V4_PREFIX}portion_suggested_portions'

# --- Алгоритм Предложения ---
SUGGESTION_DEFAULTS: Dict[str, Any] = {'min_items': 4, 'max_items': 8, 'tolerance': 0.15, 'max_attempts': 300, 'candidate_limit': 150}
SUGGESTION_SCORING_WEIGHTS: Dict[str, float] = {'deficit_reduction_base': 1.2, 'calorie_bonus_if_needed': 0.05, 'high_calorie_ratio_penalty_factor': 8.0, 'over_limit_penalty_factor': 1.5, 'diet_rating_bonus_factor': 0.20, 'added_sugar_penalty': 0.35, 'whole_grains_bonus': 0.30, 'fiber_factor': 2.5, 'satiety_factor': 1.8, 'density_factor': 1.2, 'level_score_high': 0.15, 'level_score_medium': 0.0, 'level_score_low': -0.15}

# --- Алгоритм Распределения (Smart Distribution V3.0 с порциями) ---
MEAL_SLOTS_ORDER: List[str] = ["Завтрак", "Обед", "Ужин", "Перекус"]
MEAL_TARGET_PERCENTAGES: Dict[str, float] = {"Завтрак": 0.25, "Обед": 0.35, "Ужин": 0.25, "Перекус": 0.15}
assert abs(sum(MEAL_TARGET_PERCENTAGES.values()) - 1.0) < 1e-6, "Сумма процентов для приемов пищи должна быть 100%"
MEAL_RATING_COLS_MAP: Dict[str, str] = {slot: f"{ANALYSIS_PREFIX}meal_{slot.lower()}_rating" for slot in ["Breakfast", "Lunch", "Dinner", "Snack"]}
MEAL_RATING_COLS_MAP_RU: Dict[str, str] = {"Завтрак": MEAL_RATING_COLS_MAP["Breakfast"], "Обед": MEAL_RATING_COLS_MAP["Lunch"], "Ужин": MEAL_RATING_COLS_MAP["Dinner"], "Перекус": MEAL_RATING_COLS_MAP["Snack"]}
MIN_RATING_MEAL: int = 3
MIN_RATING_SNACK: int = 2
BREAKFAST_ROLE_PENALTY: int = 2

# --- Ожидаемые Схемы для Колонок Анализа (V3 и V4) ---
# (Определяем обе схемы для обработки данных)
EXPECTED_ANALYSIS_SCHEMA: Dict[str, Dict[str, Any]] = {
    f"{ANALYSIS_PREFIX}meal_suitability_reasoning": {'dtype': str, 'default': "N/A"}, f"{ANALYSIS_PREFIX}meal_breakfast_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}meal_lunch_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}meal_dinner_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}meal_snack_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}diet_goals_reasoning": {'dtype': str, 'default': "N/A"}, f"{ANALYSIS_PREFIX}diet_weight_loss_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}diet_muscle_gain_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}diet_general_health_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}diet_low_calorie_snack_rating": {'dtype': int, 'default': 1}, f"{ANALYSIS_PREFIX}meal_component_role": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}satiety_index_estimate": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}nutrient_density_estimate": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}fiber_level_estimate": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}sodium_level_estimate": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}likely_contains_added_sugar": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}likely_contains_whole_grains": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}health_benefit_tags": {'dtype': object, 'default': []}, f"{ANALYSIS_PREFIX}preparation_complexity": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}cooking_method_guess": {'dtype': object, 'default': None}, f"{ANALYSIS_PREFIX}primary_flavor_profile": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}primary_texture": {'dtype': str, 'default': "Uncertain"}, f"{ANALYSIS_PREFIX}pairing_suggestion": {'dtype': object, 'default': None}, f"{ANALYSIS_PREFIX}is_potential_source_of_calcium": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}is_potential_source_of_iron": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}is_potential_source_of_potassium": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}is_potential_source_of_vitamin_c": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}is_potential_source_of_vitamin_d": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}is_potential_source_of_vitamin_b12": {'dtype': bool, 'default': False}, f"{ANALYSIS_PREFIX}micronutrient_comment": {'dtype': object, 'default': None},
}
EXPECTED_V4_SCHEMA: Dict[str, Dict[str, Any]] = {
     f'{V4_PREFIX}portion_suggested_portions': {'dtype': pd.Int64Dtype(), 'default': 1},
     f'{V4_PREFIX}portion_portion_reasoning': {'dtype': str, 'default': None}, # Allow None
     f'{V4_PREFIX}context_dominant_macro': {'dtype': str, 'default': 'uncertain'},
     f'{V4_PREFIX}context_consumption_temperature': {'dtype': str, 'default': 'any'},
     f'{V4_PREFIX}context_is_dessert': {'dtype': bool, 'default': False},
}

# --- UI Настройки ---
PRIMARY_GOAL_OPTIONS: List[str] = ['None', 'Weight Loss', 'Muscle Gain', 'General Health']
FILTER_SUGAR_OPTIONS: Dict[str, Optional[bool]] = {"Любой": None, "Без добав.": False, "С добав.": True}
FILTER_GRAINS_OPTIONS: Dict[str, Optional[bool]] = {"Любые": None, "С цельн.": True, "Без цельн.": False}
FILTER_ESTIMATE_COLS: Dict[str, str] = {f'{ANALYSIS_PREFIX}fiber_level_estimate': 'Клетчатка', f'{ANALYSIS_PREFIX}sodium_level_estimate': 'Натрий', f'{ANALYSIS_PREFIX}satiety_index_estimate': 'Сытость', f'{ANALYSIS_PREFIX}nutrient_density_estimate': 'Плотн.нутр.'}
FILTER_ROLE_COL: str = f'{ANALYSIS_PREFIX}meal_component_role'
FILTER_PREP_COL: str = f'{ANALYSIS_PREFIX}preparation_complexity'
FILTER_TAGS_COL: str = f'{ANALYSIS_PREFIX}health_benefit_tags'
MICRONUTRIENT_MAP: Dict[str, str] = {'Ca': f'{ANALYSIS_PREFIX}is_potential_source_of_calcium', 'Fe': f'{ANALYSIS_PREFIX}is_potential_source_of_iron', 'K': f'{ANALYSIS_PREFIX}is_potential_source_of_potassium', 'C': f'{ANALYSIS_PREFIX}is_potential_source_of_vitamin_c', 'D': f'{ANALYSIS_PREFIX}is_potential_source_of_vitamin_d', 'B12': f'{ANALYSIS_PREFIX}is_potential_source_of_vitamin_b12'}

# =============================================
# --- Проверка Наличия Файла ---
# =============================================
if not os.path.exists(JSONL_FILE_PATH):
    st.error(f"Критическая ошибка: Не найден файл данных '{JSONL_FILE_PATH}'.")
    st.stop()

# =============================================
# --- Настройка Страницы Streamlit ---
# =============================================
st.set_page_config(page_title="Конструктор Рациона V6.1 (Порции)", layout="wide") # Версия 6.1

# =============================================
# --- Загрузка и Предобработка Данных ---
# =============================================

@st.cache_data
def load_jsonl(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Читает JSONL файл и возвращает список словарей."""
    data: List[Dict[str, Any]] = []
    lines_processed_log: int = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try: data.append(json.loads(line))
                except json.JSONDecodeError:
                    lines_processed_log += 1
                    if lines_processed_log < 20 or i % 500 == 0:
                        st.warning(f"Ошибка JSON в строке {i+1} файла {file_path}. Строка пропущена.")
                    pass
            if not data: st.error(f"Не найдено валидных JSON строк в файле: {file_path}"); return None
            return data
    except Exception as e: st.error(f"Критическая ошибка при чтении файла '{file_path}': {e}"); return None

def flatten_analysis(analysis_data: Any) -> Dict[str, Any]:
    """Разворачивает вложенный словарь анализа V3 в плоский."""
    flat_data: Dict[str, Any] = {}
    if not isinstance(analysis_data, dict): return {}
    for key, value in analysis_data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items(): flat_data[f"{ANALYSIS_PREFIX}{key}_{sub_key}"] = sub_value
        elif isinstance(value, list) and key == 'health_benefit_tags': flat_data[f'{ANALYSIS_PREFIX}{key}'] = value
        elif not isinstance(value, (list, dict)): flat_data[f'{ANALYSIS_PREFIX}{key}'] = value
    return flat_data

def flatten_v4_enrichment(v4_data: Any) -> Dict[str, Any]:
    """Разворачивает вложенный словарь V4 (portion_info, additional_context) в плоский."""
    flat_data: Dict[str, Any] = {}
    if not isinstance(v4_data, dict): return {}

    # Обработка portion_info
    portion_info = v4_data.get('portion_info', {})
    if isinstance(portion_info, dict):
        for p_key, p_value in portion_info.items():
            flat_data[f"{V4_PREFIX}portion_{p_key}"] = p_value

    # Обработка additional_context
    context_info = v4_data.get('additional_context', {})
    if isinstance(context_info, dict):
        for c_key, c_value in context_info.items():
            flat_data[f"{V4_PREFIX}context_{c_key}"] = c_value

    return flat_data

def calculate_total_nutrition(row: pd.Series) -> pd.Series:
    """Рассчитывает КБЖУ на весь продукт."""
    weight_value = row.get(WEIGHT_VALUE_COL); kcal_per_basis = row.get(NUTRITION_COLS_BASE['kcal']); basis_string = row.get(NUTRITION_BASE_COL); weight_unit = str(row.get(WEIGHT_UNIT_COL, '')).lower()
    results_na = pd.Series({CALCULABLE_COLUMN: False, **{col: pd.NA for col in TOTAL_NUTRITION_COLUMNS.values()}})
    if pd.isna(weight_value) or pd.isna(kcal_per_basis) or pd.isna(basis_string): return results_na
    weight_multiplier: Optional[float] = None
    if weight_unit in ['кг', 'kg', 'л', 'l']: weight_multiplier = 1000.0
    elif weight_unit in ['мл', 'ml', 'г', 'g']: weight_multiplier = 1.0
    if weight_multiplier is None: return results_na
    try: total_weight_grams: float = float(weight_value) * weight_multiplier
    except (ValueError, TypeError): return results_na
    basis_string_lower: str = str(basis_string).lower()
    is_per_100_basis: bool = '100' in basis_string_lower and any(unit_part in basis_string_lower for unit_part in ['грамм', 'грам', 'г', 'ml', 'мл'])
    if not is_per_100_basis: return results_na
    scaling_factor: float = total_weight_grams / 100.0
    results: Dict[str, Any] = {CALCULABLE_COLUMN: True}
    for key, nutrient_col in NUTRITION_COLS_BASE.items():
        try:
            value_per_100 = pd.to_numeric(row.get(nutrient_col, pd.NA), errors='coerce')
            results[TOTAL_NUTRITION_COLUMNS[key]] = pd.NA if pd.isna(value_per_100) else float(value_per_100) * scaling_factor
        except (ValueError, TypeError): results[TOTAL_NUTRITION_COLUMNS[key]] = pd.NA
    # Если калории не рассчитались, считаем весь расчет невалидным
    if pd.isna(results.get(TOTAL_NUTRITION_COLUMNS['kcal'])):
         results[CALCULABLE_COLUMN] = False
         for col in TOTAL_NUTRITION_COLUMNS.values(): results[col] = pd.NA # Обнуляем все КБЖУ
    return pd.Series(results)


@st.cache_data
def load_and_preprocess_data(file_path: str) -> Optional[pd.DataFrame]:
    """Загружает и обрабатывает данные V4 (включая V3 и V4 анализы)."""
    st.write(f"Загрузка данных из: {file_path}")
    raw_data = load_jsonl(file_path);
    if raw_data is None: st.error("Не удалось загрузить данные."); return None
    df = pd.DataFrame(raw_data)
    if df.empty: st.warning("Файл данных пуст."); return None
    st.success(f"Загружено {len(df)} строк.")

    with st.spinner("Предобработка данных (КБЖУ, Анализ V3, Анализ V4)..."):
        df_processed = df.copy()

        # --- Базовая обработка типов ---
        numeric_cols = ['price', 'rating', WEIGHT_VALUE_COL];
        for col in numeric_cols:
            if col in df_processed.columns: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # --- Обработка 'nutrition' и расчет 'total_...' КБЖУ ---
        if 'nutrition' in df_processed.columns:
            def safe_normalize_nutrition(x): return x if isinstance(x, dict) else {}
            normalized_nutrition = df_processed['nutrition'].apply(safe_normalize_nutrition)
            if normalized_nutrition.apply(lambda d: bool(d)).any():
                try:
                    nutrition_data = pd.json_normalize(normalized_nutrition).add_prefix(NUTRITION_PREFIX)
                    df_processed = df_processed.reset_index(drop=True); nutrition_data = nutrition_data.reset_index(drop=True)
                    if 'nutrition' in df_processed.columns: df_processed = df_processed.drop(columns=['nutrition'])
                    df_processed = pd.concat([df_processed, nutrition_data], axis=1)
                    for col in NUTRITION_COLS_BASE.values():
                        if col in df_processed.columns: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                except Exception as e_norm: st.warning(f"Ошибка нормализации 'nutrition': {e_norm}.")
            else:
                 if 'nutrition' in df_processed.columns: df_processed = df_processed.drop(columns=['nutrition'])
        for col in list(NUTRITION_COLS_BASE.values()) + [NUTRITION_BASE_COL]:
             if col not in df_processed.columns: df_processed[col] = pd.NA
        try:
            total_nutrition_df = df_processed.apply(calculate_total_nutrition, axis=1)
            df_processed = df_processed.join(total_nutrition_df)
            if CALCULABLE_COLUMN not in df_processed.columns: df_processed[CALCULABLE_COLUMN] = False
        except Exception as e_calc: st.error(f"Критическая ошибка расчета КБЖУ: {e_calc}"); df_processed[CALCULABLE_COLUMN] = False

        # --- Разворачивание данных ИИ-анализа V3 ('product_analysis_v3') ---
        if ANALYSIS_FIELD_V3 in df_processed.columns:
            flattened_analysis_v3_list = df_processed[ANALYSIS_FIELD_V3].apply(
                lambda x: flatten_analysis(x) if pd.notna(x) and isinstance(x, dict) else {}
            )
            if not flattened_analysis_v3_list.empty:
                analysis_v3_df = pd.DataFrame(flattened_analysis_v3_list.tolist(), index=df_processed.index)
                cols_to_drop_v3 = [col for col in analysis_v3_df.columns if col in df_processed.columns];
                analysis_v3_df_filtered = analysis_v3_df.drop(columns=cols_to_drop_v3, errors='ignore')
                df_processed = pd.concat([df_processed.reset_index(drop=True), analysis_v3_df_filtered.reset_index(drop=True)], axis=1)
                st.sidebar.info("Данные анализа V3 обработаны.")
        else: st.sidebar.warning(f"Столбец V3 '{ANALYSIS_FIELD_V3}' не найден.")

        # --- Разворачивание данных ИИ-анализа V4 ('enrichment_v4') ---
        if ENRICHMENT_V4_KEY in df_processed.columns:
            flattened_analysis_v4_list = df_processed[ENRICHMENT_V4_KEY].apply(
                lambda x: flatten_v4_enrichment(x) if pd.notna(x) and isinstance(x, dict) else {}
            )
            if not flattened_analysis_v4_list.empty:
                analysis_v4_df = pd.DataFrame(flattened_analysis_v4_list.tolist(), index=df_processed.index)
                # Проверяем на дубликаты с УЖЕ СУЩЕСТВУЮЩИМИ колонками (включая V3)
                cols_to_drop_v4 = [col for col in analysis_v4_df.columns if col in df_processed.columns];
                analysis_v4_df_filtered = analysis_v4_df.drop(columns=cols_to_drop_v4, errors='ignore')
                df_processed = pd.concat([df_processed.reset_index(drop=True), analysis_v4_df_filtered.reset_index(drop=True)], axis=1)
                st.sidebar.info("Данные анализа V4 (порции) обработаны.")
            # Гарантируем наличие колонки с порциями даже если ключ V4 был, но пустой
            if V4_PORTIONS_COL not in df_processed.columns:
                 df_processed[V4_PORTIONS_COL] = 1
        else:
            st.sidebar.warning(f"Ключ V4 '{ENRICHMENT_V4_KEY}' не найден. Деление на порции будет недоступно.")
            # Гарантируем наличие колонки с порциями (значение по умолчанию 1)
            if V4_PORTIONS_COL not in df_processed.columns:
                 df_processed[V4_PORTIONS_COL] = 1

        # --- Фильтрация продуктов без рассчитанного КБЖУ ---
        if CALCULABLE_COLUMN in df_processed.columns and df_processed[CALCULABLE_COLUMN].any():
             df_calculable = df_processed[df_processed[CALCULABLE_COLUMN] == True].copy()
             df_calculable = df_calculable.drop(columns=[CALCULABLE_COLUMN], errors='ignore')
        else: st.warning("Не найдено продуктов с успешно рассчитанным КБЖУ."); df_calculable = pd.DataFrame()
        if df_calculable.empty: st.error("Нет продуктов с валидным КБЖУ для работы конструктора."); return None

        df_final = df_calculable.reset_index(drop=True) # Сброс индекса для чистоты

        # --- Обработка пропусков и типов в колонках анализа V3 и V4 ---
        st.write("Финальная обработка типов и пропусков V3/V4...")
        # Объединяем схемы для общей обработки
        combined_schema = {**EXPECTED_ANALYSIS_SCHEMA, **EXPECTED_V4_SCHEMA}

        for col, spec in combined_schema.items():
             default_val = spec['default']
             final_dtype = spec['dtype']

             # 1. Создаем колонку с дефолтом, если ее нет
             if col not in df_final.columns:
                 df_final[col] = default_val

             # 2. Заполняем пропуски (NaN/None) дефолтным значением (кроме случаев, когда дефолт = None)
             # Особая обработка для списков
             if isinstance(default_val, list): # Проверяем тип дефолта
                 # Применяем функцию только к тем значениям, которые не являются списком
                 df_final[col] = df_final[col].apply(lambda x: default_val if not isinstance(x, list) else x)
             elif default_val is not None:
                 df_final[col] = df_final[col].fillna(default_val)
             # Если default_val is None, fillna не используется

             # 3. Приводим к целевому типу
             try:
                 current_dtype_str = str(df_final[col].dtype)
                 target_dtype_str = str(final_dtype) # Преобразуем pandas dtype в строку для сравнения

                 # Пропускаем object, если тип уже object или категория (они могут быть смешанными)
                 if target_dtype_str == 'object' or target_dtype_str == 'category':
                     continue

                 # Проверяем, нужно ли вообще приведение типа
                 if current_dtype_str != target_dtype_str:
                      # Целочисленный тип с поддержкой NA
                      if isinstance(final_dtype, pd.Int64Dtype):
                          numeric_col = pd.to_numeric(df_final[col], errors='coerce').fillna(default_val if default_val is not None else 0) # Заполняем NA перед округлением
                          df_final[col] = numeric_col.round().astype(final_dtype)
                      # Булев тип
                      elif final_dtype == bool:
                          bool_map = {True: True, False: False, 1: True, 0: False, 'True': True, 'False': False, 'true': True, 'false': False}
                          # Заполняем NA/None дефолтом *после* map, если что-то не распозналось
                          df_final[col] = df_final[col].map(bool_map).fillna(default_val).astype(bool)
                      # Остальные типы (float, str)
                      else:
                           # Если целевой тип строка и дефолт строка, заполним NA перед конвертацией
                           if final_dtype == str and isinstance(default_val, str):
                                df_final[col] = df_final[col].fillna(default_val)
                           # Конвертируем, если тип не object
                           df_final[col] = df_final[col].astype(final_dtype)

             except Exception as e_cast_final:
                 # Выводим более подробное предупреждение
                 st.warning(f"Не удалось привести колонку '{col}' к типу {final_dtype}. Текущий тип: {df_final[col].dtype}. Ошибка: {e_cast_final}")

        st.success("Предобработка данных V3/V4 завершена!")
        return df_final

# =============================================
# --- Алгоритмы Подбора и Распределения ---
# =============================================

# Функция calculate_deviation (без изменений)
def calculate_deviation(totals: Dict[str, float], goals: Dict[str, float]) -> float:
    """Рассчитывает среднеквадратичное относительное отклонение итогов от целей КБЖУ."""
    deviation: float = 0.0; num_metrics = 0
    for key in NUTRIENT_KEYS:
        goal_value: float = goals.get(key, 0.0); total_value: float = totals.get(key, 0.0)
        if goal_value > 1e-9: deviation += ((total_value - goal_value) / goal_value) ** 2; num_metrics += 1
        elif total_value > 1e-9: norm: float = 2000.0 if key == 'kcal' else 100.0; deviation += (total_value / norm) ** 2 * 2.0; num_metrics += 1
    return (deviation / num_metrics) ** 0.5 if num_metrics > 0 else 0.0

# Функция suggest_daily_set (без изменений)
def suggest_daily_set( df_available: pd.DataFrame, goals: Dict[str, float], primary_goal: Literal['Weight Loss', 'Muscle Gain', 'General Health', 'None'], params: Dict[str, Any], scoring_weights: Dict[str, float]) -> Tuple[Optional[List[Dict]], Dict[str, float]]:
    """Предлагает набор продуктов на ДЕНЬ (без изменений)."""
    min_items: int = params['min_items']; max_items: int = params['max_items']; tolerance: float = params['tolerance']; max_attempts: int = params['max_attempts']; candidate_limit: int = params['candidate_limit']
    best_plan_indices: List[int] = []; min_deviation: float = float('inf'); best_plan_totals: Dict[str, float] = {}
    if df_available is None or df_available.empty: st.warning("Нет доступных продуктов."); return None, {}
    required_total_cols = list(TOTAL_NUTRITION_COLUMNS.values())
    if not all(col in df_available.columns for col in required_total_cols): st.error(f"Отсутствуют колонки КБЖУ: {required_total_cols}."); return None, {}
    available_indices: List[int] = df_available.index.tolist();
    if not available_indices: st.warning("Индекс продуктов пуст."); return None, {}
    st.write(f"Генерация набора на день (Цель ИИ: {primary_goal})...")
    diet_rating_col: Optional[str] = None;
    if primary_goal != 'None': diet_rating_col_map = {'Weight Loss': f'{ANALYSIS_PREFIX}diet_weight_loss_rating', 'Muscle Gain': f'{ANALYSIS_PREFIX}diet_muscle_gain_rating', 'General Health': f'{ANALYSIS_PREFIX}diet_general_health_rating'}; diet_rating_col = diet_rating_col_map.get(primary_goal);
    sugar_col: str = f'{ANALYSIS_PREFIX}likely_contains_added_sugar'; grains_col: str = f'{ANALYSIS_PREFIX}likely_contains_whole_grains'; fiber_col: str = f'{ANALYSIS_PREFIX}fiber_level_estimate'; satiety_col: str = f'{ANALYSIS_PREFIX}satiety_index_estimate'; density_col: str = f'{ANALYSIS_PREFIX}nutrient_density_estimate';
    analysis_cols_to_check: List[str] = [col for col in [diet_rating_col, sugar_col, grains_col, fiber_col, satiety_col, density_col] if col]; analysis_cols_present: bool = all(c in df_available.columns for c in analysis_cols_to_check);
    for attempt in range(max_attempts):
        current_plan_indices: List[int] = []; current_totals: Dict[str, float] = {k: 0.0 for k in goals}; candidate_indices: List[int] = available_indices[:]; random.shuffle(candidate_indices); items_added_count: int = 0
        while items_added_count < max_items:
            remaining_goals: Dict[str, float] = {k: goals[k] - current_totals[k] for k in goals}; goals_met: bool = True
            for k in goals:
                 goal_val = goals[k]; total_val = current_totals[k];
                 if goal_val > 1e-9:
                      if not (abs(total_val - goal_val) <= goal_val * tolerance): goals_met = False; break
            if goals_met and items_added_count >= min_items: break
            best_candidate_idx: Optional[int] = None; best_candidate_score: float = -float('inf'); needed_nutrient: Optional[str] = None; max_deficit_ratio: float = -1.0
            for k in goals:
                 goal_val = goals[k];
                 if goal_val > 1e-9: deficit = remaining_goals.get(k, 0.0);
                 if deficit > 1e-9: deficit_ratio = deficit / goal_val;
                 if deficit_ratio > max_deficit_ratio: max_deficit_ratio = deficit_ratio; needed_nutrient = k
            if needed_nutrient is None and goals.get('kcal', 0) > 1e-9 and remaining_goals.get('kcal', 0) > 1e-9: needed_nutrient = 'kcal'
            indices_to_evaluate: List[int] = candidate_indices[:min(candidate_limit, len(candidate_indices))]
            if not indices_to_evaluate: break
            for idx in indices_to_evaluate:
                 if idx in current_plan_indices: continue
                 try:
                     candidate_item: pd.Series = df_available.loc[idx]
                     candidate_nutrients: Dict[str, float] = {k: float(candidate_item.get(TOTAL_NUTRITION_COLUMNS[k], 0.0) or 0.0) for k in goals}
                 except (KeyError, ValueError, TypeError): continue
                 score: float = 0.0; projected_kcal: float = current_totals['kcal'] + candidate_nutrients['kcal']; calorie_limit: float = goals['kcal'] * (1 + tolerance + 0.05)
                 if projected_kcal > calorie_limit: continue
                 goal_kcal = goals.get('kcal', 0.0);
                 if goal_kcal > 1e-9: calorie_ratio = candidate_nutrients['kcal'] / goal_kcal;
                 if calorie_ratio > 0.5: score -= (calorie_ratio - 0.5) * scoring_weights['high_calorie_ratio_penalty_factor']
                 nutrient_gain: float = 0.0;
                 if needed_nutrient and candidate_nutrients.get(needed_nutrient, 0.0) > 1e-9: deficit = remaining_goals.get(needed_nutrient, 0.0);
                 if deficit > 1e-9: gain_ratio = min(candidate_nutrients[needed_nutrient] / deficit, 1.0); nutrient_gain = gain_ratio * scoring_weights['deficit_reduction_base']
                 elif remaining_goals.get('kcal', 0.0) > 1e-9 and goal_kcal > 1e-9: nutrient_gain = (candidate_nutrients.get('kcal', 0.0) / goal_kcal) * scoring_weights['calorie_bonus_if_needed']
                 score += nutrient_gain; over_limit_penalty: float = 0.0
                 for k in goals.keys():
                      if k != needed_nutrient: goal_val = goals.get(k, 0.0);
                      if goal_val > 1e-9: projected_total: float = current_totals[k] + candidate_nutrients[k]; limit: float = goal_val * (1 + tolerance);
                      if projected_total > limit: over_limit_penalty += ((projected_total - limit) / goal_val) * scoring_weights['over_limit_penalty_factor']
                 score -= over_limit_penalty
                # Внутри функции suggest_daily_set:
                 # 5. Учет данных ИИ-анализа (если доступны)
                 if analysis_cols_present:
                     try:
                         if diet_rating_col:
                             rating_value = candidate_item.get(diet_rating_col, 1) # Получаем значение, дефолт 1
                             # Безопасно конвертируем в число
                             numeric_rating = pd.to_numeric(rating_value, errors='coerce')
                             # Проверяем на NaN/None и конвертируем в int
                             if pd.isna(numeric_rating):
                                 rating = 1 # Дефолт, если не число или NA
                             else:
                                 try:
                                     rating = int(numeric_rating)
                                 except (ValueError, TypeError):
                                     rating = 1 # Дефолт при ошибке конвертации в int
                             # Ограничиваем диапазон 1-5
                             rating = max(1, min(5, rating))
                             # Добавляем к оценке
                             score += (rating - 3) * scoring_weights['diet_rating_bonus_factor']
                         # Штраф за добавленный сахар
                         if candidate_item.get(sugar_col, True): # Считаем True по умолчанию
                              score -= scoring_weights['added_sugar_penalty']

                         # Бонус за цельные злаки
                         if candidate_item.get(grains_col, False): # Считаем False по умолчанию
                              score += scoring_weights['whole_grains_bonus']

                         # Бонус/штраф за оценки (Клетчатка, Сытость, Плотность)
                         # Функция get_level_score остается без изменений
                         def get_level_score(level: Optional[str]) -> float:
                              level_map = {"High": scoring_weights['level_score_high'], "Medium": scoring_weights['level_score_medium'], "Low": scoring_weights['level_score_low'], "Uncertain": 0.0, None: 0.0}
                              # Преобразуем level в строку на всякий случай перед поиском в словаре
                              return level_map.get(str(level) if pd.notna(level) else None, 0.0)

                         score += get_level_score(candidate_item.get(fiber_col)) * scoring_weights['fiber_factor']
                         score += get_level_score(candidate_item.get(satiety_col)) * scoring_weights['satiety_factor']
                         score += get_level_score(candidate_item.get(density_col)) * scoring_weights['density_factor']

                     except (ValueError, TypeError, KeyError) as e_ai_score:
                          # Ловим возможные ошибки при доступе к данным или конвертации
                          # st.warning(f"Ошибка скоринга AI для idx {idx}: {e_ai_score}") # Можно раскомментировать для отладки
                          pass # Игнорируем ошибку скоринга ИИ для этого кандидата, чтобы не прерывать процесс
                 if score > best_candidate_score: best_candidate_score = score; best_candidate_idx = idx
            if best_candidate_idx is not None:
                 try:
                     selected_item = df_available.loc[best_candidate_idx]; current_plan_indices.append(best_candidate_idx)
                     for k in goals: current_totals[k] += float(selected_item.get(TOTAL_NUTRITION_COLUMNS[k], 0.0) or 0.0)
                     items_added_count += 1; candidate_indices.remove(best_candidate_idx)
                 except (KeyError, ValueError, TypeError): break
            else: break
        if current_plan_indices and items_added_count >= min_items :
             final_deviation = calculate_deviation(current_totals, goals)
             if final_deviation < min_deviation: min_deviation = final_deviation; best_plan_indices = current_plan_indices[:]; best_plan_totals = current_totals.copy()
    if best_plan_indices:
        st.success(f"Набор на день подобран (отклонение: {min_deviation:.3f})")
        # Конвертируем DataFrame в список словарей для дальнейшей обработки
        best_plan_df = df_available.loc[best_plan_indices]
        # Используем where + to_dict для корректной обработки NaN -> None
        best_plan_records = best_plan_df.where(pd.notna(best_plan_df), None).to_dict('records')
        return best_plan_records, best_plan_totals
    else: st.warning("Не удалось подобрать набор продуктов на день."); return None, {}



def calculate_meal_targets(daily_goals: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Рассчитывает целевые КБЖУ для каждого приема пищи."""
    meal_targets: Dict[str, Dict[str, float]] = {meal: {} for meal in MEAL_SLOTS_ORDER}
    for meal, percentage in MEAL_TARGET_PERCENTAGES.items():
        for key in NUTRIENT_KEYS:
            meal_targets[meal][key] = daily_goals.get(key, 0.0) * percentage
    return meal_targets

def calculate_meal_deviation(current_meal_totals: Dict[str, float], meal_targets: Dict[str, float]) -> float:
    """Рассчитывает отклонение текущих итогов приема пищи от его целей."""
    return calculate_deviation(current_meal_totals, meal_targets)

# --- ВАЖНО: Функция скоринга остается V2.3 ---
def score_item_for_meal( item: Dict[str, Any], meal_name: str, meal_target: Dict[str, float], current_meal_totals: Dict[str, float]) -> float:
    """Оценивает, насколько хорошо продукт подходит для данного приема пищи (Версия 2.3)."""
    score: float = 0.0; W_RATING = 0.8; W_DEVIATION = 0.7; W_ROLE = 0.4; W_EXCESS_PENALTY = 2.5
    rating_col = MEAL_RATING_COLS_MAP_RU.get(meal_name); ai_rating = 1
    if rating_col and rating_col in item:
        rating_value = item.get(rating_col); numeric_rating = pd.to_numeric(rating_value, errors='coerce')
        if pd.isna(numeric_rating): ai_rating = 1
        else:
            try: ai_rating = int(numeric_rating)
            except (ValueError, TypeError): ai_rating = 1
    ai_rating = max(1, min(5, ai_rating)); normalized_rating = (ai_rating - 1) / 4.0; score += normalized_rating * W_RATING
    item_nutrients = {k: float(item.get(TOTAL_NUTRITION_COLUMNS[k], 0.0) or 0.0) for k in NUTRIENT_KEYS}
    deviation_before = calculate_meal_deviation(current_meal_totals, meal_target)
    projected_totals = {k: current_meal_totals.get(k, 0.0) + item_nutrients.get(k, 0.0) for k in NUTRIENT_KEYS}
    deviation_after = calculate_meal_deviation(projected_totals, meal_target); improvement = deviation_before - deviation_after
    if improvement > 0: score += improvement * W_DEVIATION
    elif improvement < -0.01: score += improvement * W_DEVIATION * 0.5
    item_role = item.get(f'{ANALYSIS_PREFIX}meal_component_role', 'Uncertain')
    if meal_name == "Завтрак" and item_role in UNLIKELY_BREAKFAST_ROLES: score -= W_ROLE
    elif meal_name == "Перекус" and item_role in SNACK_ROLES: score += W_ROLE * 0.5
    target_kcal = meal_target.get('kcal', 0.0); projected_kcal = projected_totals.get('kcal', 0.0); kcal_excess_threshold_perc = 1.15; kcal_excess_threshold_abs = 100
    if target_kcal > 1e-6 and (projected_kcal > target_kcal * kcal_excess_threshold_perc or projected_kcal > target_kcal + kcal_excess_threshold_abs):
        absolute_excess = max(0, projected_kcal - target_kcal); excess_penalty = (absolute_excess / target_kcal) * W_EXCESS_PENALTY if target_kcal > 1 else (absolute_excess / 100) * W_EXCESS_PENALTY
        score -= excess_penalty
        if projected_kcal > target_kcal * 1.4: score -= W_EXCESS_PENALTY
    return score

def distribute_meals_smartly(
    items_list: List[Dict[str, Any]],
    daily_goals: Dict[str, float]
) -> Dict[str, List[Dict]]:
    """
    Распределяет продукты по приемам пищи (Версия 3.1 - Приоритет деления V4 порций).
    """
    st.write("Распределение продуктов V3.1 (Приоритет деления V4 порций)...") # Обновили версию
    meal_slots: Dict[str, List[Dict]] = {slot_name: [] for slot_name in MEAL_SLOTS_ORDER}
    meal_totals: Dict[str, Dict[str, float]] = {slot: {k: 0.0 for k in NUTRIENT_KEYS} for slot in MEAL_SLOTS_ORDER}
    remaining_items: List[Dict] = copy.deepcopy(items_list)
    random.shuffle(remaining_items)

    if not remaining_items: return meal_slots
    meal_targets = calculate_meal_targets(daily_goals)

    items_processed_in_loop = 0
    max_loops = len(remaining_items) * 3

    while remaining_items and items_processed_in_loop < max_loops:
        items_processed_in_loop += 1
        item_to_place = remaining_items.pop(0)

        item_to_place['portion_factor'] = item_to_place.get('portion_factor', 1.0)
        item_to_place['portion_desc'] = item_to_place.get('portion_desc', "")

        best_slot_for_item: Optional[str] = None
        best_score_for_item: float = -float('inf')
        best_option_is_portion: bool = False

        original_portions = int(item_to_place.get(V4_PORTIONS_COL, 1) or 1)
        # Делим только если исходно > 1 порции и текущий кусок = целый
        is_splittable_item = original_portions > 1 and item_to_place['portion_factor'] == 1.0

        scores_per_slot = {}
        scores_per_slot_half = {}
        item_half_cache = {} # Кэш для рассчитанной половины

        # --- Оцениваем варианты для каждого слота ---
        for meal_name in MEAL_SLOTS_ORDER:
            # --- Оценка ЦЕЛОГО ---
            score_whole = score_item_for_meal(item_to_place, meal_name, meal_targets[meal_name], meal_totals[meal_name])
            # Применяем штрафы/запреты к score_whole
            # ... (код штрафов как в V2.3/V3.0) ...
            rating_col = MEAL_RATING_COLS_MAP_RU.get(meal_name); ai_rating = 1
            if rating_col and rating_col in item_to_place:
                 try: ai_rating = int(pd.to_numeric(item_to_place.get(rating_col), errors='coerce').fillna(1))
                 except: pass
            ai_rating = max(1, min(5, ai_rating))
            min_rating_threshold = MIN_RATING_SNACK if meal_name == "Перекус" else MIN_RATING_MEAL
            if ai_rating < min_rating_threshold: score_whole -= 1.0
            item_role = item_to_place.get(f'{ANALYSIS_PREFIX}meal_component_role', 'Uncertain')
            if meal_name == "Завтрак" and item_role in UNLIKELY_BREAKFAST_ROLES and ai_rating <= 1: score_whole = -float('inf')


            scores_per_slot[meal_name] = score_whole
            current_best_score = score_whole
            current_best_slot = meal_name
            current_best_is_portion = False

            # --- Оценка ПОЛОВИНЫ (если применимо) ---
            if is_splittable_item:
                # Рассчитываем половину (если еще не рассчитали для этого item_to_place)
                if not item_half_cache:
                    item_half_temp = item_to_place.copy(); item_half_temp['portion_factor'] = 0.5; item_half_temp['portion_desc'] = "1/2"; valid_half_nutrition = True
                    for k, total_col in TOTAL_NUTRITION_COLUMNS.items():
                        if total_col in item_half_temp and pd.notna(item_half_temp[total_col]):
                            try: item_half_temp[total_col] = float(item_half_temp[total_col]) / 2.0
                            except Exception: valid_half_nutrition = False; break
                        else: item_half_temp[total_col] = pd.NA;
                        if k == 'kcal' and pd.isna(item_half_temp[total_col]): valid_half_nutrition = False; break
                    item_half_cache = item_half_temp if valid_half_nutrition else None

                # Если половина рассчитана корректно
                if item_half_cache:
                    item_half = item_half_cache
                    score_half = score_item_for_meal(item_half, meal_name, meal_targets[meal_name], meal_totals[meal_name])
                    # Применяем штрафы/запреты к score_half
                    # ... (тот же код штрафов) ...
                    rating_col_h = MEAL_RATING_COLS_MAP_RU.get(meal_name); ai_rating_h = 1
                    if rating_col_h and rating_col_h in item_half:
                        try: ai_rating_h = int(pd.to_numeric(item_half.get(rating_col_h), errors='coerce').fillna(1))
                        except: pass
                    ai_rating_h = max(1, min(5, ai_rating_h))
                    min_rating_threshold_h = MIN_RATING_SNACK if meal_name == "Перекус" else MIN_RATING_MEAL
                    if ai_rating_h < min_rating_threshold_h: score_half -= 1.0
                    item_role_h = item_half.get(f'{ANALYSIS_PREFIX}meal_component_role', 'Uncertain')
                    if meal_name == "Завтрак" and item_role_h in UNLIKELY_BREAKFAST_ROLES and ai_rating_h <= 1: score_half = -float('inf')

                    scores_per_slot_half[meal_name] = score_half

                    split_bonus = MULTI_PORTION_SPLIT_BONUS

                    if score_half + split_bonus > current_best_score:
                        current_best_score = score_half + split_bonus
                        current_best_slot = meal_name
                        current_best_is_portion = True # Предпочитаем порцию из-за бонуса

            # Обновляем общий лучший результат для ДАННОГО продукта по ВСЕМ слотам
            if current_best_score > best_score_for_item:
                best_score_for_item = current_best_score
                best_slot_for_item = current_best_slot
                best_option_is_portion = current_best_is_portion
        # --- Конец цикла по слотам ---

        # --- Помещение продукта (целого или порции) ---
        # Логика помещения остается такой же, как в V3.0: проверяем флаг best_option_is_portion
        if best_slot_for_item:
            target_slot = best_slot_for_item; item_to_actually_place = None
            if best_option_is_portion and item_half_cache: # Если решили делить и половина валидна
                item_half_final = item_half_cache # Используем кэшированную половину
                item_to_actually_place = item_half_final
                # Создаем вторую половину и возвращаем в пул
                other_half = item_half_final.copy()
                remaining_items.insert(0, other_half) # Вставляем в начало для скорейшей обработки
            else: # Если решили не делить (или не могли)
                item_to_actually_place = item_to_place # Помещаем то, что было (целое или исходную порцию)
            # Добавляем выбранный объект в слот и обновляем итоги
            if item_to_actually_place:
                 meal_slots[target_slot].append(item_to_actually_place)
                 item_nutrients = {k: float(item_to_actually_place.get(TOTAL_NUTRITION_COLUMNS[k], 0.0) or 0.0) for k in NUTRIENT_KEYS}
                 for k in NUTRIENT_KEYS: meal_totals[target_slot][k] += item_nutrients[k]
        else: # Fallback
             target_slot = "Перекус"; item_to_place_fallback = item_to_place
             score_snack = scores_per_slot.get(target_slot, -float('inf'))
             if score_snack > -float('inf'): # Добавляем, только если не было строго запрещено
                 meal_slots[target_slot].append(item_to_place_fallback)
                 item_nutrients = {k: float(item_to_place_fallback.get(TOTAL_NUTRITION_COLUMNS[k], 0.0) or 0.0) for k in NUTRIENT_KEYS}
                 for k in NUTRIENT_KEYS: meal_totals[target_slot][k] += item_nutrients[k]
             else: st.warning(f"Не удалось найти слот для '{item_to_place.get(NAME_COL, '?')}' (score={best_score_for_item:.2f})")


    # --- Обработка оставшихся и Балансировка ---
    # (Этот код остается таким же, как в V3.0 / V2.2) ...
    if remaining_items: st.error(f"Осталось {len(remaining_items)} нераспределенных продуктов!")
    # --- Этап 2: Балансировка ---
    st.write("Балансировка для заполнения пустых Завтрак/Обед/Ужин...")
    # ... (Код балансировки) ...
    essential_slots: List[str] = ["Завтрак", "Обед", "Ужин"]; max_balancing_moves: int = len(items_list) + 5
    for _ in range(max_balancing_moves):
        empty_essential_slots = [slot for slot in essential_slots if not meal_slots.get(slot)]; donor_slots = {slot: items for slot, items in meal_slots.items() if len(items) > (1 if slot not in essential_slots else 0)}
        if not empty_essential_slots or not donor_slots: break
        source_slot_name = max(donor_slots, key=lambda s: len(donor_slots[s])); source_items = meal_slots[source_slot_name]; target_slot_name = empty_essential_slots[0]
        best_item_index: Optional[int] = None; best_target_rating: int = -1; worst_source_item_index: Optional[int] = None; worst_source_rating: int = 6
        target_rating_col = MEAL_RATING_COLS_MAP_RU.get(target_slot_name); source_rating_col = MEAL_RATING_COLS_MAP_RU.get(source_slot_name)
        for i, item_candidate in enumerate(source_items):
            current_target_rating = 1;
            if target_rating_col and target_rating_col in item_candidate:
                try: current_target_rating = int(pd.to_numeric(item_candidate.get(target_rating_col), errors='coerce').fillna(1))
                except: pass
            current_target_rating = max(1, min(5, current_target_rating));
            if current_target_rating > best_target_rating: best_target_rating = current_target_rating; best_item_index = i
            current_source_rating = 5;
            if source_rating_col and source_rating_col in item_candidate:
                try: current_source_rating = int(pd.to_numeric(item_candidate.get(source_rating_col), errors='coerce').fillna(5))
                except: pass
            current_source_rating = max(1, min(5, current_source_rating));
            if current_source_rating < worst_source_rating: worst_source_rating = current_source_rating; worst_source_item_index = i
        item_to_move_index: Optional[int] = None
        can_move_best_for_target = best_item_index is not None and best_target_rating > 1; can_move_worst_from_source = worst_source_item_index is not None
        if can_move_best_for_target: item_to_move_index = best_item_index
        elif can_move_worst_from_source: item_to_move_index = worst_source_item_index
        elif source_items: item_to_move_index = 0
        if item_to_move_index is not None and item_to_move_index < len(source_items):
            try: item_moved = meal_slots[source_slot_name].pop(item_to_move_index); meal_slots[target_slot_name].append(item_moved)
            except IndexError: pass
        else: break
    final_empty_slots = [slot for slot in essential_slots if not meal_slots.get(slot)];
    if final_empty_slots: st.warning(f"Не удалось заполнить основные слоты ({', '.join(final_empty_slots)}) после балансировки.")


    st.success("Распределение продуктов (V3.1 с приоритетом деления) завершено.")
    return meal_slots


# =============================================
# --- Вспомогательные Функции для UI ---
# =============================================

def generate_widget_key(base: str, *args: Any) -> str:
    """Генерирует уникальный ключ для виджета Streamlit."""
    key_parts = [str(arg).replace(" ", "_").replace("/", "_").replace(":", "_") for arg in args if arg is not None]
    # Ограничиваем длину ключа, чтобы избежать проблем Streamlit
    full_key = f"{base}_{'_'.join(key_parts)}"
    return full_key[:250] # Streamlit имеет ограничение на длину ключа

# --- ОБНОВЛЕНО: Отображение деталей продукта V3 + V4 ---
def display_product_details(product: Dict[str, Any], current_goal_key: Optional[str], goal_name: str):
    """Отображает карточку продукта с деталями V3 и V4 и кнопками добавления."""
    st.markdown(f"#### {product.get(NAME_COL, 'N/A')}")
    col_img, col_info = st.columns([1, 3])

    with col_img:
        img_url = product.get(IMAGE_COL);
        if pd.notna(img_url) and isinstance(img_url, str) and img_url.startswith('http'): st.image(img_url, width=150)
        else: st.caption("Нет изображения")

    with col_info:
        # --- Блок КБЖУ ---
        kcal = product.get(TOTAL_NUTRITION_COLUMNS['kcal'], 0); prot = product.get(TOTAL_NUTRITION_COLUMNS['protein'], 0); fat = product.get(TOTAL_NUTRITION_COLUMNS['fat'], 0); carb = product.get(TOTAL_NUTRITION_COLUMNS['carbs'], 0)
        st.metric("КБЖУ (продукт)", f"{kcal:.0f} ккал", f"{prot:.1f}б / {fat:.1f}ж / {carb:.1f}у")

        # --- Блок Анализа V3 ---
        role = product.get(f'{ANALYSIS_PREFIX}meal_component_role', '?'); flavor = product.get(f'{ANALYSIS_PREFIX}primary_flavor_profile', '?'); texture = product.get(f'{ANALYSIS_PREFIX}primary_texture', '?');
        st.write(f"**Роль V3:** {role} | **Вкус:** {flavor} | **Текстура:** {texture}")
        if current_goal_key and current_goal_key in product: goal_rating = product.get(current_goal_key, '?'); st.write(f"**Рейтинг '{goal_name}':** {goal_rating}/5")
        sugar_flag = product.get(f'{ANALYSIS_PREFIX}likely_contains_added_sugar', None); grains_flag = product.get(f'{ANALYSIS_PREFIX}likely_contains_whole_grains', None); sugar_text = "⚠️ Добавл." if sugar_flag else ("✅ Без добавл." if sugar_flag is False else "?"); grains_text = "✅ Цельн." if grains_flag else ("❌ Без цельн." if grains_flag is False else "?"); st.markdown(f"**Сахар / Злаки:** {sugar_text} / {grains_text}")
        fib = product.get(f'{ANALYSIS_PREFIX}fiber_level_estimate', '?'); sod = product.get(f'{ANALYSIS_PREFIX}sodium_level_estimate', '?'); sat = product.get(f'{ANALYSIS_PREFIX}satiety_index_estimate', '?'); dens = product.get(f'{ANALYSIS_PREFIX}nutrient_density_estimate', '?'); st.markdown(f"**Оценки V3 (Клетч/Натр/Сыт/Плотн):** {fib} / {sod} / {sat} / {dens}")

        # --- Блок Анализа V4 ---
        st.markdown("---") # Разделитель
        portions_val = product.get(V4_PORTIONS_COL) # Может быть <NA> или число
        portions = int(portions_val) if pd.notna(portions_val) else 1 # Дефолт 1 если NA
        portion_reason = product.get(f'{V4_PREFIX}portion_portion_reasoning')
        dominant_macro = product.get(f'{V4_PREFIX}context_dominant_macro', '?')
        temp = product.get(f'{V4_PREFIX}context_consumption_temperature', '?')
        is_dessert_val = product.get(f'{V4_PREFIX}context_is_dessert', None) # bool или None
        is_dessert_text = 'Да' if is_dessert_val else ('Нет' if is_dessert_val is False else '?')

        portion_text = f"{portions} порц." if portions > 1 else "1 порц."
        if pd.notna(portion_reason) and portion_reason: portion_text += f" ({portion_reason})" # Добавляем причину, если она есть
        st.write(f"**Делимость V4:** {portion_text}")
        st.write(f"**Дом. макро:** {dominant_macro} | **Темп.:** {temp} | **Десерт:** {is_dessert_text}")
        st.markdown("---") # Разделитель

        # --- Кнопки добавления ---
        st.write("**Добавить ЦЕЛЫЙ продукт в:**") # Подчеркиваем, что добавляется целый
        add_cols = st.columns(len(st.session_state.meal_plan))
        # ID для ключа должен быть стабильным для продукта
        item_identifier = product.get(URL_COL, product.get(NAME_COL, random.random()))
        for i, meal_name in enumerate(st.session_state.meal_plan.keys()):
            add_key = generate_widget_key("m_add", meal_name, item_identifier)
            if add_cols[i].button(f"➕ {meal_name}", key=add_key, help=f"Добавить ЦЕЛЫЙ '{product.get(NAME_COL, 'N/A')}' в {meal_name}"):
                # Добавляем КОПИЮ ЦЕЛОГО продукта при ручном добавлении
                st.session_state.meal_plan[meal_name].append(product.copy())
                st.success(f"'{product.get(NAME_COL, 'N/A')}' добавлен в {meal_name}!")
                # st.rerun() # Можно убрать, чтобы позволить добавить в несколько слотов

    # --- Expander с доп. информацией V3 ---
    with st.expander("Больше информации (Анализ V3)"):
        # ... (Отображение остальных полей V3 без изменений) ...
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Пригодность V3 к приемам пищи (рейтинги):**"); suitability = {"Завтрак": product.get(MEAL_RATING_COLS_MAP_RU["Завтрак"], '?'), "Обед": product.get(MEAL_RATING_COLS_MAP_RU["Обед"], '?'), "Ужин": product.get(MEAL_RATING_COLS_MAP_RU["Ужин"], '?'), "Перекус": product.get(MEAL_RATING_COLS_MAP_RU["Перекус"], '?')}; st.json(suitability, expanded=False)
            reasoning_suitability = product.get(f'{ANALYSIS_PREFIX}meal_suitability_reasoning');
            if reasoning_suitability and reasoning_suitability != "N/A": st.caption(f"Обоснование V3: {reasoning_suitability}")
            st.write("**Пригодность V3 к диет. целям (рейтинги):**"); diet_goals_ratings = {"Похудение": product.get(f'{ANALYSIS_PREFIX}diet_weight_loss_rating', '?'), "Набор массы": product.get(f'{ANALYSIS_PREFIX}diet_muscle_gain_rating', '?'), "Здоровье": product.get(f'{ANALYSIS_PREFIX}diet_general_health_rating', '?'), "Низкокал.перекус": product.get(f'{ANALYSIS_PREFIX}diet_low_calorie_snack_rating', '?')}; st.json(diet_goals_ratings, expanded=False)
            reasoning_diet = product.get(f'{ANALYSIS_PREFIX}diet_goals_reasoning');
            if reasoning_diet and reasoning_diet != "N/A": st.caption(f"Обоснование V3: {reasoning_diet}")
            st.write(f"**Сложность V3:** {product.get(f'{ANALYSIS_PREFIX}preparation_complexity', '?')}")
            st.write(f"**Метод V3 (предп.):** {product.get(f'{ANALYSIS_PREFIX}cooking_method_guess', 'N/A') or 'N/A'}")
            pairing = product.get(f'{ANALYSIS_PREFIX}pairing_suggestion'); st.write(f"**Сочетание V3:** {pairing or 'N/A'}")
        with c2:
            tags = product.get(f'{ANALYSIS_PREFIX}health_benefit_tags', []); st.write(f"**Теги пользы V3:** {', '.join(tags) if tags else 'Нет'}")
            st.write("**Микронутриенты V3 (потенц. источник):**"); micro_texts = [];
            for symbol, col_name in MICRONUTRIENT_MAP.items(): is_source = product.get(col_name, False); micro_texts.append(f"{symbol}: {'✅' if is_source else '❌'}")
            st.write(" / ".join(micro_texts))
            micro_comment = product.get(f'{ANALYSIS_PREFIX}micronutrient_comment');
            if micro_comment and micro_comment != "N/A": st.write(f"*Комментарий V3 (микро):* {micro_comment}")

# --- ОБНОВЛЕНО: Отображение слота с учетом порций ---
def display_meal_slot(meal_name: str, items: List[Dict], goals: Dict[str, float], is_manual: bool = True):
    """Отображает содержимое одного приема пищи (с учетом порций)."""
    st.markdown(f"##### {meal_name}")
    if not items: st.caption("_Пусто_"); return

    indices_to_remove: List[int] = []; used_keys_in_slot: Set[str] = set()

    for item_index, item in enumerate(items):
        if not isinstance(item, dict): continue # Пропускаем не-словари

        item_name = item.get(NAME_COL, 'N/A')
        # --- Получаем описание порции (если есть) ---
        portion_desc = item.get('portion_desc', '') # e.g., "1/2"
        display_name = f"{portion_desc} {item_name}" if portion_desc else item_name
        # --- Генерируем ID с учетом порции для уникальности ключей ---
        item_identifier = item.get(URL_COL, item_name) + f"_{portion_desc}_{item_index}"

        disp_col, rem_col = st.columns([0.8, 0.2])
        with disp_col:
            st.markdown(f"**{display_name}**") # Показываем имя с порцией
            # КБЖУ должны соответствовать порции (если была разделена)
            kcal = item.get(TOTAL_NUTRITION_COLUMNS['kcal'], 0)
            prot = item.get(TOTAL_NUTRITION_COLUMNS['protein'], 0)
            fat = item.get(TOTAL_NUTRITION_COLUMNS['fat'], 0)
            carb = item.get(TOTAL_NUTRITION_COLUMNS['carbs'], 0)
            st.caption(f"{kcal:.0f}к / {prot:.1f}б / {fat:.1f}ж / {carb:.1f}у")

        # Кнопка удаления (только для ручного режима)
        if is_manual:
            with rem_col:
                # Генерируем уникальный ключ для кнопки удаления
                remove_key_base = generate_widget_key("manual_remove", meal_name, item_identifier)
                remove_key = remove_key_base; k_suffix = 0
                while remove_key in used_keys_in_slot: k_suffix += 1; remove_key = f"{remove_key_base}_{k_suffix}"
                used_keys_in_slot.add(remove_key)

                if st.button("➖", key=remove_key, help=f"Удалить '{display_name}' из {meal_name}"):
                    indices_to_remove.append(item_index)

    # Удаляем элементы, если были нажаты кнопки (только для ручного режима)
    if is_manual and indices_to_remove:
        for idx_del in sorted(indices_to_remove, reverse=True):
            try: del st.session_state.meal_plan[meal_name][idx_del]
            except IndexError: pass
        st.rerun()

def calculate_plan_totals(plan: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Рассчитывает суммарное КБЖУ для всего плана (работает и с порциями)."""
    totals: Dict[str, float] = {k: 0.0 for k in NUTRIENT_KEYS}
    for meal_list in plan.values():
        for item in meal_list:
            if isinstance(item, dict):
                for key in NUTRIENT_KEYS:
                    try:
                        # Используем КБЖУ, хранящееся в словаре item
                        # (оно должно быть скорректировано для порции, если это порция)
                        value = item.get(TOTAL_NUTRITION_COLUMNS[key], 0.0)
                        totals[key] += float(value) if pd.notna(value) else 0.0
                    except (ValueError, TypeError): pass # Игнорируем ошибки конвертации
    return totals

# --- ОБНОВЛЕНО: Отображение итогов с распределением ---
def display_totals_and_progress(totals: Dict[str, float], goals: Dict[str, float], title_prefix: str = "", show_distribution: bool = False, meal_plan: Optional[Dict[str, List[Dict]]] = None):
    """Отображает метрики и прогресс-бары + опционально распределение по приемам пищи."""
    st.header(f"📊 {title_prefix} Итоги за день")
    if not totals: st.caption("Нет данных."); return

    # --- Дневные итоги ---
    m_tot_col1, m_tot_col2, m_tot_col3, m_tot_col4 = st.columns(4)
    goal_kcal = goals.get('kcal', 0); goal_prot = goals.get('protein', 0); goal_fat = goals.get('fat', 0); goal_carb = goals.get('carbs', 0)
    total_kcal = totals.get('kcal', 0); total_prot = totals.get('protein', 0); total_fat = totals.get('fat', 0); total_carb = totals.get('carbs', 0)
    m_tot_col1.metric("Калории", f"{total_kcal:.0f} / {goal_kcal:.0f}", f"{total_kcal - goal_kcal:.0f} ккал")
    m_tot_col2.metric("Белки", f"{total_prot:.1f} / {goal_prot:.1f} г", f"{total_prot - goal_prot:.1f} г")
    m_tot_col3.metric("Жиры", f"{total_fat:.1f} / {goal_fat:.1f} г", f"{total_fat - goal_fat:.1f} г")
    m_tot_col4.metric("Углеводы", f"{total_carb:.1f} / {goal_carb:.1f} г", f"{total_carb - goal_carb:.1f} г")

    # --- Прогресс по дневным целям ---
    st.write(f"Прогресс по дневным целям ({title_prefix.lower()}):")
    def calculate_progress(total, goal): return min(total / goal, 1.0) if goal > 1e-9 else 0.0
    kcal_prog = calculate_progress(total_kcal, goal_kcal); prot_prog = calculate_progress(total_prot, goal_prot); fat_prog = calculate_progress(total_fat, goal_fat); carb_prog = calculate_progress(total_carb, goal_carb)
    st.progress(kcal_prog, text=f"Ккал: {kcal_prog*100:.0f}% ({total_kcal:.0f} / {goal_kcal:.0f})"); st.progress(prot_prog, text=f"Белки: {prot_prog*100:.0f}% ({total_prot:.1f} / {goal_prot:.1f} г)")
    st.progress(fat_prog, text=f"Жиры: {fat_prog*100:.0f}% ({total_fat:.1f} / {goal_fat:.1f} г)"); st.progress(carb_prog, text=f"Углеводы: {carb_prog*100:.0f}% ({total_carb:.1f} / {goal_carb:.1f} г)")

    # --- Распределение калорий по приемам пищи (%) ---
    if show_distribution and meal_plan is not None and total_kcal > 1e-6:
        st.markdown("---")
        st.subheader("Распределение калорий по приемам пищи (%)")
        dist_cols = st.columns(len(MEAL_SLOTS_ORDER))
        meal_kcal_totals = {meal: 0.0 for meal in MEAL_SLOTS_ORDER}
        # Считаем калории для каждого приема пищи на основе ПЛАНА
        for meal_name, items in meal_plan.items():
             if meal_name in meal_kcal_totals:
                  meal_kcal_totals[meal_name] = sum(float(item.get(TOTAL_NUTRITION_COLUMNS['kcal'], 0.0) or 0.0) for item in items if isinstance(item, dict))

        # Отображаем метрики для каждого приема пищи
        for i, meal_name in enumerate(MEAL_SLOTS_ORDER):
             with dist_cols[i]:
                 meal_kcal = meal_kcal_totals.get(meal_name, 0.0)
                 percentage = (meal_kcal / total_kcal * 100.0) if total_kcal > 0 else 0.0
                 target_perc = MEAL_TARGET_PERCENTAGES.get(meal_name, 0.0) * 100.0
                 # Показываем отклонение от целевого процента
                 st.metric(
                     label=f"{meal_name}",
                     value=f"{percentage:.0f}%",
                     delta=f"{percentage - target_perc:.0f}% (Цель: {target_perc:.0f}%)",
                     delta_color="off" # Используем нейтральный цвет для дельты
                 )
                 st.caption(f"{meal_kcal:.0f} ккал") # Показываем абсолютные калории

# =============================================
# --- Основное Приложение Streamlit ---
# =============================================

st.title("🥗 Конструктор Рациона V6.1 (Деление Порций)")

# --- Загрузка данных ---
df_products = load_and_preprocess_data(JSONL_FILE_PATH) # Загружаем V4 данные
if df_products is None or df_products.empty:
    st.error("Ошибка загрузки данных V4. Приложение не может продолжить.")
    st.stop()

# --- Инициализация Состояния ---
if 'meal_plan' not in st.session_state: st.session_state.meal_plan: Dict[str, List[Dict]] = {slot: [] for slot in MEAL_SLOTS_ORDER}
if 'suggested_distributed_plan' not in st.session_state: st.session_state.suggested_distributed_plan: Optional[Dict[str, List[Dict]]] = None
if 'suggested_totals' not in st.session_state: st.session_state.suggested_totals: Optional[Dict[str, float]] = None

# --- Боковая Панель (Sidebar) ---
with st.sidebar:
    # --- Цели ---
    st.header("🎯 Ваши цели на день")
    g_col1, g_col2 = st.columns(2); g_col3, g_col4 = st.columns(2)
    with g_col1: goal_kcal = st.number_input("Ккал", 0, 10000, 2000, 50, key="g_kcal_input")
    with g_col2: goal_protein = st.number_input("Белки, г", 0, 500, 100, 5, key="g_prot_input")
    with g_col3: goal_fat = st.number_input("Жиры, г", 0, 500, 70, 5, key="g_fat_input")
    with g_col4: goal_carbs = st.number_input("Углев, г", 0, 1000, 250, 10, key="g_carb_input")
    current_goals: Dict[str, float] = {'kcal': float(goal_kcal), 'protein': float(goal_protein), 'fat': float(goal_fat), 'carbs': float(goal_carbs)}

    # --- Настройки предложения ---
    st.markdown("---"); st.header("⚙️ Настройки Предложения")
    primary_goal_selected: str = st.radio("Цель для ИИ:", PRIMARY_GOAL_OPTIONS, index=0, key="primary_goal_radio", horizontal=True)
    filter_ready_meals: bool = st.checkbox("Только 'Готовая еда'", value=True, key="filter_ready_check")

    # --- Кнопка генерации ---
    if st.button("🤖 Предложить рацион (с порциями)", key="suggest_button", type="primary", help="Подобрать рацион с учетом делимости продуктов"):
        st.session_state.suggested_distributed_plan = None; st.session_state.suggested_totals = None
        with st.spinner("🧠 Подбираю продукты и распределяю (с делением)..."):
            df_to_suggest = df_products.copy()
            # --- Фильтрация по Готовой еде ---
            if filter_ready_meals:
                if BREADCRUMBS_COL in df_to_suggest.columns:
                    def is_ready_meal(crumbs): return isinstance(crumbs, list) and any(cat in READY_MEAL_CATEGORIES for cat in crumbs)
                    mask_ready = df_to_suggest[BREADCRUMBS_COL].apply(is_ready_meal); df_filtered = df_to_suggest[mask_ready]
                    if not df_filtered.empty: df_to_suggest = df_filtered; st.write(f"Фильтр 'Готовая еда': используется {len(df_to_suggest)} прод.")
                    else: st.warning("Нет 'Готовой еды', генерирую из всех.")
                else: st.warning(f"Нет колонки '{BREADCRUMBS_COL}', фильтр 'Готовая еда' не применен.")

            # 1. Генерация набора продуктов (без изменений)
            suggested_items_raw, _ = suggest_daily_set( # Игнорируем totals_raw, т.к. пересчитаем
                df_to_suggest, current_goals, primary_goal_selected, # type: ignore
                params=SUGGESTION_DEFAULTS, scoring_weights=SUGGESTION_SCORING_WEIGHTS
            )

            # 2. Распределение с делением порций
            if suggested_items_raw:
                 st.write("Набор продуктов подобран, запускаю распределение с делением порций...")
                 # !!! ВЫЗЫВАЕМ ФУНКЦИЮ С ЛОГИКОЙ ДЕЛЕНИЯ !!!
                 suggested_distributed = distribute_meals_smartly(suggested_items_raw, current_goals)
                 st.session_state.suggested_distributed_plan = suggested_distributed
                 # Пересчитываем итоги на основе ФАКТИЧЕСКИ РАСПРЕДЕЛЕННЫХ продуктов (и порций)
                 final_suggested_totals = calculate_plan_totals(suggested_distributed)
                 st.session_state.suggested_totals = final_suggested_totals
                 st.success("Предложение с учетом порций готово и распределено!")
                 st.write("Итоги предложения:"); st.json({k: f"{v:.1f}" for k, v in final_suggested_totals.items()})
            else:
                 st.warning("Не удалось подобрать набор продуктов.")
                 st.session_state.suggested_distributed_plan = None; st.session_state.suggested_totals = None
        st.rerun()

    # --- Кнопка скрытия предложения ---
    if st.session_state.suggested_distributed_plan:
        if st.button("Скрыть предложение", key="hide_suggestion_button"):
            st.session_state.suggested_distributed_plan = None; st.session_state.suggested_totals = None; st.rerun()

    # --- Фильтры для Ручного Поиска ---
    st.markdown("---"); st.header("🔍 Фильтры Ручного Поиска")
    current_goal_rating_col: Optional[str] = None; goal_col_map_filter = {'Weight Loss': f'{ANALYSIS_PREFIX}diet_weight_loss_rating', 'Muscle Gain': f'{ANALYSIS_PREFIX}diet_muscle_gain_rating', 'General Health': f'{ANALYSIS_PREFIX}diet_general_health_rating'};
    if primary_goal_selected != 'None': current_goal_rating_col = goal_col_map_filter.get(primary_goal_selected)
    min_goal_rating: int = 1;
    if current_goal_rating_col and current_goal_rating_col in df_products.columns: min_goal_rating = st.slider(f"Мин. рейтинг '{primary_goal_selected}':", 1, 5, 1, key="fgr_slider")
    sugar_filter_key: str = st.radio("Добавленный сахар:", list(FILTER_SUGAR_OPTIONS.keys()), index=0, key="fs_radio", horizontal=True); selected_sugar_filter: Optional[bool] = FILTER_SUGAR_OPTIONS[sugar_filter_key]
    grains_filter_key: str = st.radio("Цельные злаки:", list(FILTER_GRAINS_OPTIONS.keys()), index=0, key="fg_radio", horizontal=True); selected_grains_filter: Optional[bool] = FILTER_GRAINS_OPTIONS[grains_filter_key]
    # --- НОВОЕ: Добавляем фильтры по V4 полям ---
    with st.expander("Доп. фильтры (оценки V3, роль, теги, V4)..."): # Добавил V4 в заголовок
         # Фильтры V3 (оценки, роль, сложность, теги) - без изменений
         selected_estimates: Dict[str, List[str]] = {};
         for col, label in FILTER_ESTIMATE_COLS.items(): # V3 оценки
              if col in df_products.columns: options = sorted([lvl for lvl in df_products[col].unique() if pd.notna(lvl) and lvl != 'Uncertain']);
              if options: selected_levels = st.multiselect(f"{label} (V3):", options, key=f"f_est_{col}"); # Добавил V3
              if selected_levels: selected_estimates[col] = selected_levels
         selected_roles: List[str] = [];
         if FILTER_ROLE_COL in df_products.columns: role_options = sorted([r for r in df_products[FILTER_ROLE_COL].unique() if pd.notna(r) and r != 'Uncertain']);
         if role_options: selected_roles = st.multiselect("Роль V3:", role_options, key="fr_multiselect") # Добавил V3
         selected_prep: List[str] = [];
         if FILTER_PREP_COL in df_products.columns: prep_options = sorted([p for p in df_products[FILTER_PREP_COL].unique() if pd.notna(p) and p != 'Uncertain']);
         if prep_options: selected_prep = st.multiselect("Сложность V3:", prep_options, key="fp_multiselect") # Добавил V3
         selected_tag: Optional[str] = None;
         if FILTER_TAGS_COL in df_products.columns:
              try:
                    all_tags: Set[str] = set(tag for tag_list in df_products[FILTER_TAGS_COL].dropna() if isinstance(tag_list, list) for tag in tag_list);
                    if all_tags: tag_options: List[str] = ["Любой"] + sorted(list(all_tags)); selected_tag = st.selectbox("Тег пользы V3:", tag_options, key="ft_selectbox") # Добавил V3
              except Exception as e_tags: st.warning(f"Ошибка тегов V3: {e_tags}")

         # --- Фильтры V4 ---
         st.markdown("---") # Разделитель для V4
         # Фильтр по делимости
         portion_col = V4_PORTIONS_COL
         filter_portions = st.radio("Делимость (V4):", ["Любая", "Только делимые (>1)", "Только неделимые (1)"], index=0, key="f_portions")
         # Фильтр по доминантному макро
         macro_col = f'{V4_PREFIX}context_dominant_macro'
         selected_macros: List[str] = []
         if macro_col in df_products.columns:
              macro_options = sorted([m for m in df_products[macro_col].unique() if pd.notna(m) and m != 'uncertain'])
              if macro_options: selected_macros = st.multiselect("Дом. макро (V4):", macro_options, key="f_macro")
         # Фильтр по температуре
         temp_col = f'{V4_PREFIX}context_consumption_temperature'
         selected_temps: List[str] = []
         if temp_col in df_products.columns:
              temp_options = sorted([t for t in df_products[temp_col].unique() if pd.notna(t) and t != 'uncertain'])
              if temp_options: selected_temps = st.multiselect("Темп. употребл. (V4):", temp_options, key="f_temp")
         # Фильтр по десертам
         dessert_col = f'{V4_PREFIX}context_is_dessert'
         filter_dessert = st.radio("Десерт (V4):", ["Любой", "Только десерты", "Кроме десертов"], index=0, key="f_dessert")


# =============================================
# --- Основная Область Отображения ---
# =============================================
st.markdown("---")

# --- 1. Отображение Предложенного Рациона (если есть) ---
suggested_plan = st.session_state.suggested_distributed_plan
suggested_totals = st.session_state.suggested_totals

if suggested_plan and suggested_totals:
    # Используем expander для сгенерированного плана
    with st.expander("🤖 Предложенный рацион (с делением порций)", expanded=True):
        sugg_cols = st.columns(len(MEAL_SLOTS_ORDER))
        for i, meal_name in enumerate(MEAL_SLOTS_ORDER):
            with sugg_cols[i]:
                # Используем обновленную функцию для отображения порций
                display_meal_slot(meal_name, suggested_plan.get(meal_name, []), current_goals, is_manual=False)
        st.markdown("---")
        # Используем обновленную функцию для отображения итогов и распределения
        display_totals_and_progress(
            suggested_totals, current_goals, title_prefix="Предложение:",
            show_distribution=True, meal_plan=suggested_plan # Включаем отображение распределения
        )
        # Кнопка применения (без изменений)
        if st.button("✅ Применить предложение к ручному плану", key="apply_sugg"):
            # Копируем план (он уже содержит порции)
            st.session_state.meal_plan = {meal: [item.copy() for item in items] for meal, items in suggested_plan.items()}
            st.session_state.suggested_distributed_plan = None; st.session_state.suggested_totals = None;
            st.success("Предложение применено!"); st.rerun()

# --- 2. Ручной Конструктор ---
st.markdown("---"); st.header("✍️ Ваш рацион (Ручное добавление / Редактирование)")

# --- Применение Фильтров ---
df_manual_filtered = df_products.copy() # Начинаем с полного датафрейма
try:
    # --- Фильтры V3 (без изменений) ---
    if current_goal_rating_col and current_goal_rating_col in df_manual_filtered.columns and min_goal_rating > 1: rating_col_data = pd.to_numeric(df_manual_filtered[current_goal_rating_col], errors='coerce').fillna(1.0); df_manual_filtered = df_manual_filtered[rating_col_data >= min_goal_rating]
    sugar_col = f'{ANALYSIS_PREFIX}likely_contains_added_sugar';
    if selected_sugar_filter is not None and sugar_col in df_manual_filtered.columns: df_manual_filtered = df_manual_filtered[df_manual_filtered[sugar_col].fillna(False) == selected_sugar_filter]
    grains_col = f'{ANALYSIS_PREFIX}likely_contains_whole_grains';
    if selected_grains_filter is not None and grains_col in df_manual_filtered.columns: df_manual_filtered = df_manual_filtered[df_manual_filtered[grains_col].fillna(False) == selected_grains_filter]
    for col, levels in selected_estimates.items():
        if col in df_manual_filtered.columns and levels: df_manual_filtered = df_manual_filtered[df_manual_filtered[col].isin(levels)]
    if selected_roles and FILTER_ROLE_COL in df_manual_filtered.columns: df_manual_filtered = df_manual_filtered[df_manual_filtered[FILTER_ROLE_COL].isin(selected_roles)]
    if selected_prep and FILTER_PREP_COL in df_manual_filtered.columns: df_manual_filtered = df_manual_filtered[df_manual_filtered[FILTER_PREP_COL].isin(selected_prep)]
    if selected_tag and selected_tag != "Любой" and FILTER_TAGS_COL in df_manual_filtered.columns:
        def check_tag(tags_list): return isinstance(tags_list, list) and selected_tag in tags_list
        df_manual_filtered = df_manual_filtered[df_manual_filtered[FILTER_TAGS_COL].apply(check_tag)]

    # --- НОВОЕ: Применение фильтров V4 ---
    # Фильтр по делимости
    if portion_col in df_manual_filtered.columns:
         # Убедимся, что колонка числовая (Int64)
         df_manual_filtered[portion_col] = pd.to_numeric(df_manual_filtered[portion_col], errors='coerce').fillna(1).astype(pd.Int64Dtype())
         if filter_portions == "Только делимые (>1)":
              df_manual_filtered = df_manual_filtered[df_manual_filtered[portion_col] > 1]
         elif filter_portions == "Только неделимые (1)":
              df_manual_filtered = df_manual_filtered[df_manual_filtered[portion_col] <= 1] # Включая NA/1

    # Фильтр по доминантному макро
    if selected_macros and macro_col in df_manual_filtered.columns:
        df_manual_filtered = df_manual_filtered[df_manual_filtered[macro_col].isin(selected_macros)]

    # Фильтр по температуре
    if selected_temps and temp_col in df_manual_filtered.columns:
        df_manual_filtered = df_manual_filtered[df_manual_filtered[temp_col].isin(selected_temps)]

    # Фильтр по десертам
    if dessert_col in df_manual_filtered.columns:
         # Убедимся, что колонка булева
         df_manual_filtered[dessert_col] = df_manual_filtered[dessert_col].map({True: True, False: False}).fillna(False).astype(bool)
         if filter_dessert == "Только десерты":
              df_manual_filtered = df_manual_filtered[df_manual_filtered[dessert_col] == True]
         elif filter_dessert == "Кроме десертов":
              df_manual_filtered = df_manual_filtered[df_manual_filtered[dessert_col] == False]

except Exception as e_filter:
    st.error(f"Ошибка при применении фильтров: {e_filter}. Отображены не все результаты.")

# --- Выбор Продукта ---
st.subheader("Добавить продукт вручную")
# ... (логика selectbox без изменений) ...
if NAME_COL not in df_manual_filtered.columns: product_names_manual = []
else: product_names_manual = sorted(df_manual_filtered[NAME_COL].dropna().astype(str).unique())
if not product_names_manual: st.warning("Нет продуктов, соотв. фильтрам."); selected_product_manual_name: str = "---"
else: selected_product_manual_name = st.selectbox(f"Выберите продукт ({len(product_names_manual)} найдено):", options=["---"] + product_names_manual, key="manual_select_filtered")


# --- Отображение Деталей и Добавление ---
if selected_product_manual_name != "---":
    product_details_df = df_manual_filtered[df_manual_filtered[NAME_COL] == selected_product_manual_name]
    if not product_details_df.empty:
        product_details_dict = product_details_df.iloc[0].where(pd.notna(product_details_df.iloc[0]), None).to_dict()
        # Используем ОБНОВЛЕННУЮ функцию для отображения V3 и V4 деталей
        display_product_details(product_details_dict, current_goal_rating_col, primary_goal_selected)
    else: st.warning(f"Не найдены детали для '{selected_product_manual_name}'.")

# --- Отображение Ручного Рациона ---
st.subheader("Составленный рацион")
manual_plan = st.session_state.meal_plan
if not any(manual_plan.values()): st.info("Ваш ручной план пока пуст.")
else:
    meal_plan_cols = st.columns(len(MEAL_SLOTS_ORDER))
    for i, meal_name in enumerate(MEAL_SLOTS_ORDER):
        with meal_plan_cols[i]:
            # Используем ОБНОВЛЕННУЮ функцию для отображения порций
            display_meal_slot(meal_name, manual_plan.get(meal_name, []), current_goals, is_manual=True)

# --- Отображение Итогов Ручного Рациона ---
st.markdown("---")
manual_totals = calculate_plan_totals(manual_plan) # Работает и с порциями
# Используем ОБНОВЛЕННУЮ функцию для отображения распределения
display_totals_and_progress(
    manual_totals, current_goals, title_prefix="Ручной План:",
    show_distribution=True, meal_plan=manual_plan
)

# --- Кнопка Очистки Ручного Плана ---
if any(st.session_state.meal_plan.values()):
     if st.button("🗑️ Очистить ручной план", key="clear_manual_plan"):
          st.session_state.meal_plan = {slot: [] for slot in MEAL_SLOTS_ORDER}; st.success("Ручной план очищен."); st.rerun()

# --- Footer Info (Обновлено) ---
st.markdown("---")
st.info("""
**Примечания V6.1 (Порции):**
* **Деление Порций:** Предложение рациона теперь пытается делить продукты (помеченные как делимые >1 порции) пополам, если это помогает лучше сбалансировать КБЖУ по приемам пищи (Цели: З≈25%, О≈35%, У≈25%, П≈15%). В плане такие продукты отображаются с префиксом "1/2".
* **Ручное Добавление:** При ручном добавлении всегда добавляется **целый** продукт. Логика деления работает только для авто-предложения.
* **Новые Фильтры/Инфо:** Добавлены фильтры и информация по делимости, доминантному макронутриенту, температуре употребления и признаку десерта (данные V4).
* **Точность и Ограничения:** Качество зависит от данных. Распределение и деление - это эвристика, возможны неидеальные результаты.
""")