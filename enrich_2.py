import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio
import nest_asyncio
from openai import AsyncOpenAI, RateLimitError, APIError
from pydantic import BaseModel, Field, ValidationError, conint # conint для проверки >=1
from typing import Optional, List, Literal, Dict, Any

# --- Применяем nest_asyncio ---
nest_asyncio.apply()

st.set_page_config(page_title="Дообогащение Данных V4 (Порции + Контекст)", layout="wide")

# --- Конфигурация ---
# !!! ВНИМАНИЕ: Безопасное хранение ключа !!!
api_key = '...' # ЗАМЕНИТЬ НА БЕЗОПАСНЫЙ СПОСОБ

if not api_key:
    st.error("Ошибка: Ключ OpenAI API не найден.")
    st.stop()

# --- Пути к файлам ---
# !!! Читаем из V3, пишем в V4 !!!
INPUT_JSONL_FILE = 'vkusvill_data_enriched_v3.jsonl'
OUTPUT_JSONL_FILE = 'vkusvill_data_enriched_v4.jsonl'

# --- Настройки API ---
MODEL_NAME = "gpt-4o-2024-08-06" # Или другая актуальная модель
MAX_RETRIES = 3
INITIAL_DELAY = 5
MAX_CONCURRENT_REQUESTS = 10

# --- Pydantic Модели ТОЛЬКО для НОВЫХ полей V4 ---

# --- Вспомогательная функция для загрузки JSONL ---
# (Вставьте этот блок кода в enrich_2.py)

def load_jsonl(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Читает JSONL файл и возвращает список словарей.
    Обрабатывает ошибки JSON в отдельных строках.
    """
    data: List[Dict[str, Any]] = []
    lines_processed_log: int = 0 # Счетчик для ограничения логов
    log_func = st.warning # Используем Streamlit для вывода предупреждений

    # Проверка существования файла перед открытием
    if not os.path.exists(file_path):
        st.error(f"Файл не найден по пути: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    lines_processed_log += 1
                    # Ограничиваем вывод сообщений об ошибках JSON
                    if lines_processed_log < 20 or i % 500 == 0:
                        log_func(f"Ошибка декодирования JSON в строке {i+1} файла {file_path}. Строка пропущена.")
                    pass # Пропускаем строку с ошибкой JSON
            # Проверка, что данные вообще были загружены
            if not data:
                st.error(f"В файле '{file_path}' не найдено валидных JSON строк.")
                return None
            # Не выводим success здесь, он будет в вызывающей функции
            return data
    except Exception as e:
        # Ловим другие возможные ошибки чтения файла
        st.error(f"Критическая ошибка при чтении файла '{file_path}': {e}")
        return None


# --- Вспомогательная функция для проверки категории "Готовая еда" ---
# (Вставьте этот блок кода в enrich_2.py)

# Определим константы прямо здесь, если они не заданы глобально в этом скрипте
BREADCRUMBS_COL_NAME = 'breadcrumbs'
READY_MEAL_CATEGORIES_SET = {"Готовая еда", "Готовая ета"}

def is_ready_meal_category(item_data: Dict[str, Any]) -> bool:
    """
    Проверяет, относится ли товар к категории 'Готовая еда' по breadcrumbs.
    """
    # Получаем список "хлебных крошек" из данных продукта
    crumbs = item_data.get(BREADCRUMBS_COL_NAME)

    # Проверяем, что это действительно список
    if isinstance(crumbs, list):
        # Проверяем, есть ли ХОТЯ БЫ ОДНА категория из нашего набора READY_MEAL_CATEGORIES_SET
        # в списке категорий продукта (crumbs)
        return any(cat in READY_MEAL_CATEGORIES_SET for cat in crumbs)

    # Если 'breadcrumbs' не список или отсутствует, считаем, что это не готовая еда
    return False

# --- Функция load_jsonl(...) ---
# (Определение функции load_jsonl должно быть здесь)
# def load_jsonl(...): ...

# --- Pydantic Модели (Версия 4 - ТОЛЬКО для НОВЫХ полей) ---
# (Здесь идет ваш код с Pydantic моделями...)
# ...
# --- Pydantic Модели (Версия 4 - ТОЛЬКО для НОВЫХ полей) ---
# (Здесь идет ваш код с Pydantic моделями...)
# ...

class PortionInfo(BaseModel):
    """Анализ делимости продукта на порции."""
    suggested_portions: Optional[conint(ge=1)] = Field( # conint(ge=1) - целое >= 1
        default=1, # По умолчанию 1 порция (не делимый)
        description="Предполагаемое кол-во разумных порций (1, 2, 3...). 1 = делить нецелесообразно. null если не удалось определить."
    )
    portion_reasoning: Optional[str] = Field(
        default=None,
        description="Обоснование для suggested_portions (напр., 'Большая упаковка супа 1кг' или 'Стандартная одиночная порция')."
    )

class AdditionalContext(BaseModel):
    """Дополнительные контекстные теги продукта."""
    dominant_macro: Literal['protein', 'carb', 'fat', 'balanced', 'other', 'uncertain'] = Field(
        default='uncertain',
        description="Какой макронутриент доминирует или сбалансирован?"
    )
    consumption_temperature: Literal['hot', 'cold', 'any', 'uncertain'] = Field(
        default='any',
        description="Как обычно употребляется: горячим, холодным, без разницы?"
    )
    is_dessert: bool = Field(
        default=False,
        description="Является ли продукт десертом/сладким блюдом?"
    )

class ProductEnrichmentV4(BaseModel):
    """Модель ТОЛЬКО для ДОБАВЛЯЕМОЙ информации V4."""
    # Эти поля будут добавлены к существующему словарю продукта
    # Мы НЕ включаем сюда поля из ComprehensiveProductAnalysisV3
    portion_info: PortionInfo = Field(description="Информация о делимости на порции.")
    additional_context: AdditionalContext = Field(description="Дополнительный контекст продукта.")


# --- Инициализация Клиента OpenAI ---
try:
    async_client = AsyncOpenAI(api_key=api_key)
    st.sidebar.success("Клиент AsyncOpenAI инициализирован.")
except Exception as e:
    st.error(f"Ошибка инициализации клиента AsyncOpenAI: {e}")
    async_client = None
    st.stop()

# --- Асинхронная Функция API (ТОЛЬКО для V4 полей) ---
async def get_enrichment_v4_async(
    item_data: Dict[str, Any], # Принимает весь словарь продукта из V3 файла
    semaphore: asyncio.Semaphore
) -> tuple[str | None, dict | None]:
    """Асинхронно вызывает API для получения ТОЛЬКО полей V4 (порции, контекст)."""
    item_url = item_data.get('url')
    if not item_url: return None, None

    # Извлекаем ключевую информацию для промпта
    product_name = item_data.get('name', 'N/A')
    # Вес может отсутствовать или быть нечисловым, обрабатываем это
    weight_str = ""
    weight_val = item_data.get('weight_value')
    weight_unit = item_data.get('weight_unit')
    if pd.notna(weight_val) and pd.notna(weight_unit):
         try:
              weight_str = f"{float(weight_val):.0f} {weight_unit}"
         except: # Если вес не конвертируется
              weight_str = f"{weight_val} {weight_unit}" # Используем как есть
    elif pd.notna(weight_val): # Если есть только значение
         weight_str = f"{weight_val}"

    ingredients = item_data.get('ingredients', 'Состав не указан')
    # Можно также передать роль из V3 анализа, если она есть и полезна
    # role_v3 = item_data.get('product_analysis_v3', {}).get('meal_component_role', '')

    ingredients_truncated = ingredients[:800] + ("..." if len(ingredients) > 800 else "") # Уменьшил для краткости

    # --- Промпт для получения ТОЛЬКО полей V4 ---
    system_prompt_v4_add = """
    Ты - внимательный аналитик продуктов питания. Твоя задача - определить практическую делимость продукта на порции и добавить полезные контекстные теги.
    Основывайся на названии, весе (если указан), составе и здравом смысле.
    Верни ТОЛЬКО JSON объект, соответствующий схеме `ProductEnrichmentV4`.

    ЗАПОЛНИ ВСЕ ПОЛЯ в `ProductEnrichmentV4`:
    1.  `portion_info`:
        * `suggested_portions`: На сколько РАЗУМНЫХ порций можно разделить продукт для употребления? (1, 2, 3, 4...). Укажи 1, если продукт явно на одну порцию или делить нецелесообразно (напр., бутерброд, маленький йогурт, одна котлета). Используй здравый смысл для больших упаковок (супы 1кг, наборы роллов, большие салаты). Если не уверен, ставь 1.
        * `portion_reasoning`: Кратко обоснуй количество порций (напр., "Стандартная порция салата", "Большая упаковка 1кг", "Набор из 4 сырников").
    2.  `additional_context`:
        * `dominant_macro`: Какой макронутриент (protein, carb, fat) преобладает по составу/типу блюда? Или 'balanced', если явно смешанное. 'other' для напитков и т.п., 'uncertain' если неясно.
        * `consumption_temperature`: Обычно едят горячим ('hot'), холодным ('cold'), или без разницы ('any')? 'uncertain' если неясно.
        * `is_dessert`: Это десерт или очевидно сладкое блюдо (True/False)?

    Строго следуй схеме `ProductEnrichmentV4` и допустимым значениям! Не включай никакие другие поля.
    """
    user_prompt = f"""
    Проанализируй продукт для добавления информации V4:
    Название: {product_name}
    Вес/Объем: {weight_str if weight_str else "Не указан"}
    Состав: {ingredients_truncated}
    # Роль (из V3, если есть): {item_data.get('product_analysis_v3', {}).get('meal_component_role', 'N/A')}

    Предоставь ТОЛЬКО JSON по схеме `ProductEnrichmentV4`.
    """

    retries = 0; delay = INITIAL_DELAY
    async with semaphore:
        while retries < MAX_RETRIES:
            try:
                response = await async_client.chat.completions.create(
                     model=MODEL_NAME,
                     messages=[
                         {"role": "system", "content": system_prompt_v4_add},
                         {"role": "user", "content": user_prompt},
                     ],
                     response_format={"type": "json_object"}, # Запрос JSON
                     temperature=0.2 # Низкая температура для большей предсказуемости
                 )
                content = response.choices[0].message.content
                if content:
                    try:
                        # Пытаемся распарсить и валидировать с помощью Pydantic
                        parsed_data = ProductEnrichmentV4.model_validate_json(content)
                        # Возвращаем как словарь
                        return item_url, parsed_data.model_dump()
                    except ValidationError as e_val:
                         print(f"Ошибка Pydantic ValidationError для '{product_name}' ({item_url}): {e_val}. Ответ API: {content}")
                         # Не повторяем попытку при ошибке валидации схемы
                         return item_url, None
                    except json.JSONDecodeError as e_json:
                         print(f"Ошибка JSONDecodeError для '{product_name}' ({item_url}): {e_json}. Ответ API: {content}")
                         # Попробуем еще раз, вдруг временный сбой формата
                         pass # Переходим к блоку except Exception -> retry
                else:
                    print(f"Предупреждение [Попытка {retries+1}/{MAX_RETRIES}]: API вернул пустой контент для '{product_name}' ({item_url})")

            except RateLimitError as e:
                print(f"Предупреждение [Попытка {retries+1}/{MAX_RETRIES}]: RateLimitError для '{product_name}' ({item_url}). Повтор через {delay} сек. {e}")
            except APIError as e:
                # Обработка ошибок API (например, 400 Bad Request из-за промпта)
                print(f"Предупреждение [Попытка {retries+1}/{MAX_RETRIES}]: APIError ({e.status_code}) для '{product_name}' ({item_url}). Повтор через {delay} сек. {e}")
                if e.status_code == 400: # Не повторяем при Bad Request
                     return item_url, None
            except Exception as e:
                print(f"Предупреждение [Попытка {retries+1}/{MAX_RETRIES}]: Неизвестная ошибка для '{product_name}' ({item_url}). Повтор через {delay} сек. {e}")

            # Ожидание перед повтором
            await asyncio.sleep(delay)
            retries += 1
            delay *= 2 # Экспоненциальная задержка

    print(f"Ошибка: Не удалось получить данные V4 для '{product_name}' ({item_url}) после {MAX_RETRIES} попыток.")
    return item_url, None

# --- Основная АСИНХРОННАЯ функция дообогащения ---
# --- Убедитесь, что эта функция определена где-то ВЫШЕ ---
# (И что константы BREADCRUMBS_COL и READY_MEAL_CATEGORIES определены глобально)
# def is_ready_meal_category(item_data: Dict[str, Any]) -> bool:
#     """Проверяет, относится ли товар к категории 'Готовая еда' по breadcrumbs."""
#     crumbs = item_data.get(BREADCRUMBS_COL) # Используйте константу BREADCRUMBS_COL
#     if isinstance(crumbs, list):
#         return any(cat in READY_MEAL_CATEGORIES for cat in crumbs) # Используйте константу READY_MEAL_CATEGORIES
#     return False
# -------------------------------------------------------------

# --- Основная АСИНХРОННАЯ функция дообогащения ---
async def enrich_dataset_v4_async():
    """Читает V3 файл, добавляет V4 поля ТОЛЬКО для Готовой еды, пишет в V4 файл."""
    st.title("🧬 Дообогащение V4 (Порции + Контекст)")

    if not async_client:
        st.error("Клиент AsyncOpenAI не инициализирован.")
        return

    # --- Загрузка ИСХОДНОГО V3 файла ---
    st.info(f"Чтение исходного файла: {INPUT_JSONL_FILE}")
    all_items_v3 = load_jsonl(INPUT_JSONL_FILE) # Используем нашу функцию загрузки
    if all_items_v3 is None:
        st.error(f"Не удалось прочитать файл {INPUT_JSONL_FILE}.")
        return
    st.success(f"Прочитано {len(all_items_v3)} товаров из V3 файла.")

    # --- ФИЛЬТРАЦИЯ: Обрабатываем ТОЛЬКО "Готовую еду" с URL ---
    # (Возвращаем фильтр, как в оригинальном V3 скрипте)
    items_to_process = [
        item for item in all_items_v3
        if item.get('url') and is_ready_meal_category(item) # <<< ВОЗВРАЩАЕМ ФИЛЬТР ПО КАТЕГОРИИ
    ]
    # --- КОНЕЦ ФИЛЬТРАЦИИ ---

    if not items_to_process:
         st.warning("Не найдено товаров 'Готовой еды' с URL для дообогащения V4.")
         # Копируем V3 в V4 без изменений, если нечего обрабатывать
         try:
             import shutil
             shutil.copy2(INPUT_JSONL_FILE, OUTPUT_JSONL_FILE) # Более надежное копирование
             st.info(f"Файл {INPUT_JSONL_FILE} скопирован в {OUTPUT_JSONL_FILE}, так как нечего было дообогащать.")
         except Exception as e_copy:
             st.error(f"Ошибка копирования файла: {e_copy}")
         return

    total_to_process = len(items_to_process)
    # Уточняем сообщение для пользователя
    st.info(f"Будет обработано {total_to_process} товаров 'Готовой еды' для добавления полей V4.")

    # --- Подготовка к параллельным запросам ---
    enriched_v4_data_map: Dict[str, Dict] = {}
    processed_count = 0
    error_count = 0
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [
         asyncio.create_task(
            get_enrichment_v4_async(item, semaphore),
            name=f"EnrichV4_{item.get('url', i)}"
        ) for i, item in enumerate(items_to_process) # Создаем задачи только для отфильтрованных
    ]

    # --- Отображение в Streamlit ---
    st.markdown("---")
    # Уточняем заголовок
    st.subheader("🚀 Выполнение запросов к OpenAI API (V4 поля для Готовой еды)")
    progress_bar = st.progress(0.0, text="Начинаем обработку...")
    log_container = st.container(height=300)
     # Уточняем сообщение
    log_container.info(f"Запускаем {total_to_process} задач для 'Готовой еды'...")

    # --- Асинхронное выполнение и обработка результатов ---
    for future in asyncio.as_completed(tasks):
        processed_count += 1
        try:
            item_url, analysis_result_v4 = await future
            if item_url and analysis_result_v4:
                enriched_v4_data_map[item_url] = analysis_result_v4
                # Логируем краткий результат V4
                portions = analysis_result_v4.get('portion_info', {}).get('suggested_portions', '?')
                macro = analysis_result_v4.get('additional_context', {}).get('dominant_macro', '?')
                temp = analysis_result_v4.get('additional_context', {}).get('consumption_temperature', '?')
                is_dessert = analysis_result_v4.get('additional_context', {}).get('is_dessert', '?')
                # Ищем имя только среди items_to_process (отфильтрованных)
                product_name = next((item['name'] for item in items_to_process if item['url'] == item_url), '?')
                log_container.write(f"✅ {processed_count}/{total_to_process}. {product_name}: Порций={portions}, Макро={macro}, Темп={temp}, Десерт={is_dessert}")
            elif item_url:
                error_count += 1
                product_name = next((item['name'] for item in items_to_process if item['url'] == item_url), '?')
                log_container.error(f"❌ Ошибка V4 для: {product_name} ({item_url})")
        except Exception as e_future:
            # ... (обработка ошибок future) ...
             error_count += 1
             task_name = future.get_name() if hasattr(future, 'get_name') else f"Задача {processed_count}"
             log_container.error(f"❌ Критическая ошибка задачи '{task_name}': {e_future}")


        # Обновляем прогресс-бар
        progress_value = processed_count / total_to_process
        progress_bar.progress(progress_value, text=f"Обработано: {processed_count}/{total_to_process} (Ошибок V4: {error_count})")


    # --- Завершение ---
     # Уточняем сообщение
    st.success(f"API обработка V4 завершена. Успешно получены данные для {len(enriched_v4_data_map)} из {total_to_process} продуктов ('Готовая еда'). Ошибок: {error_count}.")

    # --- Запись результата в НОВЫЙ V4 файл ---
    # Логика записи остается прежней: проходим по всем из V3, добавляем V4 если есть
    st.info(f"Запись обогащенных данных в файл: {OUTPUT_JSONL_FILE}")
    items_written = 0
    enriched_v4_added_count = 0 # Считаем, сколько РЕАЛЬНО добавили V4
    try:
        with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as outfile:
            for item_v3 in all_items_v3: # Итерируем по ВСЕМ товарам из V3 файла
                item_url = item_v3.get('url')
                item_to_write = item_v3.copy() # Копируем, чтобы не менять исходный
                # Если для этого URL (который прошел фильтр и был обработан) есть новые данные V4
                if item_url and item_url in enriched_v4_data_map:
                    item_to_write['enrichment_v4'] = enriched_v4_data_map[item_url] # Добавляем
                    enriched_v4_added_count += 1
                # Записываем в любом случае (либо V3, либо V3 + V4)
                outfile.write(json.dumps(item_to_write, ensure_ascii=False) + '\n')
                items_written += 1
        # Уточняем сообщение
        st.success(f"Успешно записано {items_written} товаров в {OUTPUT_JSONL_FILE}. Новые данные V4 добавлены для {enriched_v4_added_count} товаров ('Готовая еда').")
    except Exception as e:
        st.error(f"Ошибка при записи в файл {OUTPUT_JSONL_FILE}: {e}")


# --- Запуск асинхронной функции ---
# (блок if __name__ == "__main__": остается без изменений)
# ...

# --- Запуск асинхронной функции ---
if __name__ == "__main__":
    try:
        asyncio.run(enrich_dataset_v4_async())
    except Exception as main_e:
        st.error(f"Критическая ошибка выполнения: {main_e}")
        import traceback
        st.code(traceback.format_exc())