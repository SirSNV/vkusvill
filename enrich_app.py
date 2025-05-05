import streamlit as st
import pandas as pd
import json
import os
import time
import asyncio # Добавлено для асинхронности
import nest_asyncio # Добавлено для совместимости asyncio со Streamlit/Jupyter

# Применяем nest_asyncio В САМОМ НАЧАЛЕ
nest_asyncio.apply()

# Импорты OpenAI и Pydantic
from openai import AsyncOpenAI, RateLimitError, APIError # Используем AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal, Dict, Any

st.set_page_config(page_title="Обогащение Данных VkusVill", layout="wide") # Настройка страницы

# --- Конфигурация ---
# Загрузка переменных окружения (рекомендуемый способ для API ключа)
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# !!! ВНИМАНИЕ: Хранить API ключ в коде небезопасно! Используйте .env или переменные окружения !!!
api_key = '...' # Ваш ключ (ЗАМЕНИТЬ НА БЕЗОПАСНЫЙ СПОСОБ)


# Проверка наличия ключа
if not api_key:
    st.error("Ошибка: Ключ OpenAI API не найден. Установите переменную окружения OPENAI_API_KEY или добавьте его в код (не рекомендуется).")
    st.stop() # Останавливаем выполнение, если ключа нет

# Пути к файлам
INPUT_JSONL_FILE = 'vkusvill_data.jsonl' # ИСПОЛЬЗУЕМ ФАЙЛ БЕЗ ДУБЛИКАТОВ
OUTPUT_JSONL_FILE = 'vkusvill_data_enriched_v3.jsonl' # Новое имя для файла с полным анализом

# Настройки API
MODEL_NAME = "gpt-4o-2024-08-06" # Актуальная модель GPT-4o
MAX_RETRIES = 3
INITIAL_DELAY = 5 # Секунды
# Настройки параллелизма
MAX_CONCURRENT_REQUESTS = 10 # Ограничиваем кол-во одновременных запросов к API

# --- Pydantic Модели (Версия 3 - Комплексная) ---

# Определения типов Literal/Enum для детерминизма
FlavorProfileEnum = Literal["Sweet", "Savory", "Spicy", "Sour", "Umami", "Bitter", "Balanced", "Neutral", "Other/Mixed", "Uncertain"]
TextureEnum = Literal["Creamy", "Crunchy", "Chewy", "Soft", "Liquid", "Crispy", "Firm", "Tender", "Mixed", "Other", "Uncertain"]
CookingMethodEnum = Literal["Fried", "Baked", "Steamed", "Grilled", "Boiled", "Stewed", "Roasted", "Raw/Salad", "Microwaved", "Sous-Vide", "Smoked", "Other", "N/A"]
HealthBenefitTagEnum = Literal[
    "Probiotic Source", "Prebiotic Source", "Antioxidant Rich", "Omega-3 Source",
    "High Fiber", "Good Source of Protein", "Low Glycemic Index (Estimate)",
    "Hydrating", "Source of Calcium", "Source of Iron", "Source of Potassium",
    "Source of Vitamin C", "Source of Vitamin D", "Source of B12"
]
EstimationLevel = Literal["Low", "Medium", "High", "Uncertain"]
PrepComplexity = Literal["Ready-to-Eat", "Requires Heating", "Minimal Prep", "Requires Cooking", "Uncertain"]
ComponentRole = Literal[
    "Primary Protein Source", "Primary Carb Source", "Primary Fat Source",
    "Vegetable/Fiber Source", "Fruit/Dessert", "Condiment/Sauce",
    "Complete Meal", "Snack", "Drink", "Other", "Uncertain"
]

class MealSuitability(BaseModel):
    breakfast_rating: int = Field(..., description="Рейтинг для Завтрака (1-5 целое число, 1=Не подходит, 5=Отлично)")
    lunch_rating: int = Field(..., description="Рейтинг для Обеда (1-5 целое число, 1=Не подходит, 5=Отлично)")
    dinner_rating: int = Field(..., description="Рейтинг для Ужина (1-5 целое число, 1=Не подходит, 5=Отлично)")
    snack_rating: int = Field(..., description="Рейтинг для Перекуса (1-5 целое число, 1=Не подходит, 5=Отлично)")
    suitability_reasoning: str = Field(..., description="Краткое обоснование рейтингов пригодности (1-2 предложения).")

class DietGoalRatings(BaseModel):
    weight_loss_rating: int = Field(..., description="Пригодность для Похудения (1-5 целое число)")
    muscle_gain_rating: int = Field(..., description="Пригодность для Набора массы (1-5 целое число)")
    general_health_rating: int = Field(..., description="Пригодность для Общего здоровья (1-5 целое число)")
    low_calorie_snack_rating: int = Field(..., description="Пригодность как Низкокалорийный перекус (1-5 целое число)")
    goal_reasoning: str = Field(..., description="Краткое обоснование рейтингов по целям.")

class ComprehensiveProductAnalysisV3(BaseModel):
    """Комплексный анализ готового блюда ВкусВилл для планирования диеты v3."""
    meal_suitability: MealSuitability = Field(description="Рейтинги и обоснование пригодности к разным приемам пищи.")
    diet_goals: DietGoalRatings = Field(description="Рейтинги и обоснование пригодности для разных диетических целей.")
    meal_component_role: ComponentRole = Field(description="Основная роль продукта в сбалансированном приеме пищи.")
    satiety_index_estimate: EstimationLevel = Field(description="Оценка уровня сытости.")
    nutrient_density_estimate: EstimationLevel = Field(description="Оценка плотности нутриентов.")
    fiber_level_estimate: EstimationLevel = Field(description="Оценка уровня клетчатки.")
    sodium_level_estimate: EstimationLevel = Field(description="Оценка уровня натрия/соли.")
    likely_contains_added_sugar: bool = Field(description="Вероятно содержит ДОБАВЛЕННЫЕ сахара?")
    likely_contains_whole_grains: bool = Field(description="Вероятно содержит ЦЕЛЬНОЗЕРНОВЫЕ злаки?")
    health_benefit_tags: List[HealthBenefitTagEnum] = Field(description="Список тегов пользы (или пустой список []).") # Убрали default=[]
    preparation_complexity: PrepComplexity = Field(description="Сложность подготовки продукта пользователем.")
    cooking_method_guess: Optional[CookingMethodEnum] = Field(description="Предполагаемый метод приготовления (или null).") # Убрали default=None
    primary_flavor_profile: FlavorProfileEnum = Field(description="Основной вкусовой профиль.")
    primary_texture: TextureEnum = Field(description="Основная текстура.")
    pairing_suggestion: Optional[str] = Field(description="Краткое предложение по сочетанию (или null).") # Убрали default=None
    is_potential_source_of_calcium: bool = Field(description="Вероятно ЗНАЧИМЫЙ источник Кальция?") # Убрали default=False
    is_potential_source_of_iron: bool = Field(description="Вероятно ЗНАЧИМЫЙ источник Железа?") # Убрали default=False
    is_potential_source_of_potassium: bool = Field(description="Вероятно ЗНАЧИМЫЙ источник Калия?") # Убрали default=False
    is_potential_source_of_vitamin_c: bool = Field(description="Вероятно ЗНАЧИМЫЙ источник Витамина C?") # Убрали default=False
    is_potential_source_of_vitamin_d: bool = Field(description="Вероятно ЗНАЧИМЫЙ источник Витамина D?") # Убрали default=False
    is_potential_source_of_vitamin_b12: bool = Field(description="Вероятно ЗНАЧИМЫЙ источник Витамина B12?") # Убрали default=False
    micronutrient_comment: Optional[str] = Field(description="Опциональный комментарий по микронутриентам (или null).") # Убрали default=None

# --- Инициализация АСИНХРОННОГО клиента OpenAI ---
try:
    # Используем AsyncOpenAI для асинхронных вызовов
    async_client = AsyncOpenAI(api_key=api_key)
    st.sidebar.success("Клиент AsyncOpenAI инициализирован.") # Сообщение в sidebar
except Exception as e:
    st.error(f"Ошибка инициализации клиента AsyncOpenAI: {e}")
    async_client = None
    st.stop()

# --- Вспомогательная функция для проверки категории "Готовая еда" ---
def is_ready_meal_category(item_data: Dict[str, Any]) -> bool:
    """Проверяет, относится ли товар к категории 'Готовая еда' по breadcrumbs."""
    crumbs = item_data.get('breadcrumbs')
    if isinstance(crumbs, list):
        return any(cat in ["Готовая еда", "Готовая ета"] for cat in crumbs)
    return False

# --- АСИНХРОННАЯ Функция для вызова API ---
async def get_product_analysis_async(
    item_data: Dict[str, Any],
    semaphore: asyncio.Semaphore # Семафор для ограничения параллельных запросов
) -> tuple[str | None, dict | None]:
    """Асинхронно вызывает OpenAI API для получения комплексного анализа продукта."""
    # Получаем данные продукта из словаря
    product_name = item_data.get('name', 'Название не указано')
    category = item_data.get('category', 'Категория не указана')
    ingredients = item_data.get('ingredients')
    item_url = item_data.get('url') # URL важен для идентификации результата

    if not item_url: return None, None # Не можем обработать без URL

    if not isinstance(ingredients, str) or not ingredients:
         ingredients = "Состав не указан или некорректен"

    # Ограничиваем длину ингредиентов
    max_ingredient_length = 1000
    ingredients_truncated = ingredients[:max_ingredient_length] + ("..." if len(ingredients) > max_ingredient_length else "")

    # Обновленный системный промпт V4
    system_prompt_v4 = """
    Ты - эксперт-аналитик продуктов питания и зарегистрированный диетолог, специализирующийся на готовых блюдах русской кухни из ассортимента ВкусВилл.
    Твоя задача - предоставить комплексный и строго структурированный анализ продукта на основе его названия, категории и состава.
    Ты ДОЛЖЕН вернуть свой анализ в виде JSON объекта, точно соответствующего предоставленной схеме `ComprehensiveProductAnalysisV3`. Используй ТОЛЬКО предопределенные значения для полей с ограниченным выбором (Literal/Enum).

    Компоненты анализа (ЗАПОЛНИ ВСЕ ПОЛЯ):
    1.  `meal_suitability`: Рейтинги (1-5 целое) для Завтрака, Обеда, Ужина, Перекуса + краткое обоснование (`suitability_reasoning`).
    2.  `diet_goals`: Рейтинги (1-5 целое) для Похудения, Набора массы, Общего здоровья, Низкокалорийного перекуса + краткое обоснование (`goal_reasoning`).
    3.  `meal_component_role`: Выбери ОДНУ основную роль из списка: [Primary Protein Source, Primary Carb Source, Primary Fat Source, Vegetable/Fiber Source, Fruit/Dessert, Condiment/Sauce, Complete Meal, Snack, Drink, Other, Uncertain].
    4.  `satiety_index_estimate`: Выбери одно: Low, Medium, High, Uncertain.
    5.  `nutrient_density_estimate`: Выбери одно: Low, Medium, High, Uncertain.
    6.  `fiber_level_estimate`: Выбери одно: Low, Medium, High, Uncertain.
    7.  `sodium_level_estimate`: Выбери одно: Low, Medium, High, Uncertain.
    8.  `likely_contains_added_sugar`: True или False.
    9.  `likely_contains_whole_grains`: True или False.
    10. `health_benefit_tags`: Выбери релевантные теги из СПИСКА: ["Probiotic Source", "Prebiotic Source", "Antioxidant Rich", "Omega-3 Source", "High Fiber", "Good Source of Protein", "Low Glycemic Index (Estimate)", "Hydrating", "Source of Calcium", "Source of Iron", "Source of Potassium", "Source of Vitamin C", "Source of Vitamin D", "Source of B12"]. Выбирай ТОЛЬКО если явно поддерживается ключевыми ингредиентами. Будь консервативен. Верни пустой список `[]`, если не уверен или ничего не подходит.
    11. `preparation_complexity`: Выбери одно: Ready-to-Eat, Requires Heating, Minimal Prep, Requires Cooking, Uncertain.
    12. `cooking_method_guess`: Выбери одно из списка: [Fried, Baked, Steamed, Grilled, Boiled, Stewed, Roasted, Raw/Salad, Microwaved, Sous-Vide, Smoked, Other, N/A] или используй `null`.
    13. `primary_flavor_profile`: Выбери одно из списка: [Sweet, Savory, Spicy, Sour, Umami, Bitter, Balanced, Neutral, Other/Mixed, Uncertain].
    14. `primary_texture`: Выбери одно из списка: [Creamy, Crunchy, Chewy, Soft, Liquid, Crispy, Firm, Tender, Mixed, Other, Uncertain].
    15. `pairing_suggestion`: Краткий текст предложения по сочетанию или `null`.
    16. **Индикаторы Микронутриентов:** Для КАЖДОГО поля `is_potential_source_of_...` установи True или False, основываясь на наличии ЗНАЧИМЫХ источников в составе. Будь консервативен (False при сомнениях). Добавь комментарий в `micronutrient_comment` при необходимости (или `null`).

    Основывай анализ ТОЛЬКО на предоставленных данных и общих знаниях. Строго следуй схеме и допустимым значениям!
    """
    user_prompt = f"""
    Информация о продукте (Готовая еда из ВкусВилл):
    Название: {product_name}
    Категория: {category}
    Состав: {ingredients_truncated}

    Пожалуйста, предоставь комплексный анализ этого продукта согласно схеме ComprehensiveProductAnalysisV3.
    """

    retries = 0
    delay = INITIAL_DELAY
    # Используем семафор для ограничения параллелизма
    async with semaphore:
        while retries < MAX_RETRIES:
            try:
                # Используем асинхронный клиент и await
                response = await async_client.responses.parse(
                    model=MODEL_NAME,
                    input=[
                        {"role": "system", "content": system_prompt_v4},
                        {"role": "user", "content": user_prompt},
                    ],
                    # Передаем Pydantic модель V3
                    text_format=ComprehensiveProductAnalysisV3,
                )
                parsed_data = response.output_parsed
                if parsed_data:
                    return item_url, parsed_data.model_dump() # Возвращаем URL и результат
                else:
                    # Логируем предупреждение (но не в Streamlit напрямую из async функции)
                    print(f"Предупреждение [Attempt {retries+1}/{MAX_RETRIES}]: API вернул пустой результат для '{product_name}' ({item_url})")

            except RateLimitError as e:
                print(f"Предупреждение [Attempt {retries+1}/{MAX_RETRIES}]: RateLimitError для '{product_name}' ({item_url}). Повтор через {delay} сек. {e}")
            except APIError as e:
                 # Проверяем статус-код 400 на случай невалидной схемы, хотя parse должен это ловить
                 if e.status_code == 400:
                     print(f"Ошибка: API вернул 400 Bad Request для '{product_name}' ({item_url}). Вероятно, проблема со схемой или промптом. Пропускаем. {e}")
                     return item_url, None # Не повторяем при 400
                 else:
                     print(f"Предупреждение [Attempt {retries+1}/{MAX_RETRIES}]: APIError ({e.status_code}) для '{product_name}' ({item_url}). Повтор через {delay} сек. {e}")
            except ValidationError as e:
                print(f"Ошибка: ValidationError Pydantic для '{product_name}' ({item_url}). Модель вернула неверный формат. Пропускаем. {e}")
                return item_url, None # Не повторяем
            except Exception as e:
                print(f"Предупреждение [Attempt {retries+1}/{MAX_RETRIES}]: Неизвестная ошибка для '{product_name}' ({item_url}). Повтор через {delay} сек. {e}")

            # Ожидание перед повтором (используем asyncio.sleep)
            await asyncio.sleep(delay)
            retries += 1
            delay *= 2 # Экспоненциальная задержка

    print(f"Ошибка: Не удалось получить данные для '{product_name}' ({item_url}) после {MAX_RETRIES} попыток.")
    return item_url, None # Возвращаем URL и None при неудаче

# --- Основная АСИНХРОННАЯ функция обогащения ---
async def enrich_dataset_async():
    """Асинхронно обрабатывает датасет, вызывая API параллельно."""
    st.title("🧬 Обогащение данных VkusVill: Полный Анализ")

    if not async_client:
         st.error("Клиент AsyncOpenAI не инициализирован. Выполнение невозможно.")
         return

    # --- Загрузка данных ---
    st.info(f"Чтение исходного файла: {INPUT_JSONL_FILE}")
    try:
        # Кэшируем чтение файла
        @st.cache_data
        def read_all_items(file_path):
             if not os.path.exists(file_path): return None # Проверка внутри кэшируемой функции
             with open(file_path, 'r', encoding='utf-8') as f_in:
                  # Добавляем обработку ошибок JSON при чтении
                  items = []
                  for line_num, line in enumerate(f_in):
                       if line.strip():
                            try:
                                 items.append(json.loads(line))
                            except json.JSONDecodeError:
                                 st.warning(f"Ошибка JSON в строке {line_num + 1} файла {file_path}. Строка пропущена.")
                  return items

        all_items = read_all_items(INPUT_JSONL_FILE)

        if all_items is None:
             st.error(f"Не удалось прочитать исходный файл {INPUT_JSONL_FILE} или он не найден.")
             return
        st.success(f"Прочитано {len(all_items)} товаров.")
    except Exception as e:
        st.error(f"Критическая ошибка при чтении файла {INPUT_JSONL_FILE}: {e}")
        return

    # --- Фильтрация "Готовой еды" ---
    ready_meal_items = [item for item in all_items if is_ready_meal_category(item) and item.get('url')]
    st.info(f"Найдено {len(ready_meal_items)} товаров 'Готовой еды' с URL для обработки.")

    if not ready_meal_items:
         st.warning("Не найдено подходящих товаров 'Готовой еды' для обогащения.")
         # Записываем пустой или исходный файл? Пока просто выходим.
         # Можно записать исходный файл в output, чтобы было что использовать.
         try:
             with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as outfile:
                 for item in all_items:
                     outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
             st.info(f"Исходные данные скопированы в {OUTPUT_JSONL_FILE}, так как нечего было обогащать.")
         except Exception as e:
             st.error(f"Ошибка при копировании исходных данных в {OUTPUT_JSONL_FILE}: {e}")
         return

    # --- Подготовка к параллельным запросам ---
    enriched_data_map = {} # Результаты [url -> analysis_dict]
    total_to_process = len(ready_meal_items)
    processed_count = 0
    error_count = 0

    # Семафор для ограничения одновременных запросов
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Создаем задачи asyncio
    tasks = [
        asyncio.create_task(
            get_product_analysis_async(item, semaphore),
            name=f"Analyze_{item.get('url', i)}" # Даем имя задаче для отладки
        )
        for i, item in enumerate(ready_meal_items)
    ]

    # --- Отображение в Streamlit ---
    st.markdown("---")
    st.subheader("🚀 Выполнение запросов к OpenAI API (Параллельно)")
    progress_bar = st.progress(0.0, text="Начинаем обработку...")
    log_container = st.container(height=450) # Увеличил высоту
    log_container.info(f"Запускаем {total_to_process} задач с ограничением {MAX_CONCURRENT_REQUESTS} одновременных запросов...")

    # --- Асинхронное выполнение и обработка результатов ---
    for future in asyncio.as_completed(tasks):
        try:
            # Ожидаем результат выполнения задачи
            item_url, analysis_result = await future
            processed_count += 1

            if item_url and analysis_result:
                enriched_data_map[item_url] = analysis_result
                product_name = next((item['name'] for item in ready_meal_items if item['url'] == item_url), 'Неизвестный продукт') # Найдем имя по URL
                # Выводим только часть информации в лог, чтобы не перегружать
                log_message = f"""
                **✅ {processed_count}/{total_to_process}. {product_name}**
                * **Роль:** {analysis_result.get('meal_component_role', '?')}
                * **Сытость:** {analysis_result.get('satiety_index_estimate', '?')} | **Плотн.нутр:** {analysis_result.get('nutrient_density_estimate', '?')}
                * **Сахар:** {analysis_result.get('likely_contains_added_sugar', '?')} | **Злаки:** {analysis_result.get('likely_contains_whole_grains', '?')}
                * **Рейтинг (З/О/У/П):** {analysis_result.get('meal_suitability',{}).get('breakfast_rating','?')}/{analysis_result.get('meal_suitability',{}).get('lunch_rating','?')}/{analysis_result.get('meal_suitability',{}).get('dinner_rating','?')}/{analysis_result.get('meal_suitability',{}).get('snack_rating','?')}
                """
                log_container.markdown(log_message)
                log_container.caption(f"Обоснование: {analysis_result.get('meal_suitability', {}).get('suitability_reasoning', 'N/A')[:100]}...") # Краткое обоснование
                log_container.markdown("---")

            elif item_url: # Если был URL, но результат None (ошибка произошла внутри get_product_analysis_async)
                error_count += 1
                product_name = next((item['name'] for item in ready_meal_items if item['url'] == item_url), 'Неизвестный продукт')
                log_container.error(f"❌ Ошибка обработки для: {product_name} ({item_url}) - см. консоль/логи для деталей.")
                log_container.markdown("---")
            # else: # Случай если URL был None (уже отфильтрован)

        except Exception as e_future:
            # Ловим ошибки, которые могли произойти при ожидании future
            processed_count += 1 # Считаем как обработанную с ошибкой
            error_count += 1
            task_name = future.get_name() if hasattr(future, 'get_name') else f"Задача {processed_count}"
            log_container.error(f"❌ Критическая ошибка при выполнении задачи '{task_name}': {e_future}")
            log_container.markdown("---")


        # Обновляем прогресс-бар
        progress_value = processed_count / total_to_process
        progress_bar.progress(progress_value, text=f"Обработано: {processed_count}/{total_to_process} (Ошибок: {error_count})")

    # --- Завершение ---
    st.success(f"API обработка завершена. Успешно получены данные для {len(enriched_data_map)} из {total_to_process} продуктов. Ошибок: {error_count}.")

    # --- Запись результата в новый файл ---
    st.info(f"Запись обогащенных данных в файл: {OUTPUT_JSONL_FILE}")
    items_written = 0
    enriched_saved_count = 0
    try:
        with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as outfile:
            for item in all_items: # Итерируем по ВСЕМ исходным товарам
                item_url = item.get('url')
                # Если для этого URL есть обогащенные данные, добавляем их
                if item_url in enriched_data_map:
                    # Добавляем весь блок анализа под ключом 'product_analysis_v3'
                    item['product_analysis_v3'] = enriched_data_map[item_url]
                    enriched_saved_count += 1
                # Записываем строку (оригинальную или обогащенную)
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                items_written += 1
        st.success(f"Успешно записано {items_written} товаров в {OUTPUT_JSONL_FILE}. Из них {enriched_saved_count} обогащено новым анализом.")
    except Exception as e:
        st.error(f"Ошибка при записи в файл {OUTPUT_JSONL_FILE}: {e}")

# --- Запуск асинхронной функции ---
# --- Запуск асинхронной функции ---
if __name__ == "__main__":
    # Используем asyncio.run() для запуска главной асинхронной функции
    # из синхронного контекста __main__
    try:
        # nest_asyncio позволяет запускать новый event loop даже если он уже есть (как в Streamlit)
        asyncio.run(enrich_dataset_async())
    except Exception as main_e:
        # Ловим и отображаем любые ошибки, возникшие при запуске
        st.error(f"Произошла критическая ошибка во время выполнения: {main_e}")
        # Выводим полный traceback в Streamlit для отладки
        import traceback
        st.code(traceback.format_exc())