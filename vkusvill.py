import asyncio
import json
import logging
import re
import time
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
import os

# --- Конфигурация ---
CATEGORY_URLS = [
    # "https://vkusvill.ru/goods/set-na-uzhin/", # Пропускаем, т.к. вызывает таймаут
    "https://vkusvill.ru/goods/khity/",
    "https://vkusvill.ru/goods/gotovaya-eda/",
    "https://vkusvill.ru/goods/sladosti-i-deserty/",
    "https://vkusvill.ru/goods/ovoshchi-frukty-yagody-zelen/",
    "https://vkusvill.ru/goods/vvstrechaem-vesnu/",
    "https://vkusvill.ru/goods/khleb-i-vypechka/",
    "https://vkusvill.ru/goods/molochnye-produkty-yaytso/",
    "https://vkusvill.ru/goods/myaso-ptitsa/",
    "https://vkusvill.ru/goods/ryba-ikra-i-moreprodukty/",
    "https://vkusvill.ru/goods/kolbasa-sosiski-delikatesy/",
    "https://vkusvill.ru/goods/zamorozhennye-produkty/",
    "https://vkusvill.ru/goods/syry/",
    "https://vkusvill.ru/goods/napitki/",
    "https://vkusvill.ru/goods/orekhi-chipsy-i-sneki/",
    "https://vkusvill.ru/goods/krupy-makarony-muka/",
    "https://vkusvill.ru/goods/vegetarianskoe-i-postnoe/",
    "https://vkusvill.ru/goods/osoboe-pitanie/",
    "https://vkusvill.ru/goods/konservatsiya/",
    "https://vkusvill.ru/goods/chay-i-kofe/syry",
    "https://vkusvill.ru/goods/masla-sousy-spetsii-sakhar-i-sol/",
]

# ЗАМЕНИТЕ НА ВАШ USER AGENT
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36" # Пример! Замените!

OUTPUT_FILE = "vkusvill_data.jsonl"
LOG_FILE = "scraper.log"
ERROR_LOG_FILE = "error_urls.log"

DELAY_PRODUCT = 0.3 # Краткая пауза между *запусками* задач парсинга товаров
DELAY_PAGINATION = 1 # Уменьшаем, т.к. основное время - парсинг товаров
CONCURRENT_PRODUCT_SCRAPES = 5 # Количество одновременно парсящихся товаров

SELECTORS = {
    "category_product_link": 'div.ProductCards__item a.ProductCard__link',
    "next_page_button": '.VV_Pager a.VV_Pager__Item:last-child:not(._current)',
    "product_name": 'h1.Product__title',
    "product_image": 'meta[itemprop="image"]',
    "product_weight": 'div.ProductCard__weight',
    "product_price": 'div[itemprop="offers"] span.Price--lg',
    "product_rating": '.Product__rating div.Rating__text',
    # КБЖУ Стандарт
    "nutrition_wrap": 'div.VV23_DetailProdPageAccordion__EnergyWrap',
    "nutrition_basis": 'h4.VV23_DetailProdPageInfoDescItem__Title',
    "nutrition_items_container": 'div.VV23_DetailProdPageAccordion__Energy',
    "nutrition_item": '.VV23_DetailProdPageAccordion__EnergyItem',
    "nutrition_value": 'div.VV23_DetailProdPageAccordion__EnergyValue',
    "nutrition_desc": 'div.VV23_DetailProdPageAccordion__EnergyDesc',
    # КБЖУ Альтернатива
    "alt_nutrition_parent_selector": 'div.VV23_DetailProdPageInfoDescItem',
    "alt_nutrition_h4_selector": 'h4.VV23_DetailProdPageInfoDescItem__Title',
    "alt_nutrition_desc_selector": 'div.VV23_DetailProdPageInfoDescItem__Desc',
    # Состав
    "ingredients": 'div.VV23_DetailProdPageInfoDescItem__Desc._sostav',
    "ingredients_more_button": 'button.js-vv-text-cut-showmore',
    # Хлебные крошки
    "breadcrumbs_container": '#js-nav-chain .Breadcrumbs',
    "breadcrumbs_name_span": 'span[itemprop="name"]'
}

# Паттерны Regex для альтернативного КБЖУ
NUTRITION_REGEX = {
    "protein": re.compile(r'белки\s+([0-9.,]+)\s*г', re.IGNORECASE),
    "fat": re.compile(r'жиры\s+([0-9.,]+)\s*г', re.IGNORECASE),
    "carbs": re.compile(r'углеводы\s+([0-9.,]+)\s*г', re.IGNORECASE),
    "kcal": re.compile(r'([0-9.,]+)\s*(?:ккал|кал)', re.IGNORECASE) # Non-capturing group
}

# --- Настройка Логгирования ---
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'),
                              logging.StreamHandler()])
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)

# --- Вспомогательные функции ---
def parse_weight_volume(text):
    if not text: return None, None
    match = re.search(r'([0-9.,]+)\s*([а-яА-Яa-zA-Z]+)', text)
    if match:
        try:
            value = float(match.group(1).replace(',', '.'))
            unit = match.group(2).strip()
            return value, unit
        except ValueError: return None, None
    return None, None

def parse_price(text):
    if not text: return None
    match = re.search(r'([0-9]+(?:[.,][0-9]+)?)', text)
    if match:
        try: return float(match.group(1).replace(',', '.'))
        except ValueError: return None
    return None

def extract_category_name(url):
    try:
        path = url.strip('/').split('/')[-1]
        return path if path else "unknown"
    except Exception: return "unknown"

# --- Основные функции скрапинга ---
async def scrape_product_page(context, product_url, category_name):
    """Скрапит данные с одной страницы товара, создает свою страницу."""
    logging.info(f"Парсим товар: {product_url}")
    product_data = {
        "name": None, "url": product_url, "category": category_name, "breadcrumbs": None,
        "image_url": None, "weight_value": None, "weight_unit": None, "price": None,
        "rating": None, "ingredients": None, "nutrition": None, "nutrition_text": None
    }
    page = None # Инициализируем page как None
    try:
        # Создаем новую страницу для каждого товара для изоляции
        page = await context.new_page()
        await page.set_extra_http_headers({"User-Agent": USER_AGENT}) # Устанавливаем User-Agent для новой страницы
        await page.goto(product_url, wait_until='domcontentloaded', timeout=30000)
        await page.wait_for_timeout(500)

        # --- Базовая информация ---
        name_loc = page.locator(SELECTORS["product_name"])
        if await name_loc.count() > 0:
            product_data["name"] = (await name_loc.first.text_content(timeout=5000) or "").strip()

        img_loc = page.locator(SELECTORS["product_image"])
        if await img_loc.count() > 0:
             product_data["image_url"] = await img_loc.first.get_attribute('content', timeout=5000)

        weight_loc = page.locator(SELECTORS["product_weight"])
        if await weight_loc.count() > 0:
            weight_text = await weight_loc.first.text_content(timeout=5000)
            product_data["weight_value"], product_data["weight_unit"] = parse_weight_volume(weight_text)

        price_loc = page.locator(SELECTORS["product_price"])
        if await price_loc.count() > 0:
             price_text = await price_loc.first.text_content(timeout=5000)
             product_data["price"] = parse_price(price_text)

        rating_loc = page.locator(SELECTORS["product_rating"])
        if await rating_loc.count() > 0:
            rating_text = await rating_loc.first.text_content(timeout=5000)
            if rating_text and re.search(r'[0-9.,]+', rating_text): # Улучшенная проверка на число
                 product_data["rating"] = rating_text.strip()

        # --- Хлебные крошки ---
        breadcrumbs_container = page.locator(SELECTORS["breadcrumbs_container"])
        if await breadcrumbs_container.count() > 0:
            name_spans = await breadcrumbs_container.locator(SELECTORS["breadcrumbs_name_span"]).all()
            # Используем list comprehension для краткости
            product_data["breadcrumbs"] = [text for span in name_spans if (text := (await span.text_content() or "").strip())]
            logging.debug(f"Спарсены хлебные крошки: {product_data['breadcrumbs']}")
        else:
            logging.warning(f"Не найден контейнер хлебных крошек для {product_url}")

        # --- Состав ---
        ingredients_loc = page.locator(SELECTORS["ingredients"])
        if await ingredients_loc.count() > 0:
             logging.debug(f"Найден блок состава для {product_url}")
             more_button = ingredients_loc.locator(SELECTORS["ingredients_more_button"])
             is_visible = False
             try: is_visible = await more_button.is_visible(timeout=1000)
             except PlaywrightTimeoutError: is_visible = False

             if is_visible:
                 logging.debug(f"Кликаем 'еще' для состава на {product_url}")
                 try:
                     await more_button.click(timeout=2000)
                     await page.wait_for_timeout(200)
                 except (PlaywrightTimeoutError, PlaywrightError) as e:
                     logging.warning(f"Не удалось кликнуть 'еще' для состава на {product_url}: {e}")
             else: logging.debug(f"Кнопка 'еще' для состава не видима/отсутствует на {product_url}")

             ing_text = await ingredients_loc.text_content(timeout=5000)
             product_data["ingredients"] = ing_text.strip() if ing_text else None
             logging.debug(f"Текст состава получен для {product_url}")
        else:
            logging.warning(f"Блок состава не найден для {product_url}")

        # --- КБЖУ ---
        parsed_nutrition = False
        nutrition_wrap = page.locator(SELECTORS["nutrition_wrap"]) # Стандартный блок
        if await nutrition_wrap.count() > 0:
            logging.debug(f"Найден СТАНДАРТНЫЙ блок КБЖУ для {product_url}")
            # (Логика парсинга стандартного блока остается той же)
            nutrition_data = {}
            basis_element = nutrition_wrap.locator(SELECTORS["nutrition_basis"])
            basis_text_std = await basis_element.first.text_content(timeout=5000) if await basis_element.count() > 0 else None
            nutrition_data["basis"] = basis_text_std.strip() if basis_text_std else "на 100 г" # Дефолт, если не найден

            items_container = nutrition_wrap.locator(SELECTORS["nutrition_items_container"])
            if await items_container.count() > 0:
                nutrition_items = await items_container.locator(SELECTORS["nutrition_item"]).all()
                for item in nutrition_items:
                    value_el = item.locator(SELECTORS["nutrition_value"])
                    desc_el = item.locator(SELECTORS["nutrition_desc"])
                    value_text = await value_el.first.text_content(timeout=1000) if await value_el.count() > 0 else None
                    desc_text = await desc_el.first.text_content(timeout=1000) if await desc_el.count() > 0 else None

                    if value_text and desc_text:
                        value = None
                        try: value = float(value_text.strip().replace(',', '.'))
                        except ValueError: pass
                        key_raw = desc_text.lower().replace(', г', '').strip()
                        key = {'ккал': 'kcal', 'белки': 'protein', 'жиры': 'fat', 'углеводы': 'carbs'}.get(key_raw, f"unknown_{key_raw}")
                        nutrition_data[key] = value
                if len(nutrition_data) > 1:
                    product_data["nutrition"] = nutrition_data
                    parsed_nutrition = True

        if not parsed_nutrition:
            logging.debug(f"Стандартный блок КБЖУ не найден/не распарсен, ищем альтернативный текст для {product_url}")
            alt_nutrition_text = None
            alt_basis = "на 100 г"
            # Ищем H4 по тексту, затем берем текст из соседнего div
            parent_divs = await page.locator(SELECTORS["alt_nutrition_parent_selector"]).all()
            target_desc_div = None
            for parent_div in parent_divs:
                h4 = parent_div.locator(SELECTORS["alt_nutrition_h4_selector"])
                if await h4.count() > 0:
                     h4_text_content = await h4.first.text_content(timeout=1000)
                     if h4_text_content and 'Пищевая и энергетическая ценность' in h4_text_content.replace('\xa0', ' '):
                         desc_div = parent_div.locator(SELECTORS["alt_nutrition_desc_selector"])
                         if await desc_div.count() > 0:
                             target_desc_div = desc_div.first # Нашли нужный div
                             # Пытаемся извлечь основу
                             basis_match = re.search(r'(?:на|в)\s+([0-9.,]+\s*[а-яА-Яa-zA-Z]+)\.?$', h4_text_content.replace('\xa0', ' '), re.IGNORECASE)
                             if basis_match and basis_match.group(1): alt_basis = basis_match.group(1).strip()
                             break # Выходим, нашли что искали

            if target_desc_div:
                alt_nutrition_text = await target_desc_div.text_content(timeout=1000)
                logging.debug(f"Найден альтернативный текст КБЖУ (основа: {alt_basis}): \"{alt_nutrition_text[:100]}...\"")
                nutrition_data_alt = {"basis": alt_basis}
                found_any_alt = False
                for key, regex_pattern in NUTRITION_REGEX.items():
                    match = regex_pattern.search(alt_nutrition_text)
                    if match and match.group(1):
                        try:
                            value = float(match.group(1).replace(',', '.'))
                            if not isinstance(value, (int, float)): continue
                            nutrition_data_alt[key] = value
                            found_any_alt = True
                            logging.debug(f"Спарсен КБЖУ (regex): {key} = {value}")
                        except ValueError:
                             logging.warning(f"Не удалось извлечь число для {key} из текста: {match.group(1)}")
                    else:
                         logging.debug(f"Не найдено значение (regex) для: {key}")

                if found_any_alt:
                    product_data["nutrition"] = nutrition_data_alt
                    parsed_nutrition = True # Отмечаем, что успешно распарсили
                else:
                    logging.warning(f"Не удалось извлечь КБЖУ из альтернативного текста для {product_url}. Сохраняем как текст.")
                    product_data["nutrition_text"] = alt_nutrition_text.strip() if alt_nutrition_text else None
            else:
                 logging.debug(f"Альтернативный блок КБЖУ не найден для {product_url}")

        # --- Финальная очистка ---
        product_data_clean = {k: v for k, v in product_data.items() if v is not None and v != ""}
        logging.info(f"Успешно спарсен: {product_data_clean.get('name', product_url)}")
        return product_data_clean

    except PlaywrightTimeoutError as e:
        logging.error(f"Таймаут при парсинге {product_url}: {e}")
        error_logger.error(f"TIMEOUT: {product_url} - {e}")
        return None
    except Exception as e:
        logging.exception(f"Неизвестная ошибка при парсинге {product_url}")
        error_logger.error(f"ERROR: {product_url} - {e}")
        return None
    finally:
        if page: # Закрываем страницу товара, если она была создана
             try: await page.close()
             except Exception as close_e: logging.error(f"Ошибка закрытия страницы товара {product_url}: {close_e}")

async def worker(url, category_name, context, semaphore):
    """Обертка для запуска scrape_product_page с семафором."""
    async with semaphore: # Ожидаем свободный слот
        logging.debug(f"Worker запущен для {url}")
        data = await scrape_product_page(context, url, category_name)
        logging.debug(f"Worker завершен для {url}")
        return data

async def scrape_category(context, category_url):
    category_name = extract_category_name(category_url)
    logging.info(f"Начинаем парсинг категории: {category_name} ({category_url})")
    page = None # Страница для навигации по категориям
    scraped_count_in_category = 0
    current_page_num = 1
    current_url = category_url
    semaphore = asyncio.Semaphore(CONCURRENT_PRODUCT_SCRAPES) # Ограничитель параллельных задач

    try:
        page = await context.new_page()

        while True:
            logging.info(f"Загружаем страницу {current_page_num} категории {category_name}: {current_url}")
            try:
                logging.debug(f"Выполняем page.goto для {current_url}")
                await page.goto(current_url, wait_until='domcontentloaded', timeout=45000)
                logging.debug(f"Страница {current_url} загружена (domcontentloaded).")
                logging.debug(f"Ожидаем первый элемент по селектору: {SELECTORS['category_product_link']} (таймаут 30с)")
                await page.locator(SELECTORS["category_product_link"]).first.wait_for(timeout=30000)
                logging.info(f"Карточки товаров найдены на стр {current_page_num}.")
            except PlaywrightTimeoutError:
                logging.warning(f"Не найдены карточки товаров (или таймаут ожидания) на стр {current_page_num} категории {category_name}: {current_url}")
                break
            except Exception as e:
                 logging.error(f"Ошибка загрузки страницы {current_page_num} категории {category_name}: {e}")
                 break

            product_links_locators = await page.locator(SELECTORS["category_product_link"]).all()
            logging.debug(f"Найдено локаторов товаров: {len(product_links_locators)}")

            page_product_urls = []
            for link_locator in product_links_locators:
                 href = await link_locator.get_attribute('href')
                 if href:
                     full_url = f"https://vkusvill.ru{href}" if href.startswith('/') else href
                     if full_url.startswith('http'): page_product_urls.append(full_url)
                 else: logging.warning("Найден локатор ссылки без href атрибута.")

            logging.info(f"Найдено {len(page_product_urls)} валидных URL товаров на странице {current_page_num}.")

            if not page_product_urls:
                logging.warning(f"Ссылки на товары не найдены на стр {current_page_num} категории {category_name}, возможно конец.")
                break

            # Запуск парсинга товаров параллельно
            tasks = [worker(url, category_name, context, semaphore) for url in page_product_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Запись результатов после завершения всех задач для данной страницы
            successful_results_count = 0
            logging.debug(f"Завершено {len(results)} задач парсинга для страницы {current_page_num}.")
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    original_url = page_product_urls[i] # Получаем URL для логирования ошибок
                    if isinstance(result, Exception):
                        logging.error(f"Задача парсинга товара {original_url} завершилась с ошибкой: {result}")
                        error_logger.error(f"TASK_ERROR: {original_url} - {result}")
                    elif result: # Успешный парсинг
                        try:
                            json_string = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
                            f.write(json_string + '\n')
                            successful_results_count += 1
                        except Exception as e:
                             logging.error(f"Ошибка записи в файл {OUTPUT_FILE} для товара {original_url}: {e}")
                    # Если result is None, значит была ошибка внутри scrape_product_page, она уже залогирована

            logging.info(f"Успешно записано {successful_results_count} товаров со страницы {current_page_num}.")
            scraped_count_in_category += successful_results_count

            # Проверка наличия кнопки "Следующая страница"
            next_button = page.locator(f'{SELECTORS["next_page_button"]}')
            logging.debug(f"Ищем кнопку 'Вперед' по селектору: {SELECTORS['next_page_button']}")
            is_next_visible = False
            try: is_next_visible = await next_button.is_visible(timeout=5000)
            except PlaywrightTimeoutError: logging.debug("Кнопка 'Вперед' не найдена по таймауту.")

            if is_next_visible:
                 logging.debug("Кнопка 'Вперед' найдена и видима.")
                 next_page_href = await next_button.first.get_attribute('href')
                 if next_page_href:
                     next_url = f"https://vkusvill.ru{next_page_href}" if next_page_href.startswith('/') else next_page_href
                     if page.url == next_url: # Исправлено: page.url без скобок
                          logging.warning(f"URL следующей страницы ({next_url}) совпадает с текущим. Завершаем категорию {category_name}.")
                          break
                     current_url = next_url
                     current_page_num += 1
                     logging.info(f"Переход на следующую страницу: {current_url}")
                     await asyncio.sleep(DELAY_PAGINATION) # Задержка перед загрузкой след. страницы
                 else:
                     logging.info(f"Не удалось получить href кнопки 'Вперед' на стр {current_page_num}. Завершаем категорию {category_name}.")
                     break
            else:
                logging.info(f"Кнопка 'Вперед' не найдена или не видима. Завершаем категорию {category_name}.")
                break

    except Exception as e:
        logging.exception(f"Критическая ошибка в категории {category_name}")
    finally:
        if page: await page.close()
        # product_page закрывается внутри scrape_product_page (если используется async with)
        # или должна была быть закрыта в scrape_category, если бы создавалась там один раз.
        # В текущей реализации она создается и закрывается в worker/scrape_product_page
        logging.info(f"Завершена категория: {category_name}. Спарсено товаров: {scraped_count_in_category}")
    return scraped_count_in_category


async def main():
    start_time = time.time()
    logging.info("Запуск скрапера ВкусВилл...")
    total_scraped = 0

    try:
        for file_path in [LOG_FILE, ERROR_LOG_FILE, OUTPUT_FILE]:
             dir_name = os.path.dirname(os.path.abspath(file_path))
             if dir_name: os.makedirs(dir_name, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f: pass
        with open(ERROR_LOG_FILE, 'w') as f: pass
        with open(LOG_FILE, 'w') as f: pass

        root_logger = logging.getLogger()
        for hdlr in root_logger.handlers[:]:
             if isinstance(hdlr, logging.FileHandler) and hdlr.baseFilename == os.path.abspath(LOG_FILE):
                  root_logger.removeHandler(hdlr)
                  hdlr.close()
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)
        logging.getLogger('error_logger').handlers[0].setFormatter(formatter)
        logging.info("Логгеры настроены.")

    except Exception as e:
        print(f"Критическая ошибка: Не удалось настроить файлы логов/вывода: {e}")
        logging.error(f"Не удалось настроить файлы логов/вывода: {e}")
        return

    async with async_playwright() as p:
        browser = None
        try:
            # Установите headless=False, если хотите видеть окно браузера для отладки
            logging.info("Запускаем браузер (headless=True)...")
            browser = await p.chromium.launch(headless=True)
            logging.info("Браузер запущен.")
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={'width': 1280, 'height': 1024}
            )
            logging.info("Контекст браузера создан.")
            context.set_default_navigation_timeout(60000)
            context.set_default_timeout(35000)
            logging.info("Таймауты установлены.")

            for category_url in CATEGORY_URLS:
                count = await scrape_category(context, category_url)
                total_scraped += count
                logging.info(f"--- Пауза 5 секунд перед следующей категорией ---")
                await asyncio.sleep(5)

        except Exception as e:
             logging.exception("Произошла ошибка на уровне запуска браузера или основного цикла")
        finally:
             if browser:
                 await browser.close()
                 logging.info("Браузер закрыт.")

    end_time = time.time()
    logging.info(f"Скрапинг завершен. Всего обработано товаров: {total_scraped}")
    logging.info(f"Затрачено времени: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    logging.info(f"Данные сохранены в: {OUTPUT_FILE}")
    logging.info(f"Ошибки записаны в: {ERROR_LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())