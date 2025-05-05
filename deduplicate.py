import json
import os
from tqdm import tqdm # Убедитесь, что установлен: pip install tqdm

# --- Настройки ---
INPUT_JSONL_FILE = 'vkusvill_data.jsonl'        # Исходный файл с дубликатами
OUTPUT_JSONL_FILE = 'vkusvill_data_deduplicated.jsonl' # Файл без дубликатов по имени
NAME_FIELD = 'name'                             # Поле для проверки уникальности

# --- Вспомогательная функция для подсчета строк ---
def count_lines(filepath):
    """Эффективно считает строки в файле."""
    try:
        with open(filepath, 'rb') as f: # Читаем в бинарном режиме для скорости
            lines = 0
            buf_size = 1024 * 1024 # Читаем по 1MB
            read_f = f.raw.read
            buf = read_f(buf_size)
            while buf:
                lines += buf.count(b'\n') # Считаем байты новой строки
                buf = read_f(buf_size)
            # Добавляем 1, если файл не пустой и не заканчивается на \n
            # Проще вернуть lines + 1, если файл не пустой, или 0 если пустой
            return lines + 1 if os.path.getsize(filepath) > 0 else 0
    except Exception as e:
        print(f"Ошибка при подсчете строк: {e}")
        return 0

# --- Основная логика дедупликации ---
def deduplicate_jsonl_by_name(input_path, output_path, name_key):
    """
    Читает JSONL файл, удаляет дубликаты по значению ключа 'name_key',
    сохраняя первое вхождение, и записывает результат в новый файл.
    """
    if not os.path.exists(input_path):
        print(f"Ошибка: Исходный файл не найден: {input_path}")
        return

    seen_names = set()
    lines_read = 0
    lines_written = 0
    lines_skipped_duplicate = 0
    lines_skipped_no_name = 0
    lines_skipped_bad_json = 0

    print(f"Начинаю дедупликацию файла: {input_path}")
    print(f"Уникальность проверяется по полю: '{name_key}'")
    print(f"Результат будет сохранен в: {output_path}")

    # Считаем строки для TQDM
    print("Подсчет строк в исходном файле...")
    total_lines = count_lines(input_path)
    if total_lines == 0:
        print("Ошибка: Не удалось посчитать строки или файл пуст.")
        # Создадим пустой выходной файл на всякий случай
        open(output_path, 'w').close()
        return
    print(f"Найдено строк (приблизительно): {total_lines}")

    # Открываем файлы для чтения и записи
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile, \
             tqdm(total=total_lines, unit=' строк', desc="Обработка") as pbar: # Инициализируем TQDM с общим числом строк

            for line in infile:
                lines_read += 1
                pbar.update(1) # Обновляем прогресс-бар на 1 строку

                line_stripped = line.strip()
                if not line_stripped:
                    continue # Пропускаем пустые строки

                try:
                    data = json.loads(line_stripped)
                    name_value = data.get(name_key)

                    if name_value is None:
                        lines_skipped_no_name += 1
                        continue # Пропускаем строки без имени

                    name_value_str = str(name_value)

                    if name_value_str not in seen_names:
                        seen_names.add(name_value_str)
                        outfile.write(line) # Записываем оригинальную строку (с переводом строки)
                        lines_written += 1
                    else:
                        lines_skipped_duplicate += 1

                except json.JSONDecodeError:
                    lines_skipped_bad_json += 1
                    # Используем pbar.write для вывода сообщений без порчи прогресс-бара
                    pbar.write(f"Предупреждение: Строка {lines_read} содержит невалидный JSON. Пропущена: {line_stripped[:100]}...")
                except Exception as e:
                     lines_skipped_bad_json += 1
                     pbar.write(f"Предупреждение: Неизвестная ошибка при обработке строки {lines_read}: {e}. Строка пропущена.")

    except Exception as e:
        print(f"\nКритическая ошибка при работе с файлами: {e}")
        # Можно добавить удаление частично созданного файла при ошибке
        # if os.path.exists(output_path):
        #     try: os.remove(output_path)
        #     except OSError: pass
        return
    finally:
        # Убедимся, что pbar закрыт, если он был создан
        if 'pbar' in locals() and pbar:
            pbar.close()


    # Выводим статистику
    print("\n--- Статистика дедупликации ---")
    print(f"Прочитано строк: {lines_read}")
    print(f"Найдено уникальных имен (записано строк): {lines_written}")
    print(f"Пропущено строк (дубликаты по имени '{name_key}'): {lines_skipped_duplicate}")
    print(f"Пропущено строк (без поля '{name_key}' или null): {lines_skipped_no_name}")
    print(f"Пропущено строк (ошибка JSON): {lines_skipped_bad_json}")
    print(f"Результат сохранен в: {output_path}")

# --- Запуск функции ---
if __name__ == "__main__":
    deduplicate_jsonl_by_name(INPUT_JSONL_FILE, OUTPUT_JSONL_FILE, NAME_FIELD)