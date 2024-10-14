#bot.py

import os
import shutil
import tempfile
import re
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import uuid

from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from CRNN_model import recognize_text_with_crnn, load_crnn_model
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from db import search_by_type, search_by_mode, search_by_year, clean_input
from datetime import datetime


tf.get_logger().setLevel('ERROR')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

import threading
import asyncio
import logging

# Настраиваем глобальный логгер для вывода в консоль и файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Для вывода в консоль
        logging.FileHandler('global_bot_logs.log', mode='w')  # Глобальный лог-файл
    ]
)

# Загрузка переменных окружения
load_dotenv()
BOT_TOKEN = os.getenv('TOKEN')

if not BOT_TOKEN:
    raise ValueError("Не удалось загрузить токен. Проверьте переменные окружения.")

storage = MemoryStorage()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)
dp.middleware.setup(LoggingMiddleware())

user_df = {}

LOGO_PATH = 'D:/EnerTest/EnerTest/Tokarev_Aleksandr/TelegramBot/logo.png'


# Функция для настройки логирования в папке пользователя
def setup_user_logging(user_id):
    logger = logging.getLogger(f"user_{user_id}")

    # Проверяем и очищаем существующие обработчики
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    user_folder = f"./{user_id}"
    os.makedirs(user_folder, exist_ok=True)
    log_file_path = os.path.join(user_folder, "bot_logs.log")

    # Создаем обработчик для записи в файл
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)

    # Устанавливаем форматирование для файлового обработчика
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Добавляем файловый обработчик к логгеру
    logger.addHandler(file_handler)

    return logger, log_file_path


def copy_file_to_user_folder(user_id, img_name, file_path, new_file_name, logger):
    user_folder = f"./{user_id}"
    image_folder = os.path.join(user_folder, img_name)
    os.makedirs(image_folder, exist_ok=True)
    new_file_path = os.path.join(image_folder, new_file_name)
    shutil.copy(file_path, new_file_path)
    logger.info(f"Скопирован файл: {new_file_path}")
    return new_file_path


def delete_folder(folder_path, logger):
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                try:
                    file_path = os.path.join(root, name)
                    os.remove(file_path)
                    logger.info(f"Удален файл: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка при удалении файла '{file_path}': {e}")
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        try:
            os.rmdir(folder_path)
            logger.info(f"Папка '{folder_path}' успешно удалена.")
        except Exception as e:
            logger.error(f"Ошибка при удалении папки '{folder_path}': {e}")


def process_logo(input_path, output_path, final_size=(400, 150), logger=None):
    with Image.open(input_path) as img:
        background = Image.new('RGB', final_size, (255, 255, 255))
        img.thumbnail((final_size[0] - 20, final_size[1] - 20), Image.LANCZOS)
        logo_position = (
            (final_size[0] - img.width) // 2,
            (final_size[1] - img.height) // 2
        )
        background.paste(img, logo_position)
        background.save(output_path, 'PNG')
    if logger:
        logger.info(f"Логотип сохранен: {output_path}")


YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'D:/EnerTest/EnerTest/Tokarev_Aleksandr/TelegramBot/best.pt')
CRNN_MODEL_PATH = os.getenv('CRNN_MODEL_PATH', 'D:/EnerTest/EnerTest/Tokarev_Aleksandr/TelegramBot/best_model_cer.keras')

if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLO модель не найдена по пути: {YOLO_MODEL_PATH}")
if not os.path.exists(CRNN_MODEL_PATH):
    raise FileNotFoundError(f"CRNN модель не найдена по пути: {CRNN_MODEL_PATH}")

model = YOLO(YOLO_MODEL_PATH)
model_CRNN = load_crnn_model(CRNN_MODEL_PATH)

labels = ['year', 'type', 'mode', 'number', 'logo', 'amper', 'volt', '1phase', '3phase']


def extract_numbers_and_format(text, is_year=False, logger=None):
    if pd.isna(text):
        return text
    if not isinstance(text, str):
        text = str(text)
    numbers = ''.join(re.findall(r'\d+', text))
    if is_year and len(numbers) == 2:
        numbers = "20" + numbers
    if logger:
        logger.info(f"Извлечены числа: {numbers}")
    return int(numbers) if numbers.isdigit() else numbers


def save_cropped_image(image, points, output_path, logger):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    cropped_img = cv2.bitwise_and(image, mask)
    rect = cv2.boundingRect(points)
    cropped_img = cropped_img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    cv2.imwrite(output_path, cropped_img)
    logger.info(f"Сохранено вырезанное изображение: {output_path}")


AUTHORIZED_USERS = [123456789, 987654321]
user_states = {}


class Form(StatesGroup):
    year = State()
    type = State()
    modification = State()
    number = State()

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    logger, _ = setup_user_logging(user_id)
    logger.info(f"Пользователь {user_id} начал работу с ботом.")
    processed_logo_path = 'processed_logo.png'
    process_logo(LOGO_PATH, processed_logo_path, logger=logger)
    await bot.send_photo(message.chat.id, photo=open(processed_logo_path, 'rb'))
    await message.reply("Добро пожаловать! \nПожалуйста, введите ваш ID:")


@dp.message_handler(lambda message: message.from_user.id not in user_states, state=None)
async def handle_user_id(message: types.Message):
    user_id = message.from_user.id
    logger, _ = setup_user_logging(user_id)
    if message.text and message.text.isdigit():
        input_id = int(message.text.strip())
        if input_id in AUTHORIZED_USERS:
            user_states[user_id] = {'authorized': True, 'image_results': {}}
            logger.info(f"ID пользователя {user_id} подтвержден.")
            await message.reply("ID подтвержден. \nПожалуйста, отправьте фотографию для распознавания.")
        else:
            logger.info(f"Неправильный ID введен пользователем {user_id}: {input_id}")
            await message.reply("ID введен неправильно. \nПовторите попытку или покиньте данный бот.")
    else:
        logger.info(f"Неправильный формат ID от пользователя {user_id}.")
        await message.reply("ID введен неправильно. \nПовторите попытку или покиньте данный бот.")


def highlight_text(key, text):
    if key == "image_name":
        return text
    if not isinstance(text, str):  # Проверяем, является ли значение строкой
        text = str(text)  # Преобразуем в строку, если это не строка
    highlighted_text = ""
    for char in text:
        if char.isdigit():
            highlighted_text += f'|{char}|'
        elif 'A' <= char <= 'Z' or 'a' <= char <= 'z':
            highlighted_text += f'<u>{char}</u>'
        else:
            highlighted_text += char
    return highlighted_text


@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_states or not user_states[user_id].get('authorized'):
        await message.reply("Сначала авторизуйтесь, введя ваш ID.")
        return

    logger, log_file_path = setup_user_logging(user_id)

    # Генерация уникального имени для изображения на основе временной метки и уникального ID
    unique_id = uuid.uuid4()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_img_path = os.path.join(temp_dir, f"downloaded_image_{timestamp}.jpg")
        await message.photo[-1].download(temp_img_path)
        logger.info(f"Изображение загружено: {temp_img_path}")

        img_name = f"{os.path.splitext(os.path.basename(temp_img_path))[0]}_{unique_id}"
        user_folder = f"./{user_id}"
        image_folder = os.path.join(user_folder, img_name)
        os.makedirs(image_folder, exist_ok=True)

        img_path_in_folder = copy_file_to_user_folder(user_id, img_name, temp_img_path, "original_image.png", logger)

        img = cv2.imread(img_path_in_folder)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model(img_rgb, imgsz=640, iou=0.4, conf=0.25, verbose=True)
        annotated_frame = results[0].plot()

        annotated_img_path = os.path.join(image_folder, f'annotated_image_{timestamp}.png')
        cv2.imwrite(annotated_img_path, annotated_frame)
        logger.info(f"Сохранено аннотированное изображение: {annotated_img_path}")

        with open(annotated_img_path, 'rb') as img_file:
            await bot.send_photo(message.chat.id, photo=img_file)
        await message.reply("Распознавание завершено. \nТеперь идет распознавание текстовых полей.")

        image_results = {label: 'не распознано' for label in labels if label != 'logo'}
        image_results['image_name'] = img_path_in_folder

        for i, box in enumerate(results[0].obb.xyxyxyxy):
            if box is not None:
                points = box.cpu().numpy().astype(int).reshape(-1, 2)
                label = labels[int(results[0].obb.cls[i])]
                if label != 'logo':  # Исключаем только лого, остальные метки обрабатываем
                    cropped_image_path = os.path.join(image_folder, f"cropped_{label}_{i}.png")
                    save_cropped_image(img_rgb, points, cropped_image_path, logger)

                    # Если метка 1phase или 3phase, записываем соответствующую информацию
                    if label == '1phase':
                        image_results[label] = 'однофазный'
                    elif label == '3phase':
                        image_results[label] = 'трехфазный'
                    else:
                        # Распознаем текст для других меток
                        detected_text = recognize_text_with_crnn(cv2.imread(cropped_image_path), model_CRNN)
                        if label == 'year':
                            detected_text = extract_numbers_and_format(detected_text, is_year=True, logger=logger)
                        image_results[label] = detected_text or 'не распознано'

        user_states[user_id]['image_results'] = image_results

        logger.info(f"Распознанная информация для пользователя {user_id}: {image_results}")

        response_text = "Распознанная информация:\n" + "\n".join(
            [f"{key}: {highlight_text(key, value)}" for key, value in image_results.items() if value != 'не распознано']
        )
        response_text += "\n\n*Примечание: английские символы подчеркнуты, цифры в прямых скобках*"

        await message.reply(response_text, parse_mode=types.ParseMode.HTML)

        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("Изменить год", callback_data="edit_year"))
        markup.add(InlineKeyboardButton("Изменить тип СИ", callback_data="edit_type"))
        markup.add(InlineKeyboardButton("Изменить модификацию СИ", callback_data="edit_mode"))
        markup.add(InlineKeyboardButton("Изменить номер № СИ", callback_data="edit_number"))
        markup.add(InlineKeyboardButton("Обновить данные", callback_data="update_data"))
        markup.add(InlineKeyboardButton("Поиск гос номера по type", callback_data="search_by_type"))

        await message.reply("Что вы хотите сделать?", reply_markup=markup)

        logger.info(f"Обработка фотографии для пользователя {user_id} завершена. Логи сохранены в: {log_file_path}")

# Обработчик нажатия на кнопки
@dp.callback_query_handler(
    lambda c: c.data in ['edit_year', 'edit_type', 'edit_mode', 'edit_number', 'update_data',  'search_by_type', 'search_by_mode',
                         'search_by_year'])
async def process_callback(callback_query: CallbackQuery, state: FSMContext):
    user_id = callback_query.from_user.id

    # Настройка логгера пользователя
    logger, _ = setup_user_logging(user_id)

    data = callback_query.data

    if data == 'edit_year':
        await callback_query.message.reply("Введите новый год:")
        await Form.year.set()
    elif data == 'edit_type':
        await callback_query.message.reply("Введите новый тип:")
        await Form.type.set()
    elif data == 'edit_mode':
        await callback_query.message.reply("Введите новую модификацию:")
        await Form.modification.set()
    elif data == 'edit_number':
        await callback_query.message.reply("Введите новый номер СИ:")
        await Form.number.set()
    elif data == 'update_data':
        # Получение распознанных данных из user_states
        image_results = user_states.get(user_id, {}).get('image_results', {})

        # Логирование обновления данных
        logger.info(f"Обновление данных для пользователя {user_id}: {image_results}")

        # Формирование текста с обновленными данными, исключая поля с "не распознано"
        response_text = "Обновленные данные:\n" + "\n".join(
            [f"{key}: {value}" for key, value in image_results.items() if value != 'не распознано']
        )

        if response_text == "Обновленные данные:\n":
            response_text = "Нет обновленных данных."

        # Отправка сообщения с обновленными данными
        await callback_query.message.reply(response_text)

    elif data == 'search_by_type':
        # Отправляем сообщение ожидания и сохраняем его ID
        waiting_message = await callback_query.message.reply("Подождите, идет поиск данных в базе...")

        try:
            # Получаем значение type, которое было распознано из изображения
            type_value = user_states[user_id]['image_results'].get('type', '')
            cleaned_type = clean_input(type_value)

            # Выполняем поиск по type в базе
            total_count, unique_gos_numbers, df = search_by_type(cleaned_type)
            logger.info(
                f"Результаты поиска по типу для пользователя {user_id}: найдено {total_count} записей, уникальные GosNumber: {unique_gos_numbers}")

            if total_count > 0:
                # Если данные найдены, создаем DataFrame и сохраняем его
                user_states[user_id]['df'] = df
                user_states[user_id]['last_search'] = {'total_count': total_count,
                                                       'unique_gos_numbers': unique_gos_numbers}
                await callback_query.message.reply(
                    f"Найдено записей: {total_count}. \nУникальные значения GosNumber: {unique_gos_numbers}")
                await show_next_search_button(user_id, callback_query, next_button='search_by_mode')
            else:
                # Если данные по type не найдены, предлагаем выполнить поиск по mode
                await callback_query.message.reply("Записей по типу не найдено. Желаете выполнить поиск по mode?",
                                                   reply_markup=InlineKeyboardMarkup().add(
                                                       InlineKeyboardButton("Поиск по mode",
                                                                            callback_data="search_by_mode")))
        finally:
            # Удаляем сообщение ожидания после выполнения поиска
            await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=waiting_message.message_id)

    elif data == 'search_by_mode':
        mode_value = user_states[user_id]['image_results'].get('mode', '')
        cleaned_mode = clean_input(mode_value)

        # Проверяем, был ли уже создан DataFrame на этапе поиска по type
        if 'df' in user_states[user_id] and user_states[user_id]['df'] is not None:
            # Если DataFrame создан, ищем в нем
            df = user_states[user_id]['df']
            df_initial_count = len(df)  # Количество записей до фильтрации по mode

            # Фильтруем DataFrame по mode
            df_filtered = df[df['SearchKey'].str.contains(cleaned_mode, na=False)]
            total_count = len(df_filtered)
            unique_gos_numbers = ', '.join(df_filtered['GosNumber'].dropna().unique())

            # Если данные найдены и количество записей уменьшилось после фильтрации
            if total_count > 0 and total_count < df_initial_count:
                user_states[user_id]['df'] = df_filtered
                user_states[user_id]['last_search'] = {'total_count': total_count,
                                                       'unique_gos_numbers': unique_gos_numbers}
                await callback_query.message.reply(
                    f"Найдено записей: {total_count}. \nУникальные значения GosNumber: {unique_gos_numbers}")
                await show_next_search_button(user_id, callback_query, next_button='search_by_year')
            else:
                # Если фильтрация не дала результата или количество записей не изменилось
                await callback_query.message.reply(
                    "Записей по mode не найдено или количество записей не изменилось. Желаете выполнить поиск по году?",
                    reply_markup=InlineKeyboardMarkup().add(
                        InlineKeyboardButton("Поиск по году", callback_data="search_by_year")))
        else:
            # Если DataFrame не был создан, выполняем поиск по mode в базе данных
            total_count, unique_gos_numbers, df = search_by_mode(cleaned_mode)

            # Если данные найдены в базе, создаем DataFrame
            if total_count > 0:
                user_states[user_id]['df'] = df
                user_states[user_id]['last_search'] = {'total_count': total_count,
                                                       'unique_gos_numbers': unique_gos_numbers}
                await callback_query.message.reply(
                    f"Найдено записей: {total_count}. \nУникальные значения GosNumber: {unique_gos_numbers}")
                await show_next_search_button(user_id, callback_query, next_button='search_by_year')
            else:
                await callback_query.message.reply("Записей по mode не найдено. Желаете выполнить поиск по году?",
                                                   reply_markup=InlineKeyboardMarkup().add(
                                                       InlineKeyboardButton("Поиск по году",
                                                                            callback_data="search_by_year")))

    elif data == 'search_by_year':
        if user_id in user_states and 'df' in user_states[user_id]:
            df = user_states[user_id]['df']
            year_value = user_states[user_id]['image_results'].get('year', '')
            df = df[df['ManufacturerYear'] == year_value]
            total_count = len(df)
            unique_gos_numbers = ', '.join(df['GosNumber'].dropna().unique())
        else:
            year_value = user_states[user_id]['image_results'].get('year', '')

            try:
                year_value = int(year_value)
            except ValueError:
                await callback_query.message.reply("Некорректное значение года.")
                return

            total_count, unique_gos_numbers, df = search_by_year(year_value)

        # Логирование результатов поиска по year
        if total_count == 0:
            last_search = user_states[user_id].get('last_search', {})
            final_count = last_search.get('total_count', 0)
            final_unique_gos_numbers = last_search.get('unique_gos_numbers', 'неизвестно')
            final_df = user_states[user_id].get('df')

            logger.info(f"Поиск по year ничего не нашел. Возвращаем результаты предыдущего поиска.")
            if final_df is not None:
                logger.info(f"Итоговый DataFrame:\n{final_df.to_string(index=False)}")

            await callback_query.message.reply(
                f"Поиск в базе завершен.\nНайдено записей: {final_count}. \nУникальные значения GosNumber: {final_unique_gos_numbers}\n"
                "Загрузите следующее изображение СИ или до встречи в следующий раз!"
            )
            # Очищаем DataFrame после завершения поиска
            user_states[user_id]['df'] = None
            user_states[user_id]['last_search'] = None
        else:
            logger.info(
                f"Результаты поиска по year для пользователя {user_id}: найдено {total_count} записей, уникальные GosNumber: {unique_gos_numbers}")
            logger.info(f"Итоговый DataFrame:\n{df.to_string(index=False)}")

            user_states[user_id]['df'] = df
            user_states[user_id]['last_search'] = {'total_count': total_count, 'unique_gos_numbers': unique_gos_numbers}
            await callback_query.message.reply(
                f"Поиск в базе завершен.\nНайдено записей: {total_count}. \nУникальные значения GosNumber: {unique_gos_numbers}\n"
                "Загрузите следующее изображение СИ или до встречи в следующий раз!"
            )
            # Очищаем DataFrame после завершения поиска
            user_states[user_id]['df'] = None
            user_states[user_id]['last_search'] = None

# Обработчики ввода данных
@dp.message_handler(state=Form.year)
async def process_year(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    try:
        year = int(message.text.strip())
        user_states[user_id]['image_results']['year'] = year
        await message.reply(f"Год обновлен: {year}")
        await state.finish()
    except ValueError:
        await message.reply("Пожалуйста, введите корректный числовой год.")

@dp.message_handler(state=Form.type)
async def process_type(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    type_ = message.text.strip()
    user_states[user_id]['image_results']['type'] = type_
    await message.reply(f"Тип обновлен: {type_}")
    await state.finish()

@dp.message_handler(state=Form.modification)
async def process_modification(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    modification = message.text.strip()
    user_states[user_id]['image_results']['mode'] = modification
    await message.reply(f"Модификация обновлена: {modification}")
    await state.finish()

@dp.message_handler(state=Form.number)
async def process_number(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    number = message.text.strip()
    user_states[user_id]['image_results']['number'] = number
    await message.reply(f"Номер № СИ обновлен: {number}")
    await state.finish()

# Дополнительные функции
async def show_next_search_button(user_id, callback_query, next_button):
    markup = InlineKeyboardMarkup()
    if next_button == 'search_by_mode':
        markup.add(InlineKeyboardButton("Поиск гос номера по mode", callback_data='search_by_mode'))
    elif next_button == 'search_by_year':
        markup.add(InlineKeyboardButton("Поиск гос номера по year", callback_data='search_by_year'))
    await callback_query.message.reply("Что вы хотите сделать дальше?", reply_markup=markup)

# Функция для запуска бота
def run_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor.start_polling(dp, skip_updates=True, loop=loop)

# Функция для запуска бота в отдельном потоке
def start_bot_in_thread():
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

# Запуск бота
if __name__ == "__main__":
    start_bot_in_thread()
    input("Бот запущен. Нажмите Enter, чтобы завершить...")