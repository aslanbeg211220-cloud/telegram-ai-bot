"""
Telegram AI bot powered by Groq (Llama 4 Scout — text + vision in one model).

Reads TELEGRAM_BOT_TOKEN and GROQ_API_KEY from the environment (set as
Replit Secrets) — never hardcode credentials in this file.
"""

import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from html import escape
from threading import Thread

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from flask import Flask
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("telegram-bot")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADMIN_TELEGRAM_ID = int(os.getenv("ADMIN_TELEGRAM_ID", "0"))

if not BOT_TOKEN:
    sys.exit("Missing TELEGRAM_BOT_TOKEN environment variable.")
if not GROQ_API_KEY:
    sys.exit("Missing GROQ_API_KEY environment variable.")

SYSTEM_PROMPT = {
    "role": "system",
    "content": "Ты — дружелюбный и умный ИИ-ассистент в Telegram. Отвечай кратко, понятно и по делу. "
    "Ты умеешь анализировать фото, которые присылает пользователь, и помнишь их в разговоре, "
    "поэтому можешь отвечать на уточняющие вопросы про уже присланные фото.",
}

# Single model that handles both plain text and images, so photos stay part
# of the same conversation history as regular messages.
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_HISTORY_MESSAGES = 16  # 1 system + up to ~7 user/assistant turns

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# In-memory per-user conversation history: {user_id: [messages]}
user_context: dict[int, list[dict]] = {}

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
groq_client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Persistent user registry (for the admin panel)
# ---------------------------------------------------------------------------

def load_users() -> dict[str, dict]:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.exception("Failed to read users.json, starting fresh")
        return {}


def save_users(users: dict[str, dict]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


known_users: dict[str, dict] = load_users()


def register_user(user: types.User) -> None:
    """Record/update a user's info the first time (and every time) they interact."""
    key = str(user.id)
    now = datetime.now(timezone.utc).isoformat()
    entry = known_users.get(key, {"first_seen": now})
    entry.update(
        {
            "id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "last_seen": now,
        }
    )
    known_users[key] = entry
    save_users(known_users)


def format_user_line(entry: dict) -> str:
    name = escape(" ".join(filter(None, [entry.get("first_name"), entry.get("last_name")])) or "Без имени")
    user_id = entry["id"]
    username = entry.get("username")

    if username:
        contact = f'@{escape(username)}'
    else:
        contact = f'<a href="tg://user?id={user_id}">Открыть профиль</a>'

    return f"<b>{name}</b>\nID: <code>{user_id}</code>\n{contact}"


# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------

def reset_context(user_id: int) -> None:
    user_context[user_id] = [dict(SYSTEM_PROMPT)]


async def download_image_as_data_url(file_id: str) -> str:
    """Download a Telegram photo and encode it as a base64 data URL so it
    stays usable in conversation history even after Telegram's temporary
    file link expires."""
    file = await bot.get_file(file_id)
    buffer = await bot.download_file(file.file_path)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


async def generate_reply(user_id: int) -> str:
    chat_completion = groq_client.chat.completions.create(
        messages=user_context[user_id],
        model=MODEL,
        temperature=0.7,
        max_tokens=1024,
    )
    ai_response = chat_completion.choices[0].message.content
    user_context[user_id].append({"role": "assistant", "content": ai_response})

    if len(user_context[user_id]) > MAX_HISTORY_MESSAGES:
        user_context[user_id] = (
            [user_context[user_id][0]] + user_context[user_id][-(MAX_HISTORY_MESSAGES - 1):]
        )

    return ai_response


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    register_user(message.from_user)
    reset_context(message.from_user.id)
    await message.answer(
        f"Привет, {message.from_user.first_name}!\n"
        "Я твой ИИ-собеседник на базе Groq. Понимаю текст и фото — можешь прислать "
        "картинку с вопросом, а потом уточнять детали, я буду помнить о чём речь.\n"
        "Команда /clear сбрасывает память диалога."
    )


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message) -> None:
    register_user(message.from_user)
    reset_context(message.from_user.id)
    await message.answer("Память очищена. Начнём заново.")


@dp.message(Command("admin"))
async def cmd_admin(message: types.Message) -> None:
    if message.from_user.id != ADMIN_TELEGRAM_ID:
        return  # silently ignore — panel is private

    if not known_users:
        await message.answer("Пока никто не запускал бота.")
        return

    users = sorted(known_users.values(), key=lambda e: e.get("last_seen", ""), reverse=True)
    header = f"<b>Пользователи бота: {len(users)}</b>\n"
    blocks = [format_user_line(u) for u in users]

    # Telegram messages are capped at 4096 chars — batch into multiple messages if needed.
    chunk = header
    for block in blocks:
        candidate = chunk + "\n\n" + block
        if len(candidate) > 3800:
            await message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)
            chunk = block
        else:
            chunk = candidate
    await message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)


@dp.message(F.photo)
async def handle_photo(message: types.Message) -> None:
    register_user(message.from_user)
    user_id = message.from_user.id

    if user_id not in user_context:
        reset_context(user_id)

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    thinking_message = await message.answer("Смотрю фото...")

    try:
        photo = message.photo[-1]  # largest resolution
        image_data_url = await download_image_as_data_url(photo.file_id)
        prompt_text = message.caption or "Опиши, что изображено на этом фото."

        user_context[user_id].append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        )

        ai_response = await generate_reply(user_id)
        await thinking_message.edit_text(ai_response)

    except Exception:
        log.exception("Groq vision request failed")
        await thinking_message.edit_text("Не получилось распознать фото. Попробуй ещё раз.")


@dp.message(F.text)
async def handle_ai_request(message: types.Message) -> None:
    register_user(message.from_user)
    user_id = message.from_user.id

    if user_id not in user_context:
        reset_context(user_id)

    user_context[user_id].append({"role": "user", "content": message.text})

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    thinking_message = await message.answer("Думаю...")

    try:
        ai_response = await generate_reply(user_id)
        await thinking_message.edit_text(ai_response)

    except Exception:
        log.exception("Groq request failed")
        await thinking_message.edit_text("Произошла ошибка на стороне ИИ-сервера. Попробуй ещё раз.")


# ---------------------------------------------------------------------------
# Keep-alive web server (so external pingers like cron-job.org can prevent
# the project from sleeping). Runs in a background thread on port 8080.
# ---------------------------------------------------------------------------

keep_alive_app = Flask(__name__)


@keep_alive_app.route("/")
def keep_alive_index():
    return "Бот работает"


def run_keep_alive_server() -> None:
    port = int(os.getenv("PORT", "8000"))
    keep_alive_app.run(host="0.0.0.0", port=port)


def start_keep_alive() -> None:
    Thread(target=run_keep_alive_server, daemon=True).start()


async def main() -> None:
    log.info("Starting bot polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    start_keep_alive()
    asyncio.run(main())
