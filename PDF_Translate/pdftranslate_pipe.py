"""
title: PDF Translator
author: local
version: 43.0
type: pipe
requirements: aiohttp, aiofiles
"""

from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import aiohttp
import aiofiles
import os
import glob
import uuid
import re
import json
import time
import asyncio

LANG_MAP = {
    "Английский": "en",
    "Русский": "ru",
    "Китайский": "zh",
    "Испанский": "es",
    "Французский": "fr",
    "Немецкий": "de",
    "Японский": "ja",
    "Корейский": "ko",
    "Итальянский": "it",
    "Португальский": "pt",
    "Арабский": "ar",
    "Турецкий": "tr",
    "Польский": "pl",
    "Нидерландский": "nl",
    "Украинский": "uk",
}


class Pipe:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(default="http://192.168.27.80:8089")
        OPENAI_BASE_URL: str = Field(default="http://192.168.27")
        OPENAI_API_KEY: str = Field(default="dummy")
        OPENAI_MODEL: str = Field(default="Qwen3-32B-AWQ")
        OWUI_URL: str = Field(default="http://192.168.27.80:8080")

    class UserValves(BaseModel):
        lang_in: str = Field(
            default="Английский", title="Язык оригинала", enum=list(LANG_MAP.keys())
        )
        lang_out: str = Field(
            default="Русский", title="Язык перевода", enum=list(LANG_MAP.keys())
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "pdf_translator"
        self.name = "PDF Translator"
        self.file_handler = True
        self.citation = False
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def _make_progress_bar(self, percent: float) -> str:
        clamped = max(0.0, min(100.0, percent))
        filled = int(clamped // 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"`[{bar}] {clamped:.0f}%`"

    async def pipe(self, body: dict, __user__: Optional[dict] = None) -> AsyncGenerator:
        last_msg = {}
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                last_msg = msg
                break

        user_message = last_msg.get("content", "")

        if (
            "### task:" in user_message.lower()
            or "predict 3-5 useful" in user_message.lower()
        ):
            yield "Фоновая задача пропущена."
            return

        file_id = None
        filename = None
        match = re.search(r'<file.*?id="([^"]+)".*?name="([^"]+)".*?>', user_message)
        if match:
            file_id = match.group(1)
            filename = match.group(2)

        if not file_id or not filename or not filename.lower().endswith(".pdf"):
            clean_text = re.sub(
                r"<attached_files>.*?</attached_files>",
                "",
                user_message,
                flags=re.DOTALL,
            ).strip()
            if not clean_text:
                clean_text = "Вы ничего не написали."
            yield f"💬 **Ваше сообщение:** {clean_text}\n\n⚠️ *Чтобы я перевел документ, прикрепите PDF-файл (📎).*"
            return

        upload_dir = "/app/backend/data/uploads/"
        search_pattern = os.path.join(upload_dir, f"{file_id}_*")
        matching_files = glob.glob(search_pattern)

        if not matching_files:
            yield f"❌ PDF прикреплен в чате, но не найден на сервере (ID: {file_id}). Попробуйте загрузить заново."
            return

        latest_pdf = next(iter(matching_files))
        original_filename = os.path.basename(latest_pdf)

        if __user__ and "valves" in __user__:
            user_valves = __user__["valves"]
            lang_in_ru = (
                user_valves.lang_in
                if hasattr(user_valves, "lang_in")
                else user_valves.get("lang_in", "Английский")
            )
            lang_out_ru = (
                user_valves.lang_out
                if hasattr(user_valves, "lang_out")
                else user_valves.get("lang_out", "Русский")
            )
        else:
            lang_in_ru = self.user_valves.lang_in
            lang_out_ru = self.user_valves.lang_out

        lang_in = LANG_MAP.get(lang_in_ru, "en")
        lang_out = LANG_MAP.get(lang_out_ru, "ru")

        yield f"⏳ **Загружаю документ и запускаю перевод** ({lang_in_ru} ➡️ {lang_out_ru})...\n\n"

        base_url = re.sub(
            r"/translate(-stream)?/?$", "", self.valves.API_BASE_URL.strip()
        )
        stream_url = f"{base_url}/translate-stream"

        try:
            async with aiohttp.ClientSession() as session:
                form_data = aiohttp.FormData()
                form_data.add_field("lang_in", lang_in)
                form_data.add_field("lang_out", lang_out)
                form_data.add_field("openai_base_url", self.valves.OPENAI_BASE_URL)
                form_data.add_field("openai_api_key", self.valves.OPENAI_API_KEY)
                form_data.add_field("openai_model", self.valves.OPENAI_MODEL)

                async with aiofiles.open(latest_pdf, "rb") as f:
                    file_content = await f.read()

                form_data.add_field(
                    "file",
                    file_content,
                    filename=original_filename,
                    content_type="application/pdf",
                )

                task_id = None
                last_reported_percent = -5.0
                last_base_stage = ""
                last_report_time = time.monotonic()

                async with session.post(
                    stream_url,
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=3600),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield f"\n❌ Ошибка API ({response.status}): {error_text[:200]}"
                        return

                    while True:
                        line = await response.content.readline()
                        if not line:
                            break

                        line_str = line.decode("utf-8").strip()
                        if not line_str:
                            continue

                        try:
                            data = json.loads(line_str)
                        except json.JSONDecodeError:
                            continue

                        msg_type = data.get("type")

                        if msg_type == "progress":
                            percent = float(data.get("progress", 0))
                            stage = data.get("stage", "")
                            base_stage, _, _ = stage.partition(" (")
                            now = time.monotonic()

                            should_report = (
                                (percent - last_reported_percent >= 5.0)
                                or (
                                    base_stage != last_base_stage
                                    and percent > last_reported_percent
                                )
                                or (
                                    now - last_report_time >= 10.0
                                    and percent > last_reported_percent
                                )
                            )

                            if should_report:
                                bar = self._make_progress_bar(percent)
                                stage_text = f" — *{stage}*" if stage else ""
                                yield f"🔄 {bar}{stage_text}\n"
                                last_reported_percent = percent
                                last_base_stage = base_stage
                                last_report_time = now

                        elif msg_type == "error":
                            yield f"\n❌ **Ошибка перевода:** {data.get('error')}"
                            return

                        elif msg_type == "finish":
                            task_id = data.get("task_id")
                            break

                if not task_id:
                    yield "\n❌ Не удалось получить идентификатор готового файла."
                    return

                download_api_url = f"{base_url}/download/{task_id}"
                async with session.get(download_api_url) as dl_resp:
                    if dl_resp.status != 200:
                        yield f"\n❌ Ошибка при скачивании результата ({dl_resp.status})."
                        return

                    pdf_bytes = await dl_resp.read()

                if pdf_bytes.startswith(b"%PDF"):
                    static_dir = "/app/backend/open_webui/static"
                    os.makedirs(static_dir, exist_ok=True)

                    unique_id = str(uuid.uuid4())[:8]
                    safe_filename = f"translated_{unique_id}.pdf"
                    result_path = os.path.join(static_dir, safe_filename)

                    async with aiofiles.open(result_path, "wb") as out_f:
                        await out_f.write(pdf_bytes)

                    download_url = f"{self.valves.OWUI_URL}/static/{safe_filename}"
                    yield f"\n✅ **Перевод готов!** ({lang_in_ru} ➡️ {lang_out_ru})\n\n### [📥 Скачать переведенный файл]({download_url})\n\n*Исходный файл: {original_filename}*"
                else:
                    yield "\n❌ API вернул поврежденный файл."

        except asyncio.TimeoutError:
            yield "\n❌ Таймаут: перевод занял слишком много времени."
        except Exception as e:
            yield f"\n❌ Ошибка при обработке: {str(e)}"
