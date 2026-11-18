"""
title: Qwen3.5 Searchable PDF OCR (Local Text Blur + Visible Fitted Text)
author: local
version: 6.9
type: pipe
requirements: aiohttp, pymupdf, pillow
"""

from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator, Callable, Awaitable, Any
from PIL import Image, ImageFilter
import aiohttp, os, glob, re, json, uuid, io, base64, hashlib, inspect, time
import fitz

try:
    from open_webui.models.files import Files, FileForm, File
except ImportError:
    Files = None
    FileForm = None
    File = None

try:
    from open_webui.internal.db import get_db
except ImportError:
    get_db = None

try:
    from open_webui.storage.provider import Storage
except ImportError:
    Storage = None


class Pipe:
    class Valves(BaseModel):
        OPENAI_BASE_URL: str = Field(
            default="http://192.168.27.80:8000/v1", description="URL vLLM"
        )
        OPENAI_API_KEY: str = Field(default="dummy", description="API-ключ")
        OPENAI_MODEL: str = Field(
            default="Qwen3.5-35B-A3B-AWQ", description="Имя модели в vLLM"
        )
        OCR_PROMPT: str = Field(
            default=(
                "Extract all text lines from this document page with coordinates. "
                "Output ONLY a valid JSON array of objects with keys 'bbox' ([x1, y1, x2, y2] in 0-1000 scale) "
                "and 'text' (exact recognized text). No markdown or explanations."
            ),
            description="Промпт для OCR с координатами",
        )
        DPI: int = Field(default=150, description="DPI рендеринга")
        JPEG_QUALITY: int = Field(default=85, description="Качество JPEG (1-100)")
        MAX_TOKENS: int = Field(
            default=4096, description="Максимум токенов на страницу"
        )
        DISABLE_THINKING: bool = Field(
            default=True, description="Отключить <think> в Qwen3.5"
        )
        BLUR_RADIUS: float = Field(
            default=4.5,
            description="Радиус размытия (блюра) только под распознанным текстом (0 = без размытия)",
        )
        BLUR_PADDING: int = Field(
            default=2,
            description="Отступ в пикселях вокруг рамки текста при локальном размытии",
        )
        MAX_FONT_SIZE: float = Field(
            default=16.0,
            description="Максимальный размер шрифта в пунктах (защита от гигантского шрифта на рукописном тексте)",
        )
        MIN_FONT_SIZE: float = Field(
            default=4.5,
            description="Минимальный размер шрифта в пунктах при сжатии длинной строки в рамку",
        )
        TEXT_COLOR: str = Field(
            default="0, 0, 0",
            description="Цвет видимого текста в RGB от 0 до 255 (например: 0, 0, 0 или 180, 0, 0)",
        )
        FONT_PATH: str = Field(
            default="/app/backend/data/fonts/DejaVuSans.ttf",
            description="Путь к TTF-шрифту с поддержкой кириллицы",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "qwen35_searchable_pdf_v6"
        self.name = "Qwen3.5 Searchable PDF OCR v6"
        self.file_handler = True
        self.citation = False
        self.valves = self.Valves()

    def _parse_color(self) -> tuple[float, float, float]:
        try:
            parts = [
                float(x.strip()) / 255.0 for x in self.valves.TEXT_COLOR.split(",")
            ]
            if len(parts) == 3:
                return (
                    max(0.0, min(1.0, parts[0])),
                    max(0.0, min(1.0, parts[1])),
                    max(0.0, min(1.0, parts[2])),
                )
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    async def _ensure_cyrillic_font(
        self, session: aiohttp.ClientSession
    ) -> Optional[str]:
        candidates = [
            self.valves.FONT_PATH,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        ]
        for path in candidates:
            if path and os.path.exists(path) and os.path.getsize(path) > 10000:
                return path

        target = self.valves.FONT_PATH or "/tmp/DejaVuSans.ttf"
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
        except Exception:
            target = "/tmp/DejaVuSans.ttf"

        urls = [
            "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
            "https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/DejaVuSans.ttf",
        ]
        for url in urls:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if len(data) > 10000:
                            with open(target, "wb") as f:
                                f.write(data)
                            return target
            except Exception:
                continue
        return None

    def _parse_json_blocks(self, raw: str) -> list:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        fence = chr(96) * 3
        cleaned = re.sub(
            rf"^{fence}(?:json)?\s*|\s*{fence}$", "", cleaned, flags=re.MULTILINE
        ).strip()
        try:
            res = json.loads(cleaned)
            if isinstance(res, list):
                return res
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                res = json.loads(match.group(0))
                if isinstance(res, list):
                    return res
            except json.JSONDecodeError:
                pass

        recovered = []
        pattern = re.compile(
            r'\{[^{}]*"(?:bbox|bbox_2d|box|coordinates)"[^{}]*"text"[^{}]*\}'
        )
        for m in pattern.finditer(cleaned):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "text" in obj:
                    recovered.append(obj)
            except json.JSONDecodeError:
                continue
        return recovered

    def _extract_bbox(self, item: dict) -> Optional[list[float]]:
        raw_box = (
            item.get("bbox")
            or item.get("bbox_2d")
            or item.get("box")
            or item.get("coordinates")
        )
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in raw_box]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1 = max(0.0, min(1000.0, x1))
            y1 = max(0.0, min(1000.0, y1))
            x2 = max(0.0, min(1000.0, x2))
            y2 = max(0.0, min(1000.0, y2))
            if x2 <= x1 or y2 <= y1:
                return None
            return [x1, y1, x2, y2]
        except Exception:
            return None

    def _find_uploaded_pdf(
        self, __files__: Optional[list] = None, body: Optional[dict] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Извлекает строго последний прикреплённый PDF из текущего сообщения.
        """
        candidates = []

        if isinstance(body, dict):
            messages = body.get("messages") or []
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    msg_files = msg.get("files") or []
                    if isinstance(msg_files, list):
                        for item in msg_files:
                            if isinstance(item, dict):
                                candidates.append(item)
                    break

        if not candidates and isinstance(__files__, list):
            for item in __files__:
                if isinstance(item, dict):
                    candidates.append(item)

        pdf_candidates = []
        for item in candidates:
            file_obj = item.get("file") if isinstance(item.get("file"), dict) else item
            meta = (
                file_obj.get("meta") if isinstance(file_obj.get("meta"), dict) else {}
            )
            name = (
                file_obj.get("filename")
                or file_obj.get("name")
                or meta.get("name")
                or item.get("name")
                or ""
            )
            path = file_obj.get("path") or meta.get("path") or item.get("path") or ""
            file_id = file_obj.get("id") or item.get("id") or ""

            if (
                str(name).lower().endswith(".pdf")
                or str(path).lower().endswith(".pdf")
                or item.get("type") == "file"
            ):
                pdf_candidates.append(
                    {"id": str(file_id), "name": str(name), "path": str(path)}
                )

        for cand in reversed(pdf_candidates):
            file_id = cand["id"]
            name = cand["name"]
            path = cand["path"]

            if Files is not None and file_id:
                try:
                    db_file = Files.get_file_by_id(file_id)
                    if db_file:
                        db_path = getattr(db_file, "path", None) or (
                            db_file.meta.get("path")
                            if getattr(db_file, "meta", None)
                            else None
                        )
                        db_name = getattr(db_file, "filename", None) or name
                        if db_path and os.path.exists(db_path):
                            return db_path, db_name or os.path.basename(db_path)
                        if Storage is not None and db_path:
                            try:
                                local_p = Storage.get_file(db_path)
                                if local_p and os.path.exists(local_p):
                                    return local_p, db_name or os.path.basename(local_p)
                            except Exception:
                                pass
                except Exception:
                    pass

            if path and os.path.exists(path) and path.lower().endswith(".pdf"):
                return path, name or os.path.basename(path)

            upload_dir = "/app/backend/data/uploads"
            if os.path.isdir(upload_dir) and file_id:
                matches = glob.glob(os.path.join(upload_dir, f"{file_id}*"))
                for m in matches:
                    if os.path.isfile(m) and m.lower().endswith(".pdf"):
                        return m, name or os.path.basename(m)

            if os.path.isdir(upload_dir) and name and name.lower().endswith(".pdf"):
                exact = os.path.join(upload_dir, name)
                if os.path.isfile(exact):
                    return exact, name
                suffix_matches = [
                    p
                    for p in glob.glob(os.path.join(upload_dir, f"*{name}"))
                    if os.path.isfile(p)
                ]
                if suffix_matches:
                    suffix_matches.sort(key=os.path.getmtime, reverse=True)
                    return suffix_matches[0], name

        return None, None

    def _blur_text_regions(self, orig_img_bytes: bytes, blocks: list) -> bytes:
        """
        Размывает ТОЛЬКО области под распознанным текстом (bbox), оставляя остальной документ чётким.
        """
        if self.valves.BLUR_RADIUS <= 0 or not blocks:
            return orig_img_bytes

        base_img = Image.open(io.BytesIO(orig_img_bytes)).convert("RGB")
        w, h = base_img.size
        pad = max(0, int(self.valves.BLUR_PADDING))

        blurred_full = base_img.filter(
            ImageFilter.GaussianBlur(radius=self.valves.BLUR_RADIUS)
        )
        result_img = base_img.copy()

        for item in blocks:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            bbox = self._extract_bbox(item)
            if not text or not bbox:
                continue

            x1, y1, x2, y2 = bbox
            left = max(0, int((x1 / 1000.0) * w) - pad)
            top = max(0, int((y1 / 1000.0) * h) - pad)
            right = min(w, int((x2 / 1000.0) * w) + pad)
            bottom = min(h, int((y2 / 1000.0) * h) + pad)

            if right > left and bottom > top:
                crop_blur = blurred_full.crop((left, top, right, bottom))
                result_img.paste(crop_blur, (left, top))

        out_buf = io.BytesIO()
        result_img.save(out_buf, format="JPEG", quality=self.valves.JPEG_QUALITY)
        return out_buf.getvalue()

    def _fit_line_fontsize(
        self,
        line_str: str,
        font_obj: fitz.Font,
        initial_size: float,
        target_width: float,
        max_page_width: float,
    ) -> float:
        """
        Ограничивает максимальный размер шрифта (для рукописного текста с крупным bbox)
        и уменьшает его, если строка шире рамки bbox или выходит за правый край страницы.
        """
        max_fs = max(float(self.valves.MAX_FONT_SIZE), 6.0)
        min_fs = max(float(self.valves.MIN_FONT_SIZE), 3.0)

        fontsize = min(initial_size, max_fs)

        try:
            text_w = font_obj.text_length(line_str, fontsize=fontsize)
        except Exception:
            text_w = len(line_str) * fontsize * 0.55

        if text_w <= 0:
            return max(fontsize, min_fs)

        # Сначала подгоняем под ширину рамки bbox (с небольшим допуском 1.05)
        if target_width > 4.0 and text_w > target_width * 1.05:
            scale = (target_width * 1.05) / text_w
            fontsize = max(fontsize * scale, min_fs)

        # Затем жёстко проверяем, чтобы текст не вылезал за правый край страницы
        try:
            final_w = font_obj.text_length(line_str, fontsize=fontsize)
        except Exception:
            final_w = len(line_str) * fontsize * 0.55

        if max_page_width > 4.0 and final_w > max_page_width:
            fontsize = max(fontsize * (max_page_width / final_w), 3.0)

        return fontsize

    def _add_visible_text_layer(
        self, page: fitz.Page, blocks: list, font_path: Optional[str]
    ) -> int:
        page_w = page.rect.width
        page_h = page.rect.height
        text_color = self._parse_color()
        font_name = "custom_cyr" if font_path else "helv"

        if font_path:
            page.insert_font(fontname=font_name, fontfile=font_path)
            try:
                font_obj = fitz.Font(fontfile=font_path)
            except Exception:
                font_obj = fitz.Font("helv")
        else:
            font_obj = fitz.Font("helv")

        margin_right = 8.0
        added = 0

        for item in blocks:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            bbox = self._extract_bbox(item)
            if not text or not bbox:
                continue

            x1, y1, x2, y2 = bbox
            rx0 = max(2.0, (x1 / 1000.0) * page_w)
            ry0 = max(2.0, (y1 / 1000.0) * page_h)
            rx1 = min(page_w - margin_right, (x2 / 1000.0) * page_w)
            ry1 = min(page_h - 2.0, (y2 / 1000.0) * page_h)

            box_w = max(rx1 - rx0, 10.0)
            box_h = max(ry1 - ry0, 6.0)
            max_avail_w = max(page_w - rx0 - margin_right, 10.0)

            lines = text.splitlines() or [text]
            line_h = box_h / len(lines)
            raw_fontsize = line_h * 0.80

            for idx, line_str in enumerate(lines):
                line_str = line_str.strip()
                if not line_str:
                    continue

                fontsize = self._fit_line_fontsize(
                    line_str=line_str,
                    font_obj=font_obj,
                    initial_size=raw_fontsize,
                    target_width=box_w,
                    max_page_width=max_avail_w,
                )

                baseline_y = ry0 + (idx + 0.82) * line_h
                if baseline_y > page_h - 2.0:
                    baseline_y = page_h - 2.0

                page.insert_text(
                    fitz.Point(rx0, baseline_y),
                    line_str,
                    fontname=font_name,
                    fontsize=fontsize,
                    color=text_color,
                    render_mode=0,
                )
                added += 1

        return added

    async def _stream_ocr_page(
        self, session: aiohttp.ClientSession, image_b64: str, page_num: int
    ):
        payload = {
            "model": self.valves.OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.valves.OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": self.valves.MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        if self.valves.DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        full_content, char_count, last_report = [], 0, 0

        try:
            async with session.post(
                f"{self.valves.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=600, sock_read=120),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    yield f"❌ Ошибка vLLM ({resp.status}): {err[:300]}\n", []
                    return

                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        piece = (
                            delta.get("content") or delta.get("reasoning_content") or ""
                        )
                        if piece:
                            full_content.append(piece)
                            char_count += len(piece)
                            if char_count - last_report >= 500:
                                last_report = char_count
                                yield f"⏳ Страница {page_num}: получено {char_count} симв...\n", None
                    except json.JSONDecodeError:
                        continue

            raw_text = "".join(full_content)
            blocks = self._parse_json_blocks(raw_text)
            yield f"✔️ Страница {page_num}: получено {len(blocks)} блоков.\n", blocks
        except Exception as e:
            yield f"❌ Ошибка соединения на странице {page_num}: {e}\n", []

    def _register_file_in_webui(
        self, pdf_bytes: bytes, display_name: str, user_id: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        file_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        storage_filename = f"{file_id}_{display_name}"
        file_path = os.path.join("/app/backend/data/uploads", storage_filename)

        tags = {"source": "qwen35_ocr"}
        try:
            if Storage is not None:
                sig = inspect.signature(Storage.upload_file)
                if len(sig.parameters) >= 3:
                    res = Storage.upload_file(
                        io.BytesIO(pdf_bytes), storage_filename, tags
                    )
                else:
                    res = Storage.upload_file(io.BytesIO(pdf_bytes), storage_filename)

                if isinstance(res, tuple) and len(res) >= 2:
                    _, file_path = res[0], res[1]
                elif isinstance(res, str):
                    file_path = res
            else:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(pdf_bytes)
        except Exception as e:
            return None, None, f"Ошибка сохранения файла: {e}"

        meta = {
            "name": display_name,
            "content_type": "application/pdf",
            "size": len(pdf_bytes),
            "source": "qwen35_ocr",
        }
        data_dict = {"status": "completed"}

        form_payload = {
            "id": file_id,
            "user_id": user_id,
            "filename": display_name,
            "path": file_path,
            "hash": file_hash,
            "data": data_dict,
            "meta": meta,
            "access_control": None,
        }
        allowed_fields = getattr(FileForm, "model_fields", {}) or getattr(
            FileForm, "__fields__", {}
        )
        filtered = (
            {k: v for k, v in form_payload.items() if k in allowed_fields}
            if allowed_fields
            else form_payload
        )

        last_err = ""
        try:
            form = FileForm(**filtered)
            record = Files.insert_new_file(user_id, form)
            if record and getattr(record, "id", None):
                return record.id, f"/api/v1/files/{record.id}/content", None
            last_err = "insert_new_file вернул None"
        except Exception as e:
            last_err = str(e)

        if File is not None and get_db is not None:
            try:
                now = int(time.time())
                db_cols = {c.name for c in File.__table__.columns}
                row_data = {
                    "id": file_id,
                    "user_id": user_id,
                    "filename": display_name,
                    "path": file_path,
                    "hash": file_hash,
                    "data": data_dict,
                    "meta": meta,
                    "access_control": None,
                    "created_at": now,
                    "updated_at": now,
                }
                clean_row = {k: v for k, v in row_data.items() if k in db_cols}
                with get_db() as db:
                    db_file = File(**clean_row)
                    db.add(db_file)
                    db.commit()
                    db.refresh(db_file)
                return file_id, f"/api/v1/files/{file_id}/content", None
            except Exception as db_e:
                last_err = f"{last_err} | DB fallback error: {db_e}"

        return None, None, last_err

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __files__: Optional[list] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:
        pdf_path, orig_name = self._find_uploaded_pdf(__files__, body)
        if not pdf_path or not orig_name:
            yield "⚠️ PDF-файл не найден. Прикрепите PDF к сообщению.\n"
            return

        yield f"📄 Обработка файла: `{orig_name}`...\n"

        src_doc = fitz.open(pdf_path)
        out_doc = fitz.open()
        total_pages = len(src_doc)

        async with aiohttp.ClientSession() as session:
            font_path = await self._ensure_cyrillic_font(session)
            if not font_path:
                yield "⚠️ Кириллический шрифт не найден, используется стандартный.\n"

            for idx in range(total_pages):
                page_num = idx + 1
                src_page = src_doc[idx]
                rect = src_page.rect

                pix = src_page.get_pixmap(dpi=self.valves.DPI)
                orig_img_bytes = pix.tobytes(
                    "jpeg", jpg_quality=self.valves.JPEG_QUALITY
                )
                img_b64 = base64.b64encode(orig_img_bytes).decode("ascii")

                blocks = []
                async for status_msg, parsed in self._stream_ocr_page(
                    session, img_b64, page_num
                ):
                    if status_msg:
                        yield status_msg
                    if parsed is not None:
                        blocks = parsed

                new_page = out_doc.new_page(width=rect.width, height=rect.height)
                blurred_bytes = self._blur_text_regions(orig_img_bytes, blocks)
                new_page.insert_image(rect, stream=blurred_bytes)

                count = self._add_visible_text_layer(new_page, blocks, font_path)
                yield f"✅ Страница {page_num}/{total_pages} готова (добавлено строк: {count}).\n"

        pdf_bytes = out_doc.tobytes(garbage=4, deflate=True)
        out_doc.close()
        src_doc.close()

        stem, _ = os.path.splitext(orig_name)
        display_name = f"{stem}_ocr.pdf"

        user_id = (__user__ or {}).get("id") or "system"
        file_id, download_url, reg_err = self._register_file_in_webui(
            pdf_bytes, display_name, user_id
        )

        if download_url and file_id:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "files",
                        "data": {
                            "files": [
                                {
                                    "type": "file",
                                    "id": file_id,
                                    "url": download_url,
                                    "name": display_name,
                                }
                            ]
                        },
                    }
                )
            yield f"\n🎉 Готово! [📥 Скачать {display_name}]({download_url})\n"
        else:
            yield f"\n❌ Не удалось зарегистрировать файл в Open WebUI: `{reg_err}` (user_id=`{user_id}`)\n"
