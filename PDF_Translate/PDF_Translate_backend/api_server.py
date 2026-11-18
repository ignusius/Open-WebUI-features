import os
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

from pdf2zh_next.high_level import do_translate_async_stream
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.config.translate_engine_model import OpenAISettings

app = FastAPI(title="pdf2zh-next Streaming API")

RESULTS_DIR = Path(tempfile.gettempdir()) / "pdf2zh_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STAGE_NAMES_RU = {
    "Parse PDF and Create Intermediate Representation": "Разбор структуры PDF",
    "DetectScannedFile": "Проверка на скан-копию",
    "Parse Page Layout": "Анализ макета страниц",
    "Parse Paragraphs": "Извлечение абзацев",
    "Parse Formulas and Styles": "Анализ формул и стилей",
    "Translate Paragraphs": "Перевод текста",
    "Typesetting": "Верстка переведенного текста",
    "Add Fonts": "Добавление шрифтов",
    "Generate drawing instructions": "Формирование страниц",
    "Subset font": "Оптимизация шрифтов",
    "Save PDF": "Сохранение итогового PDF",
}

PRE_TRANSLATE_STAGES = [
    "Parse PDF and Create Intermediate Representation",
    "DetectScannedFile",
    "Parse Page Layout",
    "Parse Paragraphs",
    "Parse Formulas and Styles",
]

POST_TRANSLATE_STAGES = [
    "Typesetting",
    "Add Fonts",
    "Generate drawing instructions",
    "Subset font",
    "Save PDF",
]


def _extract_stage_progress(event: dict, event_type: str) -> tuple[float, str]:
    """Возвращает (доля_выполнения_0_1, суффикс_счетчика)."""
    if event_type == "progress_start":
        return 0.0, ""
    if event_type == "progress_end":
        return 1.0, ""

    # 1. Проверяем пары счетчиков (где total > 1, чтобы отсечь заглушки 1/1)
    counter_pairs = [
        ("stage_current", "stage_total"),
        ("current", "total"),
        ("completed", "total"),
        ("done", "total"),
        ("page", "total_pages"),
        ("part_index", "total_parts"),
    ]
    for cur_key, tot_key in counter_pairs:
        cur = event.get(cur_key)
        tot = event.get(tot_key)
        if cur is not None and tot is not None and float(tot) > 1:
            ratio = max(0.0, min(1.0, float(cur) / float(tot)))
            return ratio, f" ({int(cur)}/{int(tot)})"

    # 2. Проверяем готовые поля процентов
    for pct_key in ("stage_progress", "progress", "percentage"):
        if event.get(pct_key) is not None:
            val = float(event[pct_key])
            ratio = val / 100.0 if val > 1.0 else val
            # Если пришло ровно 100% внутри progress_update при total_parts == 1 — это заглушка
            if ratio >= 1.0 and event.get("total_parts") == 1:
                return 0.05, ""
            return max(0.0, min(1.0, ratio)), ""

    return 0.05, ""


@app.post("/translate-stream")
async def translate_pdf_stream(
    file: UploadFile = File(...),
    lang_in: str = Form("en"),
    lang_out: str = Form("ru"),
    openai_api_key: str = Form(...),
    openai_model: str = Form("gpt-4o-mini"),
    openai_base_url: str = Form("https://openai.com"),
    dual: bool = Form(False),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Поддерживаются только PDF-файлы")

    task_id = str(uuid.uuid4())
    work_dir = Path(tempfile.mkdtemp(prefix=f"pdf2zh_{task_id}_"))
    input_path = work_dir / file.filename
    output_dir = work_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    settings = SettingsModel(
        basic={
            "input_files": [str(input_path)],
            "debug": False,
        },
        translation={
            "lang_in": lang_in,
            "lang_out": lang_out,
            "output": str(output_dir),
            "qps": 20,
            "pool_max_workers": 100,
            "no_auto_extract_glossary": True,
        },
        translate_engine_settings=OpenAISettings(
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
        ),
    )

    async def event_generator():
        try:
            async for event in do_translate_async_stream(settings, input_path):
                event_type = str(event.get("type", ""))
                raw_stage = str(event.get("stage") or "")

                # Логируем в консоль все события этапа перевода для точной диагностики
                if raw_stage == "Translate Paragraphs" or event_type not in (
                    "progress_start",
                    "progress_update",
                    "progress_end",
                ):
                    print("PDF2ZH:", event, flush=True)

                if event_type == "stage_summary":
                    yield json.dumps(
                        {"type": "progress", "progress": 0.0, "stage": "Подготовка документа"},
                        ensure_ascii=False,
                    ) + "\n"

                elif event_type in ("progress_start", "progress_update", "progress_end"):
                    stage_ru = STAGE_NAMES_RU.get(raw_stage, raw_stage or "Обработка")
                    stage_ratio, counter_suffix = _extract_stage_progress(event, event_type)

                    if raw_stage == "Translate Paragraphs":
                        overall = 15.0 + (stage_ratio * 75.0)
                        stage_ru = f"Перевод текста{counter_suffix}"
                    elif raw_stage in PRE_TRANSLATE_STAGES:
                        idx = PRE_TRANSLATE_STAGES.index(raw_stage)
                        overall = ((idx + stage_ratio) / len(PRE_TRANSLATE_STAGES)) * 15.0
                    elif raw_stage in POST_TRANSLATE_STAGES:
                        idx = POST_TRANSLATE_STAGES.index(raw_stage)
                        overall = 90.0 + ((idx + stage_ratio) / len(POST_TRANSLATE_STAGES)) * 10.0
                    else:
                        overall = stage_ratio * 100.0

                    yield json.dumps(
                        {
                            "type": "progress",
                            "progress": round(overall, 1),
                            "stage": stage_ru,
                        },
                        ensure_ascii=False,
                    ) + "\n"

                elif event_type == "finish":
                    result = event.get("translate_result")
                    target_file = result.dual_pdf_path if dual else result.mono_pdf_path

                    if not target_file or not os.path.exists(target_file):
                        yield json.dumps({"type": "error", "error": "Файл результата не был создан"}) + "\n"
                        return

                    final_path = RESULTS_DIR / f"{task_id}.pdf"
                    shutil.move(target_file, final_path)

                    yield json.dumps({"type": "finish", "task_id": task_id}) + "\n"

                elif event_type == "error":
                    yield json.dumps({"type": "error", "error": event.get("error", "Ошибка перевода")}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/download/{task_id}")
async def download_result(task_id: str):
    safe_id = Path(task_id).name
    file_path = RESULTS_DIR / f"{safe_id}.pdf"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден или уже удален")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"translated_{safe_id[:8]}.pdf",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)

