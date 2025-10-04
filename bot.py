import asyncio
import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont, ImageOps
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

LOGGER = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "Оцени и опиши черты лица по изображению: скулы, подбородок, симметрия, "
    "улыбка и т.д., дай краткий вердикт привлекательности."
)

PROMPT_TEMPLATE = (
    "Ты — эксперт по анализу черт лица. Проанализируй фото, переданное в "
    "base64 ниже. Ответь в формате JSON со следующей структурой:\n"
    "{{\n"
    "  \"verdict\": \"краткий вывод о привлекательности\",\n"
    "  \"traits\": [\n"
    "    {{\"name\": \"Название черты\", \"score\": число от 1 до 10, \"comment\": \"краткий комментарий\"}},\n"
    "    ... минимум 5 черт лица\n"
    "  ]\n"
    "}}\n"
    "Описание и комментарии давай на русском языке. Фото в base64: {image_base64}"
)


@dataclass
class TraitAssessment:
    name: str
    score: float
    comment: str


class FaceAnalyzer:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)
        self._model = "gemini-2.0-flash"

    def analyze(self, image_base64: str) -> Dict[str, Any]:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=PROMPT_TEMPLATE.format(image_base64=image_base64))],
            ),
        ]
        config = types.GenerateContentConfig(
            tools=[types.Tool(googleSearch=types.GoogleSearch())],
            system_instruction=[types.Part.from_text(text=SYSTEM_INSTRUCTION)],
        )
        LOGGER.debug("Sending request to Gemini")
        response_chunks = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )
        response_text = ""
        for chunk in response_chunks:
            if chunk.text:
                response_text += chunk.text
        LOGGER.debug("Gemini response: %s", response_text)
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            LOGGER.exception("Failed to parse Gemini response as JSON")
            raise RuntimeError("Не удалось распознать ответ модели") from exc
        return parsed


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_report_image(image_bytes: bytes, verdict: str, traits: List[TraitAssessment]) -> bytes:
    base_width = 900
    header_height = 420
    footer_height = 480
    spacing = 30
    bg_color = (20, 24, 36)
    accent_color = (93, 173, 226)
    text_color = (240, 244, 255)
    secondary_color = (140, 148, 168)

    canvas = Image.new("RGB", (base_width, header_height + footer_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    circle_size = 360
    circle_image = ImageOps.fit(original, (circle_size, circle_size))
    mask = Image.new("L", (circle_size, circle_size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, circle_size, circle_size), fill=255)
    circle_position = ((base_width - circle_size) // 2, spacing)
    canvas.paste(circle_image, circle_position, mask)

    title_font = load_font(48)
    verdict_font = load_font(32)
    trait_font = load_font(28)
    comment_font = load_font(22)

    title_text = "Портрет красоты"
    title_width, title_height = draw.textsize(title_text, font=title_font)
    draw.text(((base_width - title_width) / 2, circle_position[1] + circle_size + 20), title_text, fill=text_color, font=title_font)

    verdict_box_top = circle_position[1] + circle_size + 20 + title_height + 20
    verdict_text = verdict
    verdict_lines = wrap_text(verdict_text, verdict_font, base_width - 120, draw)
    current_y = verdict_box_top
    for line in verdict_lines:
        draw.text((60, current_y), line, fill=secondary_color, font=verdict_font)
        current_y += verdict_font.size + 8

    trait_section_top = header_height
    draw.rounded_rectangle([(0, trait_section_top), (base_width, header_height + footer_height)], radius=60, fill=(28, 34, 54))

    trait_title = "Анализ черт лица"
    tt_w, _ = draw.textsize(trait_title, font=verdict_font)
    draw.text(((base_width - tt_w) / 2, trait_section_top + 40), trait_title, fill=text_color, font=verdict_font)

    bar_left = 120
    bar_right = base_width - 120
    bar_height = 24
    bar_spacing = 80
    current_y = trait_section_top + 120

    for trait in traits:
        score_ratio = max(0.0, min(1.0, trait.score / 10))
        bar_width = bar_left + int((bar_right - bar_left) * score_ratio)

        draw.text((bar_left, current_y - 10), trait.name, fill=text_color, font=trait_font)
        score_text = f"{trait.score:.1f}/10"
        score_w, _ = draw.textsize(score_text, font=trait_font)
        draw.text((bar_right - score_w, current_y - 10), score_text, fill=accent_color, font=trait_font)

        draw.rounded_rectangle([(bar_left, current_y + 24), (bar_right, current_y + 24 + bar_height)], radius=12, fill=(44, 52, 76))
        draw.rounded_rectangle([(bar_left, current_y + 24), (bar_width, current_y + 24 + bar_height)], radius=12, fill=accent_color)

        comment_lines = wrap_text(trait.comment, comment_font, bar_right - bar_left, draw)
        comment_y = current_y + 24 + bar_height + 12
        for line in comment_lines:
            draw.text((bar_left, comment_y), line, fill=secondary_color, font=comment_font)
            comment_y += comment_font.size + 6
        current_y = comment_y + bar_spacing

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    return output.getvalue()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    for word in words:
        test_line = " ".join(current_line + [word]) if current_line else word
        width, _ = draw.textsize(test_line, font=font)
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Пришли мне селфи, и я оценю ключевые черты лица с помощью AI."
    )


async def analyze_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    if update.effective_chat:
        await update.effective_chat.send_action(action=ChatAction.UPLOAD_PHOTO)

    photo = update.message.photo[-1]
    file = await photo.get_file()
    buffer = io.BytesIO()
    await file.download_to_memory(out=buffer)
    image_bytes = buffer.getvalue()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    analyzer: FaceAnalyzer = context.application.bot_data["face_analyzer"]
    loop = asyncio.get_running_loop()
    try:
        result: Dict[str, Any] = await loop.run_in_executor(None, analyzer.analyze, image_base64)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Failed to analyze image")
        await update.message.reply_text(
            "Не удалось получить оценку от модели. Попробуйте позже."
        )
        return

    verdict = result.get("verdict", "Вердикт не получен")
    traits_raw = result.get("traits", [])
    traits: List[TraitAssessment] = []
    for item in traits_raw:
        try:
            trait = TraitAssessment(
                name=str(item.get("name", "Черта")),
                score=float(item.get("score", 0)),
                comment=str(item.get("comment", "")),
            )
        except (TypeError, ValueError):
            continue
        traits.append(trait)

    if not traits:
        await update.message.reply_text("Модель не вернула характеристики. Попробуйте другое фото.")
        return

    report_bytes = create_report_image(image_bytes, verdict, traits[:6])

    photo_file = InputFile(io.BytesIO(report_bytes), filename="face_report.png")

    message_lines = [f"<b>Вердикт:</b> {verdict}"]
    message_lines.append("")
    for trait in traits[:6]:
        message_lines.append(
            f"<b>{trait.name}</b>: {trait.score:.1f}/10 — {trait.comment}"
        )
    caption = "\n".join(message_lines)

    await update.message.reply_photo(photo=photo_file, caption=caption, parse_mode=ParseMode.HTML)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    analyzer = FaceAnalyzer(api_key=os.getenv("GEMINI_API_KEY"))

    application: Application = ApplicationBuilder().token(token).build()
    application.bot_data["face_analyzer"] = analyzer

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, analyze_photo))

    LOGGER.info("Bot started")
    application.run_polling(stop_signals=None)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        LOGGER.info("Bot stopped")
