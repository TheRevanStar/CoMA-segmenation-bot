import os
import asyncio
import logging
from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from clearml import Task
from ultralytics import YOLO
from searcher import search

logging.basicConfig(level=logging.INFO)

TOKEN = 'TOKEN'

bot = Bot(token=TOKEN)
model = YOLO(r'E:\projects\object_deleter\runs\detect\train2\weights\best.pt')

application = ApplicationBuilder().token(TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("Handling /start command")
    await update.message.reply_text('Hello! Send me a photo and I will process it.')

application.add_handler(CommandHandler('start', start))

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("Handling photo message")
    try:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        tasks = Task.get_tasks(project_name='MoCA_segmentation')
        if not tasks:
            await update.message.reply_text("No tasks in project.")
            return
        latest_task = tasks[0]

        folder_path = os.path.join(os.getcwd(), 'tg_img_cache', latest_task.name)
        os.makedirs(folder_path, exist_ok=True)

        filename = f"{photo.file_id}_{photo.file_unique_id}_{update.message.from_user.id}.jpg"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'wb') as f:
            f.write(photo_bytes)

        info_filtered,info_not_filtered = await asyncio.to_thread(search, file_path, model)
        reply = info_filtered if info_filtered else "No objects detected on image."  
        await update.message.reply_text("The filtered info about animal.")
        await update.message.reply_text(reply)
        reply_not_filtered= info_not_filtered if info_not_filtered else "No objects detected on image."  
        await update.message.reply_text("maybe you will be interested about other things connected with this animal.")
        await update.message.reply_text( reply_not_filtered)
    except Exception as e:
        logging.error(f"Error in photo_handler: {e}", exc_info=True)
        await update.message.reply_text("Error processing your photo.")

application.add_handler(MessageHandler(filters.PHOTO, photo_handler))

if __name__ == '__main__':

    logging.info("Starting bot with polling")
    application.run_polling()
