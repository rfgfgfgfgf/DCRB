import nextcord
from nextcord.ext import commands
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

intents = nextcord.Intents.default()
intents.guilds = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    learning = nextcord.Game('In development')
    await bot.change_presence(status=nextcord.Status.idle, activity=learning)

@bot.command()
async def check(ctx):
    attachments = ctx.message.attachments
    if attachments:
        for attachment in attachments:
            file_name = attachment.filename
            await attachment.save(f'images/{file_name}')
            embed = nextcord.Embed(title="Operation result", description='Your image was uploaded successfully!')
            await ctx.send(embed=embed)

            try:
                result = code_detector(f'images/{file_name}')
                response_embed = nextcord.Embed(title="Result", description=f'The model predicts that this code is writen on: {result}')
            except Exception as e:
                response_embed = nextcord.Embed(title="Error", description=f'An error occurred: {str(e)}')

            await ctx.send(embed=response_embed)
    else:
        embed = nextcord.Embed(title="Operation failed", description='Please upload your image first!')
        await ctx.send(embed=embed)

def code_detector(path):
    np.set_printoptions(suppress=True)

    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name

bot.run('Your token')
