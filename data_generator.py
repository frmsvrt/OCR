import itertools
import os
import random

from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter

import cv2
import numpy as np

from configs import generator_cfg

def LetterRange(start, end):
    return list(map(chr, range(ord(start), ord(end) + 1)))
VOCAB = LetterRange('a', 'z') + LetterRange('A', 'Z') + LetterRange('0', '9')

file_ids = [''.join(i) for i in itertools.product(VOCAB, repeat=4)]
file_index = {f: i for (i, f) in enumerate(file_ids)}

cfg = generator_cfg()

kernel = np.ones((5,5),np.float32)/15

def generate_horizontal_text(text, font, text_color, font_size, space_width):
        image_font = ImageFont.truetype(font=font, size=font_size)

        words = text.split(' ')

        space_width = image_font.getsize(' ')[0] * space_width

        words_width = [image_font.getsize(w)[0] for w in words]
        text_width =  sum(words_width) + int(space_width) * (len(words) - 1) + 5
        text_height = max([image_font.getsize(w)[1] for w in words]) + 5

        roll = np.random.random()
        if roll > 0.77:
            txt_img = Image.open(random.choice(cfg_gen.bgs))
            h, w = txt_img.size
            xs, ys = np.random.randint(h-text_width), np.random.randint(w-text_height)
            txt_img = txt_img.crop((xs, ys, xs+text_width, ys+text_height))
        else:
            txt_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
        # print(txt_img.size)

        txt_draw = ImageDraw.Draw(txt_img)

        colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
        c1, c2 = colors[0], colors[-1]

        fill = (
            random.randint(c1[0], c2[0]),
            random.randint(c1[1], c2[1]),
            random.randint(c1[2], c2[2])
        )

        for i, w in enumerate(words):
            txt_draw.text((sum(words_width[0:i]) + i * int(space_width), 0), w, fill=fill, font=image_font)

        return txt_img
