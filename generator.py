import itertools
import os
import random
import warnings
warnings.simplefilter("ignore", UserWarning)
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter
import cv2
import numpy as np
import skimage.io as io
from tqdm import tqdm

from configs import generator_cfg

DATA_PATH = './data/'
cfg = generator_cfg()
kernel = np.ones((5,5),np.float32)/27

def generate(text, font, text_color, font_size, space_width):
        image_font = ImageFont.truetype(font=font, size=font_size)
        words = text.split(' ')
        space_width = image_font.getsize(' ')[0] * space_width
        words_width = [image_font.getsize(w)[0] for w in words]
        text_width =  sum(words_width) + int(space_width) * (len(words) - 1)
        text_height = max([image_font.getsize(w)[1] for w in words])

        roll = np.random.random()
        if roll > 0.77:
            txt_img = Image.open(random.choice(cfg.bgs))
            h, w = txt_img.size
            xs, ys = np.random.randint(h-text_width), np.random.randint(w-text_height)
            txt_img = txt_img.crop((xs, ys, xs+text_width, ys+text_height))
        else:
            txt_img = Image.new('RGB', (text_width, text_height), (255, 255, 255))
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
            txt_draw.text((sum(words_width[0:i]) + i * int(space_width), 0),
                          w,
                          fill=fill,
                          font=image_font)

        return txt_img


if __name__ == '__main__':
    cfg = generator_cfg()
    print('Total num of samples: %d' % len(cfg.d))
    for idx, word in tqdm(enumerate(cfg.d)):
        font = random.choice(cfg.fonts)
        fs = random.choice(cfg.fs)
        space_width = random.choice(cfg.sw)
        text = cfg.d[idx]
        img = generate(text,
                       font=font,
                       font_size=fs,
                       space_width=1.3,
                       text_color=cfg.colors,
                       )
        # add this guy during train as well as shadows and so on
        # if np.random.random() > 0.71:
        #    img = cv2.filter2D(np.array(img), -1, kernel)
        io.imsave(DATA_PATH+cfg.fnames[idx] + '.jpeg', np.array(img))
        with open(DATA_PATH+'data.csv', 'a') as f:
            f.write(DATA_PATH+cfg.fnames[idx]+'.jpeg'+';'+cfg.d[idx])
