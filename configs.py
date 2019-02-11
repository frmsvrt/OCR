from helpers import AttrDict

FONTS = './misc/fonts/'
BGS = './misc/pictures/'

fonts = [os.path.join(FONTS, font) for font in\
         os.listdir(FONTS)]

bgs = [os.path.join(BGS, bg) for bg in\
         os.listdir(BGS)]

with open('./dicts/en.txt', 'r') as f:
        d = f.readlines()

def generator_cfg()
    cfg = AttrDict()
    cfg.fonts = fonts
    cfg.dict = lines
    cfg.bgs = bgs
    cfg.fs = [n for n in range(15,50,3)]
    cfg.d = d
    return cfg
