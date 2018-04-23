# coding:utf-8

from configparser import ConfigParser


cfg = ConfigParser()
cfg.read('./config/global.ini')
ACCEPTED_TAGS = set(str(cfg.get('preprocess', 'accepted_tags')).split())
print(ACCEPTED_TAGS)