#coding=utf-8

import sys
import time

def log(info):
	print >> sys.stderr, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), info
