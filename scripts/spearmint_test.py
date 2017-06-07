import os
import sys
import re
import numpy as np
import pandas as pd
import subprocess

def createJson(configFile):
    f = open(configFile, 'r')
    for line in f.read():
        if f 

configFile = sys.argv[1]
projctName = sys.argv[2]

cwd = os.getcwd()
spearmintPath = "/home/alvin/Downloads/Spearmint/spearmint/main.py"
mongoPath = os.path.join(cwd,"/%s_db/"%(projectName))
if not os.path.exists(mongoPath):
    os.makedirs(mongoPath)

subprocess.call("python %s cwd"%(spearmintPath))
