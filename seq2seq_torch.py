# -*- coding: utf-8 -*-
# @Time     :3/25/19 8:59 PM
# @Auther   :Jason Lin
# @File     :seq2seq_torch.py
# @Software :PyCharm

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

