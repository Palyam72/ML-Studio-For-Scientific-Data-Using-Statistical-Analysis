import streamlit as st
import pandas as pd
from feature_engine.create import *
from sklearn.preprocessing import *

class Creators:
  def __init__(self,dataset):
    self.dataset=dataset
