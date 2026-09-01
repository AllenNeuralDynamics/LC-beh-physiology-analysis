import numpy as np
from scipy import stats
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
from utils.beh_functions import session_dirs, parseSessionID
from scipy.stats import norm
from datetime import datetime
import ast
import pickle

def get_opto_tbl(session, data_type, loc = 'soma'):
    opto_csv = os.path.join(session_dirs(session)[f'opto_dir_{data_type}'], f'{session}_opto_session_{loc}.csv')
    if not os.path.exists(opto_csv):
        raise FileNotFoundError(f'OpTO table not found for session {session} in {data_type} data type.')
    opto_tbl = pd.read_csv(opto_csv)
    return opto_tbl

class opto_metrics:
    def __init__(self, session, data_type):
        """Initialize the object with a DataFrame."""
        session_dir = session_dirs(session)
        unit_tbl_dir = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging_metrics.pkl')
        with open(unit_tbl_dir, 'rb') as f:
            unit_data = pickle.load(f)
        qm_data = unit_data['opto_tagging_df_metrics']

        self.qm_data = qm_data

    def load_unit(self, unit_id):
        return self.qm_data[self.qm_data['unit_id'] == unit_id].copy()

class load_opto_sig():
    def __init__(self, session, data_type):
        self.session = session
        self.data_type = data_type
        self.opto_sigs = self.load_opto_sigs()

    def load_opto_sigs(self):
        opto_sigs_file = os.path.join(session_dirs(self.session)[f'opto_dir_{self.data_type}'], f'{self.session}_opto_sigs.pkl')
        if os.path.exists(opto_sigs_file):
            with open(opto_sigs_file, 'rb') as f:
                return pickle.load(f)
        else:
            # print(f'No opto sigs found for {self.session}')
            return None

    def load_unit(self, unit):
        if self.opto_sigs is not None:
            unit_opto_sigs = self.opto_sigs[self.opto_sigs['unit_id'] == unit].copy()
            if not unit_opto_sigs.empty:
                return unit_opto_sigs
            else:
                # print(f'No opto sigs found for unit {unit} in session {self.session}')
                return None
        else:
            # print(f'No opto sigs loaded for session {self.session}')
            return None


