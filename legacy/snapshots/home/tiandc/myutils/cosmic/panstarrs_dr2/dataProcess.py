import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

#============================================================
# Calculate extinction
#============================================================
def correct_extinction(df):

    # 创建坐标对象(批量处理)
    l = df['l'].values
    b = df['b'].values
    coords = SkyCoord(l, b, unit='deg', frame='galactic')
    
    # 初始化SFD查询器
    sfd = SFDQuery()
    
    # 批量获取E(B-V)值
    ebv = sfd(coords)
    
    # if use bayestar, use the following
    # Bayestar19 extinction coefficients
    # http://argonaut.skymaps.info/usage
    # extinction_coeffs = {
    #     'g': 3.518,
    #     'r': 2.617,
    #     'i': 1.971,
    #     'z': 1.549,
    #     'y': 1.263
    # }
    
    # if use sfd, use the following
    # this is the extinction coeffs for the PS1
    # R_v = 3.1
    # from https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103#apj398709t6
    # Table 6
    extinction_coeffs = {
        'g': 3.172,
        'r': 2.271,
        'i': 1.682,
        'z': 1.322,
        'y': 1.087
    }
    
    # 计算校正后的星等
    for band in ['g', 'r', 'i', 'z', 'y']:
        A_band = extinction_coeffs[band] * ebv
        for col in ['PSFMag','KronMag','ApMag']:
            df[f'{band}{col}_dered'] = df[f'{band}{col}'] - A_band
    
    return df

#============================================================
# Calculate color
#============================================================
class ColorCalculator:
    """Calculate color indices between bands"""
    
    def __init__(self, df: pd.DataFrame,
                 adjacent_only: bool = True):
        self.df = df
        self.adjacent_only = adjacent_only
        self.mag_types = ['PSF', 'Ap', 'Kron']
        self.bands = ['g', 'r', 'i', 'z', 'y']
        self.band_pairs = self._get_band_pairs()

    def _get_band_pairs(self):
        if self.adjacent_only:
            band_pairs = [(self.bands[i], self.bands[i + 1]) for i in range(len(self.bands) - 1)]
        else:
            band_pairs = [(band1, band2) for band1 in self.bands for band2 in self.bands if band1 != band2]
        return band_pairs
    
    def calculate_color(self) -> pd.DataFrame:
        """Calculate color indices between bands"""
        for mag_type in self.mag_types:
            for band1, band2 in self.band_pairs:
                color_name = f'{mag_type}_{band1}{band2}'
                self.df[color_name] = self.df[f'{band1}{mag_type}Mag_dered'] - self.df[f'{band2}{mag_type}Mag_dered']

        return self.df
    
    def calculate_colorErr(self) -> pd.DataFrame:
        """Calculate color error based on magnitude error"""
        for mag_type in self.mag_types:
            for band1, band2 in self.band_pairs:
                color_name = f'{mag_type}_{band1}{band2}'
                self.df[f'{color_name}_Err'] = np.sqrt(
                    self.df[f'{band1}{mag_type}MagErr']**2 + self.df[f'{band2}{mag_type}MagErr']**2
            )
        return self.df
    
    def calculate(self) -> pd.DataFrame:
        self.df = self.calculate_color()
        self.df = self.calculate_colorErr()
        return self.df
    
