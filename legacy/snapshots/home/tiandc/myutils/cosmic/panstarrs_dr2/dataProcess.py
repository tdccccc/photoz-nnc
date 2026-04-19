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
    
#============================================================
# Calculate stellar mass
#============================================================
def get_mstar_params():
    params = {
        'char_mag_zbins_center': np.array([
            0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03 , 0.035, 0.04 , 0.045,
            0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 , 0.085, 0.09 ,
            0.095, 0.1  , 0.125, 0.15 , 0.175, 0.2  , 0.225, 0.25 , 0.275,
            0.3  , 0.325, 0.35 , 0.375, 0.4  , 0.425, 0.45 , 0.475, 0.5  ,
            0.525, 0.55 , 0.575, 0.6  , 0.625, 0.65 , 0.675, 0.7  , 0.725,
            0.75 , 0.775, 0.8  , 0.825, 0.85 , 0.875, 0.9  , 0.925, 0.95 ,
            0.975, 1.   , 1.025, 1.05 , 1.075, 1.1  , 1.125, 1.15 , 1.175,
            1.2  , 1.225, 1.25 , 1.275, 1.3  , 1.325, 1.35 , 1.375, 1.4  ,
            1.425, 1.45 , 1.475, 1.5  , 1.525, 1.55 , 1.575, 1.6  , 1.625,
            1.65 , 1.675, 1.7  , 1.725, 1.75 , 1.775, 1.8  , 1.825, 1.85 ,
            1.875, 1.9  , 1.925, 1.95 , 1.975, 2.   , 2.1  , 2.2  , 2.3  ,
            2.4  , 2.5  , 2.6  , 2.7  , 2.8  , 2.9  , 2.988]),
        'z_char_mag': np.array([
            9.6927, 11.1998, 12.0838, 12.7126, 13.2012, 13.6006, 13.9392,
            14.2334, 14.494 , 14.7281, 14.9408, 15.1355, 15.3133, 15.4772,
            15.6296, 15.771 , 15.9027, 16.0272, 16.146 , 16.26  , 16.7682,
            17.2026, 17.5643, 17.8812, 18.1703, 18.4311, 18.6559, 18.8542,
            19.0498, 19.2235, 19.3845, 19.542 , 19.6855, 19.8289, 19.9639,
            20.0889, 20.2123, 20.3315, 20.4534, 20.5811, 20.6956, 20.8026,
            20.9028, 20.9989, 21.088 , 21.1793, 21.2506, 21.3149, 21.3892,
            21.4683, 21.559 , 21.6519, 21.7532, 21.8516, 21.9447, 22.0315,
            22.1192, 22.2063, 22.3179, 22.4295, 22.5397, 22.6457, 22.7458,
            22.8256, 22.897 , 22.966 , 23.0521, 23.1529, 23.2346, 23.2771,
            23.318 , 23.347 , 23.3692, 23.4011, 23.4254, 23.4564, 23.4991,
            23.5449, 23.5807, 23.6114, 23.648 , 23.6768, 23.7073, 23.7408,
            23.7757, 23.8052, 23.8404, 23.8775, 23.9056, 23.9356, 23.9678,
            24.0039, 24.0452, 24.0992, 24.1506, 24.1984, 24.2934, 24.3757,
            24.2617, 24.0119, 23.7251, 23.258 , 22.5805, 21.8331, 20.6888,
            17.019 ]),
        'zbins_center': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
        'z_a': np.array([
            11.13861784, 10.49609211, 10.66235534, 10.26808033, 10.13127691,
            9.83012158, 11.32148531,  8.24798616]),
        'z_b': np.array([
            -0.36245079,  0.28450657,  0.12026569,  0.30011592,  0.23783054,
            0.06002614, -0.96828277,  0.47226136]),
        'z_gamma': 1.5681417103650086,
    }
    return params

def calculate_mstar(z, rKronMag_dered, zKronMag_dered, params):

    logL_z = np.zeros_like(zKronMag_dered)
    mstar_z = np.zeros_like(zKronMag_dered)

    # 找到每个红移对应的bin索引
    bin_indices_char = np.array([(abs(params['char_mag_zbins_center'] - z_val)).argmin() 
                                for z_val in z])
    bin_indices = np.array([(abs(params['zbins_center'] - z_val)).argmin() 
                           for z_val in z])

    # 计算光度
    logL_z = -0.4 * (zKronMag_dered - params['z_char_mag'][bin_indices_char])

    # 计算z波段的恒星质量
    f_z = params['z_a'][bin_indices] + params['z_b'][bin_indices] * (rKronMag_dered - zKronMag_dered)
    mstar_z = params['z_gamma'] * logL_z + f_z

    # 限制恒星质量范围
    mstar = np.clip(mstar_z, a_min=None, a_max=12.7)

    return mstar