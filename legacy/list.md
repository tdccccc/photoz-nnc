# photoz-nnc 所有相关文件和处理流程

# 一些工具函数
`cosmic`库：
- 工具函数文件：`/home/tiandc/myutils/cosmic/utils.py`
- `cosmic.utils`缩写为`cu`

# Pipline

## 用于训练的数据

### LSDR10

训练用的原始数据总目录为：`/home/tiandc/Data/LegacySurveys/DR9x10`

1. 下载数据的代码：`/home/tiandc/Data/LegacySurveys/DR9x10/code/lsdr9x10_sql_250930.py`
下载到了`/home/tiandc/Data/LegacySurveys/DR9x10/raw` （其实就是`lsdr9x10_sql_250930.py`中的那个输出目录，改过名）
2. 原始数据首先进行recollect然后按照ra每5度划分为72个文件，用到的代码应该是`cu.merge_fits_by_ra()`,输出目录是`/home/tiandc/Data/LegacySurveys/DR9x10/raw72`
3. 收集了raw72中的所有的[uid, ra,dec]然后合并成一个文件，方便交叉匹配，uid方便索引回原始数据。合并的代码在`/home/tiandc/Data/LegacySurveys/DR9x10/code/dataProcess.ipynb`文件中的`collect position of all galaxies`对应的代码块，输出的文件是`/home/tiandc/Data/LegacySurveys/DR9x10/lsdr9x10_gal_raw72_allLoc.fits`

训练用的LSDR10数据与PS1DR2交叉匹配获取PS1DR2的i波段数据，相关代码在`/home/tiandc/Data/LegacySurveys/DR9x10/code/`dataProcess.ipynb文件中的以下部分对应的代码：
- `LSDR9x10 `raw72` 与PANSTARRS `raw72` 交叉匹配，获取i波段数据： `raw72`->`raw72_xPS1DR2``
- `使用STILTS.jar进行交叉匹配`
- `add PS1DR2 i band photometric data` 
最后生成的数据目录为：`/home/tiandc/Data/LegacySurveys/DR9x10/xPS1DR2/raw72_xPS1DR2`


### unWISE

1. 原始数据下载代码：`/home/tiandc/Data/unWISE/code/unwise_download.py`，使用NOIRLab DataLab的包，原始数据下载目录是：`/home/tiandc/Data/unWISE/download`
2. 原始数据进行recollect然后按照ra每5度划分为72个文件，处理的代码在文件`/home/tiandc/Data/unWISE/code/unwise.ipynb`的`Data Recollected into 72 Files download/ -> raw/`部分，数据输出目录：`/home/tiandc/Data/unWISE/raw`
3. 数据清理和收集uid，代码在：`Data Clean raw/ -> clean/`部分，输出的数据为`/home/tiandc/Data/unWISE/clean`和`/home/tiandc/Data/unWISE/unwisedr1_clean_all_loc.fits`


### PS1DR2

1. 原始数据下载代码：`/home/tiandc/Data/PanSTARRS/DR2All/download.py`，原始数据下载的目录是`/home/tiandc/Data/PanSTARRS/DR2All/download`
> 需要注意，PS1DR2下载需要用到casjob,一次下载只能用一个账号串行下载，且casjob有并行限制，下载较慢
2. 原始数据同样首先进行recollect然后按照ra每5度划分为72个文件，代码在`/home/tiandc/Data/PanSTARRS/DR2All/process.ipynb`中的`Recollect download -> raw72`
3. 同样收集了raw72中的所有的[uid, ra,dec]然后合并成一个文件，方便交叉匹配，uid方便索引回原始数据。合并的代码在`/home/tiandc/Data/PanSTARRS/DR2All/process.ipynb`的`收集 raw72 目录下所有文件的位置信息 [uid, raStack, decStack]` 这个代码块，输出的文件是`/home/tiandc/Data/PanSTARRS/DR2All/ps1dr2_raw72_loc.fits`
4. 去红化dered,使用的是`/home/tiandc/myutils/cosmic/panstarrs_dr2/dataProcess.py`文件中的`correct_extinction()`函数，需要安装dustmaps包，对应的去红化后的数据目录是`/home/tiandc/Data/PanSTARRS/DR2All/raw72_dered`
5. 添加unWISE红外数据，参考文件`/home/tiandc/Data/PanSTARRS/DR2All/process.ipynb`中`添加unWISE红外数据`部分（`数据清理`之前），数据输出的目录是`/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE`


### SDSS DR19

1. 原始数据从SDSS casjob系统下载，下载的文件叫：`/home/tiandc/Data/SDSS/sdss_dr19/MyDB_DachuanTian.csv`
2. 数据处理流程见文件`/home/tiandc/Data/SDSS/sdss_dr19/process.ipynb`中`SDSS DR19 STAR`之前的代码，最后输出的文件是`/home/tiandc/Data/SDSS/sdss_dr19/sdss_dr19.fits`


### DESI DR1

1. 去DESI DR1官网下载原始数据，用到的原始文件为：`/home/tiandc/Data/DESI/DR1/BGS_ANY_full_noveto.dat.fits`,`/home/tiandc/Data/DESI/DR1/LRG_full_noveto.dat.fits`, `/home/tiandc/Data/DESI/DR1/ELG_LOPnotqso_full_noveto.dat.fits`，文件名同DESI原始文件名
2. 数据处理流程见文件：`/home/tiandc/Data/DESI/DR1/process.ipynb`，最后输出的文件为`/home/tiandc/Data/DESI/DR1/desidr1.fits`


### 合并SDSS DR19 & DESI DR1得到总光谱样本(以下简称Spec)

1. 交叉匹配，`/home/tiandc/Data/SDSS/sdss_dr19/sdss_dr19.fits`和`/home/tiandc/Data/DESI/DR1/desidr1.fits`使用TOPCAT交叉匹配（这里用的是TOPCAT GUI界面操作的，也可以使用`cu.stilts_crossmatch()`），交叉匹配后的数据文件是`/home/tiandc/Data/DESI/DR1/sdss_dr19/desidr1_xsdssdr19.fits`
2. 清理交叉匹配后的`/home/tiandc/Data/DESI/DR1/sdss_dr19/desidr1_xsdssdr19.fits`，相关代码在文件`/home/tiandc/Data/DESI/DR1/sdss_dr19/process.ipynb`的`重新清理的DESI DR1`部分，输出的文件是`/home/tiandc/Data/DESI/DR1/sdss_dr19/DESIDR1_xSDSSDR19.fits`


### LSDR10 x Spec

1. 先用TOPCAT对LSDR10和Spec样本进行交叉匹配，输出为`/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Loc.fits`
2. 添加LSDR10的测光数据，处理流程见文件`/home/tiandc/photoz/lsdr9x10/makeTrainSample.ipynb`，输出是`/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot.fits`
3. 数据清理,处理流程见文件`/home/tiandc/photoz/lsdr9x10/makeTrainSample.ipynb`，输出是`/home/tiandc/photoz/lsdr9x10/data/DESIDR1_xSDSSDR19_xlsdr9x10Phot_clean.fits`
4. 根据不同波段/特征来分别制作训练集、验证集和测试集，具体处理过程见文件`/home/tiandc/photoz/lsdr9x10/makeTrainSample.ipynb`

### PS1DR2 x Spec

1. 先用TOPCAT对PS1DR2和Spec样本进行交叉匹配，输出是：`/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2loc.fits`
2. 添加PS1DR2的测光数据，处理流程见文件`/home/tiandc/photoz/panstarrs/makeTrainSample.ipynb`，输出是`/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2phot.fits`
3. 数据清理，处理流程见文件`/home/tiandc/photoz/panstarrs/makeTrainSample.ipynb`，输出是`/home/tiandc/photoz/panstarrs/data/DESIDR1_xSDSSDR19_xps1dr2phot_clean.fits`
4. 划分训练集验证集和测试集，处理流程见文件`/home/tiandc/photoz/panstarrs/makeTrainSample.ipynb`

### PS1DR2 x unWISE x Spec
1. 拿PS1DR2 x Spec的训练集验证集和测试集与unWISE的`/home/tiandc/Data/unWISE/unwisedr1_clean_all_loc.fits`进行交叉匹配，得到`/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_val_xunWISEloc.fits` `/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train_xunWISEloc.fits` `/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_test_xunWISEloc.fits`
2. 添加unWISE的测光数据，处理流程见`/home/tiandc/photoz/panstarrs/makeTrainSample.ipynb`，输出文件为`/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_train_xunWISEphot.fits` `/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_test_xunWISEphot.fits` `/home/tiandc/photoz/panstarrs/data/xunWISE/DESIDR1_xSDSSDR19_xps1dr2phot_clean_val_xunWISEphot.fits`



## 用于发布的数据

### LSDR10

用于发布的LSDR10数据与训练所用的LSDR10有点不同，前者的数据中包含了Gaia的数据（虽然只有少量源有），该部分数据总目录为：`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia`

1. 原始数据下载，代码为`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/download.py`,原始数据输出目录为`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/download`
2. 原始数据首先进行recollect然后按照ra每5度划分为72个文件，相关代码见：`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/dataProcess.ipynb`，输出目录为`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/raw72`
3. 与PS1DR2xunWISE交叉匹配，相关代码见`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/crossmatch.ipynb`，输出目录为`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/xPS1DR2grizy`
4. 清理交叉匹配后的LSDR10xPS1DR2xunWISE数据，相关代码见`/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/clean.ipynb`，输出目录为`'/home/tiandc/Data/LegacySurveys/DR9x10_Gaia/xPS1DR2/xPS1DR2grizy_clean'`



### PS1DR2

用于发布的PS1DR2数据与训练所用的PS1DR2一致，直接用上面处理好的数据目录`/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE`

数据处理见文件`/home/tiandc/Data/PanSTARRS/DR2All/process.ipynb`中的`数据清理`部分，输出的目录为`/home/tiandc/Data/PanSTARRS/DR2All/xunWISE/raw72_dered_xunWISE_clean`


# 模型训练


## 模型训练/预测相关代码/配置

模型训练代码，主要模型为ann, nnc, xgboost, random forest, lstm, ensemble, knn，下面是各模型相关代码：

- ANN的训练代码:`/home/tiandc/photoz/lsdr9x10/photoz_ann/photoz_ann.py`, 配置文件:`/home/tiandc/photoz/lsdr9x10/photoz_ann/photoz_ann_config.yaml`, 预测代码：`/home/tiandc/photoz/lsdr9x10/photoz_ann/predict.py`
- NNC的训练代码、配置文件、预测代码：`/home/tiandc/photoz/lsdr9x10/photoz_bin/photoz_bin.py` `/home/tiandc/photoz/lsdr9x10/photoz_bin/photoz_bin_config.yaml` `/home/tiandc/photoz/lsdr9x10/photoz_bin/predict.ipynb`
- xgboost的代码：`/home/tiandc/photoz/lsdr9x10/photoz_xgb/photoz_xgb.py`
- random forest的代码：`/home/tiandc/photoz/lsdr9x10/photoz_rf/photoz_RF.py`
- lstm的训练代码、配置文件、预测代码：`/home/tiandc/photoz/panstarrs/photoz_lstm/photoz_lstm.py` `/home/tiandc/photoz/panstarrs/photoz_lstm/config.yaml` `/home/tiandc/photoz/panstarrs/photoz_lstm/predictWithUncertainty.py`
- ensemble的代码：`/home/tiandc/photoz/panstarrs/photoz_ensemble/predict_ensemble.py`
- knn的训练代码、配置文件、预测代码：`/home/tiandc/photoz/panstarrs/photoz_knn/photoz_knn.py` `/home/tiandc/photoz/panstarrs/photoz_knn/config.yaml`


## catalog制作

1. `/home/tiandc/photoz/catalog/models`这个目录包含了P1-P5模型的训练数据和配置以及一些结果输出。
2. `/home/tiandc/photoz/catalog/predict.py` 这个文件包含了对发布数据LSDR10和PS1DR2的红移predict, merge, publish


### catalog中的星系筛选

`/home/tiandc/photoz/catalog/predict.py`这个代码中我还对源进行了分类，评估源为星系的概率`galaxy_prob`。

1. 星系筛选我也训练了模型，我所使用的模型在`/home/tiandc/photoz/catalog/models/galaxy_clf_ls`和`/home/tiandc/photoz/catalog/models/galaxy_clf_ps`
2. 星系分类训练所在的主目录为：`/home/tiandc/photoz/galaxyClf`，包含了两类算法训练，ANN和Xgboost。ANN对应的模型训练、配置和预测代码为：`/home/tiandc/photoz/galaxyClf/galaxyClf_ann/galaxyClf_ann.py` `/home/tiandc/photoz/galaxyClf/galaxyClf_ann/galaxyClf_ann_config.yaml` `/home/tiandc/photoz/galaxyClf/galaxyClf_ann/LSDR10_predict.ipynb`， XGBOOST所用代码为：`/home/tiandc/photoz/galaxyClf/galaxyClf_xgb/galaxyClf_xgb.py` `/home/tiandc/photoz/galaxyClf/galaxyClf_xgb/LSDR10_predict.ipynb`
3. `/home/tiandc/photoz/catalog/predict.py`中所用的分类模型是`/home/tiandc/photoz/galaxyClf/galaxyClf_ann/galaxyClf_ann.py`


# 论文绘图相关

论文图片和表格以及相关数据在`/home/tiandc/photoz/review/plot_reply.ipynb`这个文件中