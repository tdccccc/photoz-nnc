# AUTO-CONVERTED FROM dataProcess.ipynb
# (markdown cells are shown as comments; code cells are verbatim)

# ===== CELL 000 [markdown] =====
# # 重新整理数据 `download` -> `raw72`

# ===== CELL 001 [code] =====
from cosmic.utils import merge_fits_by_ra

result = merge_fits_by_ra(
    input_dir="/disk_b/tiandc/DR9x10_Gaia/download",
    output_dir="/disk_b/tiandc/DR9x10_Gaia/raw72",
    output_filename="lsdr9x10_{i:02d}.fits",
    temp_dir="./temp",
    num_processes=60,
    ra_column='ra',
    fits_extension="*.fits",
    data_hdu_index=1,
    buffer_flush_size=2000000,
    bin_size=5.0,
    add_uid=True
)

