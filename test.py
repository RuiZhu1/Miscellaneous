import pandas as pd, numpy as np
d = pd.read_csv('outputs_v5/v5_kitaev_points.csv')
dom = d[d.domain]
ratio = dom.E_low / dom.E_gap
for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    sub = dom[ratio < thr]
    print(f'{thr}: n={len(sub)}  secular_max={sub.secular.max():.2e}  K_relerr_max={sub.K_relerr.max():.2e}')