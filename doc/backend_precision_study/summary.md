# Backend x precision x Ns benchmark study — summary

| backend | precision | Ns | wall[s] | RSS[GiB] | iters | converged | contact_area |
|---|---|---|---|---|---|---|---|
| h2 | double | 256 | 0.09 | 0.06 | 10 | True | 0.00204 |
| h2 | double | 512 | 0.20 | 0.09 | 11 | True | 0.00145 |
| h2 | double | 1024 | 0.80 | 0.20 | 13 | True | 0.00119 |
| h2 | double | 2048 | 3.60 | 0.63 | 16 | True | 0.00086 |
| h2 | double | 4096 | 22.24 | 2.31 | 22 | True | 0.00068 |
| h2 | double | 8192 | 305.97 | 8.07 | 40 | True | 0.00055 |
| h2 | double | 16384 | 1167.63 | 24.30 | 38 | True | 0.00046 |
| h2 | float | 256 | 0.08 | 0.06 | 8 | True | 0.00204 |
| h2 | float | 512 | 0.14 | 0.08 | 8 | True | 0.00145 |
| h2 | float | 1024 | 0.51 | 0.16 | 9 | True | 0.00119 |
| h2 | float | 2048 | 1.84 | 0.46 | 11 | True | 0.00086 |
| h2 | float | 4096 | 10.41 | 1.66 | 17 | True | 0.00068 |
| h2 | float | 8192 | 104.55 | 6.11 | 28 | True | 0.00055 |
| h2 | float | 16384 | 395.90 | 24.30 | 25 | True | 0.00046 |
| fft | double | 256 | 0.06 | 0.06 | 10 | True | 0.00204 |
| fft | double | 512 | 0.11 | 0.10 | 11 | True | 0.00145 |
| fft | double | 1024 | 1.53 | 0.24 | 13 | True | 0.00119 |
| fft | double | 2048 | 11.90 | 0.80 | 16 | True | 0.00086 |
| fft | double | 4096 | 40.67 | 3.02 | 22 | True | 0.00068 |
| fft | double | 8192 | 432.61 | 11.87 | 46 | True | 0.00055 |
| fft | float | 256 | 0.06 | 0.06 | 8 | True | 0.00204 |
| fft | float | 512 | 0.08 | 0.09 | 8 | True | 0.00145 |
| fft | float | 1024 | 0.67 | 0.19 | 9 | True | 0.00119 |
| fft | float | 2048 | 4.23 | 0.60 | 11 | True | 0.00086 |
| fft | float | 4096 | 14.51 | 2.21 | 17 | True | 0.00068 |
| fft | float | 8192 | 122.90 | 8.62 | 28 | True | 0.00055 |

## Failed / incomplete cases
- fft double Ns=16384: status=oom
- fft float Ns=16384: status=oom
