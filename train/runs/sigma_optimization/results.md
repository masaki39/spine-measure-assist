# Sigma Optimization Results — smallUNet

Evaluated on N=186 samples from `dataset/`, input size 512×512.

## Overall

| sigma | MRE(px) | MRE(mm) | SDR@2mm | SDR@4mm |
|------:|--------:|--------:|--------:|--------:|
| 3     | 10.39   | 4.46    | 29.6%   | 65.6%   |
| **4** | **5.72**| **2.46**| **59.5%**| **91.3%** |
| 5     | 11.43   | 4.90    | 26.7%   | 68.3%   |
| 15    | 13.56   | 5.82    | 11.3%   | 39.2%   |

**Conclusion: sigma=4 is optimal.**

## Per-landmark (sigma=4)

| Landmark | MRE(px) | MRE(mm) | SDR@2mm | SDR@4mm |
|----------|--------:|--------:|--------:|--------:|
| L1_ant   | 6.66    | 2.86    | 49.5%   | 90.9%   |
| L1_post  | 5.25    | 2.25    | 72.6%   | 93.0%   |
| S1_ant   | 4.98    | 2.14    | 65.1%   | 92.5%   |
| S1_post  | 5.03    | 2.16    | 56.5%   | 90.3%   |
| FH       | 6.70    | 2.87    | 53.8%   | 89.8%   |
| Overall  | 5.72    | 2.46    | 59.5%   | 91.3%   |

## Notes

- sigma=3 is sharper but performs worse — too sharp for this architecture/dataset size
- sigma=4 is the sweet spot; SDR@4mm 91.3% well above the practical threshold of 75%
- Next: learning curve analysis with sigma=4 fixed (vary training data size)
