# BENCHMARKING
Due to the stochastic nature of optuna and genopt frameworks, each benchmarking optimization process has been executed 5 times and the the results has been compared.

The first equation to optimize is the following:
```python
(x**2 - 4*y**3 / z**4) * k**3
```
The second equation to optimize is the following:
```python
np.cos(x) * np.cos(y) * np.cos(z) * np.cos(k)
```

## Genopt
| EQUATIONS   | BEST_SCORE             | WORST_SCORE            | MEAN_SCORE              | STD_SCORES             |
|-------------|------------------------|------------------------|-------------------------|------------------------|
| 1. EQUATION | -5.593052897256293e+26 | -9.784457226929366e+21 | -1.1787250956286476e+26 | 2.210172921843152e+26  |
| 2. EQUATION | -0.9999957291085184    | -0.999935191839478     | -0.9999679358322329     | 2.0933731270107305e-05 |

## Optuna
| EQUATIONS   | BEST_SCORE             | WORST_SCORE            | MEAN_SCORE              | STD_SCORES             |
|-------------|------------------------|------------------------|-------------------------|------------------------|
| 1. EQUATION | -2.410809623576981e+25 | -9992671336.298042     | -4.822497370624008e+24  | 9.642799475755541e+24  |
| 2. EQUATION | -0.935875961425059     | -0.849799179002455     | -0.9002187387500051     | 0.03290155041224169    |

## Comparison
| EQUATIONS   | GENOPT                    | OPTUNA                    |
|-------------|---------------------------|---------------------------|
| 1. EQUATION | **-5.593052897256293e+26**| -2.410809623576981e+25    |
| 2. EQUATION | **-0.9999957291085184**   | -0.935875961425059        |