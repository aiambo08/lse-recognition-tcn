# Benchmarking Comparativo de Arquitecturas de LSE

| Modelo | Parámetros | Tamaño (MB) | Latencia CPU (ms) | FPS | Acc. Signer-Dependent (%) | F1 Macro (%) | Acc. Signer-Independent (%) | Gen. Gap (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard TCN | 677130 | 2.58 | 17.38 | 57.5 | 86.67 | 83.24 | 78.4 | 8.27 |
| Multi-Scale TCN (MS-TCN) | 697493 | 2.66 | 21.84 | 45.8 | 100.0 | 100.0 | 92.55 | 7.45 |
| BiLSTM | 298250 | 1.14 | 20.54 | 48.7 | 40.0 | 29.59 | 27.16 | 12.84 |
| Attention-BiLSTM | 743434 | 2.84 | 48.31 | 20.7 | 96.67 | 96.57 | 89.59 | 7.08 |
| ST-GCN (Hand Skeleton) | 7859382 | 29.98 | 33.54 | 29.8 | 93.33 | 92.03 | 80.74 | 12.59 |

## Tabla en Formato LaTeX (IEEE/ACM):

```latex
\begin{table*}[t]
\centering
\caption{Benchmarking Comparativo de Arquitecturas de Deep Learning para Reconocimiento de LSE.}
\label{tab:lse_benchmark}
\begin{tabular}{lcccccc}
\hline
\textbf{Modelo} & \textbf{Parámetros} & \textbf{Latencia (ms)} & \textbf{Throughput (FPS)} & \textbf{Acc. Dep. (\%)} & \textbf{Acc. Indep. (\%)} & \textbf{$\Delta$ Gen. (\%)} \\
\hline
Standard TCN & 677,130 & 17.38 & 57.5 & 86.7 & 78.4 & 8.3 \\
Multi-Scale TCN (MS-TCN) & 697,493 & 21.84 & 45.8 & 100.0 & 92.5 & 7.5 \\
BiLSTM & 298,250 & 20.54 & 48.7 & 40.0 & 27.2 & 12.8 \\
Attention-BiLSTM & 743,434 & 48.31 & 20.7 & 96.7 & 89.6 & 7.1 \\
ST-GCN (Hand Skeleton) & 7,859,382 & 33.54 & 29.8 & 93.3 & 80.7 & 12.6 \\
\hline
\end{tabular}
\end{table*}
```
