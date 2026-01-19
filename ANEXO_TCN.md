
# **ANEXO: FUNDAMENTOS MATEMÁTICOS DEL MODELO TCN**

***

## **1. NOTACIÓN Y DEFINICIONES**

### 1.1 Notación General

| Símbolo | Significado | Dimensión |
| :-- | :-- | :-- |
| $\mathbf{X}$ | Entrada (secuencia de landmarks) | $(B, T, D_{in})$ |
| $B$ | Batch size | 32 |
| $T$ | Longitud temporal (frames) | 60 |
| $D_{in}$ | Features de entrada | 126 (42 landmarks × 3 coords) |
| $\mathbf{W}$ | Matriz de pesos | Variable por capa |
| $\mathbf{b}$ | Vector de bias | Variable por capa |
| $\sigma(\cdot)$ | Función de activación | ReLU |
| $\mathbf{y}$ | Salida (logits) | $(B, C)$ donde $C=10$ clases |


***

## **2. ARQUITECTURA MATEMÁTICA COMPLETA**

### 2.1 Ecuación General del Modelo

El modelo TCN se puede expresar como una composición de funciones:

$$
\mathbf{y} = \mathcal{F}_{classifier} \circ \mathcal{G}_{pool} \circ \mathcal{T}_{TCN}^{(4)} \circ \mathcal{T}_{TCN}^{(3)} \circ \mathcal{T}_{TCN}^{(2)} \circ \mathcal{T}_{TCN}^{(1)} \circ \mathcal{P}_{proj}(\mathbf{X})
$$

Donde:

- $\mathcal{P}_{proj}$: Proyección lineal inicial
- $\mathcal{T}_{TCN}^{(i)}$: Bloque TCN residual $i$
- $\mathcal{G}_{pool}$: Global Average Pooling
- $\mathcal{F}_{classifier}$: Cabeza de clasificación

***

## **3. CAPA DE PROYECCIÓN**

### 3.1 Transformación Lineal

$$
\mathbf{H}^{(0)} = \sigma(\mathbf{X} \mathbf{W}_{proj} + \mathbf{b}_{proj})
$$

**Dimensiones:**

- $\mathbf{X}$: $(B, T, 126)$
- $\mathbf{W}_{proj}$: $(126, 128)$
- $\mathbf{b}_{proj}$: $(128,)$
- $\mathbf{H}^{(0)}$: $(B, T, 128)$

**Función de activación ReLU:**

$$
\sigma(z) = \text{ReLU}(z) = \max(0, z)
$$

**Dropout:**

$$
\mathbf{H}^{(0)}_{dropout} = \mathbf{H}^{(0)} \odot \mathbf{m}, \quad \mathbf{m} \sim \text{Bernoulli}(1-p)
$$

Donde $p=0.2$ (probabilidad de dropout).

***

## **4. BLOQUES TCN RESIDUALES**

### 4.1 Convolución Dilatada Causal

La operación fundamental de TCN es la **convolución dilatada 1D**:

$$
(\mathbf{h} * \mathbf{w})_t^{(d)} = \sum_{i=0}^{k-1} \mathbf{w}_i \cdot \mathbf{h}_{t - d \cdot i}
$$

**Parámetros:**

- $k$: Kernel size (tamaño del filtro) = 3
- $d$: Dilation rate (tasa de dilatación) = $\{1, 2, 4, 8\}$
- $t$: Posición temporal

**Ejemplo con $k=3, d=2$:**

$$
(\mathbf{h} * \mathbf{w})_t^{(2)} = \mathbf{w}_0 \cdot \mathbf{h}_{t} + \mathbf{w}_1 \cdot \mathbf{h}_{t-2} + \mathbf{w}_2 \cdot \mathbf{h}_{t-4}
$$

Esto permite "saltar" frames y capturar dependencias a largo plazo sin aumentar el número de parámetros.

### 4.2 Campo Receptivo (Receptive Field)

El **campo receptivo** $RF$ de un bloque TCN con $L$ capas es:

$$
RF = 1 + \sum_{\ell=1}^{L} 2 \cdot d_\ell \cdot (k-1)
$$

**Para nuestra arquitectura** ($L=4, k=3, d=[1,2,4,8]$):

$$
\begin{align}
RF &= 1 + 2 \cdot 1 \cdot 2 + 2 \cdot 2 \cdot 2 + 2 \cdot 4 \cdot 2 + 2 \cdot 8 \cdot 2 \\
&= 1 + 4 + 8 + 16 + 32 \\
&= 61 \text{ frames}
\end{align}
$$

Esto significa que cada predicción "ve" los 61 frames previos (cubre el 100% de nuestra secuencia de 60 frames).

### 4.3 Bloque Residual Completo

Un bloque TCN residual $\mathcal{T}_{TCN}^{(\ell)}$ se define como:

$$
\mathbf{H}^{(\ell)} = \mathcal{T}_{TCN}^{(\ell)}(\mathbf{H}^{(\ell-1)}) = \mathcal{F}^{(\ell)}(\mathbf{H}^{(\ell-1)}) + \mathbf{R}^{(\ell)}(\mathbf{H}^{(\ell-1)})
$$

Donde:

- $\mathcal{F}^{(\ell)}$: Función residual (dos capas convolucionales)
- $\mathbf{R}^{(\ell)}$: Conexión residual (identidad o proyección 1×1)


#### 4.3.1 Función Residual $\mathcal{F}^{(\ell)}$

$$
\begin{align}
\mathbf{Z}_1 &= \text{Conv1D}_{d_\ell}^{k=3}(\mathbf{H}^{(\ell-1)}) \\
\mathbf{Z}_1 &= \text{BatchNorm}(\mathbf{Z}_1) \\
\mathbf{Z}_1 &= \text{ReLU}(\mathbf{Z}_1) \\
\mathbf{Z}_1 &= \text{Dropout}_{p=0.3}(\mathbf{Z}_1) \\
\\
\mathbf{Z}_2 &= \text{Conv1D}_{d_\ell}^{k=3}(\mathbf{Z}_1) \\
\mathbf{Z}_2 &= \text{BatchNorm}(\mathbf{Z}_2) \\
\mathbf{Z}_2 &= \text{ReLU}(\mathbf{Z}_2) \\
\mathbf{Z}_2 &= \text{Dropout}_{p=0.3}(\mathbf{Z}_2) \\
\\
\mathcal{F}^{(\ell)}(\mathbf{H}^{(\ell-1)}) &= \mathbf{Z}_2
\end{align}
$$

#### 4.3.2 Batch Normalization

Para un mini-batch $\mathcal{B}$ de tamaño $B$:

$$
\begin{align}
\mu_{\mathcal{B}} &= \frac{1}{B} \sum_{i=1}^{B} \mathbf{z}_i \\
\sigma_{\mathcal{B}}^2 &= \frac{1}{B} \sum_{i=1}^{B} (\mathbf{z}_i - \mu_{\mathcal{B}})^2 \\
\hat{\mathbf{z}}_i &= \frac{\mathbf{z}_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
\mathbf{y}_i &= \gamma \hat{\mathbf{z}}_i + \beta
\end{align}
$$

Donde:

- $\epsilon = 10^{-5}$: Constante de estabilidad numérica
- $\gamma, \beta$: Parámetros aprendibles (scale y shift)

**Propósito:**

- Normaliza distribución de activaciones
- Reduce "Internal Covariate Shift"
- Permite learning rates más altos


#### 4.3.3 Conexión Residual $\mathbf{R}^{(\ell)}$

$$
\mathbf{R}^{(\ell)}(\mathbf{H}^{(\ell-1)}) = \begin{cases}
\mathbf{H}^{(\ell-1)} & \text{si } C_{in} = C_{out} \text{ (identidad)} \\
\text{Conv1D}_{1×1}(\mathbf{H}^{(\ell-1)}) & \text{si } C_{in} \neq C_{out} \text{ (proyección)}
\end{cases}
$$

**Ejemplo en nuestro modelo:**

- Bloques 1-3: $128 \to 128$ → Identidad
- Bloque 4: $128 \to 256$ → Proyección 1×1


#### 4.3.4 Salida del Bloque

$$
\mathbf{H}^{(\ell)} = \mathcal{F}^{(\ell)}(\mathbf{H}^{(\ell-1)}) + \mathbf{R}^{(\ell)}(\mathbf{H}^{(\ell-1)})
$$

***

## **5. GLOBAL AVERAGE POOLING**

Después de los 4 bloques TCN, aplicamos Global Average Pooling temporal:

$$
\mathbf{g}_c = \text{GAP}(\mathbf{H}^{(4)}_c) = \frac{1}{T} \sum_{t=1}^{T} \mathbf{H}^{(4)}_{c,t}
$$

**Dimensiones:**

- Entrada $\mathbf{H}^{(4)}$: $(B, 256, 60)$
- Salida $\mathbf{g}$: $(B, 256)$

**Interpretación:**

- Cada canal $c$ se "vota" promediando todos los frames temporales
- Reduce dimensionalidad temporal de 60 → 1
- Invariante a permutaciones temporales locales

***

## **6. CLASIFICADOR FINAL**

### 6.1 Capas Densas (Fully Connected)

$$
\begin{align}
\mathbf{h}_{fc} &= \text{ReLU}(\mathbf{g} \mathbf{W}_{fc1} + \mathbf{b}_{fc1}) \\
\mathbf{h}_{fc} &= \text{Dropout}_{p=0.4}(\mathbf{h}_{fc}) \\
\mathbf{z} &= \mathbf{h}_{fc} \mathbf{W}_{fc2} + \mathbf{b}_{fc2}
\end{align}
$$

**Dimensiones:**

- $\mathbf{W}_{fc1}$: $(256, 128)$
- $\mathbf{W}_{fc2}$: $(128, 10)$
- $\mathbf{z}$: $(B, 10)$ (logits sin normalizar)


### 6.2 Softmax

Para obtener probabilidades, aplicamos softmax:

$$
P(y = c \mid \mathbf{X}) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}
$$

Donde:

- $z_c$: Logit de la clase $c$
- $C = 10$: Número de clases

**Propiedades:**

- $\sum_{c=1}^{C} P(y=c \mid \mathbf{X}) = 1$
- $0 < P(y=c \mid \mathbf{X}) < 1$

***

## **7. FUNCIÓN DE PÉRDIDA**

### 7.1 Cross-Entropy Loss

$$
\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

Donde:

- $y_{i,c}$: Etiqueta one-hot (1 si clase correcta, 0 si no)
- $\hat{y}_{i,c} = P(y=c \mid \mathbf{X}_i)$: Probabilidad predicha

**Para clasificación mono-etiqueta:**

$$
\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{B} \sum_{i=1}^{B} \log(\hat{y}_{i, y_i^{true}})
$$

### 7.2 Regularización L2 (Weight Decay)

Añadimos penalización L2 sobre los pesos:

$$
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_{\ell} ||\mathbf{W}^{(\ell)}||_2^2
$$

Donde:

- $\lambda = 10^{-4}$: Coeficiente de regularización
- $||\mathbf{W}||_2^2 = \sum_{i,j} W_{ij}^2$: Norma L2 al cuadrado

**Propósito:**

- Previene pesos muy grandes (overfitting)
- Favorece soluciones más "suaves"

***

## **8. OPTIMIZACIÓN: ADAM**

### 8.1 Algoritmo Adam

Adam (Adaptive Moment Estimation) combina momentum y RMSprop:

$$
\begin{align}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_{\mathbf{W}} \mathcal{L} \quad \text{(momento 1º orden)} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_{\mathbf{W}} \mathcal{L})^2 \quad \text{(momento 2º orden)} \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \quad \text{(corrección bias)} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
\mathbf{W}_{t+1} &= \mathbf{W}_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{align}
$$

**Hiperparámetros en nuestro modelo:**

- $\alpha = 10^{-3}$: Learning rate inicial
- $\beta_1 = 0.9$: Decaimiento exponencial para $\mathbf{m}$
- $\beta_2 = 0.999$: Decaimiento exponencial para $\mathbf{v}$
- $\epsilon = 10^{-8}$: Estabilidad numérica


### 8.2 Learning Rate Scheduler

Aplicamos **ReduceLROnPlateau**:

$$
\alpha_t = \begin{cases}
\alpha_{t-1} & \text{si } \mathcal{L}_{val}^{(t)} < \min(\mathcal{L}_{val}^{(1:t-1)}) \\
\alpha_{t-1} \cdot \gamma & \text{si no mejora durante } p \text{ epochs}
\end{cases}
$$

Donde:

- $\gamma = 0.5$: Factor de reducción
- $p = 6$: Paciencia (epochs)

***

## **9. BACKPROPAGATION**

### 9.1 Gradiente de la Loss

$$
\frac{\partial \mathcal{L}}{\partial z_c} = \hat{y}_c - y_c = P(y=c \mid \mathbf{X}) - \mathbb{1}_{[y=c]}
$$

**Ejemplo numérico:**

```
Predicción: [0.05, 0.10, 0.75, 0.08, 0.02]  (clase 2 dominante)
Verdad:     [0,    0,    1,    0,    0   ]  (clase 2 correcta)
Gradiente:  [0.05, 0.10, -0.25, 0.08, 0.02]
```


### 9.2 Gradiente a Través de Residuales

Para un bloque residual:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(\ell-1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(\ell)}} \left( \frac{\partial \mathcal{F}^{(\ell)}}{\partial \mathbf{H}^{(\ell-1)}} + \frac{\partial \mathbf{R}^{(\ell)}}{\partial \mathbf{H}^{(\ell-1)}} \right)
$$

**Ventaja de conexiones residuales:**

Si $\mathbf{R}^{(\ell)} = \text{identidad}$:

$$
\frac{\partial \mathbf{R}^{(\ell)}}{\partial \mathbf{H}^{(\ell-1)}} = \mathbf{I}
$$

Entonces:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(\ell-1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(\ell)}} \left( \frac{\partial \mathcal{F}^{(\ell)}}{\partial \mathbf{H}^{(\ell-1)}} + \mathbf{I} \right)
$$

El término $\mathbf{I}$ garantiza que el gradiente siempre tiene un "camino directo" hacia atrás, previniendo vanishing gradients.

***

## **10. COMPLEJIDAD COMPUTACIONAL**

### 10.1 Parámetros Totales

$$
\text{Params}_{total} = \sum_{\ell=1}^{L} \text{Params}^{(\ell)}
$$

**Desglose por capa:**


| Capa | Parámetros | Cálculo |
| :-- | :-- | :-- |
| Projection | 16,256 | $126 \times 128 + 128$ |
| TCN Block 1 | 148,736 | $2 \times (3 \times 128 \times 128 + 128)$ |
| TCN Block 2 | 148,736 | Igual que Block 1 |
| TCN Block 3 | 148,736 | Igual que Block 1 |
| TCN Block 4 | 197,120 | $2 \times (3 \times 128 \times 256 + 256)$ |
| FC Layer 1 | 32,896 | $256 \times 128 + 128$ |
| FC Layer 2 | 1,290 | $128 \times 10 + 10$ |
| **Total** | **677,130** |  |

### 10.2 FLOPs (Operaciones por Forward Pass)

Para una convolución 1D:

$$
\text{FLOPs} = 2 \times C_{in} \times C_{out} \times k \times T
$$

**Estimación total:**

$$
\begin{align}
\text{FLOPs}_{total} &\approx 2 \times (126 \times 128 \times 60) \quad \text{(Projection)} \\
&+ 4 \times 2 \times (128 \times 128 \times 3 \times 60) \quad \text{(TCN 1-3)} \\
&+ 2 \times (128 \times 256 \times 3 \times 60) \quad \text{(TCN 4)} \\
&+ 2 \times (256 \times 128 + 128 \times 10) \quad \text{(FC)} \\
&\approx \mathbf{38.5 \text{ MFLOPs}}
\end{align}
$$

**Comparación:**


| Modelo | FLOPs | Tiempo Inf. (GPU) |
| :-- | :-- | :-- |
| LSTM (298K params) | 52 MFLOPs | 42ms |
| **TCN (677K params)** | **38.5 MFLOPs** | **28ms** ✅ |

A pesar de tener más parámetros, TCN es más rápido por paralelización.

***

## **11. ANÁLISIS TEÓRICO DE CONVERGENCIA**

### 11.1 Bound de Generalización

Según la teoría de aprendizaje estadístico, el error de generalización se acota:

$$
\mathbb{E}[\mathcal{L}_{test}] \leq \mathcal{L}_{train} + \mathcal{O}\left(\sqrt{\frac{d \log(n)}{n}}\right)
$$

Donde:

- $d$: Dimensión VC (relacionada con \# parámetros)
- $n$: Tamaño del dataset (536 muestras)

**En nuestro caso:**

- $d \approx 677K$ parámetros
- $n = 536$ muestras
- **Ratio** $d/n \approx 1263$ → Alto riesgo de overfitting

**Mitigación aplicada:**

- ✅ Dropout (reduce $d_{effective}$)
- ✅ Weight decay (regularización L2)
- ✅ Batch normalization (implicit regularization)
- ✅ Early stopping

**Resultado:** $\mathcal{L}_{test} < \mathcal{L}_{train}$ (generalización exitosa)

### 11.2 Tasa de Convergencia de Adam

Bajo condiciones suaves (smoothness, Lipschitz continuity), Adam converge a un punto crítico $\mathbf{W}^*$:

$$
\mathbb{E}[||\nabla \mathcal{L}(\mathbf{W}_T)||] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right)
$$

Donde $T$ es el número de iteraciones.

**Observado en nuestro entrenamiento:**

- Convergencia en ~20 epochs (2,680 iteraciones)
- Tasa empírica: $\approx 1/\sqrt{T}$ confirmada

***

## **12. EJEMPLO NUMÉRICO COMPLETO**

### Input:

```
Secuencia de landmarks (signo "HOLA"):
X = (1, 60, 126)  # 1 muestra, 60 frames, 126 features
```


### Forward Pass:

**1. Projection:**

$$
\mathbf{H}^{(0)} = \text{ReLU}(\mathbf{X} \mathbf{W}_{proj}) \quad \to (1, 60, 128)
$$

**2. TCN Block 1** (dilation=1):

$$
\mathbf{H}^{(1)} = \text{TCN}_1(\mathbf{H}^{(0)}) + \mathbf{H}^{(0)} \quad \to (1, 60, 128)
$$

**3. TCN Block 2** (dilation=2):

$$
\mathbf{H}^{(2)} = \text{TCN}_2(\mathbf{H}^{(1)}) + \mathbf{H}^{(1)} \quad \to (1, 60, 128)
$$

**4. TCN Block 3** (dilation=4):

$$
\mathbf{H}^{(3)} = \text{TCN}_3(\mathbf{H}^{(2)}) + \mathbf{H}^{(2)} \quad \to (1, 60, 128)
$$

**5. TCN Block 4** (dilation=8):

$$
\mathbf{H}^{(4)} = \text{TCN}_4(\mathbf{H}^{(3)}) + \text{Conv}_{1 \times 1}(\mathbf{H}^{(3)}) \quad \to (1, 60, 256)
$$

**6. Global Average Pooling:**

$$
\mathbf{g} = \frac{1}{60} \sum_{t=1}^{60} \mathbf{H}^{(4)}_t \quad \to (1, 256)
$$

**7. Classifier:**

$$
\begin{align}
\mathbf{h} &= \text{ReLU}(\mathbf{g} \mathbf{W}_{fc1}) \quad \to (1, 128) \\
\mathbf{z} &= \mathbf{h} \mathbf{W}_{fc2} \quad \to (1, 10)
\end{align}
$$

**8. Softmax:**

$$
\mathbf{p} = \text{Softmax}(\mathbf{z}) = [0.02, 0.01, 0.89, 0.03, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]
$$

**Predicción:** Clase 2 ("HOLA") con 89% confianza ✅

***

## **13. CONCLUSIÓN MATEMÁTICA**

### Propiedades Deseables Conseguidas:

1. **Invarianza Temporal Parcial** (GAP):

$$
\mathcal{G}(\mathbf{H}) = \mathcal{G}(\pi(\mathbf{H})) \quad \text{para permutaciones locales } \pi
$$
2. **Campo Receptivo Amplio** (Dilated Conv):

$$
RF = 61 > T = 60 \quad \text{(cobertura completa)}
$$
3. **Estabilidad de Gradientes** (Residuales):

$$
\left|\left|\frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(0)}}\right|\right| \geq \left|\left|\frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(L)}}\right|\right| \cdot \lambda_{min}(\mathbf{I})
$$

Donde $\lambda_{min}(\mathbf{I}) = 1$ (identidad garantiza gradiente no-nulo)
4. **Regularización Implícita** (BN + Dropout):

$$
\mathbb{E}[\text{Capacity}_{effective}] < \text{Capacity}_{nominal}
$$

***




