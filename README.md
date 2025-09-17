# Estudos de Estatística

# Módulo 1: Fundamentos

Este material cobre **teoria, prática, cálculo e aplicação em projetos de crédito** sobre estatística descritiva e distribuições de probabilidade.  

---

## 1. Tipos de Variáveis  

### Teoria
- **Nominal**: categorias sem ordem (ex.: sexo, estado civil).  
- **Ordinal**: categorias com ordem (ex.: rating A, B, C, D).  
- **Discreta**: contáveis (ex.: nº de parcelas).  
- **Contínua**: escala contínua (ex.: renda, valor de fatura).  

### Exemplo prático
```python
import pandas as pd

df = pd.DataFrame({
    "sexo": ["M", "F", "F", "M"],                 # nominal
    "rating_credito": ["A", "B", "C", "A"],       # ordinal
    "parcelas": [3, 12, 6, 9],                    # discreta
    "renda": [2500.50, 4000.75, 3200.10, 5000.0]  # contínua
})

print(df.dtypes)
```
---
## 2. Medidas de Tendência Central

### Teoria
- **Média**: soma dos valores / nº de observações. Sensível a outliers.
- **Mediana**: valor central. Robusta a outliers.
- **Moda**: valor mais frequente.

### Exemplo prático
```python
import numpy as np
from statistics import mode

valores = [100, 200, 1000, 300, 250]

media = np.mean(valores)
mediana = np.median(valores)
moda = mode(valores)

print("Média:", media)
print("Mediana:", mediana)
print("Moda:", moda)
```

### Aplicação em crédito
- Média da renda de inadimplentes vs. adimplentes.
- Mediana do valor de faturas → reduz impacto de grandes devedores.
- Moda do tipo de contrato mais associado à inadimplência.

---
## 3. Medidas de Dispersão

### Teoria
- **Amplitude** (Range) = máx - mín.
- **Variância (σ²)**: medida de dispersão em relação à média.
- **Desvio-padrão (σ)**: raiz da variância, mesma unidade da variável.
- **IQR (Interquartile Range)** = Q3 - Q1, útil para outliers.

### Exemplo prático
```python
valores = [10, 12, 14, 20, 100]

amplitude = np.max(valores) - np.min(valores)
variancia = np.var(valores)
desvio_padrao = np.std(valores)
q1, q3 = np.percentile(valores, [25, 75])
iqr = q3 - q1

print("Amplitude:", amplitude)
print("Variância:", variancia)
print("Desvio padrão:", desvio_padrao)
print("IQR:", iqr)
```

### Aplicação em crédito

- Desvio-padrão da renda → medir dispersão entre clientes.
- IQR para detectar clientes com faturas muito altas → risco.
- Variância da frequência de pagamentos → instabilidade.

---
## 4. Distribuições de Probabilidade

### Teoria
- **Normal**: simétrica, útil para padronizar scores.
- **Binomial**: eventos discretos, duas categorias (pagar/não pagar).
- **Poisson**: contagem de eventos (ex.: nº de atrasos em 12 meses).
- **Exponencial**: tempo até ocorrência de um evento (dias até pagamento).

### Exemplo prático
```python
from scipy.stats import norm, binom, poisson, expon

# Normal: probabilidade de score < 650
prob_normal = norm.cdf(650, loc=600, scale=50)

# Binomial: probabilidade de 3 atrasos em 12 meses (p=0.2)
prob_binomial = binom.pmf(3, n=12, p=0.2)

# Poisson: probabilidade de 2 atrasos/mês (λ=1.5)
prob_poisson = poisson.pmf(2, mu=1.5)

# Exponencial: probabilidade de pagar até 10 dias (λ=1/20)
prob_exponencial = expon.cdf(10, scale=20)

print("Probabilidade Normal:", prob_normal)
print("Probabilidade Binomial:", prob_binomial)
print("Probabilidade Poisson:", prob_poisson)
print("Probabilidade Exponencial:", prob_exponencial)
```

### Aplicação em crédito

- Normal: padronizar score de clientes.
- Binomial: prever inadimplência (0/1).
- Poisson: nº de atrasos esperados por cliente.
- Exponencial: tempo médio até regularização da dívida.

---

## 5. Pipeline em Projetos de Crédito

### Teoria
- **Identificar variáveis**: nominal, ordinal, discreta, contínua.
- **Analisar tendência central**: renda média de inadimplentes vs. adimplentes.
- **Calcular dispersão**: desvio-padrão, IQR para outliers.

### Modelar distribuições:
- Binomial → inadimplência.
- Poisson → atrasos recorrentes.
- Exponencial → tempo até pagamento.

Aplicar em modelo preditivo: Regressão Logística ou XGBoost.

---
