# Approche RFPC 

**Imports**


```python
import pandas as pd
import numpy as np
from S_RFPC import RFPC, verification_bloc
```


```python
data = pd.read_table("mobil_init.txt").iloc[:, 1:]
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUEX1</th>
      <th>CUEX2</th>
      <th>CUEX3</th>
      <th>PERQ1</th>
      <th>PERQ2</th>
      <th>PERQ3</th>
      <th>PERQ4</th>
      <th>PERQ5</th>
      <th>PERQ6</th>
      <th>PERQ7</th>
      <th>PERV1</th>
      <th>PERV2</th>
      <th>CUSA1</th>
      <th>CUSA2</th>
      <th>CUSA3</th>
      <th>CUSL1</th>
      <th>CUSL2</th>
      <th>CUSL3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9</td>
      <td>9</td>
      <td>8</td>
      <td>9</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>7</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>7</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>7</td>
      <td>9</td>
      <td>9</td>
      <td>7</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>9</td>
      <td>9</td>
      <td>8</td>
      <td>7</td>
      <td>9</td>
      <td>8</td>
      <td>8</td>
      <td>7</td>
      <td>5</td>
      <td>5</td>
      <td>9</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
      <td>1</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



**Définition des liens**

Il faut pour chaque variable créer un vecteur contenant des 0 ou des 1 afin de définir quelles variables pointent sur quelles autres variables. La valeur 1 signifie que la variable est pointée par une autre. La position du 1 permet de savoir quelle est la variable en question.

_Par exemple, **pqual** est pointée par la variable **expect**, car on trouve un 1 dans la première position du vecteur **pqual**._


```python
expect = [0, 0, 0, 0, 0]
pqual = [1, 0, 0, 0, 0]
pval = [1, 1, 0, 0, 0]
satis = [1, 1, 1, 0, 0]
loyal = [0, 0, 0, 1, 0]

path = np.array([expect, pqual, pval, satis, loyal])
vl = ["expect", "pqual", "pval", "satis", "loyal"]
path = pd.DataFrame(path, columns=vl, index=vl)
```

On affiche la matrice sous la forme d'un DataFrame (nécéssaire pour la fonction RFPC)


```python
path
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>expect</th>
      <th>pqual</th>
      <th>pval</th>
      <th>satis</th>
      <th>loyal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>expect</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>pqual</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>pval</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>satis</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>loyal</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Définition des blocs**

La seconde étape consiste à l'attribution de chaque variable manifeste à son bloc. Pour cela, les données doivent être de la forme d'un DataFrame.

On crée donc une liste de listes avec les indices de début et de fin des variable de chaque bloc. La fonction _vérification\_bloc_ permet de vérifier si les blocs sont exacts.


```python
block = [[0, 3], [3, 10], [10, 12], [12, 15], [15, 18]]
```


```python
verification_bloc(block, vl, data)
```

    Bloc : expect 
    Variables manifestes : ['CUEX1', 'CUEX2', 'CUEX3'] 
    
    Bloc : pqual 
    Variables manifestes : ['PERQ1', 'PERQ2', 'PERQ3', 'PERQ4', 'PERQ5', 'PERQ6', 'PERQ7'] 
    
    Bloc : pval 
    Variables manifestes : ['PERV1', 'PERV2'] 
    
    Bloc : satis 
    Variables manifestes : ['CUSA1', 'CUSA2', 'CUSA3'] 
    
    Bloc : loyal 
    Variables manifestes : ['CUSL1', 'CUSL2', 'CUSL3'] 
    


__Estimation du modèle__

On renseigne les informations précédente dans la fonction _RFPC_ : 
- **les données** – (Dataframe)
- **blocs** – (liste de liste des indices)
- **nom des variables sous forme de liste** – (liste de _str_ des noms des variables) 
- **matrice des liens** – (matrice des liens sous forme de Dataframe).


```python
model = RFPC(data, block, vl, path, verbose=True)
```

    ['CUEX1', 'CUEX2', 'CUEX3']
    ['PERQ1', 'PERQ2', 'PERQ3', 'PERQ4', 'PERQ5', 'PERQ6', 'PERQ7']
    ['PERV1', 'PERV2']
    ['CUSA1', 'CUSA2', 'CUSA3']
    ['CUSL1', 'CUSL2', 'CUSL3']


On peut récupérer les différentes sorties : modèle interne, modèle externe, récapitulatif du modèle externe et le GOF


```python
model.final_inner
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R2</th>
      <th>AVE</th>
      <th>Mean_redundancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>expect</td>
      <td>0.000</td>
      <td>0.475</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>pqual</td>
      <td>0.284</td>
      <td>0.576</td>
      <td>0.163</td>
    </tr>
    <tr>
      <td>pval</td>
      <td>0.334</td>
      <td>0.846</td>
      <td>0.282</td>
    </tr>
    <tr>
      <td>satis</td>
      <td>0.653</td>
      <td>0.688</td>
      <td>0.449</td>
    </tr>
    <tr>
      <td>loyal</td>
      <td>0.403</td>
      <td>0.523</td>
      <td>0.211</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.final_outer
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bloc</th>
      <th>weight</th>
      <th>loading</th>
      <th>communality</th>
      <th>redundancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CUEX1</td>
      <td>expect</td>
      <td>0.560205</td>
      <td>0.798378</td>
      <td>0.637407</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>CUEX2</td>
      <td>expect</td>
      <td>0.504299</td>
      <td>0.718702</td>
      <td>0.516533</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>CUEX3</td>
      <td>expect</td>
      <td>0.365421</td>
      <td>0.520781</td>
      <td>0.271212</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>PERQ1</td>
      <td>pqual</td>
      <td>0.198873</td>
      <td>0.802172</td>
      <td>0.643479</td>
      <td>0.182574</td>
    </tr>
    <tr>
      <td>PERQ2</td>
      <td>pqual</td>
      <td>0.158860</td>
      <td>0.640776</td>
      <td>0.410594</td>
      <td>0.116498</td>
    </tr>
    <tr>
      <td>PERQ3</td>
      <td>pqual</td>
      <td>0.191658</td>
      <td>0.773069</td>
      <td>0.597635</td>
      <td>0.169567</td>
    </tr>
    <tr>
      <td>PERQ4</td>
      <td>pqual</td>
      <td>0.194011</td>
      <td>0.782559</td>
      <td>0.612398</td>
      <td>0.173756</td>
    </tr>
    <tr>
      <td>PERQ5</td>
      <td>pqual</td>
      <td>0.186375</td>
      <td>0.751759</td>
      <td>0.565142</td>
      <td>0.160348</td>
    </tr>
    <tr>
      <td>PERQ6</td>
      <td>pqual</td>
      <td>0.197203</td>
      <td>0.795438</td>
      <td>0.632721</td>
      <td>0.179522</td>
    </tr>
    <tr>
      <td>PERQ7</td>
      <td>pqual</td>
      <td>0.187440</td>
      <td>0.756055</td>
      <td>0.571619</td>
      <td>0.162185</td>
    </tr>
    <tr>
      <td>PERV1</td>
      <td>pval</td>
      <td>0.543726</td>
      <td>0.919580</td>
      <td>0.845628</td>
      <td>0.282182</td>
    </tr>
    <tr>
      <td>PERV2</td>
      <td>pval</td>
      <td>0.543726</td>
      <td>0.919580</td>
      <td>0.845628</td>
      <td>0.282182</td>
    </tr>
    <tr>
      <td>CUSA1</td>
      <td>satis</td>
      <td>0.388158</td>
      <td>0.801101</td>
      <td>0.641763</td>
      <td>0.419227</td>
    </tr>
    <tr>
      <td>CUSA2</td>
      <td>satis</td>
      <td>0.413414</td>
      <td>0.853225</td>
      <td>0.727992</td>
      <td>0.475556</td>
    </tr>
    <tr>
      <td>CUSA3</td>
      <td>satis</td>
      <td>0.403675</td>
      <td>0.833124</td>
      <td>0.694095</td>
      <td>0.453413</td>
    </tr>
    <tr>
      <td>CUSL1</td>
      <td>loyal</td>
      <td>0.550070</td>
      <td>0.862916</td>
      <td>0.744624</td>
      <td>0.299939</td>
    </tr>
    <tr>
      <td>CUSL2</td>
      <td>loyal</td>
      <td>0.156031</td>
      <td>0.244772</td>
      <td>0.059913</td>
      <td>0.024134</td>
    </tr>
    <tr>
      <td>CUSL3</td>
      <td>loyal</td>
      <td>0.557254</td>
      <td>0.874186</td>
      <td>0.764201</td>
      <td>0.307825</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.GOF
```




    0.5100388220518121




```python
for i in model.list_resul_inner:
    print(i)
    print('-'*55)
```

                coef        se      tval          pval
    expect  0.896124  0.090231  9.931481  8.506244e-20
    -------------------------------------------------------
                coef        se      tval          pval
    expect  0.078761  0.066719  1.180499  2.389329e-01
    pqual   0.347012  0.039658  8.750106  3.335519e-16
    -------------------------------------------------------
                coef        se       tval          pval
    expect  0.124906  0.053426   2.337936  2.018955e-02
    pqual   0.421849  0.036228  11.644296  2.869509e-25
    pval    0.257395  0.050706   5.076189  7.582194e-07
    -------------------------------------------------------
              coef        se       tval          pval
    satis  0.55333  0.042697  12.959567  1.058381e-29
    -------------------------------------------------------

