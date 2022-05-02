# Healthcare Provider Fraud Detection Analysis

### Import labriries


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
```

### Load in data


```python
train = pd.read_csv('D:/DataScienceProjects/Healthcare_Provider_Fraud_Detection_Analysis/Health/Train-1542865627584.csv');
```


```python
train.head()
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
      <th>Provider</th>
      <th>PotentialFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PRV51001</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PRV51003</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PRV51004</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PRV51005</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PRV51007</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.shape
```




    (5410, 2)




```python
outpatient= pd.read_csv("D:/DataScienceProjects/Healthcare_Provider_Fraud_Detection_Analysis/Health/Train_Outpatientdata-1542865627584.csv")
outpatient.head()
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
      <th>BeneID</th>
      <th>ClaimID</th>
      <th>ClaimStartDt</th>
      <th>ClaimEndDt</th>
      <th>Provider</th>
      <th>InscClaimAmtReimbursed</th>
      <th>AttendingPhysician</th>
      <th>OperatingPhysician</th>
      <th>OtherPhysician</th>
      <th>ClmDiagnosisCode_1</th>
      <th>...</th>
      <th>ClmDiagnosisCode_9</th>
      <th>ClmDiagnosisCode_10</th>
      <th>ClmProcedureCode_1</th>
      <th>ClmProcedureCode_2</th>
      <th>ClmProcedureCode_3</th>
      <th>ClmProcedureCode_4</th>
      <th>ClmProcedureCode_5</th>
      <th>ClmProcedureCode_6</th>
      <th>DeductibleAmtPaid</th>
      <th>ClmAdmitDiagnosisCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11002</td>
      <td>CLM624349</td>
      <td>2009-10-11</td>
      <td>2009-10-11</td>
      <td>PRV56011</td>
      <td>30</td>
      <td>PHY326117</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78943</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>56409</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE11003</td>
      <td>CLM189947</td>
      <td>2009-02-12</td>
      <td>2009-02-12</td>
      <td>PRV57610</td>
      <td>80</td>
      <td>PHY362868</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6115</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>79380</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE11003</td>
      <td>CLM438021</td>
      <td>2009-06-27</td>
      <td>2009-06-27</td>
      <td>PRV57595</td>
      <td>10</td>
      <td>PHY328821</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2723</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE11004</td>
      <td>CLM121801</td>
      <td>2009-01-06</td>
      <td>2009-01-06</td>
      <td>PRV56011</td>
      <td>40</td>
      <td>PHY334319</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>71988</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE11004</td>
      <td>CLM150998</td>
      <td>2009-01-22</td>
      <td>2009-01-22</td>
      <td>PRV56011</td>
      <td>200</td>
      <td>PHY403831</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>82382</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>71947</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
outpatient.shape
```




    (517737, 27)




```python
inpatient= pd.read_csv("D:/DataScienceProjects/Healthcare_Provider_Fraud_Detection_Analysis/Health/Train_Inpatientdata-1542865627584.csv")
inpatient.head()
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
      <th>BeneID</th>
      <th>ClaimID</th>
      <th>ClaimStartDt</th>
      <th>ClaimEndDt</th>
      <th>Provider</th>
      <th>InscClaimAmtReimbursed</th>
      <th>AttendingPhysician</th>
      <th>OperatingPhysician</th>
      <th>OtherPhysician</th>
      <th>AdmissionDt</th>
      <th>...</th>
      <th>ClmDiagnosisCode_7</th>
      <th>ClmDiagnosisCode_8</th>
      <th>ClmDiagnosisCode_9</th>
      <th>ClmDiagnosisCode_10</th>
      <th>ClmProcedureCode_1</th>
      <th>ClmProcedureCode_2</th>
      <th>ClmProcedureCode_3</th>
      <th>ClmProcedureCode_4</th>
      <th>ClmProcedureCode_5</th>
      <th>ClmProcedureCode_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>CLM46614</td>
      <td>2009-04-12</td>
      <td>2009-04-18</td>
      <td>PRV55912</td>
      <td>26000</td>
      <td>PHY390922</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-04-12</td>
      <td>...</td>
      <td>2724</td>
      <td>19889</td>
      <td>5849</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE11001</td>
      <td>CLM66048</td>
      <td>2009-08-31</td>
      <td>2009-09-02</td>
      <td>PRV55907</td>
      <td>5000</td>
      <td>PHY318495</td>
      <td>PHY318495</td>
      <td>NaN</td>
      <td>2009-08-31</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7092.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE11001</td>
      <td>CLM68358</td>
      <td>2009-09-17</td>
      <td>2009-09-20</td>
      <td>PRV56046</td>
      <td>5000</td>
      <td>PHY372395</td>
      <td>NaN</td>
      <td>PHY324689</td>
      <td>2009-09-17</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE11011</td>
      <td>CLM38412</td>
      <td>2009-02-14</td>
      <td>2009-02-22</td>
      <td>PRV52405</td>
      <td>5000</td>
      <td>PHY369659</td>
      <td>PHY392961</td>
      <td>PHY349768</td>
      <td>2009-02-14</td>
      <td>...</td>
      <td>25062</td>
      <td>40390</td>
      <td>4019</td>
      <td>NaN</td>
      <td>331.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE11014</td>
      <td>CLM63689</td>
      <td>2009-08-13</td>
      <td>2009-08-30</td>
      <td>PRV56614</td>
      <td>10000</td>
      <td>PHY379376</td>
      <td>PHY398258</td>
      <td>NaN</td>
      <td>2009-08-13</td>
      <td>...</td>
      <td>5119</td>
      <td>29620</td>
      <td>20300</td>
      <td>NaN</td>
      <td>3893.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
inpatient.shape
```




    (40474, 30)




```python
beneficiary= pd.read_csv("D:/DataScienceProjects/Healthcare_Provider_Fraud_Detection_Analysis/Health/Train_Beneficiarydata-1542865627584.csv")
beneficiary.head()
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
      <th>BeneID</th>
      <th>DOB</th>
      <th>DOD</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>State</th>
      <th>County</th>
      <th>NoOfMonths_PartACov</th>
      <th>NoOfMonths_PartBCov</th>
      <th>...</th>
      <th>ChronicCond_Depression</th>
      <th>ChronicCond_Diabetes</th>
      <th>ChronicCond_IschemicHeart</th>
      <th>ChronicCond_Osteoporasis</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>1943-01-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>230</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>36000</td>
      <td>3204</td>
      <td>60</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE11002</td>
      <td>1936-09-01</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>280</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE11003</td>
      <td>1936-08-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>52</td>
      <td>590</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>90</td>
      <td>40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE11004</td>
      <td>1922-07-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>270</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1810</td>
      <td>760</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE11005</td>
      <td>1935-09-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>680</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1790</td>
      <td>1200</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



### Exploratory Data Analysis

Looking for the most common procedure codes which are applied for the fradulent and non fradulent services to see any specific pattern

#### Inpatient


```python
df_procedures1 =  pd.DataFrame(columns = ['Procedures'])
df_procedures1['Procedures'] = pd.concat([inpatient["ClmProcedureCode_1"], inpatient["ClmProcedureCode_2"], inpatient["ClmProcedureCode_3"], inpatient["ClmProcedureCode_4"], inpatient["ClmProcedureCode_5"], inpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures1['Procedures'].head(10)
```




    1     7092.0
    3      331.0
    4     3893.0
    5      863.0
    6     4576.0
    7     9904.0
    9     3612.0
    10    9672.0
    12    9904.0
    14    9671.0
    Name: Procedures, dtype: float64




```python
df_procedures1.shape
```




    (29692, 1)




```python
grouped_procedure_df = df_procedures1['Procedures'].value_counts()
grouped_procedure_df
```




    4019.0    1953
    9904.0    1137
    2724.0    1047
    8154.0    1021
    66.0       894
              ... 
    5689.0       1
    8853.0       1
    8134.0       1
    9031.0       1
    3343.0       1
    Name: Procedures, Length: 1321, dtype: int64




```python
df_diagnosis = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis['Diagnosis'] = pd.concat([inpatient["ClmDiagnosisCode_1"], inpatient["ClmDiagnosisCode_2"], inpatient["ClmDiagnosisCode_3"], inpatient["ClmDiagnosisCode_4"], inpatient["ClmDiagnosisCode_5"], inpatient["ClmDiagnosisCode_6"], inpatient["ClmDiagnosisCode_7"], inpatient["ClmDiagnosisCode_8"], inpatient["ClmDiagnosisCode_9"], inpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis['Diagnosis'].head(10)
```




    0     1970
    1     6186
    2    29623
    3    43491
    4      042
    5     1745
    6     1536
    7    56212
    8    42823
    9    41041
    Name: Diagnosis, dtype: object




```python
df_diagnosis.shape
```




    (327328, 1)




```python
grouped_diagnosis_df = df_diagnosis['Diagnosis'].value_counts()
grouped_diagnosis_df
```




    4019     14153
    2724      7340
    25000     7334
    41401     6442
    4280      6190
             ...  
    20213        1
    34710        1
    37855        1
    9711         1
    V6141        1
    Name: Diagnosis, Length: 4716, dtype: int64




```python
grouped_procedure_df1 = grouped_procedure_df.to_frame()
grouped_procedure_df1
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
      <th>Procedures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4019.0</th>
      <td>1953</td>
    </tr>
    <tr>
      <th>9904.0</th>
      <td>1137</td>
    </tr>
    <tr>
      <th>2724.0</th>
      <td>1047</td>
    </tr>
    <tr>
      <th>8154.0</th>
      <td>1021</td>
    </tr>
    <tr>
      <th>66.0</th>
      <td>894</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>5689.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8853.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8134.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9031.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3343.0</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1321 rows × 1 columns</p>
</div>




```python
grouped_procedure_df1.columns = ['count']
grouped_procedure_df1
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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4019.0</th>
      <td>1953</td>
    </tr>
    <tr>
      <th>9904.0</th>
      <td>1137</td>
    </tr>
    <tr>
      <th>2724.0</th>
      <td>1047</td>
    </tr>
    <tr>
      <th>8154.0</th>
      <td>1021</td>
    </tr>
    <tr>
      <th>66.0</th>
      <td>894</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>5689.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8853.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8134.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9031.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3343.0</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1321 rows × 1 columns</p>
</div>




```python
grouped_procedure_df1['Procedure'] = grouped_procedure_df1.index
grouped_procedure_df1
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
      <th>count</th>
      <th>Procedure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4019.0</th>
      <td>1953</td>
      <td>4019.0</td>
    </tr>
    <tr>
      <th>9904.0</th>
      <td>1137</td>
      <td>9904.0</td>
    </tr>
    <tr>
      <th>2724.0</th>
      <td>1047</td>
      <td>2724.0</td>
    </tr>
    <tr>
      <th>8154.0</th>
      <td>1021</td>
      <td>8154.0</td>
    </tr>
    <tr>
      <th>66.0</th>
      <td>894</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5689.0</th>
      <td>1</td>
      <td>5689.0</td>
    </tr>
    <tr>
      <th>8853.0</th>
      <td>1</td>
      <td>8853.0</td>
    </tr>
    <tr>
      <th>8134.0</th>
      <td>1</td>
      <td>8134.0</td>
    </tr>
    <tr>
      <th>9031.0</th>
      <td>1</td>
      <td>9031.0</td>
    </tr>
    <tr>
      <th>3343.0</th>
      <td>1</td>
      <td>3343.0</td>
    </tr>
  </tbody>
</table>
<p>1321 rows × 2 columns</p>
</div>




```python
grouped_procedure_df1['Percentage'] = (grouped_procedure_df1['count']/sum(grouped_procedure_df1['count']))*100
grouped_procedure_df1['Percentage']
```




    4019.0    6.577529
    9904.0    3.829314
    2724.0    3.526202
    8154.0    3.438637
    66.0      3.010912
                ...   
    5689.0    0.003368
    8853.0    0.003368
    8134.0    0.003368
    9031.0    0.003368
    3343.0    0.003368
    Name: Percentage, Length: 1321, dtype: float64




```python
grouped_diagnosis_df = grouped_diagnosis_df.to_frame()
grouped_diagnosis_df.columns = ['count']
grouped_diagnosis_df['Diagnosis'] = grouped_diagnosis_df.index
grouped_diagnosis_df['Percentage'] = (grouped_diagnosis_df['count']/sum(grouped_diagnosis_df['count']))*100
grouped_diagnosis_df['Percentage']
```




    4019     4.323798
    2724     2.242399
    25000    2.240566
    41401    1.968057
    4280     1.891070
               ...   
    20213    0.000306
    34710    0.000306
    37855    0.000306
    9711     0.000306
    V6141    0.000306
    Name: Percentage, Length: 4716, dtype: float64




```python
# taking only top 20 

plot_procedure_df1 = grouped_procedure_df1.head(20)
plot_diagnosis_df1 = grouped_diagnosis_df.head(20)
```


```python
# Plotting the most commonly used diagnosis and procedures 
from matplotlib import pyplot as plt
plot_procedure_df1['Procedure'] = plot_procedure_df1['Procedure'].astype(str)
plot_procedure_df1.sort_values(by=['Percentage'])
plot_procedure_df1.plot(x ='Procedure', y='Percentage', kind='bar', color ='green',
                  title='Procedure Distribution- Inpatient', figsize=(15,10));
```

    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\2503671377.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      plot_procedure_df1['Procedure'] = plot_procedure_df1['Procedure'].astype(str)
    


    
![output_27_1](https://user-images.githubusercontent.com/75635908/166228254-f754ff46-5c5a-4304-b004-5a3dd6ebdd89.png)




```python
plot_diagnosis_df1['Diagnosis'] =  plot_diagnosis_df1['Diagnosis'].astype(str)
plot_diagnosis_df1.sort_values(by=['Percentage'])
plot_diagnosis_df1.plot(x ='Diagnosis', y='Percentage', kind='bar', color ='green',
                  title='Diagnosis Distribution- Inpatient', figsize=(15,10));
```

    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\2313010061.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      plot_diagnosis_df1['Diagnosis'] =  plot_diagnosis_df1['Diagnosis'].astype(str)
    


    
![output_28_1](https://user-images.githubusercontent.com/75635908/166228311-c583310e-4e1a-439d-8236-4a77bf13e354.png)



We see that for inpatient the most common procedure used is 4019, 9904, 2724 among others

We see that for inpatient the most common Diagnosis used is 4019, 2724,25000 among others

#### Outpatient


```python
df_procedures2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures2['Procedures'] = pd.concat([outpatient["ClmProcedureCode_1"], outpatient["ClmProcedureCode_2"], outpatient["ClmProcedureCode_3"], outpatient["ClmProcedureCode_4"], outpatient["ClmProcedureCode_5"], outpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures2['Procedures'].head(10)
```




    8167     9672.0
    9326     4573.0
    14740      66.0
    30435      66.0
    35839    5123.0
    37710    5123.0
    50003    9390.0
    50435     239.0
    51151    8154.0
    51463    7939.0
    Name: Procedures, dtype: float64




```python
grouped_procedure_df2 = df_procedures2['Procedures'].value_counts()
```


```python
df_diagnosis2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis2['Diagnosis'] = pd.concat([outpatient["ClmDiagnosisCode_1"], outpatient["ClmDiagnosisCode_2"], outpatient["ClmDiagnosisCode_3"], outpatient["ClmDiagnosisCode_4"], outpatient["ClmDiagnosisCode_5"], outpatient["ClmDiagnosisCode_6"], outpatient["ClmDiagnosisCode_7"],  outpatient["ClmDiagnosisCode_8"], outpatient["ClmDiagnosisCode_9"], outpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis2['Diagnosis'].head(10)
grouped_diagnosis_df2 = df_diagnosis2['Diagnosis'].value_counts()
```


```python
grouped_procedure_df_op = grouped_procedure_df2.to_frame()
grouped_procedure_df_op.columns = ['count']
grouped_procedure_df_op['Procedure'] = grouped_procedure_df_op.index
grouped_procedure_df_op['Percentage'] = (grouped_procedure_df_op['count']/sum(grouped_procedure_df_op['count']))*100
grouped_procedure_df_op['Percentage']
```




    9904.0    7.352941
    4516.0    3.921569
    3722.0    3.921569
    66.0      3.431373
    5123.0    3.431373
                ...   
    5369.0    0.490196
    7971.0    0.490196
    4311.0    0.490196
    4573.0    0.490196
    4299.0    0.490196
    Name: Percentage, Length: 104, dtype: float64




```python
grouped_diagnosis_df_op = grouped_diagnosis_df2.to_frame()
grouped_diagnosis_df_op.columns = ['count']
grouped_diagnosis_df_op['Diagnosis'] = grouped_diagnosis_df_op.index
grouped_diagnosis_df_op['Percentage'] = (grouped_diagnosis_df_op['count']/sum(grouped_diagnosis_df_op['count']))*100
grouped_diagnosis_df_op['Percentage']
```




    4019     4.647817
    25000    2.218285
    2724     2.100137
    V5869    1.799853
    4011     1.738895
               ...   
    36354    0.000074
    75558    0.000074
    85252    0.000074
    86510    0.000074
    E8262    0.000074
    Name: Percentage, Length: 10846, dtype: float64




```python
# taking only top 20 

plot_procedure_df2 = grouped_procedure_df_op.head(20)
plot_diagnosis_df2 = grouped_diagnosis_df_op.head(20)
```


```python
# Plotting the most commonly used diagnosis and procedures 
from matplotlib import pyplot as plt


plot_procedure_df2['Procedure'] = plot_procedure_df2['Procedure'].astype(str)
plot_procedure_df2.sort_values(by=['Percentage'])
plot_procedure_df2.plot(x ='Procedure', y='Percentage', kind='bar', color ='blue',
                   title='Procedure Distribution- Outpatient', figsize=(15,7));
```

    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\3836996547.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      plot_procedure_df2['Procedure'] = plot_procedure_df2['Procedure'].astype(str)
    


    
![output_37_1](https://user-images.githubusercontent.com/75635908/166228441-4962ad0b-639f-49ee-8ceb-d753cf8a8a75.png)

    



```python
plot_diagnosis_df2['Diagnosis'] = plot_diagnosis_df2['Diagnosis'].astype(str)
plot_diagnosis_df2.sort_values(by=['Percentage'])
plot_diagnosis_df2.plot(x ='Diagnosis', y='Percentage', kind='bar', color ='blue',
                   title='Diagnosis Distribution- Outpatient', figsize=(15,7))
```

    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\3821142940.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      plot_diagnosis_df2['Diagnosis'] = plot_diagnosis_df2['Diagnosis'].astype(str)
    




    <AxesSubplot:title={'center':'Diagnosis Distribution- Outpatient'}, xlabel='Diagnosis'>




    
![output_38_2](https://user-images.githubusercontent.com/75635908/166228740-dc531295-41d6-4bf0-be6d-24270ed8cc75.png)




We see a minor difference between the most used diagnosis and procedure codes between inpatient and outpatients

We see that for inpatient the most common procedure used is 9904, 3722, 4516 among others

We see that for inpatient the most common Diagnosis used is 4019, 25000, 2724 among others


```python
T_fraud = train['PotentialFraud'].value_counts()
grouped_train_df = T_fraud.to_frame()

grouped_train_df.columns = ['count']
grouped_train_df['Fraud'] = grouped_train_df.index
grouped_train_df['Percentage'] = (grouped_train_df['count']/sum(grouped_train_df['count']))*100
grouped_train_df['Percentage'].plot( kind='bar',color = "blue", title = 'Distribution')
```




    <AxesSubplot:title={'center':'Distribution'}>




    
![output_40_1](https://user-images.githubusercontent.com/75635908/166228686-090187de-4a6a-45fd-9e05-d8beff9afe84.png)


#### 2. What are the most common procedures and diagnosis codes performed by the potential fradulent providers

#### Inpatient


```python
Train_f =  pd.DataFrame(columns = ['PotentialFraud', 'Provider'])
Train_f = train.loc[(train['PotentialFraud'] == 'Yes')]
Train_f
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
      <th>Provider</th>
      <th>PotentialFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>PRV51003</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PRV51005</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PRV51021</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PRV51037</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>37</th>
      <td>PRV51052</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5307</th>
      <td>PRV57642</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5324</th>
      <td>PRV57667</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5329</th>
      <td>PRV57672</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5351</th>
      <td>PRV57697</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5360</th>
      <td>PRV57709</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 2 columns</p>
</div>




```python
fraud_provider_ip_df = pd.merge(inpatient, Train_f, how='inner', on='Provider')
fraud_provider_ip_df
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
      <th>BeneID</th>
      <th>ClaimID</th>
      <th>ClaimStartDt</th>
      <th>ClaimEndDt</th>
      <th>Provider</th>
      <th>InscClaimAmtReimbursed</th>
      <th>AttendingPhysician</th>
      <th>OperatingPhysician</th>
      <th>OtherPhysician</th>
      <th>AdmissionDt</th>
      <th>...</th>
      <th>ClmDiagnosisCode_8</th>
      <th>ClmDiagnosisCode_9</th>
      <th>ClmDiagnosisCode_10</th>
      <th>ClmProcedureCode_1</th>
      <th>ClmProcedureCode_2</th>
      <th>ClmProcedureCode_3</th>
      <th>ClmProcedureCode_4</th>
      <th>ClmProcedureCode_5</th>
      <th>ClmProcedureCode_6</th>
      <th>PotentialFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>CLM46614</td>
      <td>2009-04-12</td>
      <td>2009-04-18</td>
      <td>PRV55912</td>
      <td>26000</td>
      <td>PHY390922</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-04-12</td>
      <td>...</td>
      <td>19889</td>
      <td>5849</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE17521</td>
      <td>CLM34721</td>
      <td>2009-01-20</td>
      <td>2009-02-01</td>
      <td>PRV55912</td>
      <td>19000</td>
      <td>PHY349293</td>
      <td>PHY370861</td>
      <td>PHY363291</td>
      <td>2009-01-20</td>
      <td>...</td>
      <td>2753</td>
      <td>E9305</td>
      <td>NaN</td>
      <td>7769.0</td>
      <td>5849.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE21718</td>
      <td>CLM72336</td>
      <td>2009-10-17</td>
      <td>2009-11-04</td>
      <td>PRV55912</td>
      <td>17000</td>
      <td>PHY334706</td>
      <td>PHY334706</td>
      <td>NaN</td>
      <td>2009-10-17</td>
      <td>...</td>
      <td>43812</td>
      <td>4019</td>
      <td>NaN</td>
      <td>9338.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE22934</td>
      <td>CLM73394</td>
      <td>2009-10-25</td>
      <td>2009-10-29</td>
      <td>PRV55912</td>
      <td>13000</td>
      <td>PHY390614</td>
      <td>PHY323689</td>
      <td>PHY363291</td>
      <td>2009-10-25</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8154.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE24402</td>
      <td>CLM32911</td>
      <td>2009-01-08</td>
      <td>2009-01-12</td>
      <td>PRV55912</td>
      <td>3000</td>
      <td>PHY380413</td>
      <td>PHY432598</td>
      <td>NaN</td>
      <td>2009-01-08</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8543.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23397</th>
      <td>BENE142565</td>
      <td>CLM37075</td>
      <td>2009-02-05</td>
      <td>2009-02-09</td>
      <td>PRV55514</td>
      <td>18000</td>
      <td>PHY380221</td>
      <td>PHY392672</td>
      <td>NaN</td>
      <td>2009-02-05</td>
      <td>...</td>
      <td>7265</td>
      <td>7140</td>
      <td>V1582</td>
      <td>8151.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>23398</th>
      <td>BENE105735</td>
      <td>CLM52218</td>
      <td>2009-05-22</td>
      <td>2009-05-30</td>
      <td>PRV56566</td>
      <td>12000</td>
      <td>PHY344703</td>
      <td>PHY344703</td>
      <td>NaN</td>
      <td>2009-05-22</td>
      <td>...</td>
      <td>42789</td>
      <td>2948</td>
      <td>NaN</td>
      <td>9671.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>23399</th>
      <td>BENE135136</td>
      <td>CLM60037</td>
      <td>2009-07-17</td>
      <td>2009-07-19</td>
      <td>PRV55852</td>
      <td>5000</td>
      <td>PHY336944</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-07-17</td>
      <td>...</td>
      <td>51889</td>
      <td>0413</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>23400</th>
      <td>BENE135136</td>
      <td>CLM70804</td>
      <td>2009-10-05</td>
      <td>2009-10-11</td>
      <td>PRV55852</td>
      <td>5000</td>
      <td>PHY356585</td>
      <td>PHY356585</td>
      <td>NaN</td>
      <td>2009-10-05</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>23401</th>
      <td>BENE145679</td>
      <td>CLM41028</td>
      <td>2009-03-03</td>
      <td>2009-03-12</td>
      <td>PRV55852</td>
      <td>12000</td>
      <td>PHY336944</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-03-03</td>
      <td>...</td>
      <td>1977</td>
      <td>78900</td>
      <td>42842</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>23402 rows × 31 columns</p>
</div>




```python
len(fraud_provider_ip_df)
```




    23402




```python
(len(fraud_provider_ip_df)/len(inpatient)) * 100
```




    57.81983495577408



So we see there are 23402 admitted(inpatients) cases that the potential fradulent providers have interacted with at one point or the other during their services at the hospital. This is around 58% of the cases which we have in our inpatient data.


This means from our inpatient dataset for training we can have fradulent activities on more than half of them - 58% are potential fradulent encounters

#### Outpatient


```python
fraud_provider_op_df = pd.merge(outpatient, Train_f, how='inner', on='Provider')
fraud_provider_op_df
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
      <th>BeneID</th>
      <th>ClaimID</th>
      <th>ClaimStartDt</th>
      <th>ClaimEndDt</th>
      <th>Provider</th>
      <th>InscClaimAmtReimbursed</th>
      <th>AttendingPhysician</th>
      <th>OperatingPhysician</th>
      <th>OtherPhysician</th>
      <th>ClmDiagnosisCode_1</th>
      <th>...</th>
      <th>ClmDiagnosisCode_10</th>
      <th>ClmProcedureCode_1</th>
      <th>ClmProcedureCode_2</th>
      <th>ClmProcedureCode_3</th>
      <th>ClmProcedureCode_4</th>
      <th>ClmProcedureCode_5</th>
      <th>ClmProcedureCode_6</th>
      <th>DeductibleAmtPaid</th>
      <th>ClmAdmitDiagnosisCode</th>
      <th>PotentialFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11002</td>
      <td>CLM624349</td>
      <td>2009-10-11</td>
      <td>2009-10-11</td>
      <td>PRV56011</td>
      <td>30</td>
      <td>PHY326117</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78943</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>56409</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE11004</td>
      <td>CLM121801</td>
      <td>2009-01-06</td>
      <td>2009-01-06</td>
      <td>PRV56011</td>
      <td>40</td>
      <td>PHY334319</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>71988</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE11004</td>
      <td>CLM150998</td>
      <td>2009-01-22</td>
      <td>2009-01-22</td>
      <td>PRV56011</td>
      <td>200</td>
      <td>PHY403831</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>82382</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>71947</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE11004</td>
      <td>CLM173224</td>
      <td>2009-02-03</td>
      <td>2009-02-03</td>
      <td>PRV56011</td>
      <td>20</td>
      <td>PHY339887</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20381</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE11004</td>
      <td>CLM224741</td>
      <td>2009-03-03</td>
      <td>2009-03-03</td>
      <td>PRV56011</td>
      <td>40</td>
      <td>PHY345721</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>V6546</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>189389</th>
      <td>BENE144674</td>
      <td>CLM478399</td>
      <td>2009-07-19</td>
      <td>2009-07-19</td>
      <td>PRV56012</td>
      <td>200</td>
      <td>PHY349406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72401</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>7242</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>189390</th>
      <td>BENE158989</td>
      <td>CLM204673</td>
      <td>2009-02-19</td>
      <td>2009-02-20</td>
      <td>PRV56012</td>
      <td>90</td>
      <td>PHY427933</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29590</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>189391</th>
      <td>BENE111157</td>
      <td>CLM82006</td>
      <td>2008-12-15</td>
      <td>2008-12-28</td>
      <td>PRV51119</td>
      <td>95580</td>
      <td>PHY409901</td>
      <td>PHY396304</td>
      <td>PHY396304</td>
      <td>0389</td>
      <td>...</td>
      <td>2762</td>
      <td>9672.0</td>
      <td>5119.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>865</td>
      <td>51881</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>189392</th>
      <td>BENE119614</td>
      <td>CLM738809</td>
      <td>2009-12-21</td>
      <td>2009-12-21</td>
      <td>PRV55472</td>
      <td>90</td>
      <td>PHY358448</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>V187</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>189393</th>
      <td>BENE130881</td>
      <td>CLM165585</td>
      <td>2009-01-29</td>
      <td>2009-01-29</td>
      <td>PRV55472</td>
      <td>0</td>
      <td>PHY413698</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78900</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>78900</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>189394 rows × 28 columns</p>
</div>




```python
len(fraud_provider_op_df)
```




    189394




```python
(len(fraud_provider_op_df)/len(outpatient))*100
```




    36.58112130290089



So we see there are 189394 outpatient cases that the potential fradulent providers have interacted with at one point or the other during their services at the hospital. This is around 37% of the cases which we have in our inpatient data.

This means from our outpatient dataset for training we can have fradulent activities on around 38% of encounters

#### Which were the most used procedure codes and diagnosis codes used by the potential fradulent providers

#### Inpatient


```python
df_procedures2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures2['Procedures'] = pd.concat([fraud_provider_ip_df["ClmProcedureCode_1"], fraud_provider_ip_df["ClmProcedureCode_2"], fraud_provider_ip_df["ClmProcedureCode_3"], fraud_provider_ip_df["ClmProcedureCode_4"], fraud_provider_ip_df["ClmProcedureCode_5"], fraud_provider_ip_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures2['Procedures'].head(10)
```




    1     7769.0
    2     9338.0
    3     8154.0
    4     8543.0
    5     3327.0
    7     3995.0
    10    8741.0
    11    4011.0
    12    2181.0
    13    3723.0
    Name: Procedures, dtype: float64




```python
grouped_F_procedure_df = df_procedures2['Procedures'].value_counts()
grouped_F_procedure_df
```




    4019.0    1137
    2724.0     641
    9904.0     629
    8154.0     614
    66.0       535
              ... 
    5341.0       1
    8915.0       1
    8829.0       1
    4581.0       1
    3343.0       1
    Name: Procedures, Length: 1121, dtype: int64




```python
grouped_F_procedure_df2 = grouped_F_procedure_df.to_frame()
grouped_F_procedure_df2.columns = ['count']
grouped_F_procedure_df2['Procedure'] = grouped_F_procedure_df2.index
grouped_F_procedure_df2['Percentage'] = (grouped_F_procedure_df2['count']/sum(grouped_F_procedure_df2['count']))*100
grouped_F_procedure_df2['Percentage']
```




    4019.0    6.562771
    2724.0    3.699856
    9904.0    3.630592
    8154.0    3.544012
    66.0      3.088023
                ...   
    5341.0    0.005772
    8915.0    0.005772
    8829.0    0.005772
    4581.0    0.005772
    3343.0    0.005772
    Name: Percentage, Length: 1121, dtype: float64




```python
df_diagnosis2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis2['Diagnosis'] = pd.concat([fraud_provider_ip_df["ClmDiagnosisCode_1"], fraud_provider_ip_df["ClmDiagnosisCode_2"], fraud_provider_ip_df["ClmDiagnosisCode_3"], fraud_provider_ip_df["ClmDiagnosisCode_4"], fraud_provider_ip_df["ClmDiagnosisCode_5"], fraud_provider_ip_df["ClmDiagnosisCode_6"], fraud_provider_ip_df["ClmDiagnosisCode_7"],  fraud_provider_ip_df["ClmDiagnosisCode_8"], fraud_provider_ip_df["ClmDiagnosisCode_9"], fraud_provider_ip_df["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis2['Diagnosis'].head(10)
```




    0     1970
    1     4240
    2    V5789
    3    71535
    4     2330
    5     1623
    6     3501
    7    V5789
    8     4280
    9     5579
    Name: Diagnosis, dtype: object




```python
grouped_F_diagnosis_df = df_diagnosis2['Diagnosis'].value_counts()
grouped_F_diagnosis_df
```




    4019     8116
    25000    4248
    2724     4245
    41401    3744
    4280     3585
             ... 
    81101       1
    9664        1
    65811       1
    9802        1
    7284        1
    Name: Diagnosis, Length: 4167, dtype: int64




```python
grouped_F_diagnosis_df2 = grouped_F_diagnosis_df.to_frame()
grouped_F_diagnosis_df2.columns = ['count']
grouped_F_diagnosis_df2['Diagnosis'] = grouped_F_diagnosis_df2.index
grouped_F_diagnosis_df2['Percentage'] = (grouped_F_diagnosis_df2['count']/sum(grouped_F_diagnosis_df2['count']))*100
grouped_F_diagnosis_df2['Percentage']
```




    4019     4.286718
    25000    2.243713
    2724     2.242129
    41401    1.977510
    4280     1.893529
               ...   
    81101    0.000528
    9664     0.000528
    65811    0.000528
    9802     0.000528
    7284     0.000528
    Name: Percentage, Length: 4167, dtype: float64




```python
plot_F_procedure_df1 = grouped_F_procedure_df2.head(20)

plot_F_diagnosis_df1 = grouped_F_diagnosis_df2.head(20)
```


```python
plot_F_procedure_df1.plot(x ='Procedure', y='Percentage', kind = 'bar', color ='g', figsize=(15,7))
```




    <AxesSubplot:xlabel='Procedure'>




    
![output_62_1](https://user-images.githubusercontent.com/75635908/166228890-9421f3d6-b34f-4c16-84f7-8b99b93a5792.png)

    



```python
plot_F_diagnosis_df1.plot(x ='Diagnosis', y='Percentage', kind = 'bar', color ='y', figsize=(15,7))
```




    <AxesSubplot:xlabel='Diagnosis'>




    
![output_63_1](https://user-images.githubusercontent.com/75635908/166228901-a69468cf-6aa8-4db1-8607-69444b667da7.png)

    



```python
df_procedures_op2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures_op2['Procedures'] = pd.concat([fraud_provider_op_df["ClmProcedureCode_1"], fraud_provider_op_df["ClmProcedureCode_2"], fraud_provider_op_df["ClmProcedureCode_3"], fraud_provider_op_df["ClmProcedureCode_4"], fraud_provider_op_df["ClmProcedureCode_5"], fraud_provider_op_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures_op2['Procedures'].head(10)
```




    2340     3723.0
    5843     8703.0
    6666     9904.0
    6832     4573.0
    11110    8154.0
    14060      66.0
    15283    3772.0
    18162     966.0
    19199    4516.0
    21222    4311.0
    Name: Procedures, dtype: float64




```python
grouped_F_procedure_op_df = df_procedures_op2['Procedures'].value_counts()
grouped_F_procedure_op_df.head()
```




    9904.0    5
    4516.0    5
    66.0      4
    9390.0    3
    5123.0    3
    Name: Procedures, dtype: int64




```python
grouped_F_procedure_opdf2 = grouped_F_procedure_op_df.to_frame()
grouped_F_procedure_opdf2.columns = ['count']
grouped_F_procedure_opdf2['Procedure'] = grouped_F_procedure_opdf2.index
grouped_F_procedure_opdf2['Percentage'] = (grouped_F_procedure_opdf2['count']/sum(grouped_F_procedure_opdf2['count']))*100
grouped_F_procedure_opdf2['Percentage'].head()
```




    9904.0    5.555556
    4516.0    5.555556
    66.0      4.444444
    9390.0    3.333333
    5123.0    3.333333
    Name: Percentage, dtype: float64




```python
df_diagnosis_op2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis_op2['Diagnosis'] = pd.concat([fraud_provider_op_df["ClmDiagnosisCode_1"], fraud_provider_op_df["ClmDiagnosisCode_2"], fraud_provider_op_df["ClmDiagnosisCode_3"], fraud_provider_op_df["ClmDiagnosisCode_4"], fraud_provider_op_df["ClmDiagnosisCode_5"], fraud_provider_op_df["ClmDiagnosisCode_6"], fraud_provider_op_df["ClmDiagnosisCode_7"],  fraud_provider_op_df["ClmDiagnosisCode_8"], fraud_provider_op_df["ClmDiagnosisCode_9"], fraud_provider_op_df["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis_op2['Diagnosis'].head()
```




    0    78943
    1    71988
    2    82382
    3    20381
    4    V6546
    Name: Diagnosis, dtype: object




```python
grouped_F_diagnosis_op_df = df_diagnosis2['Diagnosis'].value_counts()
grouped_F_diagnosis_op_df.head()
```




    4019     8116
    25000    4248
    2724     4245
    41401    3744
    4280     3585
    Name: Diagnosis, dtype: int64




```python
grouped_F_diagnosis_opdf2 = grouped_F_diagnosis_op_df.to_frame()
grouped_F_diagnosis_opdf2.columns = ['count']
grouped_F_diagnosis_opdf2['Diagnosis'] = grouped_F_diagnosis_opdf2.index
grouped_F_diagnosis_opdf2['Percentage'] = (grouped_F_diagnosis_opdf2['count']/sum(grouped_F_diagnosis_opdf2['count']))*100
grouped_F_diagnosis_opdf2['Percentage'].head()
```




    4019     4.286718
    25000    2.243713
    2724     2.242129
    41401    1.977510
    4280     1.893529
    Name: Percentage, dtype: float64




```python
plot_F_procedure_opdf1 = grouped_F_procedure_opdf2.head(20)

plot_F_diagnosis_opdf1 = grouped_F_diagnosis_opdf2.head(20)
```


```python
plot_F_procedure_opdf1.plot(x ='Procedure', y='Percentage', kind = 'bar', color ='g', figsize=(15,7))
```




    <AxesSubplot:xlabel='Procedure'>




    
![output_71_1](https://user-images.githubusercontent.com/75635908/166228996-ba16bd62-d46d-4711-9bff-e3bd7dd3c4f5.png)

    



```python
plot_F_diagnosis_opdf1.plot(x ='Diagnosis', y='Percentage', kind = 'bar', color ='y', figsize=(15,7))
```




    <AxesSubplot:xlabel='Diagnosis'>




    
![output_72_1](https://user-images.githubusercontent.com/75635908/166229013-c1fdf4b6-94c4-4e7b-9bb3-c41db96e683a.png)

    


#### 4.Which states/localities have the highest number of potential frauds


```python
beneficiary.head()
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
      <th>BeneID</th>
      <th>DOB</th>
      <th>DOD</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>State</th>
      <th>County</th>
      <th>NoOfMonths_PartACov</th>
      <th>NoOfMonths_PartBCov</th>
      <th>...</th>
      <th>ChronicCond_Depression</th>
      <th>ChronicCond_Diabetes</th>
      <th>ChronicCond_IschemicHeart</th>
      <th>ChronicCond_Osteoporasis</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>1943-01-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>230</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>36000</td>
      <td>3204</td>
      <td>60</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE11002</td>
      <td>1936-09-01</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>280</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE11003</td>
      <td>1936-08-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>52</td>
      <td>590</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>90</td>
      <td>40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE11004</td>
      <td>1922-07-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>270</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1810</td>
      <td>760</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE11005</td>
      <td>1935-09-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>680</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1790</td>
      <td>1200</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
fraud_beneficiary_ip_op_df = pd.merge(beneficiary, fraud_provider_ip_df, how='inner', on='BeneID')
fraud_beneficiary_ip_op_df.head()
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
      <th>BeneID</th>
      <th>DOB</th>
      <th>DOD</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>State</th>
      <th>County</th>
      <th>NoOfMonths_PartACov</th>
      <th>NoOfMonths_PartBCov</th>
      <th>...</th>
      <th>ClmDiagnosisCode_8</th>
      <th>ClmDiagnosisCode_9</th>
      <th>ClmDiagnosisCode_10</th>
      <th>ClmProcedureCode_1</th>
      <th>ClmProcedureCode_2</th>
      <th>ClmProcedureCode_3</th>
      <th>ClmProcedureCode_4</th>
      <th>ClmProcedureCode_5</th>
      <th>ClmProcedureCode_6</th>
      <th>PotentialFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>1943-01-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>230</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>19889</td>
      <td>5849</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE11017</td>
      <td>1940-06-01</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>31</td>
      <td>270</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>25000</td>
      <td>25002</td>
      <td>NaN</td>
      <td>863.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE11028</td>
      <td>1941-12-01</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38</td>
      <td>230</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>4439</td>
      <td>41401</td>
      <td>NaN</td>
      <td>9904.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE11034</td>
      <td>1946-03-01</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>34</td>
      <td>760</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>4111</td>
      <td>4589</td>
      <td>NaN</td>
      <td>3612.0</td>
      <td>4139.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE11034</td>
      <td>1946-03-01</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>34</td>
      <td>760</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>591</td>
      <td>51881</td>
      <td>NaN</td>
      <td>9672.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>




```python
Train_F_Beneficiary_grouped = fraud_beneficiary_ip_op_df['State'].value_counts()
Train_F_Beneficiary_grouped.head()
```




    5     1986
    10    1909
    33    1706
    45    1401
    36    1223
    Name: State, dtype: int64




```python
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped.to_frame()
Train_F_Beneficiary_grouped1['Count'] =  Train_F_Beneficiary_grouped1['State']
Train_F_Beneficiary_grouped1['STATE'] = Train_F_Beneficiary_grouped1.index
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped1.drop(['State'], axis = 1)
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped1.head(20)
Train_F_Beneficiary_grouped1
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
      <th>Count</th>
      <th>STATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1986</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1909</td>
      <td>10</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1706</td>
      <td>33</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1401</td>
      <td>45</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1223</td>
      <td>36</td>
    </tr>
    <tr>
      <th>14</th>
      <td>998</td>
      <td>14</td>
    </tr>
    <tr>
      <th>34</th>
      <td>924</td>
      <td>34</td>
    </tr>
    <tr>
      <th>39</th>
      <td>852</td>
      <td>39</td>
    </tr>
    <tr>
      <th>31</th>
      <td>745</td>
      <td>31</td>
    </tr>
    <tr>
      <th>49</th>
      <td>702</td>
      <td>49</td>
    </tr>
    <tr>
      <th>22</th>
      <td>626</td>
      <td>22</td>
    </tr>
    <tr>
      <th>21</th>
      <td>621</td>
      <td>21</td>
    </tr>
    <tr>
      <th>23</th>
      <td>614</td>
      <td>23</td>
    </tr>
    <tr>
      <th>15</th>
      <td>604</td>
      <td>15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>583</td>
      <td>11</td>
    </tr>
    <tr>
      <th>44</th>
      <td>562</td>
      <td>44</td>
    </tr>
    <tr>
      <th>26</th>
      <td>506</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>494</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>479</td>
      <td>7</td>
    </tr>
    <tr>
      <th>52</th>
      <td>430</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>




```python
Train_F_Beneficiary_grouped1.plot(x ='STATE', y='Count', kind = 'bar', figsize= (15,7));
```


    
![output_78_0](https://user-images.githubusercontent.com/75635908/166229157-0ad51624-b1d8-438b-977d-912ad4c3fd99.png)

    


#### Average Age for the data set and as a comparison for the probable fradulent activites applied on what age range


```python
fraud_beneficiary_ip_op_df['DOB'] =  pd.to_datetime(fraud_beneficiary_ip_op_df['DOB'], format='%Y-%m-%d')  
now = pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') # Assuming this is 2009 data as the last recorded death is for 2009
fraud_beneficiary_ip_op_df['DOB'] = fraud_beneficiary_ip_op_df['DOB'].where(fraud_beneficiary_ip_op_df['DOB'] < now) 
fraud_beneficiary_ip_op_df['age'] = (now - fraud_beneficiary_ip_op_df['DOB']).astype('<m8[Y]')  
ax = fraud_beneficiary_ip_op_df['age'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), edgecolor='b')

```


    
![output_80_0](https://user-images.githubusercontent.com/75635908/166229187-6de92604-c984-4617-bc96-57b7ebe9a0e5.png)

    


This seems logical as most of the patients are of an age >65

#### Inpatient data as a whole not just the fradulent activities


```python
beneficiary['DOB'] =  pd.to_datetime(beneficiary['DOB'], format='%Y-%m-%d')  
now = pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') # Assuming this is 2009 data as the last recorded death is for 2009
beneficiary['DOB'] = beneficiary['DOB'].where(beneficiary['DOB'] < now)
beneficiary['age'] = (now - beneficiary['DOB']).astype('<m8[Y]')
ax = beneficiary['age'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), edgecolor='b')
```


    
![output_83_0](https://user-images.githubusercontent.com/75635908/166229244-fefdbaf7-84cb-49ab-89dc-ed95e1b7b234.png)



Here too we see a similar pattern

#### What is the average cost of potential fraud claims and also what is the cost as % of whole. Checking the outliers for such claims

#### Inpatient


```python
ax = inpatient['InscClaimAmtReimbursed'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), facecolor='g', edgecolor='g')
# Insurance Claim amount reimbursed.
```


    
![output_87_0](https://user-images.githubusercontent.com/75635908/166229248-031253b2-2147-44cd-b7f4-830a797a8c84.png)

    



```python
import seaborn as sns
inpatient_1 = pd.merge(inpatient, train, how='inner', on='Provider')
g = sns.FacetGrid(inpatient_1, col='PotentialFraud', height=8)
g.map(plt.hist, 'InscClaimAmtReimbursed', bins=20, color = 'g')
```




    <seaborn.axisgrid.FacetGrid at 0x1e12d3bbc40>




    
![output_88_1](https://user-images.githubusercontent.com/75635908/166229347-9cfc7394-5a4a-48c1-99de-549e6caf3209.png)

    


We see that it is a significantly large amount which might be fradulent.


```python
inpatient_1 = inpatient_1.loc[(inpatient_1['PotentialFraud'] == 'Yes')]
Total = inpatient_1['InscClaimAmtReimbursed'].sum()
print(Total)
```

    241288510
    

241288510 - around 240 Million dollars worth of claim might have some fradulent activity. Even if we assume that it has just 10% fradulent activity the amount will be quite huge

#### Outpatient


```python
ax = outpatient['InscClaimAmtReimbursed'].plot.hist(bins=100,range=[0,5000], alpha=0.5, figsize=(8, 6), facecolor='c', edgecolor='k')
```


    
![output_93_0](https://user-images.githubusercontent.com/75635908/166229424-8481f97b-1b8f-4309-8664-736fe7c49ec7.png)




```python
outpatient_1 = pd.merge(outpatient, train, how='inner', on='Provider')
g = sns.FacetGrid(outpatient_1, col='PotentialFraud', height=8)
g.map(plt.hist, 'InscClaimAmtReimbursed', bins=20, range=[0, 5000], color ='c')
```




    <seaborn.axisgrid.FacetGrid at 0x1e142227df0>




    
![output_94_1](https://user-images.githubusercontent.com/75635908/166229432-6882bb92-fa31-4732-b4da-b8060ef8f74c.png)

    


#### Checking for missing values in the data set


```python
beneficiary.isna().sum()
```




    BeneID                                  0
    DOB                                     0
    DOD                                137135
    Gender                                  0
    Race                                    0
    RenalDiseaseIndicator                   0
    State                                   0
    County                                  0
    NoOfMonths_PartACov                     0
    NoOfMonths_PartBCov                     0
    ChronicCond_Alzheimer                   0
    ChronicCond_Heartfailure                0
    ChronicCond_KidneyDisease               0
    ChronicCond_Cancer                      0
    ChronicCond_ObstrPulmonary              0
    ChronicCond_Depression                  0
    ChronicCond_Diabetes                    0
    ChronicCond_IschemicHeart               0
    ChronicCond_Osteoporasis                0
    ChronicCond_rheumatoidarthritis         0
    ChronicCond_stroke                      0
    IPAnnualReimbursementAmt                0
    IPAnnualDeductibleAmt                   0
    OPAnnualReimbursementAmt                0
    OPAnnualDeductibleAmt                   0
    age                                     0
    dtype: int64



Only the date of death is empty - makes sense for the people who are alive

#### Adding Age Column


```python
beneficiary['DOB'] = pd.to_datetime(beneficiary['DOB'] , format = '%Y-%m-%d')
beneficiary['DOD'] = pd.to_datetime(beneficiary['DOD'],format = '%Y-%m-%d',errors='ignore')
beneficiary['Age'] = round(((beneficiary['DOD'] - beneficiary['DOB']).dt.days)/365)

## As we see that last DOD value is 2009-12-01 ,which means Beneficiary Details data is of year 2009.
## so we will calculate age of other benficiaries for year 2009.
```


```python
beneficiary.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - beneficiary['DOB']).dt.days)/365),
                                 inplace=True)
```


```python
beneficiary.head()
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
      <th>BeneID</th>
      <th>DOB</th>
      <th>DOD</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>State</th>
      <th>County</th>
      <th>NoOfMonths_PartACov</th>
      <th>NoOfMonths_PartBCov</th>
      <th>...</th>
      <th>ChronicCond_IschemicHeart</th>
      <th>ChronicCond_Osteoporasis</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>age</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>1943-01-01</td>
      <td>NaT</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>230</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>36000</td>
      <td>3204</td>
      <td>60</td>
      <td>70</td>
      <td>66.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE11002</td>
      <td>1936-09-01</td>
      <td>NaT</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>280</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>50</td>
      <td>73.0</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE11003</td>
      <td>1936-08-01</td>
      <td>NaT</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>52</td>
      <td>590</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>90</td>
      <td>40</td>
      <td>73.0</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE11004</td>
      <td>1922-07-01</td>
      <td>NaT</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>39</td>
      <td>270</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1810</td>
      <td>760</td>
      <td>87.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE11005</td>
      <td>1935-09-01</td>
      <td>NaT</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>680</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1790</td>
      <td>1200</td>
      <td>74.0</td>
      <td>74.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
## Creating the master DF
inpatient['EncounterType'] = 0
outpatient['EncounterType'] = 1
frames = [inpatient, outpatient]
TrainInAndOut = pd.concat(frames)
TrainInAndOutBenf = pd.merge(TrainInAndOut, beneficiary, how='inner', on='BeneID')
Master_df = pd.merge(TrainInAndOutBenf, train, how='inner', on='Provider')
```


```python
Master_df.head()
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
      <th>BeneID</th>
      <th>ClaimID</th>
      <th>ClaimStartDt</th>
      <th>ClaimEndDt</th>
      <th>Provider</th>
      <th>InscClaimAmtReimbursed</th>
      <th>AttendingPhysician</th>
      <th>OperatingPhysician</th>
      <th>OtherPhysician</th>
      <th>AdmissionDt</th>
      <th>...</th>
      <th>ChronicCond_Osteoporasis</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>age</th>
      <th>Age</th>
      <th>PotentialFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>CLM46614</td>
      <td>2009-04-12</td>
      <td>2009-04-18</td>
      <td>PRV55912</td>
      <td>26000</td>
      <td>PHY390922</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009-04-12</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>36000</td>
      <td>3204</td>
      <td>60</td>
      <td>70</td>
      <td>66.0</td>
      <td>67.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE16973</td>
      <td>CLM565430</td>
      <td>2009-09-06</td>
      <td>2009-09-06</td>
      <td>PRV55912</td>
      <td>50</td>
      <td>PHY365867</td>
      <td>PHY327147</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>24000</td>
      <td>2136</td>
      <td>450</td>
      <td>200</td>
      <td>77.0</td>
      <td>78.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE17521</td>
      <td>CLM34721</td>
      <td>2009-01-20</td>
      <td>2009-02-01</td>
      <td>PRV55912</td>
      <td>19000</td>
      <td>PHY349293</td>
      <td>PHY370861</td>
      <td>PHY363291</td>
      <td>2009-01-20</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>19000</td>
      <td>1068</td>
      <td>100</td>
      <td>20</td>
      <td>96.0</td>
      <td>96.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE21718</td>
      <td>CLM72336</td>
      <td>2009-10-17</td>
      <td>2009-11-04</td>
      <td>PRV55912</td>
      <td>17000</td>
      <td>PHY334706</td>
      <td>PHY334706</td>
      <td>NaN</td>
      <td>2009-10-17</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>17000</td>
      <td>1068</td>
      <td>1050</td>
      <td>540</td>
      <td>87.0</td>
      <td>87.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE22934</td>
      <td>CLM73394</td>
      <td>2009-10-25</td>
      <td>2009-10-29</td>
      <td>PRV55912</td>
      <td>13000</td>
      <td>PHY390614</td>
      <td>PHY323689</td>
      <td>PHY363291</td>
      <td>2009-10-25</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>27000</td>
      <td>2136</td>
      <td>450</td>
      <td>160</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
Master_df['PotentialFraud'].value_counts()
```




    No     345415
    Yes    212796
    Name: PotentialFraud, dtype: int64




```python
Master_df.shape
```




    (558211, 58)




```python
Master_df.isnull().sum()
```




    BeneID                                  0
    ClaimID                                 0
    ClaimStartDt                            0
    ClaimEndDt                              0
    Provider                                0
    InscClaimAmtReimbursed                  0
    AttendingPhysician                   1508
    OperatingPhysician                 443764
    OtherPhysician                     358475
    AdmissionDt                        517737
    ClmAdmitDiagnosisCode              412312
    DeductibleAmtPaid                     899
    DischargeDt                        517737
    DiagnosisGroupCode                 517737
    ClmDiagnosisCode_1                  10453
    ClmDiagnosisCode_2                 195606
    ClmDiagnosisCode_3                 315156
    ClmDiagnosisCode_4                 393675
    ClmDiagnosisCode_5                 446287
    ClmDiagnosisCode_6                 473819
    ClmDiagnosisCode_7                 492034
    ClmDiagnosisCode_8                 504767
    ClmDiagnosisCode_9                 516396
    ClmDiagnosisCode_10                553201
    ClmProcedureCode_1                 534901
    ClmProcedureCode_2                 552721
    ClmProcedureCode_3                 557242
    ClmProcedureCode_4                 558093
    ClmProcedureCode_5                 558202
    ClmProcedureCode_6                 558211
    EncounterType                           0
    DOB                                     0
    DOD                                554080
    Gender                                  0
    Race                                    0
    RenalDiseaseIndicator                   0
    State                                   0
    County                                  0
    NoOfMonths_PartACov                     0
    NoOfMonths_PartBCov                     0
    ChronicCond_Alzheimer                   0
    ChronicCond_Heartfailure                0
    ChronicCond_KidneyDisease               0
    ChronicCond_Cancer                      0
    ChronicCond_ObstrPulmonary              0
    ChronicCond_Depression                  0
    ChronicCond_Diabetes                    0
    ChronicCond_IschemicHeart               0
    ChronicCond_Osteoporasis                0
    ChronicCond_rheumatoidarthritis         0
    ChronicCond_stroke                      0
    IPAnnualReimbursementAmt                0
    IPAnnualDeductibleAmt                   0
    OPAnnualReimbursementAmt                0
    OPAnnualDeductibleAmt                   0
    age                                     0
    Age                                     0
    PotentialFraud                          0
    dtype: int64




```python
## removing the column DOD and DOB also creating a new column IsDead as we already have the age we do not need date of death and date of birth 

Master_df.loc[Master_df['DOD'].isnull(), 'IsDead'] = '0'
Master_df.loc[(Master_df['DOD'].notnull()), 'IsDead'] = '1'
Master_df = Master_df.drop(['DOD'], axis = 1)
Master_df = Master_df.drop(['DOB'], axis = 1)
```


```python
Master_df = Master_df.drop(['age'], axis = 1) 
```

Calculating the number of days the patient was admitted to the dospital and removing admission and discharge date, For outpatients as they do not get admitted will put number of days admitted = 0


```python
Master_df['AdmissionDt'] = pd.to_datetime(Master_df['AdmissionDt'] , format = '%Y-%m-%d')
Master_df['DischargeDt'] = pd.to_datetime(Master_df['DischargeDt'],format = '%Y-%m-%d')
Master_df['DaysAdmitted'] = ((Master_df['DischargeDt'] - Master_df['AdmissionDt']).dt.days)+1
Master_df.loc[Master_df['EncounterType'] == 1, 'DaysAdmitted'] = '0'
Master_df[['EncounterType','DaysAdmitted','DischargeDt','AdmissionDt']].head()
Master_df = Master_df.drop(['DischargeDt'], axis = 1)
Master_df = Master_df.drop(['AdmissionDt'], axis = 1)
```


```python
Master_df.loc[Master_df['DeductibleAmtPaid'].isnull(), 'DeductibleAmtPaid'] = '0'
```


```python
cols= ['ClmAdmitDiagnosisCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_10',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6']
```


```python
Master_df[cols]= Master_df[cols].replace({np.nan:0})
Master_df
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
      <th>BeneID</th>
      <th>ClaimID</th>
      <th>ClaimStartDt</th>
      <th>ClaimEndDt</th>
      <th>Provider</th>
      <th>InscClaimAmtReimbursed</th>
      <th>AttendingPhysician</th>
      <th>OperatingPhysician</th>
      <th>OtherPhysician</th>
      <th>ClmAdmitDiagnosisCode</th>
      <th>...</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>PotentialFraud</th>
      <th>IsDead</th>
      <th>DaysAdmitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BENE11001</td>
      <td>CLM46614</td>
      <td>2009-04-12</td>
      <td>2009-04-18</td>
      <td>PRV55912</td>
      <td>26000</td>
      <td>PHY390922</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7866</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>36000</td>
      <td>3204</td>
      <td>60</td>
      <td>70</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BENE16973</td>
      <td>CLM565430</td>
      <td>2009-09-06</td>
      <td>2009-09-06</td>
      <td>PRV55912</td>
      <td>50</td>
      <td>PHY365867</td>
      <td>PHY327147</td>
      <td>NaN</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>24000</td>
      <td>2136</td>
      <td>450</td>
      <td>200</td>
      <td>78.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENE17521</td>
      <td>CLM34721</td>
      <td>2009-01-20</td>
      <td>2009-02-01</td>
      <td>PRV55912</td>
      <td>19000</td>
      <td>PHY349293</td>
      <td>PHY370861</td>
      <td>PHY363291</td>
      <td>45340</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>19000</td>
      <td>1068</td>
      <td>100</td>
      <td>20</td>
      <td>96.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BENE21718</td>
      <td>CLM72336</td>
      <td>2009-10-17</td>
      <td>2009-11-04</td>
      <td>PRV55912</td>
      <td>17000</td>
      <td>PHY334706</td>
      <td>PHY334706</td>
      <td>NaN</td>
      <td>V5789</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>17000</td>
      <td>1068</td>
      <td>1050</td>
      <td>540</td>
      <td>87.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BENE22934</td>
      <td>CLM73394</td>
      <td>2009-10-25</td>
      <td>2009-10-29</td>
      <td>PRV55912</td>
      <td>13000</td>
      <td>PHY390614</td>
      <td>PHY323689</td>
      <td>PHY363291</td>
      <td>71946</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>27000</td>
      <td>2136</td>
      <td>450</td>
      <td>160</td>
      <td>79.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>558206</th>
      <td>BENE154147</td>
      <td>CLM394122</td>
      <td>2009-06-02</td>
      <td>2009-06-04</td>
      <td>PRV54050</td>
      <td>500</td>
      <td>PHY317497</td>
      <td>NaN</td>
      <td>PHY317497</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>890</td>
      <td>120</td>
      <td>85.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>558207</th>
      <td>BENE154687</td>
      <td>CLM184358</td>
      <td>2009-02-08</td>
      <td>2009-02-08</td>
      <td>PRV54302</td>
      <td>3300</td>
      <td>PHY376238</td>
      <td>PHY376238</td>
      <td>NaN</td>
      <td>99639</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4400</td>
      <td>220</td>
      <td>83.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>558208</th>
      <td>BENE157378</td>
      <td>CLM460770</td>
      <td>2009-07-09</td>
      <td>2009-07-29</td>
      <td>PRV51577</td>
      <td>2100</td>
      <td>PHY338096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>14240</td>
      <td>2810</td>
      <td>64.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>558209</th>
      <td>BENE158295</td>
      <td>CLM306999</td>
      <td>2009-04-16</td>
      <td>2009-04-16</td>
      <td>PRV53083</td>
      <td>10</td>
      <td>PHY416646</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>640</td>
      <td>350</td>
      <td>85.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>558210</th>
      <td>BENE158736</td>
      <td>CLM589654</td>
      <td>2009-09-20</td>
      <td>2009-09-20</td>
      <td>PRV56377</td>
      <td>60</td>
      <td>PHY392440</td>
      <td>NaN</td>
      <td>PHY392440</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3220</td>
      <td>1270</td>
      <td>66.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>558211 rows × 55 columns</p>
</div>




```python
for i in cols:
    Master_df[i][Master_df[i]!=0]= 1
```

    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    C:\Users\crispin\AppData\Local\Temp\ipykernel_2984\1718562570.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Master_df[i][Master_df[i]!=0]= 1
    


```python
Master_df[cols]= Master_df[cols].astype(float)
```


```python
Master_df['TotalDiagnosis']= Master_df['ClmDiagnosisCode_1']+Master_df['ClmDiagnosisCode_10']+Master_df['ClmDiagnosisCode_2']+ Master_df['ClmDiagnosisCode_3']+ Master_df['ClmDiagnosisCode_4']+Master_df['ClmDiagnosisCode_5']+ Master_df['ClmDiagnosisCode_6']+ Master_df['ClmDiagnosisCode_7']+Master_df['ClmDiagnosisCode_8']+ Master_df['ClmDiagnosisCode_9']
```


```python
Master_df['TotalProcedure']= Master_df['ClmProcedureCode_1']+Master_df['ClmProcedureCode_2']+Master_df['ClmProcedureCode_3']+ Master_df['ClmProcedureCode_4']+ Master_df['ClmProcedureCode_5']+Master_df['ClmProcedureCode_6']
```

Removing coulmns which are not necessary


```python
Master_df.columns
```




    Index(['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
           'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
           'OtherPhysician', 'ClmAdmitDiagnosisCode', 'DeductibleAmtPaid',
           'DiagnosisGroupCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
           'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
           'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
           'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
           'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
           'ClmProcedureCode_5', 'ClmProcedureCode_6', 'EncounterType', 'Gender',
           'Race', 'RenalDiseaseIndicator', 'State', 'County',
           'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'ChronicCond_Alzheimer',
           'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
           'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
           'ChronicCond_Depression', 'ChronicCond_Diabetes',
           'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
           'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
           'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
           'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age',
           'PotentialFraud', 'IsDead', 'DaysAdmitted', 'TotalDiagnosis',
           'TotalProcedure'],
          dtype='object')




```python
remove=['Provider','BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
       'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
       'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
       'ClmAdmitDiagnosisCode','DeductibleAmtPaid','NoOfMonths_PartACov',
        'NoOfMonths_PartBCov','DiagnosisGroupCode',
        'State', 'County']
```


```python
Master_df.drop(columns=remove, axis=1, inplace=True)
```


```python
Master_df.head()
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
      <th>InscClaimAmtReimbursed</th>
      <th>EncounterType</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>ChronicCond_Alzheimer</th>
      <th>ChronicCond_Heartfailure</th>
      <th>ChronicCond_KidneyDisease</th>
      <th>ChronicCond_Cancer</th>
      <th>ChronicCond_ObstrPulmonary</th>
      <th>...</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>PotentialFraud</th>
      <th>IsDead</th>
      <th>DaysAdmitted</th>
      <th>TotalDiagnosis</th>
      <th>TotalProcedure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>36000</td>
      <td>3204</td>
      <td>60</td>
      <td>70</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>24000</td>
      <td>2136</td>
      <td>450</td>
      <td>200</td>
      <td>78.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>0</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19000</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>19000</td>
      <td>1068</td>
      <td>100</td>
      <td>20</td>
      <td>96.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>17000</td>
      <td>1068</td>
      <td>1050</td>
      <td>540</td>
      <td>87.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>19.0</td>
      <td>9.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13000</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>27000</td>
      <td>2136</td>
      <td>450</td>
      <td>160</td>
      <td>79.0</td>
      <td>Yes</td>
      <td>0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
Master_df.shape
```




    (558211, 26)




```python
Master_df['RenalDiseaseIndicator'].value_counts()
```




    0    448363
    Y    109848
    Name: RenalDiseaseIndicator, dtype: int64




```python
Master_df['RenalDiseaseIndicator']= Master_df['RenalDiseaseIndicator'].replace({'Y':1,'0':0})
```


```python
Master_df['RenalDiseaseIndicator']=Master_df['RenalDiseaseIndicator'].astype(int)
```


```python
Master_df.describe(include='O')
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
      <th>PotentialFraud</th>
      <th>IsDead</th>
      <th>DaysAdmitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>558211</td>
      <td>558211</td>
      <td>558211</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>37</td>
    </tr>
    <tr>
      <th>top</th>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>345415</td>
      <td>554080</td>
      <td>517737</td>
    </tr>
  </tbody>
</table>
</div>




```python
Master_df['IsDead']=Master_df['IsDead'].astype(float)
Master_df['DaysAdmitted']=Master_df['DaysAdmitted'].astype(float)
Master_df['PotentialFraud']=Master_df['PotentialFraud'].replace({'Yes':1, 'No':0})
Master_df['PotentialFraud']=Master_df['PotentialFraud'].astype(int)
Master_df['PotentialFraud']
```




    0         1
    1         1
    2         1
    3         1
    4         1
             ..
    558206    0
    558207    0
    558208    0
    558209    0
    558210    0
    Name: PotentialFraud, Length: 558211, dtype: int32




```python
x= Master_df.drop('PotentialFraud', axis=1)
y= Master_df.loc[:,'PotentialFraud']
x.columns
```




    Index(['InscClaimAmtReimbursed', 'EncounterType', 'Gender', 'Race',
           'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
           'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
           'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
           'ChronicCond_Depression', 'ChronicCond_Diabetes',
           'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
           'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
           'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
           'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age', 'IsDead',
           'DaysAdmitted', 'TotalDiagnosis', 'TotalProcedure'],
          dtype='object')




```python
num_col= ['InscClaimAmtReimbursed',
       'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
       'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age',
       'DaysAdmitted', 'TotalDiagnosis', 'TotalProcedure']
numerical_columns= x.loc[:,num_col]
numerical_columns.describe()
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
      <th>InscClaimAmtReimbursed</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>DaysAdmitted</th>
      <th>TotalDiagnosis</th>
      <th>TotalProcedure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>558211.000000</td>
      <td>558211.000000</td>
      <td>558211.000000</td>
      <td>558211.000000</td>
      <td>558211.000000</td>
      <td>558211.000000</td>
      <td>558211.000000</td>
      <td>558211.000000</td>
      <td>558211.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>997.012133</td>
      <td>5227.971466</td>
      <td>568.756807</td>
      <td>2278.225348</td>
      <td>649.698745</td>
      <td>73.769770</td>
      <td>0.483269</td>
      <td>3.010897</td>
      <td>0.053557</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3821.534891</td>
      <td>11786.274732</td>
      <td>1179.172616</td>
      <td>3881.846386</td>
      <td>1002.020811</td>
      <td>13.022524</td>
      <td>2.300583</td>
      <td>2.448213</td>
      <td>0.280534</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-8000.000000</td>
      <td>0.000000</td>
      <td>-70.000000</td>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>460.000000</td>
      <td>120.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>80.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1170.000000</td>
      <td>340.000000</td>
      <td>75.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>300.000000</td>
      <td>6000.000000</td>
      <td>1068.000000</td>
      <td>2590.000000</td>
      <td>790.000000</td>
      <td>82.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>125000.000000</td>
      <td>161470.000000</td>
      <td>38272.000000</td>
      <td>102960.000000</td>
      <td>13840.000000</td>
      <td>101.000000</td>
      <td>36.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
numerical_columns.head()
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
      <th>InscClaimAmtReimbursed</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>DaysAdmitted</th>
      <th>TotalDiagnosis</th>
      <th>TotalProcedure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26000</td>
      <td>36000</td>
      <td>3204</td>
      <td>60</td>
      <td>70</td>
      <td>67.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>24000</td>
      <td>2136</td>
      <td>450</td>
      <td>200</td>
      <td>78.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19000</td>
      <td>19000</td>
      <td>1068</td>
      <td>100</td>
      <td>20</td>
      <td>96.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17000</td>
      <td>17000</td>
      <td>1068</td>
      <td>1050</td>
      <td>540</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>9.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13000</td>
      <td>27000</td>
      <td>2136</td>
      <td>450</td>
      <td>160</td>
      <td>79.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cat_col= ['EncounterType', 'Gender', 'Race',
       'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke','IsDead']
x_cat= x.loc[:,cat_col]
x_cat
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
      <th>EncounterType</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>ChronicCond_Alzheimer</th>
      <th>ChronicCond_Heartfailure</th>
      <th>ChronicCond_KidneyDisease</th>
      <th>ChronicCond_Cancer</th>
      <th>ChronicCond_ObstrPulmonary</th>
      <th>ChronicCond_Depression</th>
      <th>ChronicCond_Diabetes</th>
      <th>ChronicCond_IschemicHeart</th>
      <th>ChronicCond_Osteoporasis</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IsDead</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>558206</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558207</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558208</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558209</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558210</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>558211 rows × 16 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
x_num= scale.fit_transform(x[num_col])
x_num= pd.DataFrame(x_num, columns=['InscClaimAmtReimbursed','IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt','OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age','DaysAdmitted', 'TotalDiagnosis', 'TotalProcedure'])
x= pd.concat([x_num, x_cat], axis=1)
x
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
      <th>InscClaimAmtReimbursed</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>DaysAdmitted</th>
      <th>TotalDiagnosis</th>
      <th>TotalProcedure</th>
      <th>EncounterType</th>
      <th>...</th>
      <th>ChronicCond_KidneyDisease</th>
      <th>ChronicCond_Cancer</th>
      <th>ChronicCond_ObstrPulmonary</th>
      <th>ChronicCond_Depression</th>
      <th>ChronicCond_Diabetes</th>
      <th>ChronicCond_IschemicHeart</th>
      <th>ChronicCond_Osteoporasis</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IsDead</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.542662</td>
      <td>2.610838</td>
      <td>2.234826</td>
      <td>-0.571436</td>
      <td>-0.578530</td>
      <td>-0.519851</td>
      <td>2.832646</td>
      <td>2.446318</td>
      <td>-0.190910</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.247810</td>
      <td>1.592704</td>
      <td>1.329105</td>
      <td>-0.470968</td>
      <td>-0.448792</td>
      <td>0.324840</td>
      <td>-0.210064</td>
      <td>2.446318</td>
      <td>-0.190910</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.710935</td>
      <td>1.168481</td>
      <td>0.423385</td>
      <td>-0.561132</td>
      <td>-0.628429</td>
      <td>1.707062</td>
      <td>5.440682</td>
      <td>2.446318</td>
      <td>6.938348</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.187585</td>
      <td>0.998792</td>
      <td>0.423385</td>
      <td>-0.316403</td>
      <td>-0.109478</td>
      <td>1.015951</td>
      <td>8.048719</td>
      <td>2.446318</td>
      <td>3.373719</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.140884</td>
      <td>1.847237</td>
      <td>1.329105</td>
      <td>-0.470968</td>
      <td>-0.488712</td>
      <td>0.401630</td>
      <td>1.963300</td>
      <td>1.629395</td>
      <td>3.373719</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>558206</th>
      <td>-0.130056</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>-0.357620</td>
      <td>-0.528631</td>
      <td>0.862371</td>
      <td>-0.210064</td>
      <td>-0.004451</td>
      <td>-0.190910</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558207</th>
      <td>0.602635</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>0.546590</td>
      <td>-0.428833</td>
      <td>0.708790</td>
      <td>-0.210064</td>
      <td>-0.821374</td>
      <td>-0.190910</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558208</th>
      <td>0.288625</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>3.081468</td>
      <td>2.155946</td>
      <td>-0.750222</td>
      <td>-0.210064</td>
      <td>-0.412913</td>
      <td>-0.190910</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558209</th>
      <td>-0.258277</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>-0.422023</td>
      <td>-0.299095</td>
      <td>0.862371</td>
      <td>-0.210064</td>
      <td>-0.821374</td>
      <td>-0.190910</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>558210</th>
      <td>-0.245193</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>0.242610</td>
      <td>0.619051</td>
      <td>-0.596641</td>
      <td>-0.210064</td>
      <td>0.404010</td>
      <td>-0.190910</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>558211 rows × 25 columns</p>
</div>




```python
x.columns
```




    Index(['InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt',
           'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
           'OPAnnualDeductibleAmt', 'Age', 'DaysAdmitted', 'TotalDiagnosis',
           'TotalProcedure', 'EncounterType', 'Gender', 'Race',
           'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
           'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
           'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
           'ChronicCond_Depression', 'ChronicCond_Diabetes',
           'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
           'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'IsDead'],
          dtype='object')




```python
y
```




    0         1
    1         1
    2         1
    3         1
    4         1
             ..
    558206    0
    558207    0
    558208    0
    558209    0
    558210    0
    Name: PotentialFraud, Length: 558211, dtype: int32



#### Train- Test Split


```python
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test= train_test_split(x,y, test_size=0.1, random_state=42)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
x_train1,y_train1 = rus.fit_resample(x_train, y_train)
'''from imblearn import over_sampling

ada = over_sampling.ADASYN(random_state=0)
x_train2, y_train2 = ada.fit_resample(x_train, y_train)'''
'from imblearn import over_sampling\n\nada = over_sampling.ADASYN(random_state=0)\nx_train2, y_train2 = ada.fit_resample(x_train, y_train)'
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, auc, roc_curve
from xgboost import plot_importance
from xgboost import XGBClassifier
xgb= XGBClassifier()
xgb.fit(x_train,y_train)
plot_importance(xgb)
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![output_137_1](https://user-images.githubusercontent.com/75635908/166229694-dd0d8bf7-c419-4da5-adf4-17f79d33a789.png)

    



```python
xgb= XGBClassifier()
xgb.fit(x_train1,y_train1)
plot_importance(xgb)
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![output_138_1](https://user-images.githubusercontent.com/75635908/166229736-0f2601ab-51bb-468d-b5e8-68048e33fb04.png)

    



```python
'''xgb.fit(x_train2,y_train2)
plot_importance(xgb)'''
```




    'xgb.fit(x_train2,y_train2)\nplot_importance(xgb)'




```python
acc_score=[]
```

#### Decision tree


```python
from sklearn.model_selection import GridSearchCV
```


```python
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
estimator=  DecisionTreeClassifier()
param_grid= {'criterion':['gini', 'entropy'],
             'max_depth':[3,4,5],
             'min_samples_split':[2,3,5]
             }
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid)
grid_search.fit(x_train1, y_train1)
print(grid_search.best_score_)
print(grid_search.best_params_)
```

    0.5374148255998479
    {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2}
    


```python
'''param_grid= {'criterion':['gini', 'entropy'],
             'max_depth':[3,4,5],
             'min_samples_split':[2,3,5]
             }
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid)
grid_search.fit(x_train2, y_train2)
print(grid_search.best_score_)
print(grid_search.best_params_)'''
```




    "param_grid= {'criterion':['gini', 'entropy'],\n             'max_depth':[3,4,5],\n             'min_samples_split':[2,3,5]\n             }\ngrid_search = GridSearchCV(estimator = estimator, param_grid = param_grid)\ngrid_search.fit(x_train2, y_train2)\nprint(grid_search.best_score_)\nprint(grid_search.best_params_)"




```python
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
```

    0.6301073464557836
    {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 2}
    

#### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
estimator1= RandomForestClassifier()
'''estimator1.fit(x_train, y_train)
model_score= estimator1.predict(x_train)
accuracy= estimator1.predict(x_test)
print(accuracy_score(y_train, model_score))
print(accuracy_score(y_test, accuracy))'''
```




    'estimator1.fit(x_train, y_train)\nmodel_score= estimator1.predict(x_train)\naccuracy= estimator1.predict(x_test)\nprint(accuracy_score(y_train, model_score))\nprint(accuracy_score(y_test, accuracy))'




```python
'''estimator1.fit(x_train1, y_train1)
model_score= estimator1.predict(x_train1)
accuracy= estimator1.predict(x_test)
print(accuracy_score(y_train1, model_score))
print(accuracy_score(y_test, accuracy))'''
```




    'estimator1.fit(x_train1, y_train1)\nmodel_score= estimator1.predict(x_train1)\naccuracy= estimator1.predict(x_test)\nprint(accuracy_score(y_train1, model_score))\nprint(accuracy_score(y_test, accuracy))'




```python
'''estimator1.fit(x_train2, y_train2)
model_score= estimator1.predict(x_train2)
accuracy= estimator1.predict(x_test)
print(accuracy_score(y_train2, model_score))
print(accuracy_score(y_test, accuracy))'''
```




    'estimator1.fit(x_train2, y_train2)\nmodel_score= estimator1.predict(x_train2)\naccuracy= estimator1.predict(x_test)\nprint(accuracy_score(y_train2, model_score))\nprint(accuracy_score(y_test, accuracy))'



#### Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
bayes= GaussianNB()
bayes.fit(x_train, y_train)
train_pred= bayes.predict(x_train)
test_pred= bayes.predict(x_test)
print(accuracy_score(y_train,train_pred))
print(accuracy_score(y_test,test_pred))
```

    0.6233436639735345
    0.6254881587904411
    


```python
bayes.fit(x_train, y_train)
train_pred= bayes.predict(x_train1)
test_pred= bayes.predict(x_test)
print(accuracy_score(y_train1,train_pred))
print(accuracy_score(y_test,test_pred))
```

    0.5321842618776805
    0.6254881587904411
    


```python
'''bayes.fit(x_train, y_train)
train_pred= bayes.predict(x_train2)
test_pred= bayes.predict(x_test)
print(accuracy_score(y_train2,train_pred))
print(accuracy_score(y_test,test_pred))'''
```




    'bayes.fit(x_train, y_train)\ntrain_pred= bayes.predict(x_train2)\ntest_pred= bayes.predict(x_test)\nprint(accuracy_score(y_train2,train_pred))\nprint(accuracy_score(y_test,test_pred))'



#### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train, y_train)
train_pred= lr.predict(x_train)
test_pred= lr.predict(x_test)
print(accuracy_score(y_train,train_pred))
print(accuracy_score(y_test,test_pred))
```

    0.6299401459824956
    0.6318297445451614
    


```python
lr.fit(x_train1, y_train1)
train_pred= lr.predict(x_train1)
test_pred= lr.predict(x_test)
print(accuracy_score(y_train1,train_pred))
print(accuracy_score(y_test,test_pred))
```

    0.5363008838476068
    0.6167819139407402
    


```python
'''lr.fit(x_train2, y_train2)
train_pred= lr.predict(x_train2)
test_pred= lr.predict(x_test)
print(accuracy_score(y_train2,train_pred))
print(accuracy_score(y_test,test_pred))'''
```




    'lr.fit(x_train2, y_train2)\ntrain_pred= lr.predict(x_train2)\ntest_pred= lr.predict(x_test)\nprint(accuracy_score(y_train2,train_pred))\nprint(accuracy_score(y_test,test_pred))'



#### After Evaluation of Various parameters

#### Decision Tree


```python
import time
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
estimator=  DecisionTreeClassifier(criterion= 'gini', max_depth=5, min_samples_split= 2)
estimator.fit(x_train, y_train)
model_score= estimator.predict(x_train)
accuracy= estimator.predict(x_test)
start = time.time()
estimator.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,accuracy)*100, 2)
f1_random_forest = round(f1_score(y_test,accuracy,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,accuracy,average = "binary",beta=0.5)*100, 2)

end = time.time()

acc_score.append({'Model':'Decision Tree', 'Score': accuracy_score(y_train, model_score), 'Accuracy': accuracy_score(y_test, accuracy), 'Time_Taken':end - start})
```


```python
fn= ['InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt',
    'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
    'OPAnnualDeductibleAmt', 'Age', 'DaysAdmitted',
    'TotalDiagnosis', 'TotalProcedure', 'EncounterType', 'Gender', 'Race',
    'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
    'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
    'ChronicCond_Depression', 'ChronicCond_Diabetes',
    'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'IsDead']
```


```python
cl=['No','Yes']
```


```python
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=300)
tree.plot_tree(estimator, feature_names= fn, class_names=cl);
```


    
![output_163_0](https://user-images.githubusercontent.com/75635908/166229814-b3e986fc-cff2-43c1-a10e-4398f05a79bf.png)

    


#### Destandardizing Values


```python
m1= Master_df['DaysAdmitted'].mean()
s1= Master_df['DaysAdmitted'].std()
print((0.007*s1)+m1)
print((9.57*s1)+m1)
print((9.135*s1)+m1)
```

    0.4993729535174792
    22.49984964323506
    21.499095971790204
    


```python
m2= Master_df['OPAnnualDeductibleAmt'].mean()
s2= Master_df['OPAnnualDeductibleAmt'].std()
print((10.514*s2)+m2)
```

    11184.945549354181
    


```python
m3= Master_df['Age'].mean()
s3= Master_df['Age'].std()
print((-0.481*s3)+m3)
```

    67.50593600384992
    


```python
m4= Master_df['InscClaimAmtReimbursed'].mean()
s4= Master_df['InscClaimAmtReimbursed'].std()
print((10.075*s4)+m4)
```

    39498.97616418805
    


```python
confusion_matrix(y_test,accuracy)
```




    array([[32803,  1885],
           [18684,  2450]], dtype=int64)




```python
tn, fp, fn, tp = confusion_matrix(y_test,accuracy).ravel()
(tn, fp, fn, tp)  
```




    (32803, 1885, 18684, 2450)




```python
train_fpr, train_tpr, thresholds = roc_curve(y_train, estimator.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, estimator.predict_proba(x_test)[:,1])
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()
```


    
![output_171_0](https://user-images.githubusercontent.com/75635908/166229920-d6638044-77b5-4e56-8d5b-667572b01080.png)

    


#### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
estimator1= RandomForestClassifier()
estimator1.fit(x_train, y_train)
model_score= estimator1.predict(x_train)
accuracy= estimator1.predict(x_test)
start = time.time()
estimator1.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,accuracy)*100, 2)
f1_random_forest = round(f1_score(y_test,accuracy,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,accuracy,average = "binary",beta=0.5)*100, 2)
end = time.time()
acc_score.append({'Model':'Random Forest', 'Score': accuracy_score(y_train, model_score), 'Accuracy': accuracy_score(y_test, accuracy),'Time_Taken':end - start})
```


```python
confusion_matrix(y_test,accuracy)
```




    array([[28464,  6224],
           [10704, 10430]], dtype=int64)




```python
tn, fp, fn, tp = confusion_matrix(y_test,accuracy).ravel()
(tn, fp, fn, tp)
```




    (28464, 6224, 10704, 10430)




```python
train_fpr, train_tpr, thresholds = roc_curve(y_train, estimator1.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, estimator1.predict_proba(x_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()
```


    
![output_176_0](https://user-images.githubusercontent.com/75635908/166229977-ef9969ca-b492-4128-adbb-827f8a7741ff.png)

    



```python
y_test_rf= y_test.reset_index()
y_test.head()
```




    230744    1
    27826     1
    314625    0
    140256    0
    502955    0
    Name: PotentialFraud, dtype: int32




```python
x_test_rf=  x_test.reset_index()
x_test.head()
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
      <th>InscClaimAmtReimbursed</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>DaysAdmitted</th>
      <th>TotalDiagnosis</th>
      <th>TotalProcedure</th>
      <th>EncounterType</th>
      <th>...</th>
      <th>ChronicCond_KidneyDisease</th>
      <th>ChronicCond_Cancer</th>
      <th>ChronicCond_ObstrPulmonary</th>
      <th>ChronicCond_Depression</th>
      <th>ChronicCond_Diabetes</th>
      <th>ChronicCond_IschemicHeart</th>
      <th>ChronicCond_Osteoporasis</th>
      <th>ChronicCond_rheumatoidarthritis</th>
      <th>ChronicCond_stroke</th>
      <th>IsDead</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>230744</th>
      <td>-0.229492</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>-0.535371</td>
      <td>-0.628429</td>
      <td>0.555210</td>
      <td>-0.210064</td>
      <td>-0.004451</td>
      <td>-0.19091</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27826</th>
      <td>-0.260893</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>-0.579164</td>
      <td>-0.608470</td>
      <td>-0.519851</td>
      <td>-0.210064</td>
      <td>-1.229836</td>
      <td>-0.19091</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>314625</th>
      <td>-0.253043</td>
      <td>0.769712</td>
      <td>1.329105</td>
      <td>0.708884</td>
      <td>-0.179337</td>
      <td>-0.443061</td>
      <td>-0.210064</td>
      <td>-0.412913</td>
      <td>-0.19091</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>140256</th>
      <td>-0.253043</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>-0.403990</td>
      <td>-0.458772</td>
      <td>-0.135901</td>
      <td>-0.210064</td>
      <td>0.812472</td>
      <td>-0.19091</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>502955</th>
      <td>-0.250426</td>
      <td>-0.443565</td>
      <td>-0.482336</td>
      <td>-0.403990</td>
      <td>-0.598490</td>
      <td>1.246321</td>
      <td>-0.210064</td>
      <td>-0.821374</td>
      <td>-0.19091</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
y_test_rf.shape
```




    (55822, 2)




```python
accuracy= accuracy.reshape(55822,1)
```


```python
accuracy.shape
```




    (55822, 1)




```python
accuracy= pd.DataFrame(accuracy, columns= ['Predict'])
accuracy
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
      <th>Predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>55817</th>
      <td>0</td>
    </tr>
    <tr>
      <th>55818</th>
      <td>1</td>
    </tr>
    <tr>
      <th>55819</th>
      <td>1</td>
    </tr>
    <tr>
      <th>55820</th>
      <td>0</td>
    </tr>
    <tr>
      <th>55821</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>55822 rows × 1 columns</p>
</div>




```python
predictor= pd.concat([y_test_rf,accuracy], axis=1)
predictor
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
      <th>index</th>
      <th>PotentialFraud</th>
      <th>Predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230744</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27826</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>314625</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140256</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>502955</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>55817</th>
      <td>552465</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55818</th>
      <td>215575</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55819</th>
      <td>144212</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55820</th>
      <td>201007</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55821</th>
      <td>24674</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>55822 rows × 3 columns</p>
</div>




```python
Index_label = predictor[(predictor['PotentialFraud'] ==1) & (predictor['Predict']==1)]
indicies= Index_label['index']
wrong_predictions= Master_df.iloc[indicies,:]
wrong_predictions
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
      <th>InscClaimAmtReimbursed</th>
      <th>EncounterType</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>ChronicCond_Alzheimer</th>
      <th>ChronicCond_Heartfailure</th>
      <th>ChronicCond_KidneyDisease</th>
      <th>ChronicCond_Cancer</th>
      <th>ChronicCond_ObstrPulmonary</th>
      <th>...</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>PotentialFraud</th>
      <th>IsDead</th>
      <th>DaysAdmitted</th>
      <th>TotalDiagnosis</th>
      <th>TotalProcedure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>103756</th>
      <td>53000</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>57500</td>
      <td>1068</td>
      <td>530</td>
      <td>550</td>
      <td>83.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>140961</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1670</td>
      <td>260</td>
      <td>85.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>43866</th>
      <td>300</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1070</td>
      <td>330</td>
      <td>96.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>44359</th>
      <td>70</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>4710</td>
      <td>1370</td>
      <td>95.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>270400</th>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>260</td>
      <td>240</td>
      <td>64.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>179605</th>
      <td>50</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>30000</td>
      <td>1068</td>
      <td>890</td>
      <td>390</td>
      <td>66.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>344643</th>
      <td>60</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2000</td>
      <td>1068</td>
      <td>5760</td>
      <td>1000</td>
      <td>71.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>458648</th>
      <td>90</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>290</td>
      <td>10</td>
      <td>99.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>215575</th>
      <td>9000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>22000</td>
      <td>3204</td>
      <td>1550</td>
      <td>210</td>
      <td>48.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24674</th>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>11400</td>
      <td>1068</td>
      <td>4140</td>
      <td>160</td>
      <td>73.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10430 rows × 26 columns</p>
</div>




```python
print('Fraud Insurance Claims detected - ',wrong_predictions['InscClaimAmtReimbursed'].sum())
print('Fraud Insurance Claims for Inpatients detected - ',wrong_predictions['IPAnnualReimbursementAmt'].sum())
print('Fraud Insurance Claims for Outpatients detected - ',wrong_predictions['OPAnnualReimbursementAmt'].sum())
```

    Fraud Insurance Claims detected -  23404740
    Fraud Insurance Claims for Inpatients detected -  78053570
    Fraud Insurance Claims for Outpatients detected -  27375540
    


```python
fraud_index= y_test_rf[y_test_rf['PotentialFraud']==1]
indicies1= fraud_index['index']
```


```python
frauds= Master_df.iloc[indicies1,:]
frauds
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
      <th>InscClaimAmtReimbursed</th>
      <th>EncounterType</th>
      <th>Gender</th>
      <th>Race</th>
      <th>RenalDiseaseIndicator</th>
      <th>ChronicCond_Alzheimer</th>
      <th>ChronicCond_Heartfailure</th>
      <th>ChronicCond_KidneyDisease</th>
      <th>ChronicCond_Cancer</th>
      <th>ChronicCond_ObstrPulmonary</th>
      <th>...</th>
      <th>IPAnnualReimbursementAmt</th>
      <th>IPAnnualDeductibleAmt</th>
      <th>OPAnnualReimbursementAmt</th>
      <th>OPAnnualDeductibleAmt</th>
      <th>Age</th>
      <th>PotentialFraud</th>
      <th>IsDead</th>
      <th>DaysAdmitted</th>
      <th>TotalDiagnosis</th>
      <th>TotalProcedure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>230744</th>
      <td>120</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
      <td>20</td>
      <td>81.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27826</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>40</td>
      <td>67.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53395</th>
      <td>50</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>420</td>
      <td>80</td>
      <td>68.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>159227</th>
      <td>300</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>730</td>
      <td>120</td>
      <td>78.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>93009</th>
      <td>1400</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>43200</td>
      <td>2136</td>
      <td>20190</td>
      <td>5940</td>
      <td>70.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>344643</th>
      <td>60</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2000</td>
      <td>1068</td>
      <td>5760</td>
      <td>1000</td>
      <td>71.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>458648</th>
      <td>90</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>290</td>
      <td>10</td>
      <td>99.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>47840</th>
      <td>40</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1790</td>
      <td>890</td>
      <td>85.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>215575</th>
      <td>9000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>22000</td>
      <td>3204</td>
      <td>1550</td>
      <td>210</td>
      <td>48.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24674</th>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>11400</td>
      <td>1068</td>
      <td>4140</td>
      <td>160</td>
      <td>73.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>21134 rows × 26 columns</p>
</div>




```python
print('Fraud Insurance Claims without model - ',frauds['InscClaimAmtReimbursed'].sum())
print('Fraud Insurance Claims for Inpatients without model - ',frauds['IPAnnualReimbursementAmt'].sum())
print('Fraud Insurance Claims for Outpatients without model - ',frauds['OPAnnualReimbursementAmt'].sum())
```

    Fraud Insurance Claims without model -  30673230
    Fraud Insurance Claims for Inpatients without model -  122946970
    Fraud Insurance Claims for Outpatients without model -  48642580
    


```python
print('Insurance Claim Amount Saved - $', 30673230- 23525520)
print('Inpatient Insurance Claim Amount Saved - $',122946970- 78254980)
print('Outpatient insurance Claim Amount Saved - $',48642580- 27754480)
```

    Insurance Claim Amount Saved - $ 7147710
    Inpatient Insurance Claim Amount Saved - $ 44691990
    Outpatient insurance Claim Amount Saved - $ 20888100
    

#### Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
bayes= GaussianNB()
bayes.fit(x_train, y_train)
train_pred= bayes.predict(x_train)
test_pred= bayes.predict(x_test)
start = time.time()
bayes.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,test_pred)*100, 2)
f1_random_forest = round(f1_score(y_test,test_pred,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,test_pred,average = "binary",beta=0.5)*100, 2)

end = time.time()

acc_score.append({'Model':'Naive Bayes', 'Score': accuracy_score(y_train, train_pred), 'Accuracy': accuracy_score(y_test, test_pred),'Time_Taken':end- start})
```


```python
confusion_matrix(y_test,test_pred)
```




    array([[31737,  2951],
           [17955,  3179]], dtype=int64)




```python
tn, fp, fn, tp = confusion_matrix(y_test,test_pred).ravel()
(tn, fp, fn, tp)
```




    (31737, 2951, 17955, 3179)




```python
train_fpr, train_tpr, thresholds = roc_curve(y_train, bayes.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, bayes.predict_proba(x_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()
```


    
![output_194_0](https://user-images.githubusercontent.com/75635908/166230109-60ce4cb4-8bd6-47a4-910a-d8a9c9b3e623.png)

    


#### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train, y_train)
train_pred= lr.predict(x_train)
test_pred= lr.predict(x_test)
start = time.time()
lr.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,test_pred)*100, 2)
f1_random_forest = round(f1_score(y_test,test_pred,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,test_pred,average = "binary",beta=0.5)*100, 2)

end = time.time()
acc_score.append({'Model': "Logistic Regression", 'Score': accuracy_score(y_train,train_pred), 'Accuracy': accuracy_score(y_test,test_pred), 'Time_Taken':end - start})
```


```python
confusion_matrix(y_test,test_pred)
```




    array([[32902,  1786],
           [18766,  2368]], dtype=int64)




```python
tn, fp, fn, tp = confusion_matrix(y_test,test_pred).ravel()
(tn, fp, fn, tp)
```




    (32902, 1786, 18766, 2368)




```python
train_fpr, train_tpr, thresholds = roc_curve(y_train, lr.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()

```


    
![output_199_0](https://user-images.githubusercontent.com/75635908/166230172-73510c32-b365-40c6-9c89-54553cbab668.png)

    


#### XG Boost


```python
xgb= XGBClassifier()
xgb.fit(x_train, y_train)
model_score= xgb.predict(x_train)
accuracy= xgb.predict(x_test)
start = time.time()
xgb.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,accuracy)*100, 2)
f1_random_forest = round(f1_score(y_test,accuracy,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,accuracy,average = "binary",beta=0.5)*100, 2)

end = time.time()
acc_score.append({'Model':'XG boost', 'Score': accuracy_score(y_train, model_score), 'Accuracy': accuracy_score(y_test, accuracy), 'Time_Taken':end - start})
confusion_matrix(y_test,accuracy)
```




    array([[32733,  1955],
           [18160,  2974]], dtype=int64)




```python
tn, fp, fn, tp = confusion_matrix(y_test,accuracy).ravel()
(tn, fp, fn, tp)
```




    (32733, 1955, 18160, 2974)




```python
train_fpr, train_tpr, thresholds = roc_curve(y_train, xgb.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, xgb.predict_proba(x_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()
```


    
![output_203_0](https://user-images.githubusercontent.com/75635908/166230262-636aea6a-7120-4a6b-a0f2-0723d1fd894c.png)

    



```python
accuracy= pd.DataFrame(acc_score, columns=['Model','Score','Accuracy','Time_Taken'])
accuracy.sort_values(by='Accuracy', ascending= False, inplace= True)
accuracy
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
      <th>Model</th>
      <th>Score</th>
      <th>Accuracy</th>
      <th>Time_Taken</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>0.991204</td>
      <td>0.696750</td>
      <td>33.137974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>XG boost</td>
      <td>0.646790</td>
      <td>0.639658</td>
      <td>1.048399</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression</td>
      <td>0.629940</td>
      <td>0.631830</td>
      <td>0.159908</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Decision Tree</td>
      <td>0.630275</td>
      <td>0.631525</td>
      <td>0.195590</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Naive Bayes</td>
      <td>0.623344</td>
      <td>0.625488</td>
      <td>0.536709</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,8))
sns.barplot(x= accuracy.Model, y=accuracy.Accuracy);
```


    
![output_205_0](https://user-images.githubusercontent.com/75635908/166230324-9f5ec5ad-ed18-4109-b3ea-35f876bc9e2e.png)



```python
import pickle
```


```python
pickle.dump(estimator1, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
```


```python
print(model.predict([[6.542662,2.610838,2.234826,-0.571436,-0.578530,-0.519851,2.832646,2.446318,-0.190910,0,1,1,0,1,1,1,2,2,1,1,1,2,1,1,0.0]]))
```

    D:\DataScienceProjects\Healthcare_Provider_Fraud_Detection_Analysis\Healthcare_Fraud\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
      warnings.warn(
    

    [1]
    


```python

```
