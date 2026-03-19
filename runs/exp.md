
# Wasserstein Barycenter in SSEM

## Theory
...
## Practice
### Coding
...
### Experiments:
#### WB vs FESR
- sample_A w2 400 iter (20260305)
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
      <th>NCC</th>
      <th>SSIM</th>
      <th>MI</th>
      <th>GC</th>
      <th>PCP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Original(data)</th>
      <td>0.582520</td>
      <td>0.584339</td>
      <td>0.231147</td>
      <td>0.402052</td>
      <td>0.156131</td>
    </tr>
    <tr>
      <th>FESR(stack)</th>
      <td>0.707637</td>
      <td>0.709150</td>
      <td>0.410046</td>
      <td>0.506461</td>
      <td>0.163374</td>
    </tr>
    <tr>
      <th>WassersteinBarycenter(warped_data)</th>
      <td>0.721183</td>
      <td>0.721309</td>
      <td>0.385031</td>
      <td>0.577217</td>
      <td>0.249306</td>
    </tr>
  </tbody>
</table>
</div>

- sample_A w2 1000 iter (20260305):
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
      <th>NCC</th>
      <th>SSIM</th>
      <th>MI</th>
      <th>GC</th>
      <th>PCP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Original(data)</th>
      <td>0.582520</td>
      <td>0.584339</td>
      <td>0.231147</td>
      <td>0.402052</td>
      <td>0.156131</td>
    </tr>
    <tr>
      <th>FESR(stack)</th>
      <td>0.707637</td>
      <td>0.709150</td>
      <td>0.410046</td>
      <td>0.506461</td>
      <td>0.163374</td>
    </tr>
    <tr>
      <th>WassersteinBarycenter(warped_data)</th>
      <td>0.789503</td>
      <td>0.789304</td>
      <td>0.504623</td>
      <td>0.663005</td>
      <td>0.307763</td>
    </tr>
  </tbody>
</table>
</div>

- sample_B w2 650 iter (20260306)
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
      <th>NCC</th>
      <th>SSIM</th>
      <th>MI</th>
      <th>GC</th>
      <th>PCP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Original(data)</th>
      <td>0.402786</td>
      <td>0.411785</td>
      <td>0.123682</td>
      <td>0.161980</td>
      <td>0.062130</td>
    </tr>
    <tr>
      <th>FESR(stack)</th>
      <td>0.396117</td>
      <td>0.402865</td>
      <td>0.451450</td>
      <td>0.137875</td>
      <td>0.045014</td>
    </tr>
    <tr>
      <th>WassersteinBarycenter(warped_data)</th>
      <td>0.609639</td>
      <td>0.614605</td>
      <td>0.263933</td>
      <td>0.390635</td>
      <td>0.183525</td>
    </tr>
  </tbody>
</table>
</div>

- sample_C w2 650 iter (20260306)

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
      <th>NCC</th>
      <th>SSIM</th>
      <th>MI</th>
      <th>GC</th>
      <th>PCP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Original(data)</th>
      <td>0.344804</td>
      <td>0.353844</td>
      <td>0.093570</td>
      <td>0.174383</td>
      <td>0.091764</td>
    </tr>
    <tr>
      <th>FESR(stack)</th>
      <td>0.480153</td>
      <td>0.487514</td>
      <td>0.415664</td>
      <td>0.227896</td>
      <td>0.073430</td>
    </tr>
    <tr>
      <th>WassersteinBarycenter(warped_data)</th>
      <td>0.589234</td>
      <td>0.593683</td>
      <td>0.258278</td>
      <td>0.435975</td>
      <td>0.213582</td>
    </tr>
  </tbody>
</table>
</div>