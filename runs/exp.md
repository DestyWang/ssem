
# Wasserstein Barycenter

#### experiments:
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