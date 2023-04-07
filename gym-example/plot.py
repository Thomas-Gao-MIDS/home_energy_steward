import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

filename = 'output/validation.csv'
df = pd.read_csv(filename)
print(df.columns)


timestamp = pd.to_datetime(df['timestamps'], format='%m-%d-%Y %H:%M:%S')
hours = (timestamp.dt.hour + timestamp.dt.minute/60).tolist()
hours = [x if x>=6 else x+24 for x in hours]

with PdfPages('output/charts.pdf') as pdf:

  plt.figure()

  # Electricity Cost
  plt.plot(hours, df['cum_ecost'], label='cum_ecost')
  plt.plot(hours, df['ecost'], ".", label='ecost')
  plt.plot(hours, df['grid_cost'], ".", label='grid_cost')

  plt.legend(); plt.title('Electricity Cost: '+\
    str(round(max(df['cum_ecost']),2)));
  pdf.savefig(); plt.close()

  # EV
  plt.plot(hours, df['ev_action'], label='ev_action')
  plt.plot(hours, df['ev_engy'], label='ev_engy')
  plt.plot(hours, df['ev_energy_required'], label='ev_energy_required')

  plt.legend(); plt.title('EV');
  pdf.savefig(); plt.close()

  # ES
  plt.plot(hours, df['es_action'], label='es_action')
  plt.plot(hours, df['es_engy'], label='es_engy')
  plt.plot(hours, df['es_storage'], label='es_storge')

  plt.legend(); plt.title('ES');
  pdf.savefig(); plt.close()

  # Energy
  plt.plot(hours, df['pv_engy'], label='pv_energy')
  plt.plot(hours, df['dev_engy'], label='dev_engy')
  plt.plot(hours, df['es_engy'], ".", label='es_engy')
  plt.plot(hours, df['ev_engy'], ".", label='ev_engy')
  plt.plot(hours, df['grid_engy'], ".", label='grid_engy')

  plt.legend(); plt.title('Energy');
  pdf.savefig(); plt.close()

  # Balance
  plt.plot(hours, df['engy_consumption'], label='engy_consumption')
  plt.plot(hours, df['engy_supply'], label='engy_supply')
  plt.plot(hours, df['engy_unused'], label='engy_unused')
  plt.axhline(y=0, color='b', linestyle='-')
  plt.legend(); plt.title('Agg. Energy');
  pdf.savefig(); plt.close()

  plt.plot(hours, df['cum_engy_unused'], label='cum_engy_unused')
  plt.legend(); plt.title('Cum. Energy Unused: '+\
                          str(round(max(df['cum_engy_unused']),2)));
  pdf.savefig(); plt.close()
