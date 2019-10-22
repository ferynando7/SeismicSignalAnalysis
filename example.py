from obspy import read

st = read('/home/fz/YACHAY/BMAS/BHZ.D/EC.BMAS..BHZ.D.2010.100')



tr = st[0]

print(tr.stats)

print(tr.data.mean())
print(tr.data.std())

#st.plot()