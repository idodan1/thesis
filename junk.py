import pickle

l = ['sand_h', 'silt_h', 'clay_h']
with open('texture_hydro_cols', 'wb') as f:
    pickle.dump(l, f)

print(l)