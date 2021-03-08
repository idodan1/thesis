import pickle

# l = ['sand_h', 'silt_h', 'clay_h']
# with open('texture_hydro_cols', 'wb') as f:
#     pickle.dump(l, f)

with open('cols_for_model', 'rb') as f:
    l = pickle.load(f)
l = l[1:]
print(l)
with open('cols_for_model', 'wb') as f:
    pickle.dump(l, f)