import h5py

h5_label_file = h5py.File('../../data/cocotalk_label.h5', 'r', driver='core')

for key in h5_label_file.keys():
    print(f'- {key}')
    print(h5_label_file.get(key)[:])
    print()

print(h5_label_file.get('labels')[0])

    


 
