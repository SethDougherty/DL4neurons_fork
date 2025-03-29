import h5py
h5path = "/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/stims/allen_data_stims_314831019.hdf5"
fi = h5py.File(h5path)
st = fi['34']
myfile = open("/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/stims/CompareTest.csv","w")
for yU in st:
        myfile.write(str(yU)+"\n")