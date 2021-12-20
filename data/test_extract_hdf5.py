import h5py

# example of reading the image.h5 file. key is image_filename and value is the embedding vector 

filename = 'data/SpokenCOCO/SpokenCOCO_images.h5'
data = h5py.File(filename, 'r')
for group in data.keys(): 
    print(group) 
    print(data[group].shape) # should be a 2048-dim embedding 
    break 

print(len(data.keys())) # should be 123,287 images 
