from netCDF4 import Dataset 
import numpy as np
import json 

class Net():
    def __init__(self, name, cube, grid_lon, grid_lat,n_times):
        self.name = name
        self.cube = cube
        self.grid_lon = grid_lon
        self.grid_lat = grid_lat
        self.long_name = ''
        self.units = ''
        self.ncfile = None
        self.lat_dim = None
        self.lon_dim = None
        self.time_dim = None
        self.n_times = n_times

    
    def read_json(self):
       f = open('bioclimatic.json')
       data = json.load(f)
       self.long_name = data[self.name]['long_name']
       self.units = data[self.name]['units']
       f.close()

    def open_file(self):
        #try: ncfile.close()  # just to be safe, make sure dataset is not already open.
        #except: pass
        self.ncfile = Dataset(self.name+'.nc',mode='w',format='NETCDF4_CLASSIC') 
        print(self.ncfile)

    def creating_dimensions(self):
        self.lat_dim = self.ncfile.createDimension('lat', len(self.grid_lat))     # latitude axis
        self.lon_dim = self.ncfile.createDimension('lon', len(self.grid_lon))    # longitude axis
        self.time_dim = self.ncfile.createDimension('time', None) # unlimited axis (can be appended to).
        for dim in self.ncfile.dimensions.items():
            print(dim)

    def creating_attributes(self):
        self.ncfile.title='Bioclimatic Variables'
        #self.ncfile.subtitle="My model data subtitle"
        #self.ncfile.author
    
    def creating_variables(self):
        lat = self.ncfile.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lon = self.ncfile.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        time = self.ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'years since 2007-01-15'
        time.long_name = 'time'
        # Define a 3D variable to hold the data
        z = self.ncfile.createVariable(self.name,np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
        z.units = self.units 
        z.long_name = self.long_name # this is a CF standard name

        #writing 

        ntimes = self.n_times
        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = self.grid_lat
        lon[:] = self.grid_lon
        # Write the data.  This writes the whole 3D netCDF variable all at once.
        z[:,:,:] = self.cube  # Appends data along unlimited dimension
        print("-- Wrote data, temp.shape is now ", z.shape)
        # read data back from variable (by slicing it), print min and max
        print("-- Min/Max values:", z[:,:,:].min(), z[:,:,:].max())

        import datetime as dt
        from netCDF4 import date2num,num2date
      
        year=2007
        dateList = []
        for x in range (0, ntimes):
            dateList.append(dt.datetime(year+x,1,15,0))
        print(dateList)
        times= range(0,ntimes)
        #times = date2num(dateList, time.units)
        time[:] = times
        print(time[:])
        #print(num2date(time[:],time.units))

    def closing_file(self):
        self.ncfile.close()
        print('Dataset is closed!')

    def setup(self):
        self.open_file()
        self.creating_dimensions()
        self.creating_attributes()
        self.creating_variables()
        self.closing_file()
        return print(self.name+".nc created !!")