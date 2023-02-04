#!/mnt/sdd/dongqs/venv_gcn/bin/python3
import numpy as np
import torch
import glob
import sys
import argparse


class FFT3DCNN_UTILS():
    def __init__(self,n=8):
        self.device = 'cpu'
        self.n = n
        self.Nxyz_fft = None
        self.ph_fft = None
        self.Lxyz = None

    def __perform_fft_torch(self,pha):
        ph2 = torch.fft.fftn(torch.squeeze(pha))
        slice_idx =  list(range(0,self.n,1))+list(range(-self.n,0,1))
        ph3 = ph2[slice_idx,:, :][:,slice_idx,:][:,:,slice_idx]
        ph4 = torch.abs(torch.fft.ifftn(ph3))
        Nx,Ny,Nz = pha.shape
        Nx2,Ny2,Nz2 = ph4.shape
        ph4 = ph4*Nx2*Ny2*Nz2/Nx/Ny/Nz

        return ph4


    def load_ph_ABC(self, filename): # read from ph.bin directly
        Nxyz = np.fromfile(filename, dtype=np.intc, count=3, offset=0) # intc is identical to C int (normally int32 or int64)
        # print(Nxyz[0].itemsize) # the size of intc is 4
        self.Lxyz = np.fromfile(filename, dtype=np.float64, count=3, offset=3*4)
        # print(Lxyz[0].itemsize) # the size of float64 is 8
        ph = torch.from_numpy(np.fromfile(filename, dtype=np.float64, count=-1, offset=3*(4+8))).to(torch.float32).to(self.device) # read all remaining data
        # NxNyNz = Nxyz[0]*Nxyz[1]*Nxyz[2]
        ph = ph.reshape([3,Nxyz[0],Nxyz[1],Nxyz[2]])

        if Nxyz[0] ==1:# and Nxyz[1]==Nxyz[2]:
            ph = ph.repeat(1,Nxyz[2],1,1)
            Nxyz[0] = Nxyz[2]

        pha = self.__perform_fft_torch(ph[0,:,:,:])
        phb = self.__perform_fft_torch(ph[1,:,:,:])
        phc = self.__perform_fft_torch(ph[2,:,:,:])
        self.ph_fft = torch.stack((pha,phb,phc))

        self.Nxyz_fft = list(self.ph_fft.shape)

    
    def save_ph_ABC(self,filename):
        with open(filename,"wb") as f:
            f.write(np.array(self.Nxyz_fft[1:],dtype=np.intc))
            f.write(np.array(self.Lxyz,dtype=np.float64))
            f.write(self.ph_fft.to(torch.float64).numpy())

    def load_ph_ABC_pure(self,filename):
        Nxyz = np.fromfile(filename, dtype=np.intc, count=3, offset=0) # intc is identical to C int (normally int32 or int64)
        print(Nxyz)
        # print(Nxyz[0].itemsize) # the size of intc is 4
        Lxyz = np.fromfile(filename, dtype=np.float64, count=3, offset=3*4)
        print(Lxyz)
        # print(Lxyz[0].itemsize) # the size of float64 is 8
        ph = torch.from_numpy(np.fromfile(filename, dtype=np.float64, count=-1, offset=3*(4+8))).to(torch.float32).to(self.device) # read all remaining data
        # NxNyNz = Nxyz[0]*Nxyz[1]*Nxyz[2]
        ph = ph.reshape([3,Nxyz[0],Nxyz[1],Nxyz[2]])

        print('after fft',Nxyz,Lxyz,ph[0][0][0][0:10])

        ph_diff = ph-self.ph_fft

        print('max',torch.max(ph_diff))
        print('min',torch.min(ph_diff))

def my_parse_args():
    parser = argparse.ArgumentParser(description='QingshuDong perform_fft.py')
    parser.add_argument('-n', type=int, default=8, help='num elements')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = my_parse_args()
    print(args)

    root_path = '/home/dongqs/sdd/AABCn3D/' # linux
    # root_path = 'Z:/sdd/AABCn3D/' # windows

    structureNames = ['sadc','dadc','gagc','CsCl','dg_cs','dgn','fcc_c','bcc_c','bcc_a','g_c','IWP','PP','sagc','sddn','sdgn',
                        'c4c4','c4_cs','calc','L3',
                        'c3c6','c6_c','c6c3','c6_c_cs','c6_cs','smch','splh',
                        'sigmaN']

    for idx, sName in enumerate(structureNames): # get file names and labels
        print('---------------------------------')
        print(idx,'of',len(structureNames),':',sName)
        sys.stdout.flush()

        files = glob.glob(root_path + sName + '/*/ph.bin')

        for filename in files:
            print(filename,'--> ph{}.bin'.format(args.n))
            hh = FFT3DCNN_UTILS(n = args.n)
            hh.load_ph_ABC(filename)
            hh.save_ph_ABC(filename.replace('ph.bin','ph{}.bin'.format(args.n)))
            sys.stdout.flush()
