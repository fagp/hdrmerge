import os
import numpy as np
import PIL.Image
import scipy.misc
import scipy.io
import rawpy
#########################################################################################################
class burstProvide(object):
    '''
    Management the image resources  
    '''

    def __init__(self, path_raw, path_result, ext_raw='dng',ext_res='merged.dng',size_burst=9,brange=[0]):

        exclude = ['c483_20150901_105412_265']
        
        if (os.path.isdir(path_raw) is not True) or (os.path.isdir(path_result) is not True):
            raise ValueError('Path {} is not directory'.format(path))
        
        self.size_burst=size_burst
        #load bursts info
        self.bursts = [ f for f in sorted(os.listdir(path_raw))]
        self.files = dict()
        self.range=brange
        for i in self.range:
            images = [ os.path.join(path_raw, self.bursts[i], f ) for f in sorted(os.listdir(os.path.join(path_raw, self.bursts[i]))) if (f.split('.')[-1] == ext_raw) ]
            if (self.size_burst==-1):
                self.files[i] = images
            else:
                self.files[i] = images[:size_burst]
                
        self.num = len(self.range)

        self.hdrgt = [ os.path.join(path_result, f, ext_res ) for i,f in enumerate(self.bursts) if i in self.range]

        self.path_raw=path_raw
        self.path_result=path_result
        self.index = 0

    def getimage(self, i):
        '''
        Get image i
        '''
        #check index
        if i<0 and i>self.num: raise ValueError('Index outside range');
        self.index = i;
        pathgt = self.hdrgt[i]

        gt = np.array(self._loadimage(pathgt))
        burst = self._loadbusrt(self.index)
        Flag=True
        if (burst.shape[0]!=gt.shape[0]) or (burst.shape[1]!=gt.shape[1]): #if size mismatch try to rotate
            burst=np.rot90(burst,-1,(0,1))
            if (burst.shape[0]!=gt.shape[0]) or (burst.shape[1]!=gt.shape[1]): #if size mismatch after rotation skip image (Flag=False)
                Flag=False
        #         # raise ValueError('Dimension mismatch for busrt "%s" ' % self.bursts[self.range[self.index]])
        #         ind=self.index
        #         gt, burst=self.getimage(i-1)
        #         self.index=ind

        return gt, burst,Flag

    def exclude(self,index):
        del self.hdrgt[index]
        del self.range[index]



    def getindex(self, i):
        if i<0 and i>self.num: raise ValueError('Index outside range');
        self.index = i;

    def next(self):
        '''
        Get next image
        '''
        i = self.index;        
        gt,burst = self.getimage(i)
        self.index = (i + 1) % self.num
        return gt,burst

    def getimagename(self):
        '''
        Get current image name
        '''
        return self.bursts[self.range[self.index]]

    def isempty(self): return self.num == 0;

    def raw2bayern(self,raw,color):
        bayern= np.zeros((raw.shape[0]//2,raw.shape[1]//2,4))
        for i in range(4):
            bayern[:,:,i]=np.reshape(raw[color==i], (bayern.shape[0],bayern.shape[1]))
        return bayern

    def bayern2raw(self,bayern):
        raw= np.zeros((bayern.shape[0]*2,bayern.shape[1]*2))
        raw[0::2,0::2]=bayern[:,:,0]
        raw[0::2,1::2]=bayern[:,:,1]
        raw[1::2,0::2]=bayern[:,:,2]
        raw[1::2,1::2]=bayern[:,:,3]
        return raw

    def _loadbusrt(self, index):
        busrti=self.files[self.range[index]]
        with rawpy.imread(busrti[0]) as raw:
            raw_image = raw.raw_image.copy()
            raw_colors = raw.raw_colors.copy()
        
        busrt=np.zeros((raw_image.shape[0],raw_image.shape[1],len(busrti)))
        busrt[:,:,0]=self.bayern2raw( self.raw2bayern(raw_image,raw_colors) )
        for i in range(1,len(busrti)):
            with rawpy.imread(busrti[i]) as raw:
                raw_image = raw.raw_image.copy()
                raw_colors = raw.raw_colors.copy()
            busrt[:,:,i]=self.bayern2raw( self.raw2bayern(raw_image,raw_colors) )

        return busrt.astype(float)/2**10

    def _loadimage(self, pathname):
        if '.dng'==pathname[-4:]:
            with rawpy.imread(pathname) as raw:
                raw_image = raw.raw_image.copy()
                raw_colors = raw.raw_colors.copy()
                res=self.bayern2raw( self.raw2bayern(raw_image,raw_colors) ).astype(float)/2**14
        else:
            res = self.__loadimage(pathname)

        return res

    def __loadimage(self, pathname):
        '''
        Load image using pathname
        '''

        if os.path.exists(pathname):
            try:
                image = PIL.Image.open(pathname)
                image.load()
            except IOError as e:
                raise ValueError('IOError: Trying to load "%s": %s' % (pathname, e.message) ) 
        else:
            raise ValueError('"%s" not found' % pathname)

        if image.mode in ['L', 'RGB']:
            # No conversion necessary
            return image
        elif image.mode in ['1']:
            # Easy conversion to L
            return image.convert('L')
        elif image.mode in ['LA']:
            # Deal with transparencies
            new = PIL.Image.new('L', image.size, 255)
            new.paste(image, mask=image.convert('RGBA'))
            return new
        elif image.mode in ['CMYK', 'YCbCr']:
            # Easy conversion to RGB
            return image.convert('RGB')
        elif image.mode in ['P', 'RGBA']:
            # Deal with transparencies
            new = PIL.Image.new('RGB', image.size, (255, 255, 255))
            new.paste(image, mask=image.convert('RGBA'))
            return new
        else:
            raise ValueError('Image mode "%s" not supported' % image.mode)
        
        return image;