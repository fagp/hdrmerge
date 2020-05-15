import time
from netutil.hdrnet import HDRNet

def main():
    net = HDRNet('defaults')
    start=time.time()
    net.do_train()
    print('Total Training Time {:.3f}'.format(time.time()-start))

if __name__ == '__main__':
    main()
