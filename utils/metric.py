import torch
import numpy as np

class Metric():
    def __init__(self, input:str='torch') -> None:
        self.input = input
    
    def Magnitude(self, matrix, mask=None):
        """
        :param matrix: (C,2,H,W) float
        :param mask: (C,H,W) bool
        :returns mag: (C,H,W) if mask=None else (N,)
        """
        if self.input == 'numpy':
            mag = np.sqrt(np.sum(matrix**2, axis=1))
        elif self.input == 'torch':
            mag = torch.sum(matrix ** 2, dim=1).sqrt()
        return mag if mask is None else mag[mask]
    
    def EPE(self, flow, groundtruth, mask=None):
        """
        :param flow, groundtruth: (C,(u,v),H,W) float
        :param mask: (C,H,W) bool
        :returns epe: (C,H,W) if mask=None else (N,)
        """
        return self.Magnitude(flow-groundtruth, mask)

    def AEE(self, epe):
        """
        :param epe: (C,H,W) float
        """
        if self.input == 'numpy':
            aee = np.mean(epe)
        elif self.input == 'torch':
            aee = torch.mean(epe)
        return aee
    
    def NPE(self, epe, n:int):
        """
        :param epe: (C,H,W) float
        """
        if self.input == 'numpy':
            npe = np.mean((epe > n).astype(float))
        elif self.input == 'torch':
            npe = (epe > n).float().mean()
        return npe * 100

    def Outlier(self, epe, magnitude=None):
        """
        :param epe: (C,H,W) float
        :param magnitude: (C,H,W) float
        """
        if self.input == 'numpy':
            if magnitude is None:
                outlier = (epe > 3.0).astype(np.float32)
            else:
                outlier = np.logical_and(epe > 3.0 , (epe/magnitude) > 0.05).astype(np.float32)
        elif self.input == 'torch':
            if magnitude is None:
                outlier = (epe > 3.0).float()
            else:
                outlier = ((epe > 3.0) & ((epe/magnitude) > 0.05)).float()
        return outlier.mean() * 100


    def RMSE(self, mag, mask=None):
        temp = mag if mask is None else mag[mask]
        if self.input == 'numpy':
            rmse = np.sqrt(np.mean((temp)**2))
        elif self.input == 'torch':
            rmse = torch.mean(temp ** 2).sqrt()
        return rmse

    def MAE(self, mag, mask=None):
        temp = mag if mask is None else mag[mask]
        if self.input == 'numpy':
            mae = np.sqrt(np.mean(np.abs(temp)))
        elif self.input == 'torch':
            mae = torch.mean(torch.abs(temp)).sqrt()
        return mae
    

if __name__ == '__main__':
    # Load Metric
    metric = Metric('numpy')

    a = np.ones((1,2,3))
    b = np.zeros((1,2,3))
    mask = np.ones((1,2,3)) > 0
    c = metric.mae(a,b,mask)
    print(c)
