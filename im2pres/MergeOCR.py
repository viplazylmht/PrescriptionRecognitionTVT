import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage
import timeit, os

from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from im2pres.utils import crop_image

class MergeOCRED():
    def __init__(s, threshold=0.01):
        """threshold: float in range(0..1)"""
        
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be float and in range (0..1)")
            
        s.threshold = threshold 
        
        s.hc = None # AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', linkage ='average', distance_threshold=s.threshold)
        
        os.system('wget -q https://cdn.discordapp.com/attachments/817270986423664670/908330866042863636/font.zip')
        os.system('unzip -o font.zip')
        s.font = ImageFont.truetype('fonts/Roboto Mono/RobotoMono-Regular.ttf', 17)
        os.system('rm -r fonts font.zip')
    
    def distance(self, img, box1, box2):
        l1, t1 = box1[0]
        l2, t2 = box2[0]
        
        cy1, cx1 = ndimage.measurements.center_of_mass(np.array(crop_image(img, box1).convert('L')))
        cy2, cx2 = ndimage.measurements.center_of_mass(np.array(crop_image(img, box2).convert('L')))
        
        # return (dy, dx)
        return ((t2 + cy2) - (t1 + cy1)), ((l2 + cx2) - (l1 + cx1))
    
    def distance_matrix(self, img, data, axis=0):
        """
        return: distane_matrix [NxN]
        """
        
        matrix = np.zeros(len(data) ** 2, dtype='float').reshape(-1,len(data))

        for i,x in enumerate(data): 
            for j, y in enumerate(data[i:]):
                matrix[i+j, i] = matrix[i, i+j] = np.abs(self.distance(img, x['box'], y['box'])[axis])
        return matrix
    
    def merge(self, src_img, data, verbose=False):
        """
        Usage: Merge texts of sub images into a text
        src_img: the PIL image that contain data, this var is only for reference
        data: list of dict: [{'line': str, 'box': ndarray(2,2) }, ]
        verbose: True if you want visualize the result
        return: list of dict: same as data but maybe smaller
        """
        if not isinstance(data, list):
            raise ValueError("data must be a list of dict")
        
        start = timeit.default_timer()
        height = src_img.size[1]
        
        self.hc = AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', linkage ='average', distance_threshold=self.threshold * height)
        
        distances = self.distance_matrix(src_img, data)
        
        self.hc.fit(distances)
        
        #self.plot_dendrogram(self.hc, )
        #plt.figure(figsize=(14, 14))
        #self.plot_dendrogram(self.hc, truncate_mode="level")
        #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        #plt.show()
        
        results = [np.take(data, np.where(self.hc.labels_ == i)).reshape(-1) for i in range(self.hc.n_clusters_)]
        
        if verbose:
            print("number of grouped:", self.hc.n_clusters_)
        
        results = sorted(results, key = lambda x : x[0]['box'][0][1])
        
        for i in range(len(results)):  
            r = sorted(results[i], key = lambda x : x['box'][0][0])
            
            lines = []
            boxs = []
            for b in r:
                lines.append(b['line'])
                boxs.append(b['box'])
            
            results[i] = {'line': " ".join(lines), 'box': boxs}
            
            if verbose:
                self.bbox(src_img, [results[i],])
        end = timeit.default_timer()
        return results, end - start
