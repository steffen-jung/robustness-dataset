# -*- coding: utf-8 -*-

import json, os
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches

DATA_CIFAR10 = "cifar10"
DATA_CIFAR100 = "cifar100"
DATA_IMAGENET16 = "ImageNet16-120"
DATA_ALL = [DATA_CIFAR10, DATA_CIFAR100, DATA_IMAGENET16]
KEYS_CLEAN = ["clean"]
KEYS_ADV = ["aa_apgd-ce@Linf", "aa_square@Linf", "fgsm@Linf",
            "pgd@Linf"]
KEYS_CC = ["brightness", "contrast", "defocus_blur", "elastic_transform",
           "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise",
           "jpeg_compression", "motion_blur", "pixelate", "shot_noise", 
           "snow", "zoom_blur"]
KEYS_ALL = KEYS_CLEAN + KEYS_ADV + KEYS_CC

class RobustnessDataset:
    """
    Helper class to query evaluation results.
    
    Attributes
    ----------
    keys_clean : list
        key for evaluation results on clean data: ["clean"]
    keys_adv : list
        list that contains keys for all adversarial attack types evaluated
    keys_cc : list
        list that contains keys for all corruption types evaluated
    keys_all : list
        list that contains all keys
    data_cifar10 : str = "cifar10"
    data_cifar100 : str = "cifar100"
    data_imagenet16 : str = "ImageNet16-120"
    data : list
        list that contains all data sources ["cifar10", "cifar100", "ImageNet16-120"]
    """
    
    keys_clean = KEYS_CLEAN
    keys_adv = KEYS_ADV
    keys_cc = KEYS_CC
    keys_all = KEYS_ALL
    data_cifar10 = DATA_CIFAR10
    data_cifar100 = DATA_CIFAR100
    data_imagenet16 = DATA_IMAGENET16
    data = DATA_ALL
    
    ############################################################################
    def __init__(self, path="robustness-data"):
        """
        Parameters
        ----------
        path : str
            Path to the root folder of the dataset data.
        """
        
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
            self.map_str_to_id = {m["nb201-string"]:k for k,m in self.meta["ids"].items()}
            self.non_isomorph_ids = [i for i, d in self.meta["ids"].items() if d["isomorph"]==i]
    
    ############################################################################
    def _ensure_list(self, l):
        if type(l) is not list:
            l = [l]
        return l
    
    ############################################################################
    def query(
            self,
            data = DATA_ALL,
            key = KEYS_ALL,
            measure = ["accuracy", "confidence", "cm"],
            missing_ok = False,
            tqdm = None
        ):
        """
        Query evaluation results.
        Returns a dictionary: dict[<data>][<attack/corruption>][<measure type>][<architecture id>]
        
        Parameters
        ----------
        data : str/list
            Data used for evaluation.
        key : str/list
            Adversarial attack or corruption type.
        measure : str/list
            Measure type ("accuracy", "confidence", "cm")
        """
        
        data = self._ensure_list(data)
        key = self._ensure_list(key)
        measure = self._ensure_list(measure)
        
        pbar = tqdm.tqdm(
            total = len(data)*len(key)*len(measure)
        ) if tqdm is not None else None
            
        result = {d:{k:{} for k in key} for d in data}
        for d in data:
            for k in key:
                if d == RobustnessDataset.data_imagenet16:
                    if k in RobustnessDataset.keys_cc:
                        if pbar is not None:
                            pbar.update(1)
                        continue
                
                for m in measure:
                    file = os.path.join(self.path, d, f"{k}_{m}.json")
                    if missing_ok:
                        if not os.path.isfile(file):
                            if pbar is not None:
                                pbar.update(1)
                            continue
                    with open(file, "r") as f:
                        r = json.load(f)
                    result[d][k][m] = r[d][k][m]
                    if pbar is not None:
                        pbar.update(1)
                
                if len(result[d][k]) == 0:
                    del result[d][k]
        if pbar is not None:
            pbar.close()
        
        return result
    
    ############################################################################
    def get_uid(self, i):
        """
        Returns the evaluated architecture id (if given id is isomorph to another network)
        
        Parameters
        ----------
        i : str/int
            Architecture id.
        """
        
        return self.meta["ids"][str(i)]["isomorph"]
    
    ############################################################################
    def id_to_string(self, i):
        """
        Returns the string representing an architecture in NAS-Bench-201 for the given id.
        
        Parameters
        ----------
        i : str/int
            Architecture id.
        """
        
        return self.meta["ids"][str(i)]["nb201-string"]
    
    ############################################################################
    def string_to_id(self, s):
        """
        Returns the id of a given NAS-Bench-201 architecture string.
        
        Parameters
        ----------
        s : str
            Architecture string as in NAS-Bench-201.
        """
        
        return self.map_str_to_id[s]
    
    ############################################################################
    def draw_arch(self, s=None, i=None):
        """
        Plot the cell of a given NAS-Bench-201 architecture string or architecture id.
        
        Parameters
        ----------
        i : str/int
            Architecture id.
        s : str
            Architecture string as in NAS-Bench-201.
        """
        
        assert s is not None or i is not None
        if s is None:
            s = self.id_to_string(i)
        if i is None:
            i = self.string_to_id(s)
            
        pos = {(0,1):(0.25,0.75),(0,2):(0.5,0),(0,3):(1,-0.75),(1,2):(1,0.5),(1,3):(1.75,0.75),(2,3):(1.5,0)}
        m_ops = {"avg_pool_3x3":"avg", "nor_conv_1x1":"1x1", "nor_conv_3x3":"3x3", "skip_connect":"skip", "none":"zero"}
        
        p_0_1 = patches.FancyArrowPatch(
            path=path.Path([(0, 0),(0, 1),(1-0.2, 1)],
            [path.Path.MOVETO, path.Path.CURVE3, path.Path.CURVE3]),
            fc="none", transform=plt.gca().transData, arrowstyle="-|>,head_length=5,head_width=3"
        )
        p_0_2 = patches.FancyArrowPatch(
            path=path.Path([(0,0),(1-0.2,0)],
            [path.Path.MOVETO, path.Path.LINETO]),
            fc="none", transform=plt.gca().transData, arrowstyle="-|>,head_length=5,head_width=3"
        )
        p_0_3 = patches.FancyArrowPatch(
            path=path.Path([(0, 0),(1,-1.5),(2-0.14,0-0.14)],
            [path.Path.MOVETO, path.Path.CURVE3, path.Path.CURVE3]),
            fc="none", transform=plt.gca().transData, arrowstyle="-|>,head_length=5,head_width=3"
        )
        p_1_2 = patches.FancyArrowPatch(
            path=path.Path([(1,1),(1,0+0.2)],
            [path.Path.MOVETO, path.Path.LINETO]),
            fc="none", transform=plt.gca().transData, arrowstyle="-|>,head_length=5,head_width=3"
        )
        p_1_3 = patches.FancyArrowPatch(
            path=path.Path([(1,1),(2,1),(2,0+0.2)],
            [path.Path.MOVETO, path.Path.CURVE3, path.Path.CURVE3]),
            fc="none", transform=plt.gca().transData, arrowstyle="-|>,head_length=5,head_width=3"
        )
        p_2_3 = patches.FancyArrowPatch(
            path=path.Path([(1,0),(2-0.2,0)],
            [path.Path.MOVETO, path.Path.LINETO]),
            fc="none", transform=plt.gca().transData, arrowstyle="-|>,head_length=5,head_width=3"
        )
    
        plt.gca().add_patch(p_0_1)
        plt.gca().add_patch(p_0_2)
        plt.gca().add_patch(p_0_3)
        plt.gca().add_patch(p_1_2)
        plt.gca().add_patch(p_1_3)
        plt.gca().add_patch(p_2_3)
    
        circle = plt.Circle((0,0), 0.22, color="black")
        plt.gca().add_patch(circle)
        circle = plt.Circle((0,0), 0.2, color="white")
        plt.gca().add_patch(circle)
    
        circle = plt.Circle((1,1), 0.22, color="black")
        plt.gca().add_patch(circle)
        circle = plt.Circle((1,1), 0.2, color="white")
        plt.gca().add_patch(circle)
    
        circle = plt.Circle((1,0), 0.22, color="black")
        plt.gca().add_patch(circle)
        circle = plt.Circle((1,0), 0.2, color="white")
        plt.gca().add_patch(circle)
    
        circle = plt.Circle((2,0), 0.22, color="black")
        plt.gca().add_patch(circle)
        circle = plt.Circle((2,0), 0.2, color="white")
        plt.gca().add_patch(circle)
    
        plt.text(0, 0, "in", va="center", ha="center")
        plt.text(1, 1, "1", va="center", ha="center")
        plt.text(1, 0, "2", va="center", ha="center")
        plt.text(2, 0, "out", va="center", ha="center")
        plt.text(-0.3, 1.3, f"# {i}")
        
        for v, ops in enumerate(s.split("+")):
            v += 1
            ops = ops[1:-1]
            ops = ops.split("|")
            for o in ops:
                o, v_src = o.split("~")
                o = m_ops[o]
                x,y = pos[(int(v_src),v)]
                plt.text(
                    x, y, o,
                    va = "center",
                    ha = "center",
                    backgroundcolor = "w"
                )
    
        plt.gca().set_aspect("equal")
        plt.xlim(-0.5, 2.5)
        plt.ylim(-1, 1.5)
        plt.xticks([])
        plt.yticks([])
    
        plt.show()