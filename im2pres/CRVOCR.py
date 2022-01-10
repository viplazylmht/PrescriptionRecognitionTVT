from craft_text_detector import Craft
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import hashlib, time

from im2pres.utils import vietocr_time

class CRVOCR:
    def __init__(self, output_dir=None, use_gpu=True):
        """
        output_dir: output images for detection step, optional
        """
        self.detector = Craft(output_dir=output_dir, crop_type="box", cuda=use_gpu, export_extra=True) if output_dir is not None else \
                        Craft(crop_type="box", cuda=use_gpu)

        #config = Cfg.load_config_from_name('vgg_transformer')
        config = Cfg.load_config_from_name('vgg_seq2seq')
        # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        config['cnn']['pretrained']=False
        config['device'] = 'cuda:0' if use_gpu else 'cpu'
        config['predictor']['beamsearch']=False

        self.recognigitor = Predictor(config)

        self.output_logs = {}

    def md5(self, fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_text(s, img_path):
        """
        Extract text in a given image path
        """
        start_time = time.time()

        job_id = s.md5(img_path)

        timers, contents = vietocr_time(s.recognigitor, s.detector, img_path) 

        running_time = time.time()-start_time

        s.output_logs[job_id] = {
            'img_path': img_path,
            'timer': {
            'detection': timers[0],
            'recognition': timers[-1],
            'total_time': running_time
        }, 'contents': contents,
        }

        return s.output_logs[job_id]
    
    def get_log(s, img_path):
        job_id = s.md5(img_path)

        return s.output_logs[job_id]

    def to_string(this):
        return '\n'.join(this.output_logs.values())