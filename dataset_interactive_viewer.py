# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html#trackbar

import cv2
import numpy as np

###---
from shumba import shumbacore
from dataset_experiments import extract_events_data_from_dataset, visualize_events_nonevents
import numpy as np

logger_directory = "/home/vitek/Vitek/python_codes/ox_audio_analysis_animals/DATA_BuffaloKillResampled/"
logger_filepattern =  "{:02d}/hq_{:d}.wav"

logger7pos = shumbacore.Position(-21.7278, 29.8843, 'LatLong')
logger7 = shumbacore.Logger(id=7, name="7", position=logger7pos, directory=logger_directory, filetemplate=logger_filepattern)

logger8pos = shumbacore.Position(-21.7237,29.8819,'LatLong')
logger8 = shumbacore.Logger(id=8,name="8",position=logger8pos,directory=logger_directory,filetemplate=logger_filepattern)
###---


def get_img_from_fig(fig, dpi=180):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")  # dpi=dpi
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # maybe messes it up?

    print("shape:", img.shape)
    return img

class GUIHandler(object):

    def onChangeSend(self,x,y):

        self.strong_coherence_threshold = cv2.getTrackbarPos('strong_coherence_threshold*10 (0-7)', self.window_name)
        self.weak_coherence_threshold = cv2.getTrackbarPos('weak_coherence_threshold*10 (0-4)', self.window_name)
        self.merge = cv2.getTrackbarPos('merge', self.window_name)
        self.event_length = cv2.getTrackbarPos('event_length*10', self.window_name)

        self.strong_coherence_threshold = self.strong_coherence_threshold / 10.0
        self.weak_coherence_threshold = self.weak_coherence_threshold / 10.0
        self.event_length = self.event_length / 10.0

        text = "thr strong >"+str(self.strong_coherence_threshold)+", thr non <"+str(self.weak_coherence_threshold)+\
               ", merge "+str(self.merge)+", "+str(self.event_length)+"sec."

        self.update_image()

    def update_image(self):
        #### EXECUTE THE CODE:
        #selected_logger = logger8
        #selected_coarse_timestamp = 1528419200
        selected_logger = logger7
        selected_coarse_timestamp = 1528419400

        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        print(">--------- station", selected_logger.logger_id, "@ time", selected_coarse_timestamp, ":")
        events, non_events = extract_events_data_from_dataset(selected_logger, selected_coarse_timestamp,
                                                              self.strong_coherence_threshold,
                                                              self.weak_coherence_threshold,
                                                              self.event_length, self.merge)
        fig = visualize_events_nonevents(selected_logger, selected_coarse_timestamp, events, non_events, show=False,
                                         save=None, linewidth=6, close=False)
        fig.tight_layout()

        # define a function which returns an image as numpy array from figure


        img_arr = get_img_from_fig(fig)
        image_res = cv2.resize(img_arr, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        ####
        print("Finished and displayed!")
        self.img = image_res

    def __init__(self):
        self.text = ""

    def start_window_rendering(self):

        # Create a black image, a window
        self.h = int(960 / 2.0)
        self.w = int(2400 / 2.0)
        self.img = np.zeros((self.h, self.w, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (0, self.h - 25)
        fontScale = 1.0
        fontColor = (128, 128, 128)
        lineType = 1

        self.window_name = 'InteractiveWindow'
        cv2.namedWindow(self.window_name)

        # create trackbars for color change
        x = lambda x: None
        cv2.createTrackbar('strong_coherence_threshold*10 (0-7)', self.window_name, 59, 70, x)
        cv2.createTrackbar('weak_coherence_threshold*10 (0-4)', self.window_name, 30, 40, x)
        cv2.createTrackbar('merge', self.window_name, 44, 60, x)
        cv2.createTrackbar('event_length*10', self.window_name, 40, 60, x)
        cv2.createButton("Generate", self.onChangeSend, None, cv2.QT_PUSH_BUTTON, 1)

        #self.onChangeSend(x=None,y=None) # toggle once at start
        text = " Set up the desired parameters, then press Ctrl+P and click 'Generate'."
        cv2.putText(self.img, text, position, font, fontScale, fontColor, lineType)

        while (1):
            # also keep another inf. loop

            cv2.imshow(self.window_name, self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

print("Note that every evaluation of the parameters takes time (cca 1 min).")

from threading import Thread

gui = GUIHandler()
thread = Thread(target=gui.start_window_rendering())
thread.start()
