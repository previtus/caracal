# Fast code to extract event samples from the dataset given the classical pipeline of Caracal.
# We get samples which contain events (due to coherence criteria) and events without any events.

import os
import sys

from shumba import shumbacore
from shumba import audiocore
from shumba import extract
from shumba import match
from shumba import position

import pylab as plt
import numpy as np
from matplotlib.patches import Rectangle

from scipy.io import wavfile

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def is_overlapping(x1, x2, y1, y2):
    return max(x1, y1) <= min(x2, y2)

def draw_bboxes_for_events(events, ax, boolean_mask=None):
    orig_xlim = ax.get_xlim()
    ### Mark down events
    for event_i in range(len(events)):
        if boolean_mask is not None:
            if not boolean_mask[event_i]:
                continue  # skip this event's bbox drawing
        # print(dir(events[event_i]))
        # print("logger,timestamp,time,len:", events[event_i].logger, events[event_i].coarse_timestamp, events[event_i].event_time, events[event_i].event_length, )

        # In plot coordinates
        t_start_orig = events[event_i].event_time
        t_end_orig = t_start_orig + events[event_i].event_length
        #print(t_start_orig, t_end_orig)

        # In data coordinates
        # t_start = int( events[event_i].event_time*Fs )
        # t_end = t_start + int( events[event_i].event_length*Fs )
        # print(rawAudio[t_start:t_end, : ].shape)

        # tmp_k = (1 + event_i) * (2000 / (len(events) + 1))
        # xy, width, height = (t_start_orig, tmp_k), 4, 100
        xy, width, height = (t_start_orig, 0), 4, 2000 - 30

        # Draw a diagonal line
        point1 = [xy[0], xy[1]]
        point2 = [xy[0] + width, height]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        line_handler = ax.plot(x_values, y_values, linewidth=2)
        c = line_handler[0].get_color()  # 'r'

        # Draw a rectangle around event:
        # xy, width, height
        rect = Rectangle(xy, width, height, linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)

        # Line:
        # plt.axvline(x=t_start_orig)
        # plt.axvline(x=t_end_orig)

    ax.set_xlim(orig_xlim)


def extract_events_data_from_dataset(logger, coarse_timestamp, strong_coherence_threshold = 5.9, weak_coherence_threshold = 3.0):

    # Get all positive matches for events ~ "has event"
    merge = 44 # 40 is close to having no overlap in between two 4 sec events
    bigger_than_coherence = lambda metric, coherence_thr: metric > coherence_thr

    ev = extract.CoherentEventSegmenter(coherence_threshold=strong_coherence_threshold,merge=merge,event_length=4.0, metric_comparison_function_over=bigger_than_coherence)
    bf, events = ev.extractEvents(logger, coarse_timestamp)
    print("Strong events: ", len(events), "events have coherence > ",strong_coherence_threshold)

    # Get all negative matches for events ~ "doesnt have event"
    merge = 44
    smaller_than_coherence = lambda metric, coherence_thr: metric < coherence_thr
    non_ev = extract.CoherentEventSegmenter(coherence_threshold=weak_coherence_threshold,merge=merge,event_length=4.0, metric_comparison_function_over=smaller_than_coherence)
    _, non_events = non_ev.extractEvents(logger, coarse_timestamp)
    print("Non events: ", len(non_events), "'non-events' have coherence < ",weak_coherence_threshold)

    ### Possibly evaluate overlaps
    # ranges_events contain [start_event_time, end_event_time]
    # nonoverlapping_nonevent_flags has boolean flags for items in the non_events non overlapped by items in the events
    ranges_events = []
    nonoverlapping_nonevent_flags = []

    for event_i in range(len(events)):
        t_start_orig = events[event_i].event_time
        t_end_orig = t_start_orig + events[event_i].event_length
        ranges_events.append([t_start_orig,t_end_orig]) # they are also ordered...

    for non_event_i in range(len(non_events)): # x1,x2 <?> y1,y2
        x1 = non_events[non_event_i].event_time
        x2 = x1 + non_events[non_event_i].event_length

        overlapping = False
        for r in ranges_events:
            y1, y2 = r
            if is_overlapping(x1,x2,y1,y2): # one overlaping is enough
                overlapping = True
                break
            if y2 > x2: # .. we could also stop early for efficiency here
                break
        nonoverlapping_nonevent_flags.append(not overlapping)

    final_non_events = [non_events[i] for i in range(len(non_events)) if nonoverlapping_nonevent_flags[i]]

    print("We got", len(events), "events.")
    print("And", len(final_non_events), "non-events which weren't overlapping.")
    return events, final_non_events

def visualize_events_nonevents(logger, coarse_timestamp, events, non_events, show=False, save=None):
    viz = audiocore.AudioViz()
    fig = plt.figure(figsize=(10 * 4, 4 * 4), dpi=60)
    fig.suptitle('Visualization of salient (shown in first row) and non-salient (second row) events (all have 4 channels).', fontsize=32)

    ## Load from objects:
    rawAudio = audiocore.AudioFile(logger).loadAtTime(coarse_timestamp)
    rawAudio = rawAudio.copy()

    print("rawAudio.shape", rawAudio.shape)

    viz.plotSpectrogram(rawAudio)

    ax_list = fig.axes
    draw_bboxes_for_events(events, ax_list[0])
    draw_bboxes_for_events(non_events, ax_list[1])

    #for ax in ax_list:
    #    draw_bboxes_for_events(events, ax)
    #    draw_bboxes_for_events(non_events, ax)

    if show:
        plt.show()

    if save is not None:
        name = "SaliencyViz_"+str(logger.logger_id)+"_"+str(coarse_timestamp)
        plt.savefig(save+name+".png")

    # plt.figure(figsize=(8, 6), dpi=180)
    # viz.plotSpectrolines(rawAudio,spacing=0.05,low_band=10,high_band=200,step=2)
    # plt.show()
    plt.close()

def save_audio_samples(events, directory):
    if len(events) == 0:
        return None

    viz = audiocore.AudioViz()
    opener = audiocore.AudioFile(events[0].logger)
    rawAudio = opener.loadAtTime(events[0].coarse_timestamp) # < all share one
    Fs = 44100

    for event in events:
        sample_idx = int((event.event_time * Fs))
        mid_len = int((event.event_length / 2) * Fs)
        full_audio_sample = rawAudio[sample_idx - mid_len: sample_idx + mid_len, :]

        # directory/ logerid_coarsetimestamp_(?).wav === 7_1528419100_17.wav
        filename = directory + str(event.logger.logger_id) + "_" + str(event.coarse_timestamp) + "_" + str(int(event.event_time))

        # Save PNG plot
        fig = plt.figure(figsize=(4, 4*4), dpi=60)
        fig.suptitle("Event "+str(event.logger.logger_id) + "_" + str(event.coarse_timestamp) + "_" + str(sample_idx)+", SS: "+str(event.SS))
        # << as a bonus might be useful to mark which window caused this event and what was it's coherence score ...

        viz.plotSpectrogram(full_audio_sample)
        plt.savefig(filename + ".png")
        plt.close()

        # Save WAV
        data = np.asarray(full_audio_sample, dtype=np.float)
        #print("data.shape", data.shape)
        wavfile.write(filename + ".wav",Fs,data)

###


################
#"""
# we create our logger objects and embed them into our world.
# the size of our world. note that the origin (0,0) is defined by
# the position of the lower left node
worldExtents = [-1500,3000,-1500,3000]
world = shumbacore.World(worldExtents,sample_rate=44100)
coarse_timestamps = [1528419100,1528419200,1528419300,1528419400,1528419500]
# Where the files live
logger_directory = "/home/vitek/Vitek/python_codes/ox_audio_analysis_animals/DATA_BuffaloKillResampled/"
logger_filepattern =  "{:02d}/hq_{:d}.wav"
# Coordinates of the loggers
logger7pos = shumbacore.Position(-21.7278,29.8843,'LatLong')
logger8pos = shumbacore.Position(-21.7237,29.8819,'LatLong')
logger10pos = shumbacore.Position(-21.7317,29.8876,'LatLong')
logger14pos = shumbacore.Position(-21.7359,29.8777,'LatLong')
# Setup the loggers themselves
logger7 = shumbacore.Logger(id=7,name="7",position=logger7pos,directory=logger_directory,filetemplate=logger_filepattern)
logger8 = shumbacore.Logger(id=8,name="8",position=logger8pos,directory=logger_directory,filetemplate=logger_filepattern)
logger10 = shumbacore.Logger(id=10,name="10",position=logger10pos,directory=logger_directory,filetemplate=logger_filepattern)
logger14 = shumbacore.Logger(id=14,name="14",position=logger14pos,directory=logger_directory,filetemplate=logger_filepattern)
loggerlist = [logger7,logger8,logger10,logger14]
# Add the loggers to the world
for logger in loggerlist:
    world.addLogger(logger)

lat = -21.7278
lon = 29.8797
targetpos = shumbacore.Position(lat,lon,'LatLong')
target = shumbacore.Target(0,"KillLoc",targetpos)
world.addTarget(target)

"""
logger_directory = "/home/vitek/Vitek/python_codes/ox_audio_analysis_animals/DATA_BuffaloKillResampled/"
logger_filepattern =  "{:02d}/hq_{:d}.wav"
logger8pos = shumbacore.Position(-21.7237,29.8819,'LatLong')
logger8 = shumbacore.Logger(id=8,name="8",position=logger8pos,directory=logger_directory,filetemplate=logger_filepattern)
"""

#################
strong_coherence_threshold = 5.9 # salient events have coherence > this thr
weak_coherence_threshold = 3.0   # non events have coherence < this thr

plots_directory = "samples/" # plots/
mkdir(plots_directory)
mkdir("samples/")
events_directory = "samples/events/"
mkdir(events_directory)
nonevents_directory = "samples/nonevents/"
mkdir(nonevents_directory)


total_events = []
total_non_events = []

#if True:
#    if True:

for selected_logger in loggerlist:
    for selected_coarse_timestamp in coarse_timestamps:

        #selected_logger = logger8
        #selected_coarse_timestamp = 1528419200

        print(">--------- station", selected_logger.logger_id, "@ time", selected_coarse_timestamp, ":")
        events, non_events = extract_events_data_from_dataset(selected_logger, selected_coarse_timestamp, strong_coherence_threshold, weak_coherence_threshold)
        visualize_events_nonevents(selected_logger, selected_coarse_timestamp, events, non_events, show=False, save=plots_directory)

        save_audio_samples(events, events_directory)
        save_audio_samples(non_events, nonevents_directory)

        for e in events: total_events.append(e)
        for ne in non_events: total_non_events.append(ne)


print("------ Final state ------")
print("We have in total", len(total_events), "salient samples.")
print("And", len(total_non_events), "samples without events.")

# ------ Final state ------
# We have in total 105 salient samples.
# And 268 samples without events.